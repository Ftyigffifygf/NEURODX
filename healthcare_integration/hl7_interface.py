"""
HL7 interface for legacy healthcare system integration.

This module provides HL7 v2.x message parsing, generation, and bidirectional
communication capabilities for integrating with legacy hospital systems.
"""

import logging
import socket
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import re

from src.models.patient import PatientRecord, ImagingStudy
from src.models.diagnostics import DiagnosticResult


@dataclass
class HL7Config:
    """Configuration for HL7 interface."""
    host: str = "localhost"
    port: int = 2575
    encoding: str = "utf-8"
    timeout: int = 30
    max_connections: int = 10
    message_separator: str = "\x0b"  # VT (Vertical Tab)
    message_terminator: str = "\x1c\x0d"  # FS + CR
    field_separator: str = "|"
    component_separator: str = "^"
    repetition_separator: str = "~"
    escape_character: str = "\\"
    subcomponent_separator: str = "&"


class HL7ValidationError(Exception):
    """Raised when HL7 message validation fails."""
    pass


class HL7CommunicationError(Exception):
    """Raised when HL7 communication fails."""
    pass


class HL7Message:
    """
    HL7 v2.x message parser and builder.
    
    Handles parsing and generation of HL7 messages with proper field separation
    and validation according to HL7 v2.x standards.
    """
    
    def __init__(self, message_text: str = "", config: Optional[HL7Config] = None):
        """
        Initialize HL7 message.
        
        Args:
            message_text: Raw HL7 message text
            config: HL7 configuration
        """
        self.config = config or HL7Config()
        self.segments = []
        self.message_type = ""
        self.control_id = ""
        self.timestamp = datetime.now()
        
        if message_text:
            self.parse(message_text)
    
    def parse(self, message_text: str) -> None:
        """Parse HL7 message text into segments."""
        # Clean message text
        message_text = message_text.strip()
        message_text = message_text.replace(self.config.message_separator, "")
        message_text = message_text.replace(self.config.message_terminator, "")
        
        # Split into segments
        segment_lines = message_text.split('\r')
        self.segments = []
        
        for line in segment_lines:
            line = line.strip()
            if line:
                segment = self._parse_segment(line)
                self.segments.append(segment)
        
        # Extract message type and control ID from MSH segment
        msh_segment = self.get_segment("MSH")
        if msh_segment:
            self.message_type = self._get_field_value(msh_segment, 9, 1)  # MSH.9.1
            self.control_id = self._get_field_value(msh_segment, 10)      # MSH.10
    
    def _parse_segment(self, segment_line: str) -> Dict[str, Any]:
        """Parse individual HL7 segment."""
        if not segment_line:
            raise HL7ValidationError("Empty segment line")
        
        # Handle MSH segment specially (field separator is part of the segment)
        if segment_line.startswith("MSH"):
            segment_id = "MSH"
            # MSH segment: MSH|^~\&|...
            fields = segment_line[3:].split(self.config.field_separator)
            # Insert encoding characters as field 1
            fields.insert(0, segment_line[3:8])  # |^~\&
        else:
            parts = segment_line.split(self.config.field_separator)
            segment_id = parts[0]
            fields = parts[1:] if len(parts) > 1 else []
        
        return {
            "id": segment_id,
            "fields": fields
        }
    
    def _get_field_value(self, segment: Dict[str, Any], field_num: int, component_num: int = 1) -> str:
        """Get field value from segment."""
        try:
            if field_num <= len(segment["fields"]):
                field_value = segment["fields"][field_num - 1]
                
                if component_num > 1:
                    components = field_value.split(self.config.component_separator)
                    if component_num <= len(components):
                        return components[component_num - 1]
                    return ""
                
                return field_value
            return ""
        except (IndexError, KeyError):
            return ""
    
    def get_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """Get first segment with specified ID."""
        for segment in self.segments:
            if segment["id"] == segment_id:
                return segment
        return None
    
    def get_segments(self, segment_id: str) -> List[Dict[str, Any]]:
        """Get all segments with specified ID."""
        return [seg for seg in self.segments if seg["id"] == segment_id]
    
    def add_segment(self, segment_id: str, fields: List[str]) -> None:
        """Add segment to message."""
        segment = {
            "id": segment_id,
            "fields": fields
        }
        self.segments.append(segment)
    
    def to_string(self) -> str:
        """Convert message to HL7 string format."""
        lines = []
        
        for segment in self.segments:
            if segment["id"] == "MSH":
                # MSH segment special handling
                line = "MSH" + self.config.field_separator
                line += self.config.field_separator.join(segment["fields"])
            else:
                line = segment["id"]
                if segment["fields"]:
                    line += self.config.field_separator + self.config.field_separator.join(segment["fields"])
            
            lines.append(line)
        
        message_text = '\r'.join(lines)
        return f"{self.config.message_separator}{message_text}{self.config.message_terminator}"
    
    def validate(self) -> bool:
        """Validate HL7 message structure."""
        # Check for required MSH segment
        msh_segment = self.get_segment("MSH")
        if not msh_segment:
            raise HL7ValidationError("Missing required MSH segment")
        
        # Validate MSH fields
        required_msh_fields = [1, 2, 3, 4, 7, 9, 10, 11, 12]  # Key MSH fields
        for field_num in required_msh_fields:
            if not self._get_field_value(msh_segment, field_num):
                raise HL7ValidationError(f"Missing required MSH field {field_num}")
        
        return True


class HL7Interface:
    """
    HL7 interface for bidirectional communication with legacy hospital systems.
    
    Provides server and client capabilities for HL7 message exchange,
    including message parsing, validation, and response generation.
    """
    
    def __init__(self, config: HL7Config):
        """
        Initialize HL7 interface.
        
        Args:
            config: HL7 configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.server_socket = None
        self.is_running = False
        self.message_handlers: Dict[str, Callable] = {}
        self.client_connections = []
        
        # Register default message handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default message type handlers."""
        self.register_handler("ADT", self._handle_adt_message)
        self.register_handler("ORM", self._handle_orm_message)
        self.register_handler("ORU", self._handle_oru_message)
        self.register_handler("QRY", self._handle_qry_message)
    
    def register_handler(self, message_type: str, handler: Callable[[HL7Message], HL7Message]) -> None:
        """
        Register message type handler.
        
        Args:
            message_type: HL7 message type (e.g., "ADT", "ORM")
            handler: Handler function that takes HL7Message and returns response HL7Message
        """
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for message type: {message_type}")
    
    def start_server(self) -> None:
        """Start HL7 server to listen for incoming messages."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.config.host, self.config.port))
            self.server_socket.listen(self.config.max_connections)
            
            self.is_running = True
            self.logger.info(f"HL7 server started on {self.config.host}:{self.config.port}")
            
            # Start server thread
            server_thread = threading.Thread(target=self._server_loop, daemon=True)
            server_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start HL7 server: {e}")
            raise HL7CommunicationError(f"Server startup failed: {e}")
    
    def stop_server(self) -> None:
        """Stop HL7 server."""
        self.is_running = False
        
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        
        # Close client connections
        for conn in self.client_connections:
            try:
                conn.close()
            except:
                pass
        self.client_connections.clear()
        
        self.logger.info("HL7 server stopped")
    
    def _server_loop(self) -> None:
        """Main server loop to handle incoming connections."""
        while self.is_running:
            try:
                if self.server_socket:
                    client_socket, address = self.server_socket.accept()
                    self.client_connections.append(client_socket)
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Server loop error: {e}")
                    time.sleep(1)
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple[str, int]) -> None:
        """Handle individual client connection."""
        self.logger.info(f"Client connected from {address}")
        
        try:
            while self.is_running:
                # Receive message
                data = client_socket.recv(4096)
                if not data:
                    break
                
                message_text = data.decode(self.config.encoding)
                self.logger.debug(f"Received message from {address}: {message_text[:100]}...")
                
                # Process message
                try:
                    response = self.process_message(message_text)
                    if response:
                        client_socket.send(response.encode(self.config.encoding))
                        
                except Exception as e:
                    self.logger.error(f"Error processing message from {address}: {e}")
                    # Send NAK response
                    nak_response = self._create_nak_response(str(e))
                    client_socket.send(nak_response.encode(self.config.encoding))
                
        except Exception as e:
            self.logger.error(f"Client handler error for {address}: {e}")
        finally:
            client_socket.close()
            if client_socket in self.client_connections:
                self.client_connections.remove(client_socket)
            self.logger.info(f"Client {address} disconnected")
    
    def process_message(self, message_text: str) -> Optional[str]:
        """
        Process incoming HL7 message and generate response.
        
        Args:
            message_text: Raw HL7 message text
            
        Returns:
            Response message text or None
        """
        try:
            # Parse message
            message = HL7Message(message_text, self.config)
            message.validate()
            
            # Get message type
            message_type = message.message_type.split("^")[0] if message.message_type else ""
            
            # Find handler
            handler = self.message_handlers.get(message_type)
            if handler:
                response_message = handler(message)
                if response_message:
                    return response_message.to_string()
            else:
                self.logger.warning(f"No handler for message type: {message_type}")
                return self._create_ack_response(message, "AR", f"Unsupported message type: {message_type}")
            
            # Default ACK response
            return self._create_ack_response(message, "AA", "Message processed successfully")
            
        except HL7ValidationError as e:
            self.logger.error(f"Message validation error: {e}")
            return self._create_nak_response(str(e))
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            return self._create_nak_response(str(e))
    
    def send_message(self, host: str, port: int, message: HL7Message) -> Optional[HL7Message]:
        """
        Send HL7 message to remote system.
        
        Args:
            host: Target host
            port: Target port
            message: HL7 message to send
            
        Returns:
            Response message or None
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config.timeout)
                sock.connect((host, port))
                
                # Send message
                message_text = message.to_string()
                sock.send(message_text.encode(self.config.encoding))
                
                # Receive response
                response_data = sock.recv(4096)
                if response_data:
                    response_text = response_data.decode(self.config.encoding)
                    return HL7Message(response_text, self.config)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to send message to {host}:{port}: {e}")
            raise HL7CommunicationError(f"Message send failed: {e}")
    
    def _create_ack_response(self, original_message: HL7Message, ack_code: str, text_message: str) -> str:
        """Create ACK response message."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        control_id = f"ACK{int(time.time())}"
        
        # MSH segment
        msh_fields = [
            "^~\\&",  # Encoding characters
            "NeuroDx-MultiModal",  # Sending application
            "AI-System",  # Sending facility
            "Hospital-System",  # Receiving application
            "Legacy-HIS",  # Receiving facility
            timestamp,  # Date/time of message
            "",  # Security
            "ACK^ACK^ACK",  # Message type
            control_id,  # Message control ID
            "P",  # Processing ID
            "2.5"  # Version ID
        ]
        
        # MSA segment
        msa_fields = [
            ack_code,  # Acknowledgment code
            original_message.control_id,  # Message control ID
            text_message  # Text message
        ]
        
        ack_message = HL7Message(config=self.config)
        ack_message.add_segment("MSH", msh_fields)
        ack_message.add_segment("MSA", msa_fields)
        
        return ack_message.to_string()
    
    def _create_nak_response(self, error_message: str) -> str:
        """Create NAK (negative acknowledgment) response."""
        return self._create_ack_response(
            HL7Message(config=self.config),
            "AE",
            f"Application Error: {error_message}"
        )
    
    def _handle_adt_message(self, message: HL7Message) -> Optional[HL7Message]:
        """Handle ADT (Admit, Discharge, Transfer) messages."""
        self.logger.info(f"Processing ADT message: {message.message_type}")
        
        # Extract patient information from PID segment
        pid_segment = message.get_segment("PID")
        if pid_segment:
            patient_id = self._get_field_value(pid_segment, 3)  # Patient ID
            patient_name = self._get_field_value(pid_segment, 5)  # Patient name
            
            self.logger.info(f"ADT message for patient: {patient_id} - {patient_name}")
            
            # Here you would typically update patient records
            # For now, just log the information
        
        return None  # Return None for default ACK
    
    def _handle_orm_message(self, message: HL7Message) -> Optional[HL7Message]:
        """Handle ORM (Order) messages."""
        self.logger.info(f"Processing ORM message: {message.message_type}")
        
        # Extract order information from ORC and OBR segments
        orc_segment = message.get_segment("ORC")
        obr_segment = message.get_segment("OBR")
        
        if orc_segment and obr_segment:
            order_control = self._get_field_value(orc_segment, 1)  # Order control
            order_id = self._get_field_value(orc_segment, 2)  # Placer order number
            procedure_code = self._get_field_value(obr_segment, 4)  # Universal service ID
            
            self.logger.info(f"Order {order_control}: {order_id} - {procedure_code}")
        
        return None  # Return None for default ACK
    
    def _handle_oru_message(self, message: HL7Message) -> Optional[HL7Message]:
        """Handle ORU (Observation Result) messages."""
        self.logger.info(f"Processing ORU message: {message.message_type}")
        
        # Extract observation results from OBX segments
        obx_segments = message.get_segments("OBX")
        
        for obx in obx_segments:
            observation_id = self._get_field_value(obx, 3)  # Observation identifier
            observation_value = self._get_field_value(obx, 5)  # Observation value
            
            self.logger.info(f"Observation: {observation_id} = {observation_value}")
        
        return None  # Return None for default ACK
    
    def _handle_qry_message(self, message: HL7Message) -> Optional[HL7Message]:
        """Handle QRY (Query) messages."""
        self.logger.info(f"Processing QRY message: {message.message_type}")
        
        # Extract query information from QRD segment
        qrd_segment = message.get_segment("QRD")
        if qrd_segment:
            query_date = self._get_field_value(qrd_segment, 1)  # Query date/time
            query_format = self._get_field_value(qrd_segment, 2)  # Query format code
            query_priority = self._get_field_value(qrd_segment, 3)  # Query priority
            
            self.logger.info(f"Query: {query_format} (Priority: {query_priority})")
        
        return None  # Return None for default ACK
    
    def create_diagnostic_result_message(self, diagnostic_result: DiagnosticResult, patient_id: str) -> HL7Message:
        """
        Create HL7 ORU message for diagnostic results.
        
        Args:
            diagnostic_result: NeuroDx diagnostic result
            patient_id: Patient identifier
            
        Returns:
            HL7 ORU message
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        control_id = f"ORU{int(time.time())}"
        
        # MSH segment
        msh_fields = [
            "^~\\&",  # Encoding characters
            "NeuroDx-MultiModal",  # Sending application
            "AI-System",  # Sending facility
            "Hospital-System",  # Receiving application
            "Legacy-HIS",  # Receiving facility
            timestamp,  # Date/time of message
            "",  # Security
            "ORU^R01^ORU_R01",  # Message type
            control_id,  # Message control ID
            "P",  # Processing ID
            "2.5"  # Version ID
        ]
        
        # PID segment (Patient Identification)
        pid_fields = [
            "1",  # Set ID
            "",  # Patient ID (external)
            patient_id,  # Patient identifier list
            "",  # Alternate patient ID
            "",  # Patient name (would be populated from patient record)
            "",  # Mother's maiden name
            "",  # Date/time of birth
            "",  # Administrative sex
            "",  # Patient alias
            "",  # Race
            "",  # Patient address
        ]
        
        # OBR segment (Observation Request)
        obr_fields = [
            "1",  # Set ID
            "",  # Placer order number
            "",  # Filler order number
            "NEURODX^NeuroDx Multi-Modal Analysis^L",  # Universal service identifier
            "",  # Priority
            "",  # Requested date/time
            diagnostic_result.timestamp.strftime("%Y%m%d%H%M%S"),  # Observation date/time
            "",  # Observation end date/time
            "",  # Collection volume
            "",  # Collector identifier
            "",  # Specimen action code
            "",  # Danger code
            "",  # Relevant clinical information
            "",  # Specimen received date/time
            "",  # Specimen source
            "AI-SYSTEM^NeuroDx AI System^L",  # Ordering provider
        ]
        
        message = HL7Message(config=self.config)
        message.add_segment("MSH", msh_fields)
        message.add_segment("PID", pid_fields)
        message.add_segment("OBR", obr_fields)
        
        # Add OBX segments for each classification result
        obx_set_id = 1
        for condition, probability in diagnostic_result.classification_probabilities.items():
            obx_fields = [
                str(obx_set_id),  # Set ID
                "NM",  # Value type (Numeric)
                f"PROB_{condition.upper().replace(' ', '_')}^{condition} Probability^L",  # Observation identifier
                "",  # Observation sub-ID
                str(probability),  # Observation value
                "%",  # Units
                "",  # References range
                "",  # Abnormal flags
                "",  # Probability
                "",  # Nature of abnormal test
                "F",  # Observation result status (Final)
                "",  # Effective date of reference range
                "",  # User defined access checks
                diagnostic_result.timestamp.strftime("%Y%m%d%H%M%S"),  # Date/time of observation
            ]
            message.add_segment("OBX", obx_fields)
            obx_set_id += 1
        
        # Add confidence scores as additional OBX segments
        for metric, score in diagnostic_result.confidence_scores.items():
            obx_fields = [
                str(obx_set_id),  # Set ID
                "NM",  # Value type (Numeric)
                f"CONF_{metric.upper()}^{metric} Confidence^L",  # Observation identifier
                "",  # Observation sub-ID
                str(score),  # Observation value
                "%",  # Units
                "",  # References range
                "",  # Abnormal flags
                "",  # Probability
                "",  # Nature of abnormal test
                "F",  # Observation result status (Final)
                "",  # Effective date of reference range
                "",  # User defined access checks
                diagnostic_result.timestamp.strftime("%Y%m%d%H%M%S"),  # Date/time of observation
            ]
            message.add_segment("OBX", obx_fields)
            obx_set_id += 1
        
        return message
    
    def test_connection(self, host: str, port: int) -> bool:
        """
        Test connection to HL7 system.
        
        Args:
            host: Target host
            port: Target port
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)  # Short timeout for test
                sock.connect((host, port))
                self.logger.info(f"HL7 connection test successful to {host}:{port}")
                return True
                
        except Exception as e:
            self.logger.error(f"HL7 connection test failed to {host}:{port}: {e}")
            return False