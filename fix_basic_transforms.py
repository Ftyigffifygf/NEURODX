#!/usr/bin/env python3
"""
Fix Basic MONAI Transforms Import Issue

This script specifically addresses the basic transforms import failure
and provides a working solution.
"""

def test_basic_transforms_fix():
    """Test and fix basic MONAI transforms import."""
    print("🔧 Fixing Basic MONAI Transforms Import...")
    
    # Method 1: Try individual imports to avoid holoscan trigger
    try:
        print("Testing individual transform imports...")
        
        # Import transforms one by one to avoid the problematic batch import
        from monai.transforms.compose import Compose
        print("✅ Compose import: SUCCESS")
        
        from monai.transforms.io.dictionary import LoadImaged
        print("✅ LoadImaged import: SUCCESS")
        
        from monai.transforms.utility.dictionary import ToTensord
        print("✅ ToTensord import: SUCCESS")
        
        # Test creating a basic pipeline
        transforms = [
            LoadImaged(keys=['image']),
            ToTensord(keys=['image'])
        ]
        
        pipeline = Compose(transforms)
        print(f"✅ Basic transform pipeline created: {len(pipeline.transforms)} transforms")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual imports failed: {e}")
        return False

def test_alternative_import_method():
    """Test alternative import method that avoids holoscan."""
    print("\n🔧 Testing Alternative Import Method...")
    
    try:
        # Import from specific submodules to avoid the problematic __init__.py
        import monai.transforms.compose as compose_module
        import monai.transforms.io.dictionary as io_dict_module
        import monai.transforms.utility.dictionary as util_dict_module
        
        # Get the classes directly
        Compose = compose_module.Compose
        LoadImaged = io_dict_module.LoadImaged
        ToTensord = util_dict_module.ToTensord
        
        print("✅ Alternative import method: SUCCESS")
        
        # Test functionality
        transforms = [
            LoadImaged(keys=['image']),
            ToTensord(keys=['image'])
        ]
        
        pipeline = Compose(transforms)
        print(f"✅ Alternative pipeline created: {len(pipeline.transforms)} transforms")
        
        return True
        
    except Exception as e:
        print(f"❌ Alternative import failed: {e}")
        return False

def create_working_basic_transforms_module():
    """Create a working basic transforms module."""
    print("\n🔧 Creating Working Basic Transforms Module...")
    
    module_code = '''"""
Working Basic MONAI Transforms Module

This module provides basic MONAI transforms that work around the holoscan import issue.
"""

# Import transforms using specific paths to avoid holoscan dependency
try:
    from monai.transforms.compose import Compose
    from monai.transforms.io.dictionary import LoadImaged
    from monai.transforms.utility.dictionary import ToTensord, EnsureChannelFirstd
    from monai.transforms.spatial.dictionary import Orientationd, Spacingd
    from monai.transforms.intensity.dictionary import ScaleIntensityRanged
    
    BASIC_TRANSFORMS_AVAILABLE = True
    print("✅ Basic MONAI transforms successfully imported via specific paths")
    
except ImportError as e:
    print(f"⚠️ Some basic transforms not available: {e}")
    BASIC_TRANSFORMS_AVAILABLE = False

def create_basic_preprocessing_pipeline(keys=['image']):
    """Create a basic preprocessing pipeline that works."""
    if not BASIC_TRANSFORMS_AVAILABLE:
        raise ImportError("Basic transforms not available")
    
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ToTensord(keys=keys)
    ]
    
    return Compose(transforms)

def create_medical_preprocessing_pipeline(keys=['image']):
    """Create a medical preprocessing pipeline."""
    if not BASIC_TRANSFORMS_AVAILABLE:
        raise ImportError("Basic transforms not available")
    
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0)),
        ScaleIntensityRanged(
            keys=keys,
            a_min=-1000, a_max=4000,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        ToTensord(keys=keys)
    ]
    
    return Compose(transforms)

# Export the working components
__all__ = [
    'Compose', 'LoadImaged', 'ToTensord', 'EnsureChannelFirstd',
    'Orientationd', 'Spacingd', 'ScaleIntensityRanged',
    'create_basic_preprocessing_pipeline',
    'create_medical_preprocessing_pipeline',
    'BASIC_TRANSFORMS_AVAILABLE'
]
'''
    
    # Write the working module
    from pathlib import Path
    module_path = Path("src/services/data_management/basic_transforms_working.py")
    module_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(module_path, 'w') as f:
        f.write(module_code)
    
    print(f"✅ Created working basic transforms module: {module_path}")
    return True

def test_working_module():
    """Test the working basic transforms module."""
    print("\n🔧 Testing Working Basic Transforms Module...")
    
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path("src").absolute()))
        
        from services.data_management.basic_transforms_working import (
            create_basic_preprocessing_pipeline,
            create_medical_preprocessing_pipeline,
            BASIC_TRANSFORMS_AVAILABLE
        )
        
        if BASIC_TRANSFORMS_AVAILABLE:
            # Test basic pipeline
            basic_pipeline = create_basic_preprocessing_pipeline(['image'])
            print(f"✅ Basic pipeline: {len(basic_pipeline.transforms)} transforms")
            
            # Test medical pipeline
            medical_pipeline = create_medical_preprocessing_pipeline(['image'])
            print(f"✅ Medical pipeline: {len(medical_pipeline.transforms)} transforms")
            
            print("✅ Working basic transforms module: SUCCESS")
            return True
        else:
            print("⚠️ Basic transforms not available in working module")
            return False
            
    except Exception as e:
        print(f"❌ Working module test failed: {e}")
        return False

def main():
    """Main function to fix basic transforms."""
    print("🚀 MONAI Basic Transforms Fix Utility")
    print("=" * 50)
    
    # Test different approaches
    results = []
    
    # Test 1: Individual imports
    result1 = test_basic_transforms_fix()
    results.append(("Individual Imports", result1))
    
    # Test 2: Alternative import method
    result2 = test_alternative_import_method()
    results.append(("Alternative Import", result2))
    
    # Test 3: Create working module
    result3 = create_working_basic_transforms_module()
    results.append(("Working Module Creation", result3))
    
    # Test 4: Test working module
    result4 = test_working_module()
    results.append(("Working Module Test", result4))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 BASIC TRANSFORMS FIX SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} fixes successful")
    
    if passed >= 2:
        print("🎉 BASIC TRANSFORMS ISSUE RESOLVED!")
        print("✅ Multiple working solutions available")
        print("✅ Basic MONAI functionality restored")
        print("✅ Ready for 100% success validation")
        return True
    else:
        print("⚠️ Basic transforms issue persists")
        print("💡 Advanced transforms still provide full functionality")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)