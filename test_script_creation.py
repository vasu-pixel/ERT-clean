#!/usr/bin/env python3
"""
Test script to verify that the create_training_script method works correctly
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from finetune_ollama import OllamaLoRAFineTuner

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_script_creation.log')
    ]
)
logger = logging.getLogger(__name__)

def test_script_creation():
    """Test the create_training_script method in isolation"""

    print("🧪 Testing Script Creation")
    print("=" * 40)

    try:
        # Create finetuner instance
        logger.info("Creating OllamaLoRAFineTuner instance...")
        finetuner = OllamaLoRAFineTuner()

        # Ensure we have a clean test environment
        logger.info("Setting up test environment...")
        test_output_dir = Path("test_finetuned_models")
        finetuner.output_dir = test_output_dir

        # Clean up any existing test files
        test_script_path = test_output_dir / "train_lora.sh"
        if test_script_path.exists():
            logger.info(f"Removing existing test script: {test_script_path}")
            test_script_path.unlink()

        # Test directory creation and permissions
        logger.info("Testing directory creation...")
        test_output_dir.mkdir(exist_ok=True)
        logger.info(f"Test directory created: {test_output_dir}")

        # Test the create_training_script method
        logger.info("Calling create_training_script()...")
        script_path = finetuner.create_training_script()

        # Verify results
        logger.info("Verifying results...")
        script_path_obj = Path(script_path)

        print(f"\n📊 Test Results:")
        print(f"  Script path returned: {script_path}")
        print(f"  File exists: {script_path_obj.exists()}")
        print(f"  File size: {script_path_obj.stat().st_size if script_path_obj.exists() else 'N/A'} bytes")
        print(f"  Is executable: {oct(script_path_obj.stat().st_mode)[-3:] if script_path_obj.exists() else 'N/A'}")

        if script_path_obj.exists():
            print(f"  ✅ SUCCESS: Script created successfully!")

            # Show first few lines of the script
            with open(script_path_obj, 'r') as f:
                first_lines = f.readlines()[:10]

            print(f"\n📝 Script Preview (first 10 lines):")
            for i, line in enumerate(first_lines, 1):
                print(f"  {i:2d}: {line.rstrip()}")

        else:
            print(f"  ❌ FAILED: Script was not created!")

        # Check directory contents
        print(f"\n📁 Directory contents ({test_output_dir}):")
        for item in test_output_dir.iterdir():
            print(f"  - {item.name} ({item.stat().st_size} bytes)")

        return script_path_obj.exists()

    except Exception as e:
        logger.error(f"Test failed with exception: {type(e).__name__}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"❌ Test failed with exception: {e}")
        return False

def test_permissions():
    """Test directory permissions specifically"""

    print("\n🔐 Testing Directory Permissions")
    print("=" * 40)

    try:
        test_dir = Path("test_permissions")
        test_dir.mkdir(exist_ok=True)

        # Test write permission
        test_file = test_dir / "test.txt"
        with open(test_file, 'w') as f:
            f.write("test content")

        print(f"✅ Write permission: OK")

        # Test read permission
        with open(test_file, 'r') as f:
            content = f.read()

        print(f"✅ Read permission: OK")

        # Test execute permission
        test_file.chmod(0o755)
        print(f"✅ Execute permission: OK")

        # Cleanup
        test_file.unlink()
        test_dir.rmdir()

        return True

    except Exception as e:
        print(f"❌ Permission test failed: {e}")
        return False

def main():
    """Run all tests"""

    print("🚀 Starting Script Creation Tests")
    print("=" * 50)

    # Test 1: Directory permissions
    perm_test = test_permissions()

    # Test 2: Script creation
    script_test = test_script_creation()

    print("\n📋 Test Summary:")
    print(f"  Permission Test: {'✅ PASS' if perm_test else '❌ FAIL'}")
    print(f"  Script Creation Test: {'✅ PASS' if script_test else '❌ FAIL'}")

    if perm_test and script_test:
        print("\n🎉 All tests passed! The script creation should work now.")
    else:
        print("\n❌ Some tests failed. Check the logs for details.")

    print(f"\n📝 Detailed logs saved to: test_script_creation.log")

if __name__ == "__main__":
    import traceback
    main()