"""
Main Test Runner for AI Service
Convenience script to run tests from the main AI directory
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the test suite"""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    tests_dir = script_dir / "tests"
    
    # Check if tests directory exists
    if not tests_dir.exists():
        print("[FAIL] Tests directory not found!")
        print(f"Expected: {tests_dir}")
        return 1
    
    # Change to tests directory and run the test runner
    original_dir = os.getcwd()
    
    try:
        os.chdir(tests_dir)
        print(f"TEST: Running tests from: {tests_dir}")
        print("=" * 60)
        
        # Run the interactive test runner
        result = subprocess.run([sys.executable, "run_tests.py"], 
                              cwd=tests_dir)
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n[BYE] Test runner interrupted")
        return 0
    except Exception as e:
        print(f"[FAIL] Error running tests: {e}")
        return 1
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main())
