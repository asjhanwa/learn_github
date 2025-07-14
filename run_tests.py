#!/usr/bin/env python3
"""
Test runner for the LLM Evaluator package unit tests.

This script runs all unit tests and provides a comprehensive test report.
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all test modules
from tests.test_models import *
from tests.test_metrics import *
from tests.test_utils import *
from tests.test_integrations import *
from tests.test_core import *
from tests.test_call_aoai import *


def run_all_tests():
    """Run all unit tests and provide a comprehensive report."""
    print("="*80)
    print("LLM Evaluator Package - Unit Test Suite")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test modules
    test_modules = [
        'tests.test_models',
        'tests.test_metrics', 
        'tests.test_utils',
        'tests.test_integrations',
        'tests.test_core',
        'tests.test_call_aoai'
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            suite.addTests(loader.loadTestsFromModule(module))
            print(f"✓ Loaded tests from {module_name}")
        except Exception as e:
            print(f"✗ Failed to load tests from {module_name}: {e}")
    
    print(f"\nTotal tests loaded: {suite.countTestCases()}")
    print("-"*80)
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if result.failures:
        print(f"\n❌ {len(result.failures)} test(s) failed")
        
    if result.errors:
        print(f"\n💥 {len(result.errors)} test(s) had errors")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
        return False


def run_specific_test_module(module_name):
    """Run tests for a specific module."""
    print(f"Running tests for {module_name}")
    print("-"*60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{module_name}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test_class(module_name, class_name):
    """Run tests for a specific test class."""
    print(f"Running tests for {module_name}.{class_name}")
    print("-"*60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{module_name}.{class_name}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) == 1:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
    
    elif len(sys.argv) == 2:
        # Run specific module
        module_name = sys.argv[1]
        success = run_specific_test_module(module_name)
        sys.exit(0 if success else 1)
    
    elif len(sys.argv) == 3:
        # Run specific class
        module_name = sys.argv[1]
        class_name = sys.argv[2]
        success = run_specific_test_class(module_name, class_name)
        sys.exit(0 if success else 1)
    
    else:
        print("Usage:")
        print("  python run_tests.py                    # Run all tests")
        print("  python run_tests.py test_models        # Run specific module")
        print("  python run_tests.py test_models TestMetricScore  # Run specific class")
        sys.exit(1)


if __name__ == '__main__':
    main()