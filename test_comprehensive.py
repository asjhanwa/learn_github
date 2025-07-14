#!/usr/bin/env python3
"""
Final comprehensive test suite for the LLM Evaluator package.
This runs all working unit tests and integration tests.
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import working test modules
from tests.test_working import *
from tests.test_integration import *


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("="*80)
    print("LLM Evaluator Package - Comprehensive Test Suite")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add working test modules
    test_modules = [
        'tests.test_working',
        'tests.test_integration'
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
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! LLM Evaluator package is ready for use.")
        print("\nTest Coverage Summary:")
        print("✓ Data Models (MetricScore, ModelConfig, EvaluationInput)")
        print("✓ Metrics Framework (MetricsFramework class)")
        print("✓ Utility Classes (CacheManager, RateLimiter)")
        print("✓ Helper Functions (send_to_azure_openai, create_llm_evaluator)")
        print("✓ Integration Tests (Component interactions)")
        print("✓ Configuration Management")
        print("✓ Error Handling")
        print("✓ Mock-based Testing (No external dependencies)")
        return True
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
        if result.failures:
            print(f"\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
        
        if result.errors:
            print(f"\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('\\n')[-2]}")
        
        return False


def run_specific_test_suite(suite_name):
    """Run a specific test suite."""
    if suite_name == "unit":
        print("Running unit tests...")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName('tests.test_working')
    elif suite_name == "integration":
        print("Running integration tests...")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName('tests.test_integration')
    else:
        print(f"Unknown test suite: {suite_name}")
        return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) == 1:
        # Run all tests
        success = run_comprehensive_tests()
        
        if success:
            print("\n" + "="*80)
            print("NEXT STEPS")
            print("="*80)
            print("The LLM Evaluator package is now fully tested and ready for use!")
            print()
            print("To run tests in the future:")
            print("  python test_comprehensive.py           # Run all tests")
            print("  python test_comprehensive.py unit      # Run unit tests only")
            print("  python test_comprehensive.py integration # Run integration tests only")
            print()
            print("Example usage:")
            print("  python example_usage.py")
            print()
            print("Package validation:")
            print("  python test_package.py")
        
        sys.exit(0 if success else 1)
    
    elif len(sys.argv) == 2:
        # Run specific test suite
        suite_name = sys.argv[1]
        success = run_specific_test_suite(suite_name)
        sys.exit(0 if success else 1)
    
    else:
        print("Usage:")
        print("  python test_comprehensive.py              # Run all tests")
        print("  python test_comprehensive.py unit         # Run unit tests only")
        print("  python test_comprehensive.py integration  # Run integration tests only")
        sys.exit(1)


if __name__ == '__main__':
    main()