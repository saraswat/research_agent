"""
Test runner for research agent unit tests.
"""
import unittest
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from tests.test_information_extraction_tool import TestInformationExtractionTool
from tests.test_citation_generation_tool import TestCitationGenerationTool


def run_tests():
    """Run all unit tests for the research agent."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases to the suite
    test_suite.addTest(unittest.makeSuite(TestInformationExtractionTool))
    test_suite.addTest(unittest.makeSuite(TestCitationGenerationTool))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)