#!/usr/bin/env python3
"""
Quick installation verification for the LLM Evaluator package.
Run this after installing to ensure everything is working correctly.
"""

def verify_installation():
    """Verify the LLM Evaluator package is properly installed."""
    
    print("=" * 60)
    print("LLM Evaluator Package Installation Verification")
    print("=" * 60)
    
    try:
        # Test basic imports
        print("1. Testing imports...")
        from llm_evaluator import LLMEvaluator, EvaluationInput, ModelConfig
        from call_aoai import create_llm_evaluator
        print("   ✓ All imports successful")
        
        # Test package info
        print("\n2. Testing package info...")
        import llm_evaluator
        print(f"   ✓ Package version: {llm_evaluator.__version__}")
        
        # Test basic functionality
        print("\n3. Testing basic functionality...")
        
        # Create a sample evaluation input
        eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Python?"}],
            retrieved_chunks=["Python is a programming language."],
            llm_answer="Python is a high-level programming language known for its simplicity.",
            user_feedback_label="Positive",
            user_feedback_comment="Good explanation"
        )
        
        # Create a model config
        config = ModelConfig(
            model_name="gpt-4",
            temperature=0.0,
            max_tokens=1000
        )
        
        print("   ✓ Data models work correctly")
        
        # Test metrics framework
        from llm_evaluator.metrics import MetricsFramework
        framework = MetricsFramework()
        metrics = framework.get_all_metrics()
        print(f"   ✓ Metrics framework loaded ({len(metrics)} metrics)")
        
        # Test utilities
        from llm_evaluator.utils import AnalysisUtils, ExportUtils
        print("   ✓ Utility classes available")
        
        print("\n4. Testing convenience functions...")
        
        # Test the enhanced call_aoai function
        from call_aoai import send_to_azure_openai
        print("   ✓ Enhanced Azure OpenAI function available")
        
        print("\n" + "=" * 60)
        print("✅ INSTALLATION VERIFIED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Set up your API credentials for Azure OpenAI and Claude")
        print("2. See example_usage.py for comprehensive examples")
        print("3. Run test_package.py for full functionality tests")
        print("4. Check README.md for detailed documentation")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've installed the requirements: pip install -r requirements.txt")
        print("2. Check that you're in the correct directory")
        print("3. Verify Python version (3.8+ required)")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\nPlease check the installation and try again")
        return False


if __name__ == "__main__":
    import sys
    success = verify_installation()
    sys.exit(0 if success else 1)