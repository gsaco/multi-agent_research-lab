#!/usr/bin/env python3
"""
Quick Setup Test Script
Tests that all dependencies are installed and configured correctly.
"""

import sys
import os


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    tests = []
    
    try:
        import crewai
        tests.append(("✓", "crewai"))
    except ImportError as e:
        tests.append(("✗", f"crewai: {e}"))
    
    try:
        import langchain
        tests.append(("✓", "langchain"))
    except ImportError as e:
        tests.append(("✗", f"langchain: {e}"))
    
    try:
        import langchain_community
        tests.append(("✓", "langchain_community"))
    except ImportError as e:
        tests.append(("✗", f"langchain_community: {e}"))
    
    try:
        import huggingface_hub
        tests.append(("✓", "huggingface_hub"))
    except ImportError as e:
        tests.append(("✗", f"huggingface_hub: {e}"))
    
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        tests.append(("✓", "DuckDuckGoSearchRun"))
    except ImportError as e:
        tests.append(("✗", f"DuckDuckGoSearchRun: {e}"))
    
    try:
        import crewai_tools
        tests.append(("✓", "crewai_tools"))
    except ImportError as e:
        tests.append(("✗", f"crewai_tools: {e}"))
    
    for status, name in tests:
        print(f"  {status} {name}")
    
    failed = [t for t in tests if t[0] == "✗"]
    return len(failed) == 0


def test_configuration():
    """Test that required environment variables are set."""
    print("\nTesting configuration...")
    
    # Check OpenAI API key
    if os.environ.get("OPENAI_API_KEY"):
        key = os.environ.get("OPENAI_API_KEY")
        if key.startswith("sk-") and len(key) > 20:
            print("  ✓ OPENAI_API_KEY is set and looks valid")
        else:
            print("  ⚠ OPENAI_API_KEY is set but may be invalid")
    else:
        print("  ✗ OPENAI_API_KEY is not set (REQUIRED for execution)")
        print("    Get one at: https://platform.openai.com/api-keys")
        print("    Set it with: export OPENAI_API_KEY='sk-your-key-here'")
    
    # Check Hugging Face token
    if os.environ.get("HF_TOKEN"):
        token = os.environ.get("HF_TOKEN")
        if token.startswith("hf_") and len(token) > 20:
            print("  ✓ HF_TOKEN is set and looks valid")
        else:
            print("  ⚠ HF_TOKEN is set but may be invalid")
    else:
        print("  ⚠ HF_TOKEN is not set (optional but recommended)")
        print("    Get one at: https://huggingface.co/settings/tokens")
    
    return os.environ.get("OPENAI_API_KEY") is not None


def test_agents():
    """Test that agents can be created."""
    print("\nTesting agent creation...")
    
    try:
        sys.path.insert(0, 'src')
        from agents import ResearchAgents
        
        factory = ResearchAgents()
        print("  ✓ ResearchAgents factory created")
        
        researcher = factory.create_researcher()
        print(f"  ✓ Researcher created: {researcher.role}")
        
        writer = factory.create_writer()
        print(f"  ✓ Writer created: {writer.role}")
        
        reviewer = factory.create_reviewer()
        print(f"  ✓ Reviewer created: {reviewer.role}")
        
        # Test crew creation
        crew = factory.create_crew("Test Topic")
        print(f"  ✓ Crew created with {len(crew.agents)} agents")
        
        return True
    except Exception as e:
        print(f"  ✗ Agent creation failed: {e}")
        return False


def test_search_tool():
    """Test that the search tool works."""
    print("\nTesting search functionality...")
    
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        # Just test that it can be instantiated
        print("  ✓ Search tool initialized")
        return True
    except Exception as e:
        print(f"  ✗ Search tool failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Multi-Agent Research Lab - Setup Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("Agent Creation", test_agents()))
    results.append(("Search Tool", test_search_tool()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Run: python src/agents.py 'Your Research Topic'")
        print("  2. Or open: notebooks/workflow_demo.ipynb")
    else:
        print("⚠ Some tests failed. Please check the output above.")
        print("\nFor help:")
        print("  - See CONFIGURATION.md for setup instructions")
        print("  - See README.md for usage examples")
        print("  - Check requirements.txt and run: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
