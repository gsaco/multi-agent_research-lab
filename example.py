#!/usr/bin/env python3
"""
Simple Example - Multi-Agent Research Lab
A minimal example showing how to use the research agents.
"""

import os
import sys

# Add src to path
sys.path.insert(0, 'src')

from agents import run_research_workflow


def main():
    """Run a simple research workflow example."""
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠ Warning: OPENAI_API_KEY not found.")
        print("Please set it with: export OPENAI_API_KEY='sk-your-key-here'")
        print("\nContinuing with placeholder for demonstration...")
        os.environ["OPENAI_API_KEY"] = "sk-placeholder"
    
    # Define research topic
    topic = "Impact of Synthetic Data in Healthcare"
    
    # Allow user to specify custom topic
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    
    print(f"\n{'='*60}")
    print(f"Starting Research Workflow")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")
    
    try:
        # Run the workflow
        result = run_research_workflow(
            topic=topic,
            output_file="research_summary.md"
        )
        
        print(f"\n{'='*60}")
        print("✅ Workflow completed successfully!")
        print(f"{'='*60}\n")
        
        print("Output saved to: research_summary.md")
        print("\nTo view the summary:")
        print("  cat research_summary.md")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ Workflow failed!")
        print(f"{'='*60}\n")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure OPENAI_API_KEY is set correctly")
        print("  2. Check your internet connection")
        print("  3. Run 'python test_setup.py' to verify setup")
        print("  4. See CONFIGURATION.md for detailed setup instructions")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
