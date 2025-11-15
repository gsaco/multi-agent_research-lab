"""
Multi-Agent Research Lab
A collaborative AI research system using CrewAI, LangChain, and Hugging Face.
"""

from .agents import ResearchAgents, run_research_workflow

__version__ = "1.0.0"
__all__ = ["ResearchAgents", "run_research_workflow"]
