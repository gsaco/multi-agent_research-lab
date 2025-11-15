"""
Multi-Agent Research Lab - Agent Definitions
This module defines three agents for collaborative research:
1. Researcher Agent - Conducts information search
2. Writer Agent - Synthesizes research into a summary
3. Reviewer Agent - Evaluates and refines the summary
"""

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Type
from pydantic import BaseModel, Field
import os


class SearchInput(BaseModel):
    """Input schema for DuckDuckGoSearchTool."""
    query: str = Field(..., description="Search query string")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = (
        "Search the web using DuckDuckGo search engine. "
        "Useful for finding information, articles, and research papers."
    )
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str) -> str:
        """Execute the search."""
        search = DuckDuckGoSearchRun()
        try:
            results = search.run(query)
            return results
        except Exception as e:
            return f"Search error: {str(e)}"


class ResearchAgents:
    """Factory class for creating research agents"""
    
    def __init__(self, hf_token=None, llm_model="gpt-4o-mini"):
        """
        Initialize the research agents factory
        
        Args:
            hf_token: Hugging Face API token (optional, can be set via environment)
            llm_model: LLM model to use for agents (default: gpt-4o-mini)
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            print("Warning: No Hugging Face token provided. Set HF_TOKEN environment variable.")
        
        # Set default OpenAI API key if not present (for CrewAI compatibility)
        if not os.environ.get("OPENAI_API_KEY"):
            # For demonstration purposes, we set a placeholder
            # In production, users should provide their own API key
            os.environ["OPENAI_API_KEY"] = "sk-placeholder"
            print("Note: Using placeholder OpenAI API key. For actual execution, set OPENAI_API_KEY.")
        
        self.llm_model = llm_model
        
        # Initialize search tool
        self.search_tool = DuckDuckGoSearchTool()
    
    def create_researcher(self):
        """
        Create a Researcher Agent that conducts information search
        
        Returns:
            Agent: CrewAI Researcher agent
        """
        researcher = Agent(
            role="Research Specialist",
            goal="Find reliable and relevant web sources about AI topics, particularly focusing on "
                 "peer-reviewed content, academic papers, and reputable technology publications.",
            backstory="You are an expert research assistant with years of experience in gathering "
                      "scientific and technical information from various online sources. You excel "
                      "at identifying high-quality, authoritative sources and extracting key insights.",
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool],
            llm=self.llm_model
        )
        return researcher
    
    def create_writer(self):
        """
        Create a Writer Agent that synthesizes research into summaries
        
        Returns:
            Agent: CrewAI Writer agent
        """
        writer = Agent(
            role="Technical Writer",
            goal="Synthesize research findings into a clear, well-structured 500-word summary "
                 "in Markdown format with proper sections and citations.",
            backstory="You are a skilled technical writer with expertise in AI and machine learning. "
                      "You excel at transforming complex research findings into accessible, "
                      "well-organized summaries that maintain technical accuracy while being "
                      "readable to a broad audience.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm_model
        )
        return writer
    
    def create_reviewer(self):
        """
        Create a Reviewer Agent that evaluates and refines summaries
        
        Returns:
            Agent: CrewAI Reviewer agent
        """
        reviewer = Agent(
            role="Research Reviewer",
            goal="Evaluate the coherence, factual accuracy, and structure of research summaries, "
                 "providing constructive feedback and suggesting specific improvements.",
            backstory="You are a meticulous research reviewer with a keen eye for detail. "
                      "You have extensive experience in peer review and excel at identifying "
                      "logical inconsistencies, factual errors, and structural issues in "
                      "scientific writing.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm_model
        )
        return reviewer
    
    def create_research_task(self, agent, topic):
        """
        Create a research task for the Researcher agent
        
        Args:
            agent: The researcher agent
            topic: The research topic to investigate
            
        Returns:
            Task: CrewAI task for research
        """
        task = Task(
            description=f"""
            Conduct comprehensive research on the topic: "{topic}"
            
            Your task:
            1. Search for reliable sources including academic papers, research articles, 
               and reputable technology publications
            2. Focus on recent developments (last 3-5 years preferred)
            3. Extract key findings, statistics, and insights
            4. Identify main challenges, benefits, and ethical considerations
            5. Compile at least 5-7 key points with source information
            
            Use search queries targeting:
            - Academic sites (site:researchgate.net, site:arxiv.org)
            - Tech publications (site:medium.com, site:towardsdatascience.com)
            - Industry sources (site:openai.com, site:huggingface.co)
            
            Provide a detailed summary of your findings including URLs when possible.
            """,
            agent=agent,
            expected_output="A comprehensive research report with 5-7 key findings, including sources and URLs"
        )
        return task
    
    def create_writing_task(self, agent, topic):
        """
        Create a writing task for the Writer agent
        
        Args:
            agent: The writer agent
            topic: The research topic to write about
            
        Returns:
            Task: CrewAI task for writing
        """
        task = Task(
            description=f"""
            Write a comprehensive 500-word research summary on: "{topic}"
            
            Your task:
            1. Review the research findings provided by the Researcher
            2. Create a well-structured Markdown document with these sections:
               - Introduction (75-100 words)
               - Key Findings (200-250 words)
               - Ethical & Technical Challenges (100-125 words)
               - Conclusion (75-100 words)
            3. Ensure the content is:
               - Technically accurate
               - Well-organized and flows logically
               - Accessible to readers with basic AI knowledge
               - Properly formatted in Markdown
            4. Include specific examples and data points from the research
            
            Format requirements:
            - Use ## for section headers
            - Use bullet points for lists
            - Keep paragraphs concise (3-4 sentences)
            - Aim for approximately 500 words total
            """,
            agent=agent,
            expected_output="A well-structured 500-word Markdown summary with Introduction, Key Findings, "
                          "Ethical & Technical Challenges, and Conclusion sections"
        )
        return task
    
    def create_review_task(self, agent):
        """
        Create a review task for the Reviewer agent
        
        Args:
            agent: The reviewer agent
            
        Returns:
            Task: CrewAI task for reviewing
        """
        task = Task(
            description="""
            Review and evaluate the research summary provided by the Writer
            
            Your task:
            1. Evaluate the summary for:
               - Coherence and logical flow
               - Factual accuracy and consistency
               - Structural quality and organization
               - Completeness of required sections
               - Clarity and readability
            2. Provide specific, actionable feedback including:
               - What works well
               - What needs improvement
               - Specific suggestions for enhancements
               - Any factual inconsistencies to address
            3. Rate each aspect (1-5 scale):
               - Coherence
               - Factual accuracy
               - Structure
               - Completeness
               - Clarity
            4. Provide an overall assessment and recommendation
            
            Be constructive and specific in your feedback.
            """,
            agent=agent,
            expected_output="A detailed review with ratings (1-5) for coherence, factual accuracy, structure, "
                          "completeness, and clarity, along with specific improvement suggestions"
        )
        return task
    
    def create_crew(self, topic):
        """
        Create a complete crew with all agents and tasks
        
        Args:
            topic: The research topic
            
        Returns:
            Crew: A CrewAI crew ready to execute
        """
        # Create agents
        researcher = self.create_researcher()
        writer = self.create_writer()
        reviewer = self.create_reviewer()
        
        # Create tasks
        research_task = self.create_research_task(researcher, topic)
        writing_task = self.create_writing_task(writer, topic)
        review_task = self.create_review_task(reviewer)
        
        # Create and return crew
        crew = Crew(
            agents=[researcher, writer, reviewer],
            tasks=[research_task, writing_task, review_task],
            verbose=True
        )
        
        return crew


def run_research_workflow(topic, hf_token=None, output_file="research_summary.md"):
    """
    Run the complete research workflow
    
    Args:
        topic: The research topic to investigate
        hf_token: Hugging Face API token (optional)
        output_file: Path to save the final summary
        
    Returns:
        dict: Results from the crew execution
    """
    # Initialize agents
    agents_factory = ResearchAgents(hf_token=hf_token)
    
    # Create crew
    crew = agents_factory.create_crew(topic)
    
    # Execute workflow
    print(f"\n{'='*60}")
    print(f"Starting Multi-Agent Research Workflow")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")
    
    result = crew.kickoff()
    
    # Save the writer's output to file
    # The result should contain the final summary
    if result:
        with open(output_file, 'w') as f:
            # Extract the writing task output
            if hasattr(result, 'tasks_output') and len(result.tasks_output) >= 2:
                summary = result.tasks_output[1].raw_output
            else:
                summary = str(result)
            f.write(summary)
        print(f"\n{'='*60}")
        print(f"Research summary saved to: {output_file}")
        print(f"{'='*60}\n")
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    
    topic = "Impact of Synthetic Data in Healthcare"
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    
    result = run_research_workflow(topic)
    print("\nWorkflow completed!")
    print(f"Result: {result}")
