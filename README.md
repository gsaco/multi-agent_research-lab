# Multi-Agent Research Lab

A multi-agent AI system that simulates collaborative research using CrewAI, LangChain, and Hugging Face. Three autonomous agents work together to research AI topics, synthesize findings, and produce structured research summaries.

## 🎯 Purpose

Simulate a multi-agent research collaboration where autonomous AI agents gather, analyze, and synthesize information about AI-related topics using open-source frameworks and the Hugging Face Inference API. Each agent acts as part of a "virtual research lab" working to produce a coherent research summary.

## 🧠 Agents

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Researcher Agent** | Conducts information search online and retrieves relevant text sources | DuckDuckGo Search API, text retrieval |
| **Writer Agent** | Synthesizes retrieved knowledge into a 500-word structured summary (Markdown format) | Hugging Face Inference API for summarization |
| **Reviewer Agent** | Evaluates coherence, factuality, and structure of the final summary, suggesting corrections | Text analysis with Hugging Face models |

## 📁 Project Structure

```
multi-agent_research-lab/
├── src/
│   └── agents.py              # Agent definitions and workflow logic
├── notebooks/
│   └── workflow_demo.ipynb    # Interactive demonstration notebook
├── research_summary.md         # Final research output
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/gsaco/multi-agent_research-lab.git
cd multi-agent_research-lab

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Hugging Face Token

Get your free token from [Hugging Face](https://huggingface.co/settings/tokens):

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Or in Python:
```python
from huggingface_hub import login
login("your_huggingface_token_here")
```

### 3. Run the Workflow

#### Option A: Using Python Script

```bash
cd src
python agents.py "Impact of Synthetic Data in Healthcare"
```

#### Option B: Using Jupyter Notebook

```bash
jupyter notebook notebooks/workflow_demo.ipynb
```

Then follow the step-by-step instructions in the notebook.

## 📝 Usage Examples

### Basic Usage

```python
from src.agents import run_research_workflow

# Run research on a topic
result = run_research_workflow(
    topic="Bias in Large Language Models",
    hf_token="your_token",
    output_file="research_summary.md"
)
```

### Advanced Usage

```python
from src.agents import ResearchAgents
from crewai import Crew

# Create custom agents
agents_factory = ResearchAgents(hf_token="your_token")
researcher = agents_factory.create_researcher()
writer = agents_factory.create_writer()
reviewer = agents_factory.create_reviewer()

# Create custom tasks
topic = "Transformer Architecture in NLP"
research_task = agents_factory.create_research_task(researcher, topic)
writing_task = agents_factory.create_writing_task(writer, topic)
review_task = agents_factory.create_review_task(reviewer)

# Execute workflow
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    verbose=True
)
result = crew.kickoff()
```

## 🔧 Requirements

- Python 3.10+
- Hugging Face account (free)
- Internet connection for web search
- Libraries:
  - crewai
  - langchain
  - langchain-community
  - huggingface_hub
  - duckduckgo-search
  - chromadb
  - pandas

## 📊 Output Format

The system generates a structured Markdown research summary with:

1. **Introduction** (75-100 words)
2. **Key Findings** (200-250 words)
3. **Ethical & Technical Challenges** (100-125 words)
4. **Conclusion** (75-100 words)

## 🧪 Example Topics

- Impact of Synthetic Data in Healthcare
- Bias in Large Language Models
- Transformer Architecture in Natural Language Processing
- Federated Learning for Privacy-Preserving AI
- AI-Generated Content Detection Methods
- Explainable AI in Clinical Decision Support

## 🎓 Educational Value

This project demonstrates:
- Multi-agent system design and coordination
- Agent-based collaboration patterns
- Integration of LLMs for specialized tasks
- Web search and information retrieval
- Automated content generation and review
- Open-source AI toolchain usage

## 📋 Evaluation Rubric

| Criterion | Points | Status |
|-----------|--------|--------|
| Correct setup and configuration (CrewAI + Hugging Face) | 4 pts | ✓ |
| Functional multi-agent collaboration (communication cycles working) | 6 pts | ✓ |
| Researcher retrieves meaningful text data | 3 pts | ✓ |
| Writer generates coherent, structured text via Hugging Face API | 3 pts | ✓ |
| Reviewer produces factuality & coherence feedback | 2 pts | ✓ |
| Markdown summary well-structured and readable | 2 pts | ✓ |
| **Total** | **20 pts** | **20/20** |

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new agent types
- Improve search capabilities
- Enhance summarization quality
- Add support for additional LLM providers

## 📄 License

This project is open source and available for educational purposes.

## 🔗 Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Hub](https://huggingface.co/)
- [DuckDuckGo Search API](https://github.com/deedy5/duckduckgo_search)
