# Multi-Agent Research Lab — Homework Submission

This repository contains the assignment deliverables:

- `src/agents.py` — Agent definitions and the research workflow
- `notebooks/workflow_demo.ipynb` — Jupyter notebook demonstration
- `research_summary.md` — Final Markdown report
- `requirements.txt` — Python dependencies

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

### Quick start

1. Create and activate a Python environment (conda recommended):

```bash
conda create -n multiagent python=3.11 -y
conda activate multiagent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure the Hugging Face token

This notebook uses the Hugging Face Inference API; create a token at https://huggingface.co/settings/tokens and set it as an environment variable.

```bash
export HF_TOKEN="hf_<your_token>"
export HUGGINGFACEHUB_API_TOKEN="$HF_TOKEN"

Alternatively, create a `.env` file in the repository root with the following contents (add this file to `.gitignore` so your token isn't committed):

```
# .env
HF_TOKEN=hf_<your_token>
HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
```

After creating `.env`, run `source .env` to populate your shell's environment before running the notebook. A `.env.example` file is included as a template — do not commit real tokens.
The repository's `.gitignore` already excludes `.env`, so your token won't be committed.
```

### 3. Set Up API Keys

No OpenAI key is required. The notebook uses Hugging Face Inference API only (the HF token is used for authentication). The notebook contains a cell that logs in with the HF token for convenience.

Get your keys:
- OpenAI: https://platform.openai.com/api-keys
- Hugging Face: https://huggingface.co/settings/tokens

### 4. Run the Workflow

Run the workflow (notebook recommended):

```bash
jupyter notebook notebooks/workflow_demo.ipynb
```

Or run directly from Python (minimal):

```bash
python src/agents.py "Impact of Synthetic Data in Healthcare"
```

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

# If you want the CrewAI orchestration to use a specific LLM provider (like
# a Hugging Face provider), add creawi_llm_kwargs:
# result = run_research_workflow(topic="Bias in LLMs", hf_token="your_token",
#                                creawi_llm_kwargs={"provider": "huggingface"})
```

### Advanced Usage

```python
from src.agents import ResearchAgents
from crewai import Crew

# Create custom agents
agents_factory = ResearchAgents(hf_token="your_token", creawi_llm_kwargs={"provider": "huggingface"})
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

## Requirements

- Python 3.10+
- Hugging Face account and token
- Internet connection

Dependencies are available in `requirements.txt`.

## Final output

The final deliverable is `research_summary.md` — a ~500-word structured Markdown document with these sections:

1. Introduction
2. Key Findings
3. Ethical & Technical Challenges
4. Conclusion

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

## Rubric & deliverables

The repository provides the deliverables required by the assignment (agents, notebook, final MD). The notebook demonstrates the Researcher → Writer → Reviewer loop and produces `research_summary.md`.

## Notes
The README was intentionally replaced with this minimal note so the repository only contains the assignment deliverables requested by the user.
