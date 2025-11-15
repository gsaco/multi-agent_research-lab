# Configuration Guide

This guide explains how to set up API keys and tokens for the Multi-Agent Research Lab.

## Required API Keys

### 1. OpenAI API Key (Required for Agent Execution)

CrewAI uses OpenAI's GPT models by default for agent reasoning. You need an OpenAI API key to run the agents.

**Getting your OpenAI API key:**
1. Sign up at [OpenAI](https://platform.openai.com/)
2. Go to [API Keys](https://platform.openai.com/api-keys)
3. Create a new API key
4. Copy the key

**Setting the API key:**

On Linux/Mac:
```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
```

On Windows:
```cmd
set OPENAI_API_KEY=sk-your-actual-key-here
```

Or in Python:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-actual-key-here"
```

### 2. Hugging Face Token (Optional)

While not strictly required for the basic workflow, the Hugging Face token enables:
- Direct use of Hugging Face Inference API
- Access to private models
- Higher rate limits

**Getting your Hugging Face token:**
1. Sign up at [Hugging Face](https://huggingface.co/join)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token
4. Copy the token

**Setting the HF token:**

On Linux/Mac:
```bash
export HF_TOKEN="hf_your_token_here"
```

On Windows:
```cmd
set HF_TOKEN=hf_your_token_here
```

Or in Python:
```python
from huggingface_hub import login
login("hf_your_token_here")
```

## Alternative: Using Hugging Face Models with CrewAI

If you prefer to use Hugging Face models instead of OpenAI, you can configure CrewAI to use Hugging Face endpoints:

```python
from crewai import Agent, LLM

# Use Hugging Face model
llm = LLM(
    model="huggingface/mistralai/Mistral-7B-Instruct-v0.2",
    api_key=os.environ.get("HF_TOKEN")
)

# Create agent with HF model
agent = Agent(
    role="Researcher",
    goal="Research AI topics",
    backstory="You are an expert researcher.",
    llm=llm
)
```

## Using Local Models (Advanced)

For completely offline execution, you can use locally hosted models:

1. **Ollama** (Recommended for local models):
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull mistral

# Use in CrewAI
llm = LLM(model="ollama/mistral")
```

2. **LM Studio**:
   - Download from [LM Studio](https://lmstudio.ai/)
   - Load a model
   - Start the local server
   - Configure CrewAI to use `http://localhost:1234`

## Environment Variables Summary

Create a `.env` file in the project root:

```bash
# Required for agent execution
OPENAI_API_KEY=sk-your-openai-key-here

# Optional: for Hugging Face features
HF_TOKEN=hf_your-huggingface-token-here

# Optional: for other providers
# ANTHROPIC_API_KEY=your-anthropic-key
# GOOGLE_API_KEY=your-google-key
```

Load it in your code:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Testing Your Configuration

Run this script to verify your setup:

```python
import os
from src.agents import ResearchAgents

# Test configuration
print("Testing configuration...")

if os.environ.get("OPENAI_API_KEY"):
    print("✓ OpenAI API key found")
else:
    print("✗ OpenAI API key missing - required for execution")

if os.environ.get("HF_TOKEN"):
    print("✓ Hugging Face token found")
else:
    print("⚠ Hugging Face token missing - optional but recommended")

# Try creating agents
try:
    factory = ResearchAgents()
    researcher = factory.create_researcher()
    print("✓ Agent creation successful")
    print("\nConfiguration is valid!")
except Exception as e:
    print(f"✗ Configuration error: {e}")
```

## Cost Considerations

- **OpenAI GPT-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **OpenAI GPT-4o**: ~$2.50 per 1M input tokens, ~$10 per 1M output tokens
- **Hugging Face Inference API**: Free tier available, pay-as-you-go for production
- **Local Models (Ollama)**: Free, no API costs

A typical research workflow execution costs approximately:
- With GPT-4o-mini: $0.01 - $0.05 per run
- With GPT-4o: $0.10 - $0.50 per run
- With local models: Free

## Troubleshooting

### "OPENAI_API_KEY is required" error
- Make sure you've set the environment variable
- Check that there are no typos in the key
- Verify the key hasn't expired

### "Rate limit exceeded" error
- Wait a few moments and try again
- Upgrade your OpenAI plan
- Use a different model with lower demand

### Search tool not working
- Check your internet connection
- DuckDuckGo may have rate limits - wait a moment
- Try using alternative search tools

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** or secure vaults
3. **Rotate keys regularly**
4. **Use separate keys** for development and production
5. **Set spending limits** in your API provider dashboards
6. **Monitor usage** to detect anomalies
