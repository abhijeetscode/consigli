# RAG Financial Document Analysis System

## Prerequisites

Before running the system, you need to set up your OpenAI API key:

1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Quick Start

All required components (vector database, cache, and warmup cache) are already included in the repository. The warmup cache pre-computes common queries to improve response speed.

To run this system, simply execute:

```bash
python chatbot.py
```
