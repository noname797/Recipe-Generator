# Recipe-Generator

A multi-agent, LangGraph-based system that generates personalized recipes from user-provided ingredients, dietary preferences, allergies, cuisine, and flavor profiles. The system can be run either as a CLI application (`main.py`) or as a web app (`app.py`) via Flask.

## 🎯 Project Overview

The **Multi-Agent Recipe Generator** chains together multiple specialized agents, each backed by an LLM (via AWS Bedrock / LangChain), to take a user from raw ingredients to a fully detailed, nutrition-aware recipe — with built-in sentiment monitoring and human escalation for when things go wrong.

## Agent Roles and Responsibilities

1. **Ingredient Input Agent** *(LLM)* — Handles the initial collection of user-provided ingredients, dietary preferences, allergies, cuisine, and flavor profiles. Manages ambiguity resolution for ingredient inputs, ensuring downstream agents receive precise and actionable data.
2. **Chef Agent** *(LLM)* — Generates multiple recipe suggestions based on the normalized ingredients and user preferences. Maximizes ingredient usage, aligns with dietary/flavor constraints, and provides detailed metadata for each recipe option.
3. **Nutrition Agent** *(LLM)* — Analyzes the nutritional profile of each recipe, suggesting common add-ons and new ingredients to enhance nutritional value.
4. **User Selection Agent** — Lets the user pick which recipe suggestion to proceed with.
5. **User Modification Agent** — Processes nutritional adjustment and ingredient modification requests, updating the recipe and nutritional profile accordingly.
6. **Final Recipe Agent** — Synthesizes all prior outputs into a complete, step-by-step recipe (ingredients, instructions, tips, serving suggestions) with integrated nutrition data.
7. **Sentiment & Escalation Agent** *(LLM)* — Monitors user sentiment across interactions; flips an `escalation_required` flag if frustration or urgency is detected, triggering a human handoff.
8. **Human Escalation Agent** — Packages context and suggested responses for a human reviewer/operator UI to review, continue, or override.

### Prompt Format

Prompts follow a consistent structure, with output instructions programmatically injected as a JSON dump of the expected output (Pydantic) class:

```
[INSTRUCTIONS]
{instructions for the model}

[OUTPUT INSTRUCTIONS]
{output instructions}
```

Basic rules enforced: skip preamble, stick to the output schema, avoid hallucinations.

## 📁 Project Structure

```
Recipe-Generator/
├── app.py                  # Flask web app entry point
├── main.py                 # CLI entry point
├── workflow.py              # LangGraph workflow wiring all agents together
├── pyproject.toml           # Project metadata & dependencies (uv)
├── uv.lock                  # Locked dependency versions
├── requirements.md          # Detailed project/agent overview
├── agents/
│   ├── ingredient_input_agent.py
│   ├── chef_agent.py
│   ├── nutrition_agent.py
│   ├── user_selection_agent.py
│   ├── user_modification_agent.py
│   ├── final_recipe_agent.py
│   ├── sentiment_escalation_agent.py
│   └── human_escalation_agent.py
├── utils/
│   ├── aws_bedrock.py        # AWS Bedrock client/helpers
│   ├── models.py             # Pydantic models for agent I/O
│   ├── prompt.py             # Prompt templates/formatting
│   └── state.py              # Shared `RecipeState` used across the workflow
└── templates/
    └── index.html            # Web UI template for the Flask app
```

## ⚙️ Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires **Python >= 3.12**.

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Configure AWS credentials (used by `langchain-aws` / Bedrock) in a `.env` file at the project root, e.g.:
   ```
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=...
   ```

## 🚀 Usage

### CLI

Run the workflow from the command line:

```bash
uv run main.py
```

Add `--test` to run in test mode:

```bash
uv run main.py --test
```

### Web App

Start the Flask server:

```bash
uv run app.py
```

Then open the app in your browser and interact with the chat-style UI, which walks through each workflow step (ingredient → chef → nutrition → user selection → user modification → final recipe → sentiment → human escalation, if needed).

## 🧩 Key Dependencies

- `langgraph`, `langchain`, `langchain-core`, `langchain-aws` — orchestration and LLM integration (AWS Bedrock)
- `pydantic` — structured agent input/output models
- `flask` — web app / API layer
- `boto3` — AWS SDK
- `pyyaml`, `dotenv` — configuration handling
- `ruff` — linting

See `pyproject.toml` for the full list.