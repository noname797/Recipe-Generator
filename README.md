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

### Architecture: a single shared LangGraph workflow

Both the CLI (`main.py`) and the Flask web app (`app.py`) drive the **same** compiled LangGraph graph (`workflow.interactive_workflow`) — there is no separate, hand-rolled step machine duplicated between the two entry points. Steps that need human input (`ingredient`, `user_selection`, `user_modification`, `sentiment`) use LangGraph's [`interrupt()`](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) to pause execution and checkpoint state (via an in-memory `MemorySaver`, keyed by a per-session `thread_id`); the caller resumes the graph with `Command(resume=user_input)` once it has the next piece of user input. This is what actually made the CLI path functional end-to-end (previously, `run_workflow()` invoked the graph exactly once and immediately failed, since plain LangGraph nodes can only be called with `(state)` and have no way to receive ad hoc arguments mid-run).

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
│   ├── state.py              # Shared `RecipeState` used across the workflow
│   ├── llm_parsing.py        # YAML parsing with retry/self-correction for LLM outputs
│   └── test_yaml.py          # Fixed YAML fixtures used in `test_mode`
├── tests/                    # Pytest suite (agents, workflow, LLM parsing helper)
└── templates/
    └── index.html            # Web UI template for the Flask app
```

## ⚙️ Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires **Python >= 3.12**.

1. Install dependencies (including dev/test dependencies):
   ```bash
   uv sync --group dev
   ```
2. Copy `.env.example` to `.env` and fill in real values:
   ```bash
   cp .env.example .env
   ```
   - AWS credentials (used by `langchain-aws` / Bedrock): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BEDROCK_MODEL_ID`. If using temporary/STS credentials (access key ID starting with `ASIA` rather than `AKIA`), you must also set `AWS_SESSION_TOKEN` — omitting it causes Bedrock calls to fail with `UnrecognizedClientException: The security token included in the request is invalid`.
   - `FLASK_SECRET_KEY`: required by `app.py` to sign session cookies. Generate one with:
     ```bash
     python -c "import secrets; print(secrets.token_hex(32))"
     ```
   - `SESSION_TYPE` / `SESSION_FILE_DIR`: control where the Flask web app stores session data server-side (via `flask-session`). Defaults to `filesystem` storage under `./.flask_session`; set `SESSION_TYPE=redis` (and configure `SESSION_REDIS`) for production deployments.
   - `LOG_LEVEL`: logging verbosity for `app.py` (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
   - `ESCALATION_DB_PATH`: SQLite file used to persist escalated sessions for human review (defaults to `./escalations.db`).
   - `ESCALATION_API_KEY`: shared secret required (via the `X-API-Key` header) to call the `/escalations*` human-reviewer endpoints. If unset, those endpoints are disabled (return `503`) rather than being left open.
   - `CHECKPOINT_DB_PATH`: optional path to a SQLite file for persisting LangGraph checkpoints (via `langgraph-checkpoint-sqlite`). If unset, an in-memory checkpointer is used, meaning in-flight (interrupted) sessions are lost on process restart — set this for anything beyond local development.

## 🧪 Testing

Unit and integration tests live in `tests/` and run entirely in `test_mode`, so no AWS credentials or network access are required:

```bash
uv run pytest
```

## 🛡️ Reliability Notes

- **Single shared workflow graph**: Both the CLI and web app drive the same interrupt-based `workflow.interactive_workflow` LangGraph graph (see "Architecture" above) instead of maintaining duplicate step machines.
- **Server-side sessions**: The Flask web app stores only an opaque LangGraph `thread_id` in the client-side cookie; all actual workflow state lives server-side via `flask-session` + the LangGraph checkpointer, avoiding cookie size limits and not leaking recipe/nutrition data to the browser.
- **Persistent checkpointing (optional)**: By default the graph uses an in-memory checkpointer (fine for local dev/tests), but set `CHECKPOINT_DB_PATH` to persist checkpoints to SQLite (via `langgraph-checkpoint-sqlite`) so in-flight sessions survive process restarts.
- **Escalation persistence**: When a session is escalated to a human, its context is persisted to a SQLite-backed store (`utils/escalation_store.py`) so a reviewer has somewhere to actually see and resolve it — via `GET /escalations`, `GET /escalations/<id>`, and `POST /escalations/<id>/resolve` on the Flask app. These endpoints require an `X-API-Key` header matching `ESCALATION_API_KEY`.
- **LLM output self-correction**: Every LLM-backed agent parses its YAML response via `utils/llm_parsing.parse_llm_yaml_with_retry`, which retries with a LangChain `OutputFixingParser` (asking the model to repair malformed output) and re-invokes the LLM up to a configurable number of times before failing.
- **Bedrock retry/backoff**: `utils/aws_bedrock.BedrockClient.invoke` automatically retries transient AWS Bedrock errors (throttling, timeouts, internal errors) with exponential backoff via `tenacity`.
- **XSS-safe rendering**: Recipe/nutrition text (which can be influenced by earlier free-text user input via the LLM) is HTML-escaped before being interpolated into the chat UI's HTML, and the web UI renders the user's own typed messages as plain text rather than HTML.
- **Shared agent base class**: All LLM-backed agents (`agents/*.py`) share a common `agents.base_agent.BaseLLMAgent` that handles test-mode fixtures, prompt-building dispatch, and YAML/Pydantic parsing, removing duplicated boilerplate across agents.


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