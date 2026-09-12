"""
Shared helper for invoking an LLM and parsing its response into a Pydantic
model via a LangChain YAML output parser, with automatic retry/self-correction
when the raw response fails to parse.

Strategy, in order:
1. Parse the raw response directly.
2. If that fails, ask an `OutputFixingParser` (backed by the same LLM) to
   repair the malformed output and re-parse it.
3. If that still fails, re-invoke the LLM with the original prompt up to
   `max_retries` additional times, repeating steps 1-2 for each attempt.
4. If all attempts are exhausted, raise a `RuntimeError` with full context.
"""
import logging

from langchain.output_parsers import OutputFixingParser

logger = logging.getLogger(__name__)


def parse_llm_yaml_with_retry(
    llm_client,
    yaml_parser,
    prompt: str,
    agent_name: str,
    max_retries: int = 2,
):
    """
    Invoke `llm_client` with `prompt`, parsing the response using
    `yaml_parser`. Retries with self-correction on failure.

    Args:
        llm_client: Object exposing `.invoke(prompt) -> str` and `.llm`
            (the underlying LangChain chat model, used for self-correction).
        yaml_parser: A `langchain.output_parsers.yaml.YamlOutputParser`
            instance for the expected Pydantic output model.
        prompt: The fully-formatted prompt string to send to the LLM.
        agent_name: Human-readable agent name, used in error messages/logs.
        max_retries: Number of additional LLM invocations to attempt if
            parsing (including self-correction) keeps failing.

    Returns:
        The parsed Pydantic model instance.

    Raises:
        RuntimeError: if parsing still fails after all retries.
    """
    fixing_parser = OutputFixingParser.from_llm(parser=yaml_parser, llm=llm_client.llm)

    last_error = None
    last_response = None
    attempts = max_retries + 1

    for attempt in range(1, attempts + 1):
        response = llm_client.invoke(prompt)
        last_response = response
        try:
            return yaml_parser.parse(response)
        except Exception as parse_error:
            logger.warning(
                "%s: direct YAML parse failed on attempt %d/%d: %s",
                agent_name,
                attempt,
                attempts,
                parse_error,
            )
            try:
                return fixing_parser.parse(response)
            except Exception as fixing_error:
                last_error = fixing_error
                logger.warning(
                    "%s: OutputFixingParser also failed on attempt %d/%d: %s",
                    agent_name,
                    attempt,
                    attempts,
                    fixing_error,
                )

    raise RuntimeError(
        f"{agent_name}: Failed to parse YAML from LLM response after {attempts} attempt(s): "
        f"{last_error}\nLast raw response: {last_response}"
    )
