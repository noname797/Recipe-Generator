"""
Shared base class for LLM-backed agents.

Every LLM-backed agent in `agents/` follows the same pattern:
  1. In `test_mode`, load a fixed YAML fixture instead of calling the LLM.
  2. Otherwise, build a prompt, invoke the LLM, and parse the YAML response
     into a Pydantic model (with retry/self-correction).

This class factors out that boilerplate so individual agents only need to
declare their output model / fixture key / prompt-building logic.
"""
import logging

import yaml
import yaml as pyyaml
from langchain.output_parsers.yaml import YamlOutputParser

from utils.aws_bedrock import BedrockClient
from utils.llm_parsing import parse_llm_yaml_with_retry
from utils.test_yaml import fixed_yaml_definitions

logger = logging.getLogger(__name__)


class BaseLLMAgent:
    """Base class for agents that call an LLM and parse a YAML/Pydantic output.

    Subclasses must set:
      - `output_model`: the Pydantic model class the LLM response parses into.
      - `fixture_key`: key into `fixed_yaml_definitions` used in test_mode.
      - `agent_name`: human-readable name, used in retry/error logs.

    And must implement:
      - `_build_prompt(self, *args, **kwargs) -> str`
    """

    output_model = None
    fixture_key = None
    agent_name = None

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self.llm = BedrockClient()
        if self.output_model is not None:
            self.yaml_parser = YamlOutputParser(pydantic_object=self.output_model)

    def output_schema_yaml(self) -> str:
        """Dynamically generate the YAML output-instruction block from the
        Pydantic output model's JSON schema."""
        return pyyaml.safe_dump(
            self.output_model.model_json_schema(), sort_keys=False, indent=2
        )

    def _build_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError(f"{type(self).__name__} must implement _build_prompt")

    def run(self, *prompt_args, **prompt_kwargs):
        """Execute the agent: return fixture output in test mode, otherwise
        build a prompt, invoke the LLM, and parse+validate the response."""
        if self.test_mode:
            fixed_yaml = fixed_yaml_definitions[self.fixture_key]
            return self.output_model(**yaml.safe_load(fixed_yaml))

        prompt = self._build_prompt(*prompt_args, **prompt_kwargs)
        return parse_llm_yaml_with_retry(
            self.llm, self.yaml_parser, prompt, self.agent_name
        )
