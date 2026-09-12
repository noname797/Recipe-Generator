import logging
import os

from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Bedrock/AWS error codes worth retrying on (transient/capacity related).
_RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
}


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in _RETRYABLE_ERROR_CODES
    # Fall back to retrying generic ChatBedrock/boto client errors that carry
    # the error code as a plain attribute or in their message.
    return any(code in str(exc) for code in _RETRYABLE_ERROR_CODES)


class BedrockClient:
    def __init__(self):
        self.llm = ChatBedrock(
            model_id=os.getenv("BEDROCK_MODEL_ID"),
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def invoke(self, prompt: str, max_tokens: int = 2048) -> str:
        try:
            response = self.llm.invoke(
                [HumanMessage(content=prompt)], max_tokens=max_tokens
            )
        except Exception:
            logger.warning("Bedrock invocation failed, may retry if transient", exc_info=True)
            raise
        return response.content
