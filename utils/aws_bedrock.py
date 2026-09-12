import os
import boto3
import json
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

load_dotenv()


class BedrockClient:
    def __init__(self):
        self.llm = ChatBedrock(
            model_id=os.getenv("BEDROCK_MODEL_ID"),
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )

    import json

    def invoke(self, prompt: str, max_tokens: int = 2048) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)], max_tokens=max_tokens)
        return response.content
