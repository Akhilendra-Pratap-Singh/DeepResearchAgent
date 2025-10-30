from pydantic import BaseModel,Field
from agents import Agent
import os

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

groq_client = AsyncOpenAI(api_key=groq_api_key,base_url="https://api.groq.com/openai/v1")

# model_name = "llama-3.3-70b-versatile"

# model_name = "moonshotai/kimi-k2-instruct-0905"#THis supports structured outputs
# model_name = "openai/gpt-oss-120b"
# model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"

model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

model = OpenAIChatCompletionsModel(openai_client=groq_client,model=model_name)

HOW_MANY_SEARCHES = 3


INSTRUCTIONS = f"""
You are a research assistant. Given a query, output exactly {HOW_MANY_SEARCHES} web search items
in valid JSON format. Each item must have:

- "reason": why this search is important to the query
- "query": the search term to use for the web search

Output only JSON. Do not add any text outside the JSON object. Do not explain anything.

Example output for {HOW_MANY_SEARCHES} items:

{{
  "searches": [
    {{"reason": "Example reason 1", "query": "Example query 1"}},
    {{"reason": "Example reason 2", "query": "Example query 2"}},
    {{"reason": "Example reason 3", "query": "Example query 3"}}
  ]
}}
"""


class WebSearchItem(BaseModel):
    reason:str = Field(description="Your reasoning for why this search is important to this query")

    query:str = Field(description="The search term to use for the web search")

class WebSearchPlan(BaseModel):

    searches:list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")

planner_agent = Agent(
    name = "planneragent",
    instructions=INSTRUCTIONS,
    model = model,
    output_type=WebSearchPlan
)

