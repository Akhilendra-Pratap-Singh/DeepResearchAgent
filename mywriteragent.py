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

INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")

    markdown_report: str = Field(description="The final report")

    follow_up_questions: list[str] = Field(description="Suggested topics to research further")


writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model=model,
    output_type=ReportData,
)