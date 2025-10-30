from agents import Agent,function_tool
import os
import aiohttp

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

@function_tool
async def tavily_search(query: str) -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return "ERROR: TAVILY_API_KEY not set in environment."

    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"query": query, "num_results": 6}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=30) as resp:
                data = await resp.json()
    except Exception as e:
        return f"ERROR calling Tavily: {e}"

    results = data.get("results", []) or []
    if not results:
        return "No results found."

    out_lines = []
    for r in results:
        title = r.get("title", "")
        link = r.get("url", "")
        snippet = r.get("content") or r.get("snippet") or ""
        out_lines.append(f"{title} - {link}\nSnippet: {snippet}")

    return "\n\n".join(out_lines)


INSTRUCTIONS = "You are a research assistant. Given a search term, you search the web for that term and \
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 \
words. Capture the main points. Write succintly, no need to have complete sentences or good \
grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the \
essence and ignore any fluff. Do not include any additional commentary other than the summary itself"

search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    # tools=[tavily_search],
    model=model,
    # model_settings=ModelSettings(tool_choice="required"),
)