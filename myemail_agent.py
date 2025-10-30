from agents import Agent,function_tool
import sendgrid
from sendgrid.helpers.mail import Email,To,Content,Mail
from typing import Dict
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

INSTRUCTIONS = """You are able to send a nicely formatted HTML email based on a detailed report.
You will be provided with a detailed report. You should use your tool to send one email, providing the 
report converted into clean, well presented HTML with an appropriate subject line."""

@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("akhilendrasingh2812@gmail.com") # Change this to your verified email
    to_email = To("akhilendrasingh2812@gmail.com") # Change this to your email
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}

email_agent = Agent(
    name="Email agent",
    instructions=INSTRUCTIONS,
    tools=[send_email],
    model=model,
)

