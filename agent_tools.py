from agents import Agent, Runner ,AsyncOpenAI,OpenAIChatCompletionsModel
from tools import function_tools
import asyncio
from dotenv import load_dotenv
import os

 
# Load environment variables
load_dotenv()

# Gemini API client
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta"
)

# Model setup
mymodel = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You translate the user's message to Spanish",
)

french_agent = Agent(
    name="French agent",
    instructions="You translate the user's message to French",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate. "
        "If asked for multiple translations, you call the relevant tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
)

async def main():
    result = await Runner.run(orchestrator_agent, input="Say 'Hello, how are you?' in Spanish.")
    print(result.final_output)

# یہ لائن لازمی ہے تاکہ async فنکشن execute ہو
