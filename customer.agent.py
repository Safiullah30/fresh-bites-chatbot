import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, RunConfig
from dotenv import load_dotenv
import chainlit as cl

# ------------------------
# Load environment variables
# ------------------------
load_dotenv()
set_tracing_disabled(True)

# ------------------------
# Business Information (Change for client)
# ------------------------
BUSINESS_INFO = """
Business Name: Fresh Bites Bakery
We provide fresh breads, cakes, pastries, and cookies daily. 
We also take custom cake orders for birthdays, weddings, and other special events. 
Location: Blue Area, Islamabad, Pakistan.
Timings: 9 AM to 9 PM (Monday to Saturday).
Contact: +92-300-1234567 | support@freshbites.com
"""

# ------------------------
# Setup OpenAI/Gemini Client
# ------------------------
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta"
)

mymodel = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

confiq = RunConfig(
    model=mymodel,
    model_provider=client,
    tracing_disabled=True
)

# ------------------------
# Agent (Advanced Instructions)
# ------------------------
agent = Agent(
    name="AdvancedCustomerSupportAgent",
    instructions=f"""
You are a professional, friendly, and advanced customer support assistant for this business:

{BUSINESS_INFO}

### Guidelines:
1. Always greet customers politely.
2. Understand customer intent even if wording is different.
3. Keep track of conversation context (previous questions).
4. Answer only business-related questions.
5. If user asks something unrelated, reply:
   "Sorry, I can only help with Fresh Bites Bakery related queries."
6. Always give clear, complete, and professional answers.
7. Offer help proactively if you see an opportunity (e.g. suggest menu, timings, offers).
""",
    model=mymodel
)

# ------------------------
# Chainlit Chat Events
# ------------------------
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Welcome to Fresh Bites Bakery! How can I assist you today?"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role": "user", "content": message.content})

    result = await Runner.run(
        agent,
        input=history,
        run_config=confiq,
    )

    response = result.final_output or "⚠ Sorry, I couldn't process that. Please try again."

    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)

    await cl.Message(content=response).send()