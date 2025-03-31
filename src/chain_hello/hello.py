import chainlit as cl
from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,RunConfig
from dotenv import load_dotenv,find_dotenv
import os
load_dotenv(find_dotenv())

google_api_key = os.getenv("GEMINI_API_KEY")
# Initialize the OpenAI provider
provider = AsyncOpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Initialize the OpenAI model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# Initialize the runner
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Initialize the agent
agent1 = Agent(
    name="Shahmir Agent",
    instructions="You are a helpful assistant that can answer questions and help with tasks."
)

# Initialize the runner
runner = Runner.run_sync(
    agent1,
    input = "What is the capital of the moon?",
    run_config=run_config
)

@cl.on_message
async def handle_message(message:cl.Message):
    result = Runner.run_sync(
    agent1,
    input = f"{message.content}",
    run_config=run_config
)
    await cl.Message(content=result.final_output).send()
