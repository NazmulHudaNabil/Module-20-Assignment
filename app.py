import os
import asyncio
import streamlit as st
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

load_dotenv()
set_tracing_disabled(True)



# Open ai Models Client
client = AsyncOpenAI(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
)

MODEL = "openai/gpt-4o-mini"



# Output Format
class LinkdinStylePost(BaseModel):
    title: str = Field(..., description="The title of the LinkedIn post")
    content: str = Field(..., description="The content of the LinkedIn post")
    hashtags: List[str] = Field(..., description="Relevant hashtags")



# Supported Languages
SUPPORTED_LANGUAGES = [
    "English",
    "Bengali",
    "Spanish",
    "French",
    "Hindi",
    "Chinese",
    "Arabic",
]



# Tools
@function_tool
def validate_language(language: str) -> str:
    """
    Validate whether the requested language is supported.

    Call this tool exactly once using the language mentioned
    by the user.
    """

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages are: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    return f"{language} is supported."



# Agent
linkedin_post_agent = Agent(
    name="LinkedIn Post Generator",
    instructions="""
You are an expert LinkedIn content strategist.

Workflow:
1. Identify the language mentioned by the user.
2. Call validate_language exactly once with that language.
3. If the language is supported, generate the LinkedIn post.
4. Return the final answer.

Never call validate_language more than once.

The post must:
- Be 2–4 paragraphs.
- Have a strong opening hook.
- Be engaging.
- End with a question to encourage comments.
- Be written entirely in the requested language.
""",
    model=OpenAIChatCompletionsModel(
        openai_client=client,
        model=MODEL,
    ),
    tools=[validate_language],
    output_type=LinkdinStylePost,
    )




# ======================
# STREAMLIT UI
# ======================
st.title("📝 LinkedIn Post Generator")

topic = st.text_area("Enter your topic + language", placeholder="AI in Spanish, Remote Work in English")

if st.button("Generate Post"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Generating..."):

            async def run_agent():
                return await Runner.run(linkedin_post_agent, topic, max_turns=5)

            result = asyncio.run(run_agent())

        st.subheader("Title")
        st.write(result.final_output.title)

        st.subheader("Content")
        st.write(result.final_output.content)

        st.subheader("Hashtags")
        st.write(" ".join(result.final_output.hashtags))