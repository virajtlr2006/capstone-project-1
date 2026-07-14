"""
campusai_lite/agents/langchain_chain.py
Part A – LangChain LCEL chain for CampusAI Lite.

Pipeline:
  1. Input is validated via PydanticAI models.
  2. An LLMChain classifies the intent and generates tool calls.
  3. An Agent Executor uses UniversityInfoTool + DoclingDocumentTool.
  4. The final response is returned as a string.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableSequence

from tools.university_tool import UniversityInfoTool
from tools.docling_tool import DoclingDocumentTool
from agents.pydantic_models import validate_student_query


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are CampusAI, a friendly and knowledgeable AI assistant for \
TechVista University.

Your job is to answer student questions about:
- Admissions (undergraduate and graduate)
- Tuition and financial aid
- Academic departments and programs
- Campus facilities
- Academic calendar
- Student services and support
- Research opportunities

Always use the available tools to retrieve accurate, up-to-date information.
When you cannot find specific data, clearly say so and suggest who the student
can contact.

Be friendly, clear, and concise. End your responses with a relevant next step
or contact the student can follow up with."""


# ── Build the LangChain agent ─────────────────────────────────────────────────
def _build_langchain_agent() -> AgentExecutor:
    llm = ChatOpenAI(
        model=os.getenv("BOB_MODEL", "ibm/granite-3-3-8b-instruct"),
        temperature=0.3,
        api_key=os.getenv("BOB_API_KEY"),
        base_url=os.getenv("BOB_BASE_URL", "https://api.bam.res.ibm.com/v1"),
    )

    tools = [UniversityInfoTool(), DoclingDocumentTool()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )


# ── LCEL pipeline: validate → agent → format ─────────────────────────────────
def _build_lcel_chain(agent_executor: AgentExecutor) -> RunnableSequence:
    """
    Wraps the agent executor in an LCEL chain that:
      1. Validates input with PydanticAI's StudentQuery model.
      2. Passes it to the AgentExecutor.
      3. Extracts the output string.
    """

    def validate_input(raw: str | dict) -> dict:
        question = raw if isinstance(raw, str) else raw.get("question", str(raw))
        validated = validate_student_query(question)
        return {"input": validated.question}

    def extract_output(result: dict) -> str:
        return result.get("output", str(result))

    chain = RunnableLambda(validate_input) | agent_executor | RunnableLambda(extract_output)
    return chain


# ── Public entry point ────────────────────────────────────────────────────────
_agent_executor: AgentExecutor | None = None
_chain: RunnableSequence | None = None


def _get_chain() -> RunnableSequence:
    global _agent_executor, _chain
    if _chain is None:
        _agent_executor = _build_langchain_agent()
        _chain = _build_lcel_chain(_agent_executor)
    return _chain


def run_langchain(question: str) -> str:
    """Run the LangChain LCEL pipeline and return the answer."""
    chain = _get_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    sample = "What courses are available in the Computer Science department?"
    print(run_langchain(sample))
