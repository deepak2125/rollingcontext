

import argparse
import os
import re

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

MAX_MESSAGES = 6  # summarize once state holds more than this many messages
KEEP_LAST = 2  # how many recent messages survive a summarization pass
OLLAMA_MODEL = "llama3.2"  # also try gpt-oss:20b, qwen:4b, deepseek-r1:1.5b
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class State(MessagesState):
    summary: str


def _strip_thinking(msg: AIMessage) -> AIMessage:
    """Reasoning models (deepseek-r1, ...) may inline <think> blocks; keep them out of state."""
    if isinstance(msg.content, str):
        msg.content = re.sub(r"<think>.*?</think>", "", msg.content, flags=re.DOTALL).strip()
    return msg


def make_llm(model: str | None):
    from langchain_ollama import ChatOllama

    model = model or OLLAMA_MODEL
    print(f"Using Ollama model '{model}' at {OLLAMA_URL}\n")
    return ChatOllama(model=model, base_url=OLLAMA_URL, temperature=0) | RunnableLambda(_strip_thinking)


def build_graph(llm):
    def chat(state: State):
        prompt: list[AnyMessage] = [SystemMessage("You are a helpful, concise assistant.")]
        if state.get("summary"):
            prompt.append(
                SystemMessage(
                    "Earlier messages in this conversation were condensed into this summary. "
                    f"Treat it as things you remember about the user:\n{state['summary']}"
                )
            )
        prompt += state["messages"]
        print(f"   -> sending {len(prompt)} messages to the model")
        return {"messages": [llm.invoke(prompt)]}

    def should_summarize(state: State):
        return "summarize" if len(state["messages"]) > MAX_MESSAGES else END

    def summarize(state: State):
        old, summary = state["messages"][:-KEEP_LAST], state.get("summary", "")
        rules = (
            "Write it as short bullet points. Always keep every personal fact about the user "
            "(name, location, preferences) and the topics discussed. Output only the bullets."
        )
        if summary:
            instruction = (
                f"This is the summary so far:\n{summary}\n\n"
                f"Rewrite the summary so it also covers the new messages above. {rules}"
            )
        else:
            instruction = f"Summarize the conversation above. {rules}"
        new_summary = llm.invoke(old + [HumanMessage(instruction)]).content
        print(f"   ** rolled {len(old)} old messages into the summary")
        # RemoveMessage tells the add_messages reducer to drop these ids from state.
        return {"summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in old]}

    builder = StateGraph(State)
    builder.add_node("chat", chat)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "chat")
    builder.add_conditional_edges("chat", should_summarize, ["summarize", END])
    builder.add_edge("summarize", END)
    # The checkpointer persists state per thread_id across invocations.
    return builder.compile(checkpointer=InMemorySaver())


DEMO_TURNS = [
    "Hi! My name is Deepak and I live in Noida.",
    "I'm learning LangGraph. What is a StateGraph in one sentence?",
    "What is a checkpointer for?",
    "My favourite language is Python.",
    "Give me one tip for writing good prompts.",
    "What's a reducer in LangGraph?",
    "Do you still remember my name and city?",
]


def show_state(graph, config):
    state = graph.get_state(config).values
    print(f"   state: {len(state['messages'])} messages kept, summary={'yes' if state.get('summary') else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph rolling-context demo")
    parser.add_argument("-i", "--interactive", action="store_true", help="chat yourself instead of the scripted demo")
    parser.add_argument("-m", "--model", help=f"Ollama model name (default: {OLLAMA_MODEL})")
    args = parser.parse_args()

    graph = build_graph(make_llm(args.model))
    config = {"configurable": {"thread_id": "demo"}}

    def turn(text: str):
        print(f"USER: {text}")
        result = graph.invoke({"messages": [HumanMessage(text)]}, config)
        reply = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
        print(f"AI:   {reply.content}")
        show_state(graph, config)
        print()

    if args.interactive:
        print("Type 'quit' to exit, 'summary' to print the rolling summary.\n")
        while (text := input("> ").strip()).lower() not in {"quit", "exit"}:
            if text == "summary":
                print(graph.get_state(config).values.get("summary") or "(no summary yet)", "\n")
            elif text:
                turn(text)
    else:
        for text in DEMO_TURNS:
            turn(text)
        print("Final rolling summary:\n" + (graph.get_state(config).values.get("summary") or "(none)"))
