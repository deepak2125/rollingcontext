"""Semantic memory with a vector store, using LangGraph and Ollama.

Each thing the user tells the assistant is embedded (turned into a vector of numbers
that captures its meaning) and saved in LangGraph's long-term Store. On every turn the
question is embedded too, and the memories whose vectors are closest in meaning (cosine
similarity) are added to the prompt.

Unlike the BM25 example, no shared keywords are needed: "a gift for my kid's special day"
finds "my daughter's birthday is on 14 March". And unlike a checkpointer, the Store is
shared across threads, so a brand-new conversation can still recall facts from an old one.

    START -> recall -> chat -> remember -> END
"""

import argparse
import uuid

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from rolling_context import OLLAMA_MODEL, OLLAMA_URL, make_llm

EMBED_MODEL = "nomic-embed-text"  # install with: ollama pull nomic-embed-text
EMBED_DIMS = 768  # vector size produced by nomic-embed-text
WINDOW = 4  # recent messages sent verbatim
TOP_K = 3  # memories retrieved per turn
MIN_SCORE = 0.45  # cosine similarity below this is treated as "not related"


class State(MessagesState):
    memories: list[str]


def make_store(embed_model: str = EMBED_MODEL) -> InMemoryStore:
    """A key-value store with a vector index on the 'text' field. Swap for PostgresStore in production."""
    embeddings = OllamaEmbeddings(model=embed_model, base_url=OLLAMA_URL)
    return InMemoryStore(index={"embed": embeddings, "dims": EMBED_DIMS, "fields": ["text"]})


def build_graph(llm, store: BaseStore, top_k: int = TOP_K):
    def namespace(config: RunnableConfig):
        # Memories belong to a user, not to a thread, so every thread of that user can see them.
        return ("memories", config["configurable"]["user_id"])

    def recall(state: State, config: RunnableConfig, *, store: BaseStore):
        query = state["messages"][-1].content
        hits = store.search(namespace(config), query=query, limit=top_k)
        for h in hits:
            mark = "use " if h.score >= MIN_SCORE else "skip"
            print(f"   {mark} {h.score:.3f}  {h.value['text']!r}")
        if not hits:
            print("   (memory is empty)")
        return {"memories": [h.value["text"] for h in hits if h.score >= MIN_SCORE]}

    def chat(state: State):
        prompt: list[AnyMessage] = [SystemMessage("You are a helpful, concise assistant.")]
        if state["memories"]:
            facts = "\n".join(f"- {m}" for m in state["memories"])
            prompt.append(
                SystemMessage(
                    "Things the user told you in earlier conversations (retrieved by meaning). "
                    f"Use them when relevant:\n{facts}"
                )
            )
        prompt += state["messages"][-WINDOW:]
        print(f"   -> sending {len(prompt)} messages to the model")
        return {"messages": [llm.invoke(prompt)]}

    def remember(state: State, config: RunnableConfig, *, store: BaseStore):
        text = state["messages"][-2].content  # the user message of this turn
        # Simple heuristic: statements are worth remembering, questions usually aren't.
        if not text.rstrip().endswith("?"):
            store.put(namespace(config), str(uuid.uuid4()), {"text": text})  # embedded on put
            print("   ** saved to semantic memory")
        return {}

    builder = StateGraph(State)
    builder.add_node("recall", recall)
    builder.add_node("chat", chat)
    builder.add_node("remember", remember)
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "chat")
    builder.add_edge("chat", "remember")
    builder.add_edge("remember", END)
    # Checkpointer = short-term memory per thread; store = long-term memory per user.
    return builder.compile(checkpointer=InMemorySaver(), store=store)


# Two separate conversations (threads). The second starts with an empty history and
# its questions share almost no words with the facts, so only semantic search can help.
DEMO_SESSIONS = {
    "monday": [
        "Hi, I'm Deepak and I live in Noida.",
        "I'm allergic to peanuts.",
        "My daughter turns eight on 14 March and she adores painting.",
        "I drive a red Tata Nexon electric car.",
        "At work I'm building an AI agent with LangGraph.",
    ],
    "friday": [
        "What should I get my kid for her special day?",
        "Suggest a quick snack recipe that is safe for me to eat.",
        "Any tips for a long road trip in my vehicle?",
        "What framework am I using for my project at the office?",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph semantic memory (vector store) demo")
    parser.add_argument("-i", "--interactive", action="store_true", help="chat yourself instead of the scripted demo")
    parser.add_argument("-m", "--model", help=f"Ollama chat model (default: {OLLAMA_MODEL})")
    parser.add_argument("-e", "--embed-model", default=EMBED_MODEL, help=f"Ollama embedding model (default: {EMBED_MODEL})")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="memories to retrieve per turn")
    args = parser.parse_args()

    store = make_store(args.embed_model)
    graph = build_graph(make_llm(args.model), store, args.top_k)
    user_id = "deepak"

    def turn(text: str, thread_id: str):
        print(f"USER: {text}")
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        result = graph.invoke({"messages": [HumanMessage(text)]}, config)
        reply = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
        print(f"AI:   {reply.content}\n")

    if args.interactive:
        thread = 1
        print("Type 'quit' to exit, 'new' to start a new thread (memory is kept), 'memories' to list them.\n")
        while (text := input(f"[thread {thread}]> ").strip()).lower() not in {"quit", "exit"}:
            if text == "new":
                thread += 1
                print(f"--- new thread {thread}: chat history is empty, semantic memory is not ---\n")
            elif text == "memories":
                items = store.search(("memories", user_id), limit=100)
                print("\n".join(f"- {i.value['text']}" for i in items) or "(empty)", "\n")
            elif text:
                turn(text, f"thread-{thread}")
    else:
        for thread_id, turns in DEMO_SESSIONS.items():
            print(f"=============== thread '{thread_id}' (new conversation) ===============\n")
            for text in turns:
                turn(text, thread_id)
