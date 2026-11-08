"""Full-text search (BM25) memory with LangGraph.

Only the last WINDOW messages stay in the prompt. Older messages are moved into an
inverted index; on every turn the user's question is searched with BM25 and the
top K past messages are added to the prompt.

    START -> retrieve -> chat -> archive -> END

1. Index:  tokenize -> lowercase -> drop stop words -> stem -> inverted index
2. Rank:   BM25 (term frequency, rarity of the term, document length)
3. Answer: analyze the question the same way, score, take the top K, add to prompt
"""

import argparse
import heapq
import math
import re
from collections import Counter, defaultdict

import snowballstemmer
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from rolling_context import OLLAMA_MODEL, make_llm

WINDOW = 4  # messages kept verbatim in the prompt (2 user/assistant turns)
TOP_K = 3  # past messages pulled back in by search
SNIPPET_CHARS = 300  # long hits are shortened before going into the prompt

STOP_WORDS = set(
    "a an and are as at be but by can do does for from had has have how i i'm in is it its "
    "me my of on or so that the their them then there these they this to was we what when "
    "where which who why will with you your about could would should did any all just "
    "he him his she her our us get got hi hello s m t d ll re ve".split()
)
_stemmer = snowballstemmer.stemmer("english")


def analyze(text: str) -> list[str]:
    """Tokenize, lowercase, remove stop words and stem: 'running', 'runs' -> 'run'."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return _stemmer.stemWords([t for t in tokens if t not in STOP_WORDS])


class BM25Index:
    """An in-memory inverted index ranked with BM25. Real systems keep this in a database."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)  # term -> {doc_id: term freq}

    def add(self, text: str) -> int:
        doc_id = len(self.docs)
        terms = analyze(text)
        self.docs.append(text)
        self.doc_len.append(len(terms))
        for term, tf in Counter(terms).items():
            self.postings[term][doc_id] = tf
        return doc_id

    def search(self, query: str, k: int = TOP_K) -> list[tuple[float, str]]:
        if not self.docs:
            return []
        n_docs, avg_len = len(self.docs), sum(self.doc_len) / len(self.docs)
        scores: dict[int, float] = defaultdict(float)
        for term in set(analyze(query)):
            posting = self.postings.get(term)
            if not posting:
                continue  # the index lookup skips every document without this term
            # Rare terms (few documents contain them) get a higher weight.
            idf = math.log(1 + (n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for doc_id, tf in posting.items():
                # More occurrences help, with diminishing returns; long documents are penalized.
                length_norm = 1 - self.b + self.b * self.doc_len[doc_id] / avg_len
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
        return [(score, self.docs[doc_id]) for doc_id, score in top]


class State(MessagesState):
    retrieved: list[str]


def build_graph(llm, index: BM25Index, top_k: int = TOP_K):
    def retrieve(state: State):
        query = state["messages"][-1].content
        hits = index.search(query, top_k)
        print(f"   search terms: {analyze(query)}")
        for score, text in hits:
            print(f"   hit {score:5.2f}  {text[:90]!r}")
        if not hits:
            print("   (no hits)")
        return {"retrieved": [text[:SNIPPET_CHARS] for _, text in hits]}

    def chat(state: State):
        prompt: list[AnyMessage] = [SystemMessage("You are a helpful, concise assistant.")]
        if state["retrieved"]:
            memories = "\n".join(f"- {m}" for m in state["retrieved"])
            prompt.append(
                SystemMessage(
                    "These earlier messages from this conversation were found by search. "
                    f"Use them if they help answer the user:\n{memories}"
                )
            )
        prompt += state["messages"]
        print(f"   -> sending {len(prompt)} messages to the model")
        return {"messages": [llm.invoke(prompt)]}

    def archive(state: State):
        old = state["messages"][:-WINDOW]
        for m in old:
            index.add(f"{'user' if isinstance(m, HumanMessage) else 'assistant'}: {m.content}")
        if old:
            print(f"   ** archived {len(old)} messages into the index ({len(index.docs)} docs, {len(index.postings)} terms)")
        return {"messages": [RemoveMessage(id=m.id) for m in old]}

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("chat", chat)
    builder.add_node("archive", archive)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "chat")
    builder.add_edge("chat", "archive")
    builder.add_edge("archive", END)
    return builder.compile(checkpointer=InMemorySaver())


DEMO_TURNS = [
    "Hi, I'm Deepak. I'm planning a trip to Japan in April.",
    "My laptop is a ThinkPad X1 running Fedora Linux.",
    "Invoice #4512 from Acme Corp is overdue by 30 days, I need to chase them.",
    "My son's birthday is on 14 March and she loves painting.",
    "Explain BM25 in one sentence.",
    "Suggest a name for a cat.",
    # Everything above has left the prompt window by now; only search can bring it back.
    "When is my son's birthday, and what gift could I get her?",
    "Which invoice was overdue, and from which company?",
    "What operating system runs on my laptop?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph full-text search (BM25) memory demo")
    parser.add_argument("-i", "--interactive", action="store_true", help="chat yourself instead of the scripted demo")
    parser.add_argument("-m", "--model", help=f"Ollama model name (default: {OLLAMA_MODEL})")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="past messages to retrieve per turn")
    args = parser.parse_args()

    index = BM25Index()
    graph = build_graph(make_llm(args.model), index, args.top_k)
    config = {"configurable": {"thread_id": "demo"}}

    def turn(text: str):
        print(f"USER: {text}")
        result = graph.invoke({"messages": [HumanMessage(text)]}, config)
        reply = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
        print(f"AI:   {reply.content}\n")

    if args.interactive:
        print("Type 'quit' to exit, 'index' to list the indexed messages.\n")
        while (text := input("> ").strip()).lower() not in {"quit", "exit"}:
            if text == "index":
                print("\n".join(f"[{i}] {d}" for i, d in enumerate(index.docs)) or "(empty)", "\n")
            elif text:
                turn(text)
    else:
        for text in DEMO_TURNS:
            turn(text)
