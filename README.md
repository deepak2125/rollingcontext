# Rolling context with LangGraph

A small LangGraph example that keeps a conversation's context bounded. It keeps the last
few messages word for word and folds everything older into a running summary.

```
START -> chat -> (more than MAX_MESSAGES?) -> summarize -> END
                         \------------------------------> END
```

- **`chat`** sends `system prompt + summary + recent messages` to the model.
- **`summarize`** runs when state holds more than `MAX_MESSAGES` (6). It asks the model to
  extend the summary with the older messages, then deletes them with `RemoveMessage`.
  Only the last `KEEP_LAST` (2) messages are kept.
- **`InMemorySaver`** checkpointer saves `messages` and `summary` for each `thread_id`
  between calls.

## Run

By default the demo calls a local [Ollama](https://ollama.com) server with `llama3.2`.

```bash
uv sync
uv run rolling-context                    # scripted 7-turn demo on Ollama llama3.2
uv run rolling-context -i                 # chat interactively ('summary' prints the summary)
uv run rolling-context -m gpt-oss:20b     # any model from `ollama list`
```

Set `OLLAMA_HOST` if Ollama isn't at `http://localhost:11434`. `<think>` blocks from
reasoning models such as `deepseek-r1` are stripped before they are saved to state.

Change `MAX_MESSAGES`, `KEEP_LAST` and the default models at the top of
`src/rolling_context/__init__.py`.

# Full-text search (BM25) memory

`src/rolling_context/fts_memory.py` handles old messages differently. Instead of summarizing
them, it moves them into a searchable index and pulls back only the ones relevant to the
current question.

```
START -> retrieve -> chat -> archive -> END
```

- **`archive`** keeps the last `WINDOW` (4) messages in state. Older messages are moved into
  `BM25Index`.
- **Indexing** (`analyze`): tokenize, lowercase, remove stop words, then stem with Snowball,
  so `running`/`runs` both become `run`. Each term maps to `{doc_id: term frequency}`, which
  is the inverted index.
- **`retrieve`** analyzes the question the same way and scores only the documents that
  contain its terms, using BM25 (term frequency, rarity of the term, document length). The
  top `K` hits are shortened to 300 characters.
- **`chat`** adds those hits to the prompt as a system message before the recent messages.

```bash
uv run fts-memory              # scripted demo: facts, filler turns, then recall questions
uv run fts-memory -i           # chat yourself ('index' lists what has been archived)
uv run fts-memory -k 5 -m gpt-oss:20b
```

The demo prints the analyzed search terms and the BM25 score of each hit, so you can see why
each past message was chosen. Here the index is in memory. A real system would usually keep
it in a database, for example SQLite FTS5, which has a built-in `bm25()` ranking function.

# Semantic memory with a vector store

`src/rolling_context/semantic_memory.py` finds memories by **meaning** instead of shared
words. It uses LangGraph's long-term `Store` with a vector index, and `nomic-embed-text`
from Ollama for the embeddings.

```
START -> recall -> chat -> remember -> END
```

- **`remember`** saves each statement from the user (questions are skipped) with
  `store.put(...)`. The store embeds the text into a 768-number vector.
- **`recall`** embeds the new question and runs `store.search(..., query=...)`. Memories
  with cosine similarity of at least `MIN_SCORE` (0.45) are added to the prompt.
- **Short-term vs long-term memory:** the checkpointer keeps the history for each
  `thread_id`. The store keeps memories for each `user_id` (namespace
  `("memories", user_id)`), so a brand-new thread can still recall facts.

The demo runs two threads. On "monday" the user shares facts. On "friday" a fresh thread asks
questions that share almost no words with those facts, for example *"What should I get my
kid for her special day?"* finds *"My daughter turns eight on 14 March and she adores
painting"*. BM25 would miss that match.

```bash
ollama pull nomic-embed-text      # one-time: the embedding model
uv run semantic-memory            # scripted two-thread demo
uv run semantic-memory -i         # chat yourself: 'new' = new thread, 'memories' = list them
uv run semantic-memory -m gpt-oss:20b -k 5
```

`InMemoryStore` is lost when the program exits. For persistence, use `PostgresStore`
(pgvector) with the same `index={...}` config.
