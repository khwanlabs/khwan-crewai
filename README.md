# khwan-crewai

**Memory for a CrewAI crew that outlives the run.** Plugs
[Khwan](https://khwan.ai) — a pure AI-memory layer — into CrewAI as tools and as
a wrapper around a crew run.

Khwan never runs your model. The crew's model is the "your model" step:

```
prepare()  ──►  recall memory + constitution     (no LLM)
   [ the crew runs ]                             ← Khwan never touches it
record()   ──►  persist what was actually said
```

```bash
pip install khwan-crewai
```

## Two halves, and you want both

**Tools** let an agent look something up when it decides to. **The loop** runs
whether it decided to or not — and that is what the next session recalls.

### The loop — where memory accumulates

```python
from khwan_crewai import KhwanMemory

async with KhwanMemory(api_key="kwk_live_xxx", core="acme") as mem:
    turn   = await mem.prepare(user_input)
    result = await crew.kickoff_async(inputs={
        "memory":   mem.seed_text(turn),   # "" when nothing is known
        "question": user_input,
    })
    await mem.record(turn, str(result))
```

It wraps the crew rather than living inside it, because what is worth keeping is
the answer that actually went out — not every intermediate step an agent took to
get there.

### Tools — what an agent reaches for mid-task

```python
from crewai import Agent
from khwan_crewai import khwan_tools

agent = Agent(
    role="Analyst",
    tools=khwan_tools(api_key="kwk_live_xxx", core="acme"),
    ...
)
```

| Tool | What it does |
| --- | --- |
| `khwan_recall(query)` | standing rules + past exchanges on a topic |
| `khwan_remember(fact)` | keep a decision or preference for future runs |
| `khwan_verify(draft)` | check a draft against memory before it ships |

## Everything fails open

Memory that is down must never stop an answer. A failed `prepare` means the crew
runs without recall; a failed `record` means the turn is not learned; a tool that
cannot reach Khwan returns a sentence saying so rather than raising, because a
raising tool aborts the agent's step. All of it logs at warning, so an outage is
visible rather than silent.

## An empty recall is an answer

Retrieval applies a relevance floor, so no results means the brain holds nothing
close to this question — not that something went wrong. `khwan_recall` says so in
words the agent can repeat, and `seed_text` returns `""`, which interpolates into
a prompt harmlessly.

## Isolation

One account holds many **cores** (separate brains), and each core can hold a
**sub-brain** per end-user — a complete separate brain that does not spend one of
your cores. A client with several projects is usually one core with a sub-brain
each:

```python
KhwanMemory(api_key=..., core="acme", user_id="Web")   # account::acme::@Web
KhwanMemory(api_key=..., core="acme", user_id="Api")   # shares nothing with @Web
```

Cores must exist before you point at one — create them in the
[dashboard](https://app.khwan.ai). Sub-brains are created on first write.

## Configuration

| Argument | Required | Purpose |
| --- | --- | --- |
| `api_key` | yes | From the [dashboard](https://app.khwan.ai) (`kwk_live_…`). Not your model provider's key. |
| `core` | no | Which isolated brain (default: the account's default core). |
| `user_id` | no | Sub-brain within that core (paid plans). |
| `base_url` | no | Override the API base, e.g. a local engine. |

## Related

- [`khwan`](https://pypi.org/project/khwan/) — the client underneath, sync and async
- [`khwan-mcp`](https://pypi.org/project/khwan-mcp/) — the same memory for any MCP client
- [docs.khwan.ai](https://docs.khwan.ai)

## Source

[github.com/khwanlabs/khwan-crewai](https://github.com/khwanlabs/khwan-crewai) —
this sits between your crew and its memory and runs with your key. Read it before
you install it.

## License

MIT — © Khwan Labs.
