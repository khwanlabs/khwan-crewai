"""khwan-crewai — tools an agent can call, and the loop that actually remembers.

Run against real CrewAI (no stubs) with the HTTP layer mocked, so what is pinned
is the contract with CrewAI's BaseTool and the behaviour that matters when memory
is unavailable: a tool must never abort the agent's step, and a crew must never
fail because the memory layer did.

Run: python3 test_khwan_crewai.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import httpx
import requests

from khwan import Turn
from khwan_crewai import KhwanMemory, KhwanTools, khwan_tools
from khwan_crewai.memory import seed_text

BASE = "https://example.invalid"


class _FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload or {}, text
        self.headers, self.content = {}, b"x"

    def json(self):
        return self._payload


def _patch_requests(monkey):
    """Point the SYNC client at a callable, since tools run synchronously."""
    requests.request = monkey


def test_tools_are_real_crewai_tools():
    from crewai.tools import BaseTool
    tools = khwan_tools(api_key="k", core="acme", base_url=BASE)
    assert len(tools) == 3
    assert all(isinstance(t, BaseTool) for t in tools)
    names = {t.name for t in tools}
    assert names == {"khwan_recall", "khwan_remember", "khwan_verify"}, names
    # A description is what the model routes on; an empty one makes the tool dead
    # weight in the prompt.
    assert all(len(t.description) > 40 for t in tools)
    print("✓ tools: three real CrewAI BaseTools, each described")


def test_recall_formats_lessons_first_then_facts():
    def fake(method, url, **kw):
        return _FakeResponse(200, {
            "lessons": ["Answer in Thai."],
            "sources": [{"input": "which db?", "response": "Postgres", "similarity": 0.4}],
            "turn_token": "t1",
        })
    _patch_requests(fake)
    out = KhwanTools(api_key="k", base_url=BASE).recall._run("db choice")
    assert out.index("Answer in Thai.") < out.index("Postgres"), out
    assert "which db? → Postgres" in out, out
    print("✓ recall: standing rules lead, recalled exchanges follow")


def test_recall_says_nothing_is_known_rather_than_inventing():
    """The relevance floor means empty is an ANSWER. A tool that returns nothing
    invites the agent to fill the silence."""
    _patch_requests(lambda *a, **k: _FakeResponse(200, {"lessons": [], "sources": [],
                                                        "turn_token": "t1"}))
    out = KhwanTools(api_key="k", base_url=BASE).recall._run("green curry recipe")
    assert "nothing relevant is known" in out, out
    print("✓ recall: an empty brain says so, in words the agent can repeat")


def test_a_tool_never_raises_into_the_agent():
    """A raising tool aborts the step. Memory being down is not the task failing."""
    def boom(*a, **k):
        raise requests.RequestException("connection refused")
    _patch_requests(boom)
    t = KhwanTools(api_key="k", base_url=BASE)
    assert "unavailable" in t.recall._run("x")
    assert "could not store" in t.remember._run("a fact")
    assert "unavailable" in t.verify._run("a draft")
    print("✓ tools: an unreachable Khwan returns text, never an exception")


def test_remember_reports_a_refusal_instead_of_claiming_success():
    _patch_requests(lambda *a, **k: _FakeResponse(200, {"turn_token": None,
                                                        "reason": "gate declined"}))
    out = KhwanTools(api_key="k", base_url=BASE).remember._run("something")
    assert "not stored" in out and "gate declined" in out, out
    print("✓ remember: a refused write is reported, not silently swallowed")


# ---- the loop ----

def _async_client(handler):
    mem = KhwanMemory(api_key="k", core="acme", base_url=BASE)
    mem._kw._client = httpx.AsyncClient(
        base_url=BASE, transport=httpx.MockTransport(handler),
        headers={"X-API-Key": "k"})
    return mem


def test_loop_prepares_and_records_around_the_crew():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == "/prepare":
            return httpx.Response(200, json={"turn_token": "t1",
                                             "lessons": ["Be brief."], "sources": []})
        return httpx.Response(200, json={})

    async def go():
        async with _async_client(handler) as mem:
            turn = await mem.prepare("what did we decide?")
            assert "Be brief." in mem.seed_text(turn)
            assert await mem.record(turn, "we decided X") is True

    asyncio.run(go())
    assert seen == ["/prepare", "/record"], seen
    print("✓ loop: prepare before, record after, in that order")


def test_the_crew_still_runs_when_khwan_is_down():
    """Fail open, everywhere. A memory outage must cost recall, not the answer."""
    def dead(request):
        raise httpx.ConnectError("no route to host")

    async def go():
        async with _async_client(dead) as mem:
            turn = await mem.prepare("anything")
            assert turn is None                      # ran without recall
            assert mem.seed_text(turn) == ""         # interpolates to nothing
            assert await mem.record(turn, "an answer") is False
            assert (await mem.verify(turn, "a draft"))["ok"] is True

    asyncio.run(go())
    print("✓ loop: Khwan down → no recall, no learning, and no failure")


def test_record_of_a_refused_turn_is_a_no_op():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/prepare":
            return httpx.Response(200, json={"turn_token": None, "allowed": False,
                                             "reason": "blocked"})
        return httpx.Response(200, json={})

    async def go():
        async with _async_client(handler) as mem:
            turn = await mem.prepare("x")
            assert await mem.record(turn, "an answer") is False

    asyncio.run(go())
    assert calls == ["/prepare"], f"recorded against a refused turn: {calls}"
    print("✓ loop: a refused turn has nothing to record, and nothing is sent")


def test_seed_text_shows_a_stored_fact_once():
    """khwan_remember stores input == response. Rendering it as 'X → X' is noise."""
    turn = Turn({"sources": [{"input": "We bill monthly.", "response": "We bill monthly."}]})
    assert seed_text(turn).count("We bill monthly.") == 1, seed_text(turn)
    print("✓ seed_text: a remembered fact appears once, not as a pair")


if __name__ == "__main__":
    _real = requests.request
    try:
        test_tools_are_real_crewai_tools()
        test_recall_formats_lessons_first_then_facts()
        test_recall_says_nothing_is_known_rather_than_inventing()
        test_a_tool_never_raises_into_the_agent()
        test_remember_reports_a_refusal_instead_of_claiming_success()
    finally:
        requests.request = _real
    test_loop_prepares_and_records_around_the_crew()
    test_the_crew_still_runs_when_khwan_is_down()
    test_record_of_a_refused_turn_is_a_no_op()
    test_seed_text_shows_a_stored_fact_once()
    print("\n✅ khwan-crewai verified against crewai 0.203.2")
