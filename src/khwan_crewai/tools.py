"""Khwan as CrewAI tools — what an agent can reach for mid-task.

CrewAI runs tools synchronously, so these use the blocking client. That is the
right call rather than a compromise: a tool runs inside the crew's own execution,
and bridging to an event loop there buys nothing and costs a class of bugs.

The counterpart is `KhwanMemory`, which wraps the whole run. Tools alone do not
accumulate memory — an agent calls them when it thinks to, which is not the same
as always.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool
from khwan import Khwan, KhwanError
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


class _RecallArgs(BaseModel):
    query: str = Field(description="The task or topic to recall memory for.")


class _RememberArgs(BaseModel):
    fact: str = Field(description="A durable fact, decision or preference to keep.")


class _VerifyArgs(BaseModel):
    draft: str = Field(description="The answer you are about to give, before giving it.")


class _KhwanTool(BaseTool):
    """Shared plumbing: one client for every tool, and failures that read as text.

    A tool that raises aborts the agent's step. A tool that returns a sentence
    saying it could not reach memory lets the agent carry on and say so — which
    is what should happen when the memory layer, not the task, is broken.
    """

    _kw: Khwan = PrivateAttr()

    def __init__(self, client: Khwan, **data: Any):
        super().__init__(**data)
        self._kw = client


class KhwanRecallTool(_KhwanTool):
    name: str = "khwan_recall"
    description: str = (
        "Recall what is already known about a topic from previous sessions, before "
        "working on it. Returns standing rules and past exchanges. An empty result "
        "means the memory holds nothing relevant — say so rather than guessing."
    )
    args_schema: Type[BaseModel] = _RecallArgs

    def _run(self, query: str) -> str:
        from .memory import seed_text
        try:
            turn = self._kw.prepare(query)
        except KhwanError as e:
            logger.warning("khwan_recall failed: %s", e)
            return f"khwan_recall: memory unavailable ({e}). Continue without it."
        text = seed_text(turn)
        return text or "khwan_recall: nothing relevant is known about this yet."


class KhwanRememberTool(_KhwanTool):
    name: str = "khwan_remember"
    description: str = (
        "Persist a durable fact, decision or preference so future sessions have it. "
        "Use for things that outlive this run — not for working notes."
    )
    args_schema: Type[BaseModel] = _RememberArgs

    def _run(self, fact: str) -> str:
        fact = (fact or "").strip()
        if not fact:
            return "khwan_remember: nothing to store."
        try:
            turn = self._kw.prepare(fact)
            if not turn.turn_token:
                return f"khwan_remember: not stored ({turn.reason or 'declined'})."
            self._kw.record(turn, fact)
        except KhwanError as e:
            logger.warning("khwan_remember failed: %s", e)
            return f"khwan_remember: could not store it ({e})."
        return "khwan_remember: stored."


class KhwanVerifyTool(_KhwanTool):
    name: str = "khwan_verify"
    description: str = (
        "Check a draft answer against memory before giving it. Returns BLOCKED when "
        "the draft contradicts what is known — revise and check again rather than "
        "sending a blocked draft."
    )
    args_schema: Type[BaseModel] = _VerifyArgs

    def _run(self, draft: str) -> str:
        try:
            # /verify accepts an answer with no turn_token; it scores against the
            # brain rather than against one prepared turn.
            from khwan import Turn
            data = self._kw.verify(Turn({}), draft or "")
        except KhwanError as e:
            logger.warning("khwan_verify unavailable: %s", e)
            return "khwan_verify: unavailable, treat as supported."
        verdict = "supported" if data.get("ok", True) else "BLOCKED"
        reason = data.get("reason") or ""
        return f"khwan_verify: {verdict} (coherence={data.get('coherence')}) {reason}".strip()


class KhwanTools:
    """Every Khwan tool, sharing one client.

        tools = KhwanTools(api_key="kwk_live_xxx", core="acme", user_id="Web")
        agent = Agent(..., tools=tools.as_list())
    """

    def __init__(self, *, api_key: str, core: Optional[str] = None,
                 user_id: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: int = 60):
        kwargs: dict = {"api_key": api_key, "timeout": timeout}
        if core:
            kwargs["core"] = core
        if user_id:
            kwargs["user_id"] = user_id
        if base_url:
            kwargs["base_url"] = base_url
        self.client = Khwan(**kwargs)
        self.recall = KhwanRecallTool(self.client)
        self.remember = KhwanRememberTool(self.client)
        self.verify = KhwanVerifyTool(self.client)

    def as_list(self) -> List[BaseTool]:
        return [self.recall, self.remember, self.verify]


def khwan_tools(**kwargs: Any) -> List[BaseTool]:
    """One-liner for the common case: `tools=khwan_tools(api_key=..., core=...)`."""
    return KhwanTools(**kwargs).as_list()
