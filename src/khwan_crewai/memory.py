"""The loop: prepare before the crew runs, record after it finishes.

This is the half that makes memory accumulate. Tools let an agent look something
up when it thinks to; this runs whether it thought to or not, and it is what the
next session recalls.

It wraps the crew rather than living inside it, because what is worth persisting
is the answer that actually went out — not each intermediate step an agent took
on the way there.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from khwan import AsyncKhwan, KhwanError, Turn

logger = logging.getLogger(__name__)


def seed_text(turn: Turn) -> str:
    """A memory block to drop into a crew's inputs.

    Lessons lead: a rule earned over months outranks a single turn that happens
    to sit nearby in the index, and an agent reading the block should be able to
    tell a standing rule from one recalled exchange.

    Empty when the brain has nothing relevant — retrieval applies a relevance
    floor, so that is an answer rather than a failure. Interpolating an empty
    string is the right thing to do with it.
    """
    blocks: List[str] = []
    if turn.lessons:
        blocks.append("What we have learned (applies generally):\n"
                      + "\n".join(f"- {l}" for l in turn.lessons))
    facts = []
    for s in turn.sources or []:
        if not isinstance(s, dict) or not s.get("response"):
            continue
        asked = (s.get("input") or "").strip()
        answered = (s.get("response") or "").strip()
        facts.append(f"- {answered}" if not asked or asked == answered
                     else f"- {asked} → {answered}")
    if facts:
        blocks.append("Relevant memory:\n" + "\n".join(facts))
    return "\n\n".join(blocks)


class KhwanMemory:
    """Memory around a crew run.

        async with KhwanMemory(api_key=..., core="acme", user_id="Web") as mem:
            turn   = await mem.prepare(user_input)
            result = await crew.kickoff_async(inputs={
                "memory": mem.seed_text(turn),
                "question": user_input,
            })
            await mem.record(turn, str(result))

    Every call fails **open**: memory that is down must never stop an answer. A
    failed prepare means the crew runs without recall; a failed record means the
    turn is not learned. Neither is worth an error to the user, and both are
    logged at warning so a persistent outage is visible rather than silent.
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
        self._kw = AsyncKhwan(**kwargs)

    async def __aenter__(self) -> "KhwanMemory":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the pool, after any background records have landed."""
        await self._kw.aclose()

    # ---- the loop ----
    async def prepare(self, user_input: str) -> Optional[Turn]:
        """Recall for this turn. `None` when Khwan could not be reached."""
        try:
            return await self._kw.prepare(user_input)
        except (KhwanError, Exception) as e:  # noqa: BLE001 — fail open, always
            logger.warning("khwan prepare failed, running without memory: %s", e)
            return None

    async def record(self, turn: Optional[Turn], answer: str,
                     *, background: bool = False) -> bool:
        """Persist what the crew actually said. False when it did not land.

        A turn that was refused by the coherence gate has no token, and there is
        nothing to record against — that is not an error either.
        """
        if turn is None or not turn.turn_token or not answer:
            return False
        try:
            await self._kw.record(turn, answer, background=background)
            return True
        except Exception as e:  # noqa: BLE001 — a failed learn must not break the reply
            logger.warning("khwan record failed, this turn is not learned: %s", e)
            return False

    async def verify(self, turn: Optional[Turn], draft: str) -> dict:
        """Check a draft against the brain before it ships.

        `prepare` gates the turn; this gates the *answer*. Unreachable counts as
        supported — blocking a reply on a network blip is worse than shipping an
        unchecked one.
        """
        if turn is None:
            return {"ok": True, "reason": "no prepared turn"}
        try:
            return await self._kw.verify(turn, draft)
        except Exception as e:  # noqa: BLE001
            logger.warning("khwan verify unavailable, treating as supported: %s", e)
            return {"ok": True, "reason": f"verify unavailable ({e})"}

    # ---- convenience ----
    @staticmethod
    def seed_text(turn: Optional[Turn]) -> str:
        """The recalled memory as a block for the crew's inputs ("" if none)."""
        return seed_text(turn) if turn is not None else ""
