"""Khwan memory for CrewAI.

Two halves, because a crew needs both and they are not the same shape:

* **Tools** (`KhwanTools`) — `recall`, `remember`, `verify` for the agent to call
  when it decides to. CrewAI runs tools synchronously, so these do.
* **The loop** (`KhwanMemory`) — `prepare` before the crew runs and `record`
  after, around the whole thing rather than inside it. This is where memory
  actually accumulates, and it is async because that is where a crew is driven
  from.

Khwan never runs your model. The crew's model is the "your model" step:

    prepare()  ──► recall memory + constitution   (no LLM)
      [ the crew runs ]                           ← Khwan never touches it
    record()   ──► persist what was actually said
"""

from .memory import KhwanMemory
from .tools import KhwanTools, khwan_tools

__all__ = ["KhwanMemory", "KhwanTools", "khwan_tools"]
__version__ = "0.1.0"
