"""No-memory baseline: the floor. Every question is answered cold."""
from .base import Memory


class NoMemory(Memory):
    name = "none"

    def retrieve(self, question: str) -> str:
        return ""

    def evolve(self, question: str, answer: str, correct: bool) -> None:
        pass

    def size_tokens(self) -> int:
        return 0
