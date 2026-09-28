"""The memory contract every Part B system implements (spec: blog draft §5.1).

    retrieve(question) -> str      text injected into the prompt; <= CAP o200k tokens
    evolve(question, answer, correct)   called AFTER scoring; binary feedback only,
                                        never the reference answer
    size_tokens() -> int           current stored size (reported per row)

Systems that call an LLM to update memory receive the instrumented client and tag
those calls `memory`; the runner never sees inside.
"""
from __future__ import annotations

import tiktoken

CAP = 4096
_ENC = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def fit_entries(entries: list[str], cap: int = CAP, sep: str = "\n\n") -> str:
    """Concatenate whole entries in order until adding the next would exceed `cap`."""
    out, used = [], 0
    for e in entries:
        n = count_tokens(e) + (count_tokens(sep) if out else 0)
        if used + n > cap:
            break
        out.append(e); used += n
    return sep.join(out)


class Memory:
    name = "base"

    def retrieve(self, question: str) -> str:
        raise NotImplementedError

    def evolve(self, question: str, answer: str, correct: bool) -> None:
        raise NotImplementedError

    def size_tokens(self) -> int:
        raise NotImplementedError
