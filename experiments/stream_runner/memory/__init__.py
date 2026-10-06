from pathlib import Path

from .base import CAP, Memory, count_tokens, fit_entries
from .none import NoMemory
from .exprag import ExpRAG
from .dynamic_cheatsheet import DynamicCheatsheet
from .frozen import FrozenSheet

__all__ = ["CAP", "Memory", "count_tokens", "fit_entries", "NoMemory", "ExpRAG", "DynamicCheatsheet", "make_memory"]


def make_memory(name: str, *, embedder=None, client=None, state_path=None) -> Memory:
    """Registry. Systems that need the embedder or the instrumented client get them here;
    later additions (dynamic_cheatsheet, mem0, amem, hindsight, reasoningbank) register in the same table."""
    if name == "none":
        return NoMemory()
    if name == "exprag":
        assert embedder is not None, "exprag needs the shared Embedder"
        return ExpRAG(embedder)
    if name == "dc":
        assert client is not None, "dc needs the instrumented client (curator calls)"
        return DynamicCheatsheet(client, state_path=state_path)
    if name == "dc-frozen":
        return FrozenSheet()                               # FROZEN_SHEET_PATH env
    if name == "mem0":
        from .mem0_adapter import Mem0Memory               # lazy: needs mem0ai + faiss-cpu
        assert client is not None and embedder is not None and state_path is not None
        return Mem0Memory(client, embedder, state_dir=Path(state_path).parent / "mem0")
    if name == "ace":
        from .ace_adapter import ACEMemory                 # lazy: needs the ace checkout (code-repo/ace)
        assert client is not None and state_path is not None
        return ACEMemory(client, state_path=Path(state_path).with_suffix(".ace.json"))
    if name == "amem":
        from .amem_adapter import AMemMemory               # lazy: needs the A-mem checkout deps
        assert client is not None and embedder is not None and state_path is not None
        return AMemMemory(client, embedder, state_path=Path(state_path).with_suffix(".amem.pkl"))
    if name == "hindsight":
        from .hindsight_adapter import HindsightMemory     # lazy: needs hindsight-all (embedded server)
        assert client is not None and embedder is not None and state_path is not None
        return HindsightMemory(client, embedder, state_path=Path(state_path).with_suffix(".hindsight"))
    if name == "reasoningbank":
        from .reasoningbank_adapter import ReasoningBankMemory   # lazy: from-scratch implementation (no official code)
        assert client is not None and embedder is not None and state_path is not None
        return ReasoningBankMemory(client, embedder, state_path=Path(state_path).with_suffix(".reasoningbank.json"))
    raise KeyError(f"unknown memory system {name!r}")
