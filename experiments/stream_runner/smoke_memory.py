"""Sub-step 3 smoke: the memory contract on real Philosophy items, no LLM involved.

    python smoke_memory.py            # needs data/ + manifests/ from sub-step 2
"""
from manifests import load_manifest
from embed import Embedder
from memory import make_memory, count_tokens, CAP

m, items = load_manifest("mmlu_pro_philosophy", 0)
emb = Embedder()
vecs = emb.embed_stream("mmlu_pro_philosophy", items)
print(f"embedded {len(vecs)} items → cache/embeddings/mmlu_pro_philosophy.npy")

none, rag = make_memory("none"), make_memory("exprag", embedder=emb)
order = m["order"][:40]
for pos, iid in enumerate(order):
    q = items[iid]["input_text"]
    block_none, block_rag = none.retrieve(q), rag.retrieve(q)
    assert block_none == ""
    n = count_tokens(block_rag)
    assert n <= CAP, (pos, n)
    fake_answer, correct = items[iid]["target"], (pos % 3 != 0)   # pretend 2/3 correct
    none.evolve(q, fake_answer, correct); rag.evolve(q, fake_answer, correct)
    if pos in (0, 1, 10, 39):
        print(f"pos {pos:2d}: injected {n:4d} tok · stored {rag.size_tokens():5d} tok · "
              f"entries {len(rag._entries)}")

print("\n--- last injected block (head) ---")
print(rag.retrieve(items[order[-1]]["input_text"])[:600])
print("\nOK: cap enforced, memory grows, no-memory stays empty")
