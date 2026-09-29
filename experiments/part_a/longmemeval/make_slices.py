"""Pin the frozen LongMemEval-S slices (step 10): nested, stratified, seeded.

    python make_slices.py            # writes slices.json next to this file

Strata = (question_type, is_abs) — 12 cells, so abstention items sit inside their type in the
same proportion as in the full 500 (the official aggregation keeps them there). Each stratum
gets one seeded permutation; a slice of size n takes the first k_s items of every stratum
with k_s from largest-remainder rounding of n · |s| / 500. Because every slice is a prefix of
the same permutations, 20 ⊂ 50 ⊂ 100 ⊂ 150 hold by construction, and any system run on the
150 has every row a 20/50/100 system was run on → paired comparisons.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from lme_data import load_all, is_abs

HERE = Path(__file__).resolve().parent
SIZES = [20, 50, 100, 150]
SEED = 0


def allocate(n: int, counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder apportionment of n over strata with sizes `counts` (sum = total)."""
    raw = {s: n * c / total for s, c in counts.items()}
    k = {s: int(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - k[s], reverse=True)[: n - sum(k.values())]:
        k[s] += 1
    return k


def main():
    data = load_all()
    strata: dict[str, list[str]] = defaultdict(list)
    for e in data:
        strata[f"{e['question_type']}|{'abs' if is_abs(e) else 'ans'}"].append(e["question_id"])
    rng = random.Random(SEED)
    perms = {s: rng.sample(ids, len(ids)) for s, ids in sorted(strata.items())}
    counts = {s: len(ids) for s, ids in strata.items()}
    slices = {}
    for n in SIZES:
        k = allocate(n, counts, len(data))
        ids = [qid for s in sorted(perms) for qid in perms[s][: k[s]]]
        assert len(ids) == n
        slices[str(n)] = ids
    for a, b in zip(SIZES, SIZES[1:]):                       # nesting check
        assert set(slices[str(a)]) <= set(slices[str(b)])
    out = {"seed": SEED, "strata": "question_type × is_abs", "counts": counts, "sizes": SIZES, "slices": slices,
           "per_type": {n: {s: sum(1 for q in slices[n] if q in set(perms[s])) for s in perms} for n in slices}}
    (HERE / "slices.json").write_text(json.dumps(out, indent=1))
    for n in SIZES:
        by = out["per_type"][str(n)]
        print(f"{n:>4}: " + " · ".join(f"{s.split('|')[0][:12]}{'*' if s.endswith('abs') else ''} {c}" for s, c in by.items() if c))
    print("wrote", HERE / "slices.json")


if __name__ == "__main__":
    main()
