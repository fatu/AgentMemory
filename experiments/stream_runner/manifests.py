"""Stream manifests — WHICH items, in WHICH order, from WHICH dataset revision.

    python manifests.py pin                      # resolve + write pins.json (HF dataset revisions)
    python manifests.py build --stream mmlu_pro_philosophy --seed 0
    python manifests.py build-all                # every stream × its seeds
    python manifests.py check                    # re-download and verify every manifest's items_sha256

Outputs (manifests/ is committed; data/ is gitignored):
    manifests/<stream>_s<seed>.json  {stream, dataset, config, split, revision, filter, n, seed,
                                      order: [ids], items_sha256}
    data/<stream>.jsonl              items {id, input_text, target, meta} in dataset order

Every count is asserted against the value fixed in the plan; any mismatch or download
failure raises — there are no fallbacks and no sample data (lesson from the evo_mem audit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST_DIR = HERE / "manifests"
DATA_DIR = HERE / "data"
PINS = HERE / "pins.json"

# stream → (hf repo, config, split, filter, expected n)
STREAMS = {
    "mmlu_pro_engineering": ("TIGER-Lab/MMLU-Pro", None, "test", {"category": "engineering"}, 969),
    "mmlu_pro_economics":   ("TIGER-Lab/MMLU-Pro", None, "test", {"category": "economics"}, 844),
    "mmlu_pro_philosophy":  ("TIGER-Lab/MMLU-Pro", None, "test", {"category": "philosophy"}, 499),
    "gpqa_diamond":         ("Idavidrein/gpqa", "gpqa_diamond", "train", {}, 198),
    "aime24":               ("HuggingFaceH4/aime_2024", None, "train", {}, 30),
    "aime25":               ("math-ai/aime25", None, "test", {}, 30),
}
SEEDS = {"mmlu_pro_engineering": [0, 1, 2], "mmlu_pro_economics": [0, 1, 2],
         "mmlu_pro_philosophy": [0, 1, 2], "gpqa_diamond": [0], "aime24": [0], "aime25": [0]}
LETTERS = string.ascii_uppercase


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


# ------------------------------------------------------------------ loaders
def _pins() -> dict:
    if not PINS.exists():
        raise SystemExit("pins.json missing — run `python manifests.py pin` first")
    return json.loads(PINS.read_text())


def _load(repo, config, split):
    from datasets import load_dataset
    rev = _pins()[repo]
    return load_dataset(repo, config, split=split, revision=rev), rev


def _items_mmlu_pro(ds, flt):
    out = []
    for r in ds:
        if r["category"] != flt["category"]:
            continue
        opts = [o for o in r["options"] if o is not None and o != "N/A"]
        assert 2 <= len(opts) <= 10, r["question_id"]
        body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        target = r["answer"].strip().upper()
        assert target == LETTERS[r["answer_index"]], r["question_id"]
        out.append({
            "id": f"mmlu_pro-{r['question_id']}",
            "input_text": f"{r['question'].strip()}\n\n{body}",
            "target": target,
            "meta": {"category": r["category"], "n_options": len(opts), "src": r.get("src")},
        })
    return out


def _items_gpqa(ds, _flt):
    out = []
    for r in ds:
        rid = str(r["Record ID"])
        correct = r["Correct Answer"].strip()
        options = [correct, r["Incorrect Answer 1"].strip(), r["Incorrect Answer 2"].strip(),
                   r["Incorrect Answer 3"].strip()]
        rng = random.Random(stable_int(f"gpqa-{rid}"))   # per-item, seed-independent shuffle
        rng.shuffle(options)
        target = LETTERS[options.index(correct)]
        body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        out.append({
            "id": f"gpqa-{rid}",
            "input_text": f"{r['Question'].strip()}\n\n{body}",
            "target": target,
            "meta": {"domain": r.get("High-level domain"), "subdomain": r.get("Subdomain")},
        })
    return out


def _items_aime(ds, _flt, prefix):
    out = []
    for i, r in enumerate(ds):
        problem = r.get("problem") or r.get("question")
        ans = str(r["answer"]).strip()
        assert ans.lstrip("-").isdigit(), (prefix, i, ans)
        out.append({
            "id": f"{prefix}-{r.get('id', i)}",
            "input_text": problem.strip(),
            "target": str(int(ans)),
            "meta": {"year": r.get("year"), "url": r.get("url")},
        })
    return out


def build_items(stream: str):
    repo, config, split, flt, n_expected = STREAMS[stream]
    ds, rev = _load(repo, config, split)
    if stream.startswith("mmlu_pro"):
        items = _items_mmlu_pro(ds, flt)
    elif stream == "gpqa_diamond":
        items = _items_gpqa(ds, flt)
    else:
        items = _items_aime(ds, flt, stream)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), f"{stream}: duplicate ids"
    assert len(items) == n_expected, f"{stream}: got {len(items)} items, expected {n_expected}"
    return items, rev


def items_sha256(items) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps(it, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ commands
def cmd_pin(_args):
    from huggingface_hub import HfApi
    api = HfApi()
    pins = {repo: api.dataset_info(repo).sha for repo, *_ in STREAMS.values()}
    PINS.write_text(json.dumps(pins, indent=2) + "\n")
    print(json.dumps(pins, indent=2))


def cmd_build(args):
    streams = [args.stream] if args.stream else list(STREAMS)
    MANIFEST_DIR.mkdir(exist_ok=True); DATA_DIR.mkdir(exist_ok=True)
    for stream in streams:
        items, rev = build_items(stream)
        (DATA_DIR / f"{stream}.jsonl").write_text(
            "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
        sha = items_sha256(items)
        seeds = [args.seed] if args.seed is not None else SEEDS[stream]
        for seed in seeds:
            order = [it["id"] for it in items]
            random.Random(seed).shuffle(order)
            repo, config, split, flt, n = STREAMS[stream]
            m = {"stream": stream, "dataset": repo, "config": config, "split": split,
                 "revision": rev, "filter": flt, "n": n, "seed": seed, "order": order,
                 "items_sha256": sha}
            path = MANIFEST_DIR / f"{stream}_s{seed}.json"
            path.write_text(json.dumps(m, indent=1) + "\n")
            print(f"{path.name}: n={n} revision={rev[:8]} sha={sha[:8]} first={order[:3]}")


def cmd_check(_args):
    ok = True
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        m = json.loads(path.read_text())
        items, rev = build_items(m["stream"])
        sha = items_sha256(items)
        good = sha == m["items_sha256"] and rev == m["revision"] and set(m["order"]) == {i["id"] for i in items}
        ok &= good
        print(f"{path.name}: {'OK' if good else 'MISMATCH'}")
    raise SystemExit(0 if ok else 1)


def load_manifest(stream: str, seed: int):
    """Used by run.py: returns (manifest, {id: item}) after verifying the local data file."""
    m = json.loads((MANIFEST_DIR / f"{stream}_s{seed}.json").read_text())
    items = [json.loads(l) for l in (DATA_DIR / f"{stream}.jsonl").read_text().splitlines()]
    assert items_sha256(items) == m["items_sha256"], f"{stream}: data/ does not match manifest — rebuild"
    return m, {it["id"]: it for it in items}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pin")
    b = sub.add_parser("build"); b.add_argument("--stream", choices=list(STREAMS)); b.add_argument("--seed", type=int)
    sub.add_parser("build-all")
    sub.add_parser("check")
    a = ap.parse_args()
    {"pin": cmd_pin, "build": cmd_build, "build-all": cmd_build, "check": cmd_check}[a.cmd](a)
