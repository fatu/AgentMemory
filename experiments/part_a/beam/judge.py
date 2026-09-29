"""Sonnet-5 judge running BEAM's official scoring VERBATIM (src/evaluation/compute_metrics.py
@ b2da22e; the public implementation configures gpt-4.1-mini at temperature 0):

  nugget judge   `unified_llm_judge_base_prompt` (imported from src/prompts.py) with
                 <rubric_item> / <llm_response> filled — the question itself is NOT in the
                 prompt (official) — one call per rubric nugget, JSON {score, reason},
                 score ∈ {1, 0.5, 0}; question score = mean over nuggets.
  event ordering `event_ordering_score(rubric, response.split("\\n"), align_type="llm")`:
                 `llm_equivalence` YES/NO alignment of each response line against the
                 unmatched rubric events, then Kendall τ_b → tau_norm = (τ_b + 1) / 2 —
                 the field report_results.py aggregates for this ability; the nugget judge
                 runs on it too (llm_judge_score), as the official evaluator does.
`event_ordering_score` and `parse_json_response` are copied from compute_metrics.py with two
edits: the alignment dispatch takes a callable (our client-bound align_with_llm) instead of
(align_type, llm), and a NaN τ_b (degenerate ranks) maps to 0 instead of propagating.
(Importing compute_metrics would pull sentence-transformers / rouge / nltk for metrics BEAM
does not report.)
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple

from scipy.stats import kendalltau

from beam_data import unified_llm_judge_base_prompt

NUGGET_MAX_TOKENS = 400          # {"score", "reason"} — the reason is free text
_SCORE_RE = re.compile(r'"score"\s*:\s*"?([01](?:\.[05])?)')

# llm_equivalence (compute_metrics.py:105-133), messages verbatim incl. whitespace
EQUIV_SYSTEM = """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """
EQUIV_USER = """First snippet: {first} \n
                       Second snippet: {second}
                    """


# ------------------------------------------------------------ copied from compute_metrics.py
def parse_json_response(response: str):
    response = response.strip()

    if response.startswith("```"):
        match = re.search(
            r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r'(\{.*?\}|\[.*?\])', response, re.DOTALL)
    if match:
        json_part = match.group(1)
        try:
            return json.loads(json_part)
        except Exception as e:
            raise ValueError(
                f"Found possible JSON but failed to parse it: {e}")

    raise ValueError("No valid JSON found in response.")


def event_ordering_score(reference_list: List[str],
                         system_list: List[str],
                         align: "callable") -> dict:
    """compute_metrics.py:270-308 with align_type='llm'; `align` = align_with_llm bound to
    our judge client."""
    reference_canon, system_canon = align(reference_list, system_list)

    tp = len(set(reference_canon) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference_canon])
    fn = len([x for x in reference_canon if x not in system_canon])

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0

    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq):
        r = {item: i+1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b, p_value = kendalltau(to_rank(reference_canon),
                                to_rank(system_canon),
                                variant="b", method="auto")
    tau_b_norm = (tau_b + 1) / 2 if tau_b is not None and tau_b == tau_b else 0   # NaN-safe
    final_score = tau_b_norm * f1
    return dict(precision=precision, recall=recall, f1=f1,
                tau_norm=tau_b_norm, final_score=final_score)
# ------------------------------------------------------------ end of copied code


class Judge:
    def __init__(self, client):
        self.client = client
        self.calls = 0
        self.equiv_log: list[dict] = []          # every alignment decision (ref, line, yes) — judge audit

    def _nugget(self, item: str, response: str, row_id: str) -> tuple[float | None, str]:
        prompt = unified_llm_judge_base_prompt.replace("<rubric_item>", item).replace("<llm_response>", response)
        raw = self.client.complete("", prompt, tag="judge", row_id=row_id, max_tokens=NUGGET_MAX_TOKENS,
                                   temperature=0.0, thinking=False) or ""
        self.calls += 1
        try:
            return float(parse_json_response(raw)["score"]), raw
        except Exception:
            m = _SCORE_RE.search(raw)
            return (float(m.group(1)) if m else None), raw

    def _equivalent(self, first: str, second: str, row_id: str) -> bool:
        raw = self.client.complete(EQUIV_SYSTEM, EQUIV_USER.format(first=first, second=second), tag="judge",
                                   row_id=row_id, max_tokens=10, temperature=0.0, thinking=False) or ""
        self.calls += 1
        verdict = raw.lower().replace("*", "").strip()          # "**YES**" / "Yes." / "\nNO" → yes|no
        yes = verdict.startswith("yes")                          # official: "yes" in response.lower()
        self.equiv_log.append({"row_id": row_id, "ref": first, "line": second, "raw": raw, "yes": yes})
        return yes

    def _align_with_llm(self, reference: List[str], system: List[str], row_id: str) -> Tuple[List[str], List[str]]:
        """align_with_llm (compute_metrics.py:136-162)."""
        used, system_out = set(), []
        for s in system:
            matched_index = None
            for index, r in enumerate(reference):
                if index in used:
                    continue
                if self._equivalent(r, s, row_id):
                    matched_index = index
                    break
            if matched_index is not None:
                system_out.append(reference[matched_index]); used.add(matched_index)
            else:
                system_out.append(s)
        return reference, system_out

    def score(self, *, ability: str, rubric: list[str], response: str, row_id: str) -> dict:
        """Returns {score, llm_judge_score, tau_norm, f1, n_nuggets, judge_calls, nugget_scores,
        parse_fails}; `score` is the field the official report aggregates for the ability."""
        c0 = self.calls
        e0 = len(self.equiv_log)
        nuggets, fails, raws = [], 0, []
        for item in rubric:
            s, raw = self._nugget(item, response, row_id)
            raws.append(raw)
            if s is None:
                fails += 1
            else:
                nuggets.append(s)
        llm_judge = (sum(nuggets) / len(nuggets)) if nuggets else None
        out = {"llm_judge_score": llm_judge, "tau_norm": None, "f1": None, "n_nuggets": len(rubric),
               "nugget_scores": nuggets, "parse_fails": fails, "raw": raws}
        if ability == "event_ordering":
            eo = event_ordering_score(list(rubric), response.split("\n"),
                                      lambda ref, sys_: self._align_with_llm(ref, sys_, row_id))
            out.update(tau_norm=eo["tau_norm"], f1=eo["f1"])
            out["score"] = eo["tau_norm"]
        else:
            out["score"] = llm_judge
        out["judge_calls"] = self.calls - c0
        out["equiv"] = self.equiv_log[e0:]
        return out
