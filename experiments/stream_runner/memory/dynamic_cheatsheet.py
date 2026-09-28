"""Dynamic Cheatsheet, ours — DC-CU style with binary feedback (blog draft §5.1).

One evolving cheatsheet of generalizable strategies, heuristics and pitfalls. The WHOLE
sheet is injected before every question (no retrieval); after every scored answer a
curator LLM call REWRITES the whole sheet from (previous sheet, question, the model's
final answer, correct|wrong). The curator never sees the reference answer.

Reference: Suzgun et al. 2025 (arXiv 2504.07952), curator prompt for DC-Cu
(`dc_curator_reference.txt`, MIT). Ours is adapted: multiple-choice reasoning + integer
math instead of code, a feedback rule (wrong → pitfall, never a guessed answer), and a
ban on storing question–answer pairs (that is ExpRAG's job).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .base import CAP, Memory, count_tokens

CURATOR_MAX_TOKENS = 6000    # smoke: 3000 cut 11/20 rewrites mid-sheet → sheet froze for 12 steps
TARGET_WORDS = 2000          # ≈ 2.7K tokens; under the 4,096 injection cap with margin. Pilot: curators
                             # overshoot any target (~350 tok/step growth) → the count is stated per call
_SHEET_RE = re.compile(r"<\s*cheatsheet\s*>(.*?)<\s*/\s*cheatsheet\s*>", re.S | re.I)
_OPEN_RE = re.compile(r"<\s*cheatsheet\s*>(.*)$", re.S | re.I)
_ITEM_CLOSE = re.compile(r"<\s*/\s*memory_item\s*>", re.I)
_JUNK_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")      # zero-width chars / BOM some models emit


def _normalize(reply: str) -> str:
    reply = _JUNK_RE.sub("", reply or "")
    return reply.replace("```", "")                           # tags inside a code fence still parse

CURATOR_SYSTEM = f"""# CHEATSHEET CURATOR

You maintain a single, continuously evolving cheatsheet that helps a model answer a
stream of questions: multiple-choice questions in philosophy, formal logic, economics,
engineering and the sciences, and integer-answer competition math problems.

After each question you receive: the PREVIOUS CHEATSHEET, the CURRENT QUESTION, the
MODEL'S FINAL ANSWER, and the RESULT (correct or wrong). You never receive the reference
answer. Produce the NEW CHEATSHEET.

## What belongs in the cheatsheet
- Reusable strategies and procedures (e.g. how to translate "X is a sufficient condition
  for Y" into a conditional; how to test a syllogism for validity; elimination tactics
  for multiple-choice questions; sanity checks for integer answers).
- Domain facts and definitions that recur across questions (named positions, theorems,
  formulas), stated generally.
- Pitfalls: patterns that led to wrong answers and what to do instead.
- Meta-reasoning heuristics that make answers more reliable.

## What must NOT be in the cheatsheet
- Specific question–answer pairs, or anything that only works for one question.
- Guesses about the correct answer to a question that was answered wrongly. When the
  RESULT is wrong, record what went wrong or what to check next time — do not invent
  the right answer.
- Verbatim copies of questions.

## How to update
1. Assess the model's answer in light of the RESULT.
   - correct: if the reasoning strategy is reusable, add or strengthen a memory item;
     increase its usage count if an existing item was used.
   - wrong: add or refine a pitfall item describing the failure mode and a check to
     apply next time; do not state the right answer.
2. Preserve useful content. Anything in the previous cheatsheet that you do not copy
   into the new one is LOST. Copy forward every item that remains useful; merge
   duplicates; drop only trivial or superseded items.
3. Keep it general, concise and actionable. Tag items with the question numbers they
   came from, e.g. (Q3, Q17).
4. Keep the whole cheatsheet under about {TARGET_WORDS} words.

## Format
Output ONLY the new cheatsheet, wrapped exactly like this:

<cheatsheet>
Version: <n>

STRATEGIES AND PROCEDURES
<memory_item>
<description>[what it is for, when to use it] (Q…)</description>
<content>[the strategy, procedure or fact, stated generally]</content>
<count>[times used successfully]</count>
</memory_item>

PITFALLS
<memory_item>
[...]
</memory_item>

META-REASONING
<memory_item>
[...]
</memory_item>
</cheatsheet>
"""

SHORTEN_SUFFIX = ("\n\nThe previous version was too long. Rewrite the NEW CHEATSHEET to at most "
                  "{words} words, keeping the most reusable items.")


class DynamicCheatsheet(Memory):
    name = "dc"

    def __init__(self, client, cap: int = CAP, state_path: str | Path | None = None):
        self.client = client
        self.cap = cap
        self.sheet = ""
        self.version = 0
        self.n_updates = 0
        self.parse_fails = 0
        self.salvaged = 0                       # closing tag missing → kept complete items only
        self.compressions = 0                   # sheet hit the cap → shorten retry fired
        self.truncations = 0
        self.position = 0                       # question counter for (Q<n>) tags
        self.state_path = Path(state_path) if state_path else None

    # ------------------------------------------------------------ interface
    def retrieve(self, question: str) -> str:
        if not self.sheet:
            return ""
        return f"## Memory (your cheatsheet, v{self.version})\n{self.sheet}"

    def evolve(self, question: str, answer: str, correct: bool) -> None:
        self.position += 1
        user = self._user_message(question, answer, correct)
        new = self._curate(user)
        if new is None:
            self.parse_fails += 1               # keep the old sheet
        else:
            if count_tokens(self._header() + new) > self.cap:
                self.compressions += 1
                shorter = self._curate(user + SHORTEN_SUFFIX.format(words=TARGET_WORDS // 2))
                new = shorter if shorter is not None else new
            self.sheet = self._fit(new)
            self.version += 1
        self.n_updates += 1
        self.dump()

    def size_tokens(self) -> int:
        return count_tokens(self.sheet)

    # ------------------------------------------------------------ internals
    def _header(self) -> str:
        return f"## Memory (your cheatsheet, v{self.version + 1})\n"

    def _user_message(self, question: str, answer: str, correct: bool) -> str:
        return (
            f"## PREVIOUS CHEATSHEET (~{len(self.sheet.split())} words; keep the new one under {TARGET_WORDS} words)\n"
            f"{self.sheet if self.sheet else '(empty — this is the first question)'}\n\n"
            f"## CURRENT QUESTION (Q{self.position})\n{question}\n\n"
            f"## MODEL'S FINAL ANSWER\n{answer if answer else '(no parsable answer)'}\n\n"
            f"## RESULT\n{'correct' if correct else 'wrong'}\n\n"
            f"Now write the NEW CHEATSHEET (Version: {self.version + 1})."
        )

    def _curate(self, user: str) -> str | None:
        row_id = getattr(self.client, "current_row_id", None) or f"curate:{self.position}"
        raw = self.client.complete(
            CURATOR_SYSTEM, user, tag="memory", row_id=row_id,
            max_tokens=CURATOR_MAX_TOKENS, thinking=False,
        ) or ""
        reply = _normalize(raw)
        m = _SHEET_RE.search(reply)
        if m:
            return m.group(1).strip()
        # No closing tag: output cut off or malformed. Salvage every complete item, whether
        # or not the opening tag survived.
        m = _OPEN_RE.search(reply)
        body = m.group(1) if m else reply
        closes = list(_ITEM_CLOSE.finditer(body))
        if closes:
            body = body[: closes[-1].end()]
            self.salvaged += 1
            self._dump_reply(raw, "salvaged")
            print(f"[dc] Q{self.position}: no </cheatsheet> (len={len(raw)}) — salvaged "
                  f"{len(closes)} complete items")
            return body.strip()
        self._dump_reply(raw, "fail")
        print(f"[dc] Q{self.position}: no cheatsheet content (len={len(raw)}, head={raw[:60]!r}); "
              f"keeping previous sheet")
        return None

    def _dump_reply(self, reply: str, kind: str) -> None:
        if self.state_path:
            d = self.state_path.parent / "curator_debug"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"q{self.position:04d}_{kind}.txt").write_text(reply)

    def _fit(self, sheet: str) -> str:
        """Hard cap: drop trailing <memory_item> blocks until header + sheet <= cap."""
        budget = self.cap - count_tokens(self._header())
        if count_tokens(sheet) <= budget:
            return sheet
        self.truncations += 1
        parts = re.split(r"(?=<memory_item>)", sheet)
        out = parts[0]
        for part in parts[1:]:
            if count_tokens(out + part) > budget:
                break
            out += part
        return out.rstrip() + "\n</cheatsheet-truncated>"

    # ------------------------------------------------------------ persistence
    def dump(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "sheet": self.sheet, "version": self.version, "n_updates": self.n_updates,
                "parse_fails": self.parse_fails, "salvaged": self.salvaged,
                "compressions": self.compressions, "truncations": self.truncations,
                "position": self.position,
            }, indent=1))

    def load(self) -> bool:
        if self.state_path and self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            self.sheet, self.version = st["sheet"], st["version"]
            self.n_updates, self.parse_fails = st["n_updates"], st["parse_fails"]
            self.truncations, self.position = st.get("truncations", 0), st["position"]
            self.salvaged = st.get("salvaged", 0)
            self.compressions = st.get("compressions", 0)
            return True
        return False
