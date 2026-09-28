"""Which switch (if any) turns thinking OFF on the served Qwen model?

    QWEN_BASE_URL=http://box:8000/v1 python probe_thinking.py

Prints, per variant: content, reasoning length, output tokens, finish reason.
A variant "works" when reasoning_len == 0 and out is small.
"""
import os
import openai

c = openai.OpenAI(base_url=os.environ["QWEN_BASE_URL"], api_key=os.environ.get("QWEN_API_KEY", "x"))
model = os.environ.get("QWEN_MODEL") or c.models.list().data[0].id
Q = "What is 17*23? Reply with the number only."


def ask(label, user=Q, **kw):
    r = c.chat.completions.create(
        model=model, max_tokens=600, temperature=0, seed=0,
        messages=[{"role": "system", "content": "Answer tersely."}, {"role": "user", "content": user}],
        **kw,
    )
    msg = r.choices[0].message
    rc = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    print(f"{label:34s} content={(msg.content or '')[:30]!r:34} reasoning_len={len(rc):5d} "
          f"out={r.usage.completion_tokens:4d} finish={r.choices[0].finish_reason}")


print("model:", model)
ask("default")
ask("chat_template enable_thinking=False", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
ask("/no_think suffix", user=Q + " /no_think")
ask("both", user=Q + " /no_think", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
