import json, urllib.request, pathlib, hashlib, statistics, collections, concurrent.futures, tiktoken
ROOT=pathlib.Path(__file__).resolve().parent
SHA='b2da22eac88bb0874c64665f13457eb99835774a'
BASE=f'https://raw.githubusercontent.com/mohammadtavakoli78/BEAM/{SHA}/'
ENC=tiktoken.get_encoding('o200k_base')
def fetch(p):
 out=ROOT/'data'/p
 if not out.exists():
  out.parent.mkdir(parents=True,exist_ok=True)
  out.write_bytes(urllib.request.urlopen(BASE+p,timeout=120).read())
 return json.loads(out.read_text())
def run(task):
 group,i=task;p=f'chats/{group}/{i}/chat.json';d=fetch(p);q=fetch(f'chats/{group}/{i}/probing_questions/probing_questions.json')
 turns=[t for b in d for t in b['turns']];msgs=[m for t in turns for m in t]
 # Match upstream count_message_tokens' tiktoken path: serialize role/content and count each turn.
 n=sum(len(ENC.encode(''.join(f"{m['role']}: {m['content']}\n" for m in t),disallowed_special=())) for t in turns)
 content=sum(len(ENC.encode(m['content'],disallowed_special=())) for m in msgs)
 nq=sum(len(v) for v in q.values())
 row=dict(group=group,conversation_id=i,history_tokens=n,content_only_tokens=content,questions=nq,tokens_per_question=n/nq,tokens_div_20=n/20,messages=len(msgs),turns=len(turns),sha256=hashlib.sha256((ROOT/'data'/p).read_bytes()).hexdigest())
 schema=dict(batch_fields=sorted(set().union(*(b.keys() for b in d))),message_fields=sorted(set().union(*(m.keys() for m in msgs))),question_fields={k:sorted(set().union(*(v.keys() for v in arr))) for k,arr in q.items()},question_types=sorted(set(m.get('question_type','') for m in msgs)))
 print(f'{group}/{i}: {n:,} tokens, {nq} questions',flush=True)
 return row,schema
if __name__=='__main__':
 tasks=[(g,i) for g,n in [('100K',20),('1M',35)] for i in range(1,n+1)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: out=list(pool.map(run,tasks))
 rows=[r for r,s in out]
 summary={}
 for g in ['100K','1M']:
  rs=[r for r in rows if r['group']==g];ns=[r['history_tokens'] for r in rs]
  summary[g]=dict(conversations=len(rs),questions=sum(r['questions'] for r in rs),total_tokens=sum(ns),min=min(ns),max=max(ns),mean=statistics.mean(ns),median=statistics.median(ns),amortized_tokens=sum(ns)/sum(r['questions'] for r in rs))
 result=dict(commit=SHA,tiktoken_version=tiktoken.__version__,encoding='o200k_base',method='Sum over turns of encode(concat(role + ": " + content + "\\n")); no JSON metadata, no API chat template; original chat.json only',summary=summary,conversations=rows,schemas={f"{r['group']}/{r['conversation_id']}":s for r,s in out})
 (ROOT/'token-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 md=['# BEAM token audit',f'Commit: {SHA}; tiktoken {tiktoken.__version__}; o200k_base.','Original chat.json, per-turn role/content serialization; excludes metadata and API chat-template overhead.','| Group | Conversation | History tokens | Questions | Tokens / question |','|---|---:|---:|---:|---:|']
 for r in rows:md.append(f"| {r['group']} | {r['conversation_id']} | {r['history_tokens']:,} | {r['questions']} | {r['tokens_per_question']:,.2f} |")
 (ROOT/'token-audit.md').write_text('\n'.join(md)+'\n')
 print(json.dumps(summary,indent=2))
