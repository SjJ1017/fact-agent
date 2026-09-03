"""B 在 B1、B2 都说了同一事实，C 只在 C1 说过，A|3 采纳它。
   list 版本 -> B 拿 2/3、C 拿 1/3；set 版本 -> 各 1/2。"""
import sys, json
sys.path.insert(0,'experiments')
import analyze_stance_flow as M
M.attach_labels = lambda *a, **k: 0

FID="f_shared"
store={"mentions":{},"facts":{},"mention_to_fact":{}}
def add(mid,a,r,fid,t):
    store["mentions"][mid]={"mention_id":mid,"text":t,"quote":t,
      "provenance":{"execution_id":"t","agent_id":a,"round":r,"channel":"output","doc_id":None,"extra":{}}}
    store["mention_to_fact"][mid]=fid
    store["facts"].setdefault(fid,{"canonical_text":t,"mention_ids":[],"properties":{"stance":"SUPPORT"}})["mention_ids"].append(mid)
add("m1","B",1,FID,"shared"); add("m2","B",2,FID,"shared")   # B 重申
add("m3","C",1,FID,"shared")                                  # C 只说一次
add("m4","A",3,FID,"shared")                                  # A 第 3 轮采纳

d={"transcript":{f"{a}|{r}":"w" for a in "ABC" for r in (1,2,3)},"delivery":{}}
for a in "ABC":
    for r in (1,2,3):
        d["delivery"][f"{a}|{r}"]={
            "peer_turns":[f"{p}|{r-1}" for p in "ABC" if p!=a] if r>1 else [],
            "visible_peer_turns":[f"{p}|{k}" for p in "ABC" if p!=a for k in range(1,r)],
            "visible_self_turns":[f"{a}|{k}" for k in range(1,r)]}

tot={}
for sp,li,st,cr,tk in M.uptake_edges(json.loads(json.dumps(store)), d, "t"):
    if li=="A" and tk: tot[sp]=tot.get(sp,0)+cr
print("A|3 采纳该事实时的来源 credit:", {k:round(v,3) for k,v in sorted(tot.items())})
print("期望: B 和 C 各 0.5（均分给来源 peer，与 B 重申几次无关）")
