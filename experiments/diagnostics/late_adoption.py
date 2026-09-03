"""构造一个 cumulative 场景：B|1 说了一个事实，A 到第 3 轮才说。
   用投递事件算 -> novel；用可见窗口算 -> adopted。"""
import sys, json, importlib
sys.path.insert(0,'experiments')
import analyze_flow_profile as M

FID = "f_late"
store = {"mentions": {}, "facts": {}, "mention_to_fact": {}}
def add(mid, agent, rnd, fid, text):
    store["mentions"][mid] = {"mention_id": mid, "text": text, "quote": text,
        "provenance": {"execution_id":"t","agent_id":agent,"round":rnd,
                       "channel":"output","doc_id":None,"extra":{}}}
    store["mention_to_fact"][mid] = fid
    store["facts"].setdefault(fid, {"canonical_text": text, "mention_ids": [],
                                    "properties": {}})["mention_ids"].append(mid)
add("m1","B",1,FID,"late fact")          # B 第 1 轮说
add("m2","A",3,FID,"late fact")          # A 第 3 轮才说
for a in "ABC":                           # 给每人补点别的，凑齐轮次
    for r in (1,2,3):
        add(f"x{a}{r}", a, r, f"f_{a}{r}", f"filler {a}{r}")

def debate(visible):
    d = {"transcript": {f"{a}|{r}": "w "*40 for a in "ABC" for r in (1,2,3)},
         "delivery": {}}
    for a in "ABC":
        for r in (1,2,3):
            peers = [f"{p}|{r-1}" for p in "ABC" if p != a] if r > 1 else []
            info = {"peer_turns": peers}
            if visible:
                info["visible_peer_turns"] = [f"{p}|{k}" for p in "ABC" if p != a
                                              for k in range(1, r)]
                info["visible_self_turns"] = [f"{a}|{k}" for k in range(1, r)]
            d["delivery"][f"{a}|{r}"] = info
    return d

import types
M.attach_labels = lambda *a, **k: 0          # 跳过标注 sidecar
M.output_tokens = lambda d: 1000
for visible in (False, True):
    r = M.profile(json.loads(json.dumps(store)), debate(visible), "t")
    print(f"visible_* {'有' if visible else '无'}：novel={r['novel']:3d} "
          f"adopted={r['adopted']:3d}  adopted_share={r['adopted_share']:.1%} "
          f"reception={r['reception']:.1%}")
