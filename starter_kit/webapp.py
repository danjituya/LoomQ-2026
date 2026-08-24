#!/usr/bin/env python3
"""LoomQ Web Studio - 用大白话指挥量子计算机.

A zero-CDN single-page web app built for the L2 interaction / inclusivity
experience. Users with no quantum background type plain language, the agent
(agent_chat) returns OpenQASM 2.0, and the page renders the circuit as an SVG
diagram plus a measurement histogram - all rendered client-side with no
external resources.

Run:
    export LOOMQ_LLM_BASE_URL=...
    export LOOMQ_LLM_API_KEY=...
    export LOOMQ_LLM_MODEL=deepseek-v4-flash
    python webapp.py            # serves on http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import os
import re

from flask import Flask, jsonify, request

try:
    from adapter import agent_chat, run
    from adapter import _parse_qasm2
except ImportError:
    from starter_kit.adapter import agent_chat, run, _parse_qasm2

app = Flask(__name__)


def _qasm_ops_to_circuit(qasm_str: str) -> tuple:
    """Return (n_qubits, [gate dicts]) ready for the SVG renderer."""
    try:
        qregs, cregs, ops = _parse_qasm2(qasm_str)
    except Exception:
        return 0, []
    n_qubits = sum(qregs.values())
    gates = []
    for op in ops:
        if op[0] == "gate":
            _, gate, params, targets = op
            idxs = []
            for t in targets:
                m = re.search(r"\[(\d+)\]", t)
                idxs.append(int(m.group(1)) if m else 0)
            gates.append({
                "gate": gate.lower(),
                "qubits": idxs,
                "param": params[0] if params else None,
            })
        elif op[0] == "measure":
            _, q, qi, c, ci = op
            gates.append({
                "gate": "measure",
                "qubits": [int(qi) if qi else 0],
                "param": None,
            })
    return n_qubits, gates


def _explain_counts(counts: dict, n_qubits: int) -> str:
    """Plain-language explanation of the measurement outcome."""
    if not counts:
        return ""
    total = sum(counts.values())
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    top_str = "、".join(f"{k}（{v/total*100:.0f}%）" for k, v in top)
    if len(top) == 1 or abs(top[0][1] - top[1][1]) / total < 0.06:
        return (
            f"测量结果高度集中在 {top_str} 附近。这类「确定性叠加」状态（如 GHZ / Bell 态）"
            "正是量子计算最经典的入门实验——你刚才让一个量子系统进入了两个状态的叠加，"
            "并在测量时「坍缩」到了其中一种。"
        )
    return (
        f"测量分布较分散，主要出现在 {top_str}。这通常是因为电路含参数门（旋转门），"
        "产生的是「概率性」结果——这是量子世界与经典世界的核心差异：同一电路重复运行，"
        "每次得到的结果可以不同。"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "请输入你想做的事情"}), 400

    result = {"reply": "", "qasm": None, "gates": [], "qubits": 0,
              "counts": None, "shots": 1024, "explain": "", "error": None}
    try:
        result["reply"] = agent_chat(prompt)
        m = re.search(r"OPENQASM\s+2\.0;.*?(?=^\s*```|\Z)", result["reply"],
                      re.DOTALL | re.MULTILINE)
        qasm = m.group(0).strip() if m else None
        if qasm:
            result["qasm"] = qasm
            n_q, gates = _qasm_ops_to_circuit(qasm)
            result["qubits"], result["gates"] = n_q, gates
            try:
                r = run(qasm, "braket", 1024)
                result["counts"] = {str(k): int(v) for k, v in r["counts"].items()}
                result["explain"] = _explain_counts(result["counts"], n_q)
            except Exception as exc:
                result["error"] = f"电路运行失败: {type(exc).__name__}: {exc}"
    except Exception as exc:
        result["error"] = f"智能体调用失败: {type(exc).__name__}: {exc}"
    return jsonify(result)


@app.route("/")
def index():
    return PAGE_HTML


PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoomQ 量子小工坊 · 用大白话指挥量子计算机</title>
<style>
:root{
  --bg:#f7f6f2; --card:#ffffff; --ink:#2c2c2a; --muted:#5f5e5a; --faint:#888780;
  --line:#e5e2d8; --accent:#0f6e56; --accent2:#378add; --soft:#e1f5ee;
  --danger:#a32d2d;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif;background:var(--bg);color:var(--ink);line-height:1.7;padding:24px 16px 60px}
.wrap{max-width:860px;margin:0 auto}
.hero{background:linear-gradient(135deg,#0f6e56 0%,#0c447c 100%);border-radius:20px;padding:34px 34px 28px;color:#fff;margin-bottom:20px}
.hero h1{font-size:26px;font-weight:500;letter-spacing:.5px}
.hero p{font-size:14px;opacity:.92;margin-top:8px;max-width:640px}
.chips{margin-top:18px;display:flex;flex-wrap:wrap;gap:8px}
.chip{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.28);color:#fff;border-radius:999px;padding:6px 14px;font-size:13px;cursor:pointer;transition:.15s}
.chip:hover{background:rgba(255,255,255,.28)}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 22px;margin-bottom:16px}
.card h2{font-size:15px;font-weight:500;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.step{display:inline-flex;width:22px;height:22px;border-radius:50%;background:var(--soft);color:var(--accent);font-size:12px;align-items:center;justify-content:center;flex:none}
textarea{width:100%;min-height:72px;border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:14px;font-family:inherit;resize:vertical;outline:none;background:#fff}
textarea:focus{border-color:var(--accent)}
.row{display:flex;gap:10px;align-items:flex-end}
button{background:var(--accent);color:#fff;border:0;border-radius:12px;padding:11px 22px;font-size:14px;cursor:pointer;font-family:inherit;white-space:nowrap}
button:disabled{opacity:.5;cursor:wait}
.btn-ghost{background:#fff;color:var(--accent);border:1px solid var(--accent)}
.muted{color:var(--muted);font-size:13px}
.spin{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;border-radius:50%;animation:r 1s linear infinite;vertical-align:-2px;margin-right:6px}
@keyframes r{to{transform:rotate(360deg)}}
.out{display:none;margin-top:14px}
.out.show{display:block}
.reply{background:#fbfaf6;border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:14px;white-space:pre-wrap;word-break:break-word;max-height:260px;overflow:auto}
.section{margin-top:18px}
.section h3{font-size:13px;font-weight:500;color:var(--muted);margin-bottom:8px}
.err{background:#fcebeb;color:var(--danger);border-radius:10px;padding:10px 12px;font-size:13px;margin-top:12px}
.guide{font-size:14px}
.guide details{background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin-bottom:8px}
.guide summary{cursor:pointer;font-weight:500;font-size:14px}
.guide p{color:var(--muted);font-size:13px;margin-top:6px}
code{background:#f1efe8;border-radius:5px;padding:1px 6px;font-size:12px;font-family:Consolas,monospace}
.foot{text-align:center;color:var(--faint);font-size:12px;margin-top:24px}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>量子小工坊 · 用大白话指挥量子计算机</h1>
    <p>不用学物理，不用写代码。告诉我你想要什么样的量子态，剩下的交给智能体——它会生成电路、画出线路图、运行模拟器，并用大白话解释结果。</p>
    <div class="chips" id="chips"></div>
  </div>

  <div class="card">
    <h2><span class="step">1</span> 说出你的量子实验</h2>
    <div class="row">
      <textarea id="input" placeholder="例如：生成一个 3 比特的 GHZ 态并测量"></textarea>
      <button id="go">运行</button>
    </div>
    <div class="out" id="out">
      <div class="section"><h3>智能体回答</h3><div class="reply" id="reply"></div></div>
      <div class="section" id="circuitSec" style="display:none"><h3>电路图</h3><div id="circuit"></div></div>
      <div class="section" id="histSec" style="display:none"><h3>测量结果</h3><div id="hist"></div><p class="muted" id="explain" style="margin-top:10px"></p></div>
      <div class="err" id="err" style="display:none"></div>
    </div>
  </div>

  <div class="card guide">
    <h2><span class="step">2</span> 量子计算是什么？（30 秒入门）</h2>
    <details open><summary>量子比特：一个可以"既是 0 又是 1"的比特</summary>
    <p>经典比特只有 0 或 1。量子比特（qubit）在测量前处于叠加态——既有 0 的成分又有 1 的成分。测量瞬间才"坍缩"成确定的 0 或 1。</p></details>
    <details><summary>叠加与纠缠：量子计算的两大魔法</summary>
    <p><b>叠加</b>：多个量子比特可以同时表示所有可能的组合（2 个比特同时是 00、01、10、11）。<br>
    <b>纠缠</b>：两个量子比特建立起神秘关联，测量其中一个，另一个立即"知道"结果——这就是贝尔态、GHZ 态的基础。</p></details>
    <details><summary>门操作：像积木一样搭电路</summary>
    <p>H 门创造叠加，CX（CNOT）门创造纠缠，Rz/Ry 旋转门调节概率。把门按顺序排好，就是一张量子电路图——你在上面看到的就是它。</p></details>
    <details><summary>测量结果怎么读？</summary>
    <p>模拟器会把同一个电路重复运行很多次（如 1024 次），每次得到一个位串。柱状图展示每种结果出现的比例——比如 GHZ 态会得到约 50% 的 000 和 50% 的 111。</p></details>
  </div>

  <p class="foot">LoomQ · Quantum Accessibility Equality Initiative · 让不懂"黑话"的人也能指挥最前沿的算力</p>
</div>

<script>
const EXAMPLES = [
  {label:"首次实验 · 生成 3 比特 GHZ 态", prompt:"生成一个 3 比特的 GHZ 态（最大纠缠态），并进行全测量"},
  {label:"修复报错代码", prompt:"我想制备一个贝尔态，但这段代码报错了，帮我修好：H q[0]; CX q[0] q[1]（未定义寄存器且门名大小写错误）"},
  {label:"帮我选平台", prompt:"我需要运行一个 15 比特电路，且零排队等待，选哪个平台？"}
];
const chips = document.getElementById("chips");
EXAMPLES.forEach((e, i) => {
  const b = document.createElement("button");
  b.className = "chip";
  b.textContent = (i + 1) + ". " + e.label;
  b.onclick = () => { document.getElementById("input").value = e.prompt; document.getElementById("go").click(); };
  chips.appendChild(b);
});

const input = document.getElementById("input");
const go = document.getElementById("go");
const out = document.getElementById("out");
const replyEl = document.getElementById("reply");
const circuitEl = document.getElementById("circuit");
const histEl = document.getElementById("hist");
const explainEl = document.getElementById("explain");
const errEl = document.getElementById("err");

function setBusy(b) {
  go.disabled = b;
  go.innerHTML = b ? '<span class="spin"></span>思考中…' : "运行";
}

async function run() {
  const prompt = input.value.trim();
  if (!prompt) return;
  out.classList.add("show");
  errEl.style.display = "none";
  replyEl.textContent = "智能体正在生成量子电路并自检…";
  document.getElementById("circuitSec").style.display = "none";
  document.getElementById("histSec").style.display = "none";
  setBusy(true);
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt})
    });
    const data = await resp.json();
    if (data.error) {
      errEl.textContent = data.error;
      errEl.style.display = "block";
      replyEl.textContent = data.reply || "";
    } else {
      replyEl.textContent = data.reply;
      if (data.gates && data.gates.length) {
        circuitEl.innerHTML = renderCircuit(data.qubits, data.gates);
        document.getElementById("circuitSec").style.display = "block";
      }
      if (data.counts && Object.keys(data.counts).length) {
        histEl.innerHTML = renderHist(data.counts);
        explainEl.textContent = data.explain || "";
        document.getElementById("histSec").style.display = "block";
      }
    }
  } catch (e) {
    errEl.textContent = "网络错误：" + e.message;
    errEl.style.display = "block";
  } finally {
    setBusy(false);
  }
}
go.onclick = run;
input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(); } });

/* ---- client-side SVG rendering (mirrors backend, zero deps) ---- */
function esc(s){return s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function renderCircuit(n, gates){
  const colW=46,rowH=56,top=40,left=60;
  const cols=Math.max(2,Math.min(26,gates.length+1));
  const w=left+cols*colW+30,h=top+n*rowH+30;
  const used=new Array(n).fill(0);
  let s=`<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:auto;max-width:760px"><rect x="0" y="0" width="${w}" height="${h}" rx="12" fill="#fff"/>`;
  for(let i=0;i<n;i++){const y=top+i*rowH;s+=`<line x1="${left-22}" y1="${y}" x2="${w-14}" y2="${y}" stroke="#b4b2a9"/><text x="${left-30}" y="${y}" text-anchor="end" dominant-baseline="central" font-size="13" fill="#5f5e5a">q[${i}]</text>`;}
  const STY={h:"#e6f1fb",x:"#fcebeb",s:"#faeeda",sdg:"#faeeda",t:"#fbeaf0",tdg:"#fbeaf0",rz:"#eaf3de",ry:"#eaf3de",cx:"#e1f5ee",cu1:"#e1f5ee",swap:"#e1f5ee",ccx:"#fcebeb",measure:"#f1efe8"};
  const LBL={h:"H",x:"X",s:"S",sdg:"S†",t:"T",tdg:"T†",rz:"Rz",ry:"Ry",cx:"⊕",cu1:"U1",swap:"×",ccx:"•",measure:"M"};
  gates.forEach(g=>{
    const col=Math.max(...g.qubits.map(i=>used[i]))+1;
    g.qubits.forEach(i=>used[i]=col);
    const cx=left+col*colW-colW/2, ys=g.qubits.map(i=>top+i*rowH);
    const fill=STY[g.gate]||"#f1efe8", lb=LBL[g.gate]||g.gate.toUpperCase().slice(0,3);
    if(g.gate==="cx"||g.gate==="cu1"||g.gate==="swap"){
      const a=ys[0],b=ys[ys.length-1];
      s+=`<line x1="${cx}" y1="${a}" x2="${cx}" y2="${b}" stroke="#0f6e56" stroke-width="1.5"/>`;
      if(g.gate==="swap"){ys.forEach(y=>s+=`<circle cx="${cx}" cy="${y}" r="7" fill="none" stroke="#0f6e56" stroke-width="1.5"/>`);s+=`<text x="${cx}" y="${(a+b)/2}" text-anchor="middle" dominant-baseline="central" fill="#0f6e56" font-size="13">×</text>`;}
      else{s+=`<circle cx="${cx}" cy="${a}" r="5" fill="#0f6e56"/>`;
        if(g.gate==="cu1"){const t="U1("+fmt(g.param)+")";s+=`<rect x="${cx-16}" y="${b-13}" width="32" height="26" rx="6" fill="${fill}" stroke="#1d9e75" stroke-width=".5"/><text x="${cx}" y="${b}" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">${t}</text>`;}
        else{s+=`<circle cx="${cx}" cy="${b}" r="5" fill="none" stroke="#0f6e56" stroke-width="1.5"/><text x="${cx}" y="${b}" text-anchor="middle" dominant-baseline="central" fill="#0f6e56" font-size="13">${lb}</text>`;}}
    }else if(g.gate==="ccx"){
      const a=ys[0],c=ys[ys.length-1];
      s+=`<line x1="${cx}" y1="${a}" x2="${cx}" y2="${c}" stroke="#a32d2d" stroke-width="1.5"/>`;
      ys.slice(0,-1).forEach(y=>s+=`<circle cx="${cx}" cy="${y}" r="5" fill="#a32d2d"/>`);
      s+=`<circle cx="${cx}" cy="${c}" r="5" fill="none" stroke="#a32d2d" stroke-width="1.5"/><text x="${cx}" y="${c}" text-anchor="middle" dominant-baseline="central" fill="#a32d2d" font-size="12">⊕</text>`;
    }else{
      ys.forEach(y=>{const bw=g.param?58:40;
        s+=`<rect x="${cx-bw/2}" y="${y-14}" width="${bw}" height="28" rx="7" fill="${fill}" stroke="#888780" stroke-width=".5"/><text x="${cx}" y="${y}" text-anchor="middle" dominant-baseline="central" fill="#2c2c2a" font-size="12">${g.param?lb+"("+fmt(g.param)+")":lb}</text>`;});
    }
  });
  return s+"</svg>";
}
function fmt(p){if(p==null)return "";const v=parseFloat(p);if(isNaN(v))return esc(String(p));if(Math.abs(v)<1e-9)return "0";if(Math.abs(v-Math.round(v))<1e-9)return String(Math.round(v));return v.toFixed(2).replace(/\.?0+$/,"");}
function renderHist(counts){
  const items=Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,12);
  const total=items.reduce((s,e)=>s+e[1],0), max=Math.max(...items.map(e=>e[1]),1);
  const bw=46,n=items.length,w=Math.max(320,n*(bw+22)+60),h=220,base=h-40;
  let s=`<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:auto;max-width:640px"><rect x="0" y="0" width="${w}" height="${h}" rx="12" fill="#fff"/>`;
  items.forEach((e,i)=>{const [key,val]=e;const hh=Math.max(4,val/max*(base-56)),x=40+i*(bw+22),y=base-hh;
    s+=`<rect x="${x}" y="${y}" width="${bw}" height="${hh}" rx="5" fill="#378add"/><text x="${x+bw/2}" y="${y-6}" text-anchor="middle" font-size="12" fill="#444441">${(val/total*100).toFixed(1)}%</text><text x="${x+bw/2}" y="${base+18}" text-anchor="middle" font-size="12" fill="#5f5e5a">${key}</text>`;});
  return s+"</svg>";
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("LoomQ Web Studio 启动: http://127.0.0.1:8765")
    app.run(host="127.0.0.1", port=8765, debug=False)
