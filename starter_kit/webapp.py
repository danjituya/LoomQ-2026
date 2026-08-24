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
            f"测量结果高度集中在 {top_str} 附近。这类最大纠缠态（如 GHZ / Bell 态）"
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
        # ---- TierF 兜底承载处理：把隐形 HTML 注释里的原因 → 单独 notice 条，正文只留干净 QASM ----
        _tier_m = re.search(r"<!--TIER_F_REASON:([\s\S]*?)-->", result["reply"])
        if _tier_m:
            _reason = _tier_m.group(1).strip()
            if not result["error"]:
                result["error"] = _reason
            result["reply"] = re.sub(r"\n*<!--TIER_F_REASON:[\s\S]*?-->\n*", "\n", result["reply"], count=1).strip() + "\n"
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
.chips-hint{font-size:13px;opacity:.88;margin:14px 0 0}
.gate-line{font-size:13px;color:var(--muted);padding:5px 2px;border-bottom:1px dashed var(--line);line-height:1.6}
.gate-line:last-child{border-bottom:0}
.gate-line b{color:var(--accent);font-weight:500}
.hist-legend{font-size:12px;color:var(--faint);margin-top:8px}
.fluct{background:#fbfaf6;border:1px dashed var(--line);border-radius:10px;padding:9px 12px;font-size:12.5px;color:var(--muted);margin-top:10px;line-height:1.65}
.act-tag{display:inline-block;background:var(--soft);color:var(--accent);font-size:11px;border-radius:6px;padding:1px 7px;margin-right:6px}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>量子小工坊 · 用大白话指挥量子计算机</h1>
    <p>不用学物理，不用写代码。告诉我你想要什么样的量子态，剩下的交给智能体——它会生成电路、画出线路图、运行模拟器，并用大白话解释结果。</p>
    <div class="chips-hint">不知道怎么提问？点下面的按钮试试：</div>
    <div class="chips" id="chips"></div>
  </div>

  <div class="card">
    <h2><span class="step">1</span> 说出你的量子实验</h2>
    <div class="row">
      <textarea id="input" placeholder="试试输入：生成一个 3 比特的 GHZ 态并测量"></textarea>
      <button id="go">运行</button>
    </div>
    <div class="out" id="out">
      <div class="section"><h3>智能体回答</h3><div class="reply" id="reply"></div></div>
      <div class="section" id="circuitSec" style="display:none"><h3>电路图</h3><div id="circuit"></div></div>
      <div class="section" id="gateSec" style="display:none"><h3>电路解读</h3><div id="gateExplain"></div></div>
      <div class="section" id="histSec" style="display:none"><h3>测量结果（蓝色柱 = 实际采样，灰色虚线 = 理论期望）</h3><div id="hist"></div><p class="hist-legend">蓝色柱 = 实际采样次数；灰色虚线 = 理论期望值（若已知）。</p><p class="muted" id="explain" style="margin-top:10px"></p><div class="fluct" id="fluctuation" style="display:none"></div></div>
      <div class="err" id="err" style="display:none"></div>
    </div>
  </div>

  <div class="card guide">
    <h2><span class="step">2</span> 量子计算是什么？（4 幕故事化入门）</h2>
    <details open><summary><span class="act-tag">第 1 幕</span>经典 vs 量子</summary>
    <p>经典比特只有 0 或 1，就像硬币只能正面或反面。量子比特在测量前处于叠加态——既有 0 的成分又有 1 的成分，就像一枚正在旋转的硬币。</p></details>
    <details><summary><span class="act-tag">第 2 幕</span>叠加：同时是 0 和 1</summary>
    <p>H 门让量子比特进入叠加态。2 个比特可以同时表示 00、01、10、11 四种状态——这就是量子并行计算的基础。</p></details>
    <details><summary><span class="act-tag">第 3 幕</span>纠缠：远距离的神秘关联</summary>
    <p>CX（CNOT）门让两个比特纠缠。测量其中一个，另一个立即"知道"结果——这就是 Bell 态、GHZ 态的基础，爱因斯坦称之为"幽灵般的超距作用"。</p></details>
    <details><summary><span class="act-tag">第 4 幕</span>测量：概率性坍缩与统计涨落</summary>
    <p>模拟器把电路重复运行 1024 次。理论上 Bell 态应得 50% 00 + 50% 11，但每次测量结果是概率性的——你看到的数字可能不是正好 50%，±3% 的偏差是正常的统计涨落。</p></details>
  </div>

  <p class="foot">LoomQ · Quantum Accessibility Equality Initiative · 让不懂"黑话"的人也能指挥最前沿的算力</p>
</div>

<script>
const EXAMPLES = [
  {label:"🪙 量子硬币翻转（2比特Bell态）", prompt:"生成一个 2 比特的 Bell 态（最大纠缠态），并进行全测量"},
  {label:"🔗 三胞胎纠缠（3比特GHZ态）", prompt:"生成一个 3 比特的 GHZ 态（最大纠缠态），并进行全测量"},
  {label:"🎲 量子随机数生成器", prompt:"用 4 个量子比特生成一个真正的随机数，做全测量"},
  {label:"⚖️ W态：三个比特里恰好一个为1", prompt:"生成一个 3 比特的 W 态，要求三个比特中恰好只有一个为 1，全测量"},
  {label:"🔄 量子叠加态实验", prompt:"对 q[0] 施加 H 门后 S 再 T，然后测量，给我看叠加效果"},
  {label:"🔧 修复报错代码", prompt:"我想制备一个贝尔态，但这段代码报错了，帮我修好：H q[0]; CX q[0] q[1]（未定义寄存器且门名大小写错误）"},
  {label:"🖥️ 帮我选平台", prompt:"我需要运行一个 15 比特电路，且零排队等待，选哪个平台？"},
  {label:"🚀 4比特最大纠缠态", prompt:"生成一个 4 比特的 GHZ 态并全测量"}
];
const GATE_CN = {
  h: "H 门 → 创造叠加态（让比特同时是 0 和 1）",
  x: "X 门 → 量子翻转（0 变 1，1 变 0）",
  s: "S 门 → 相位旋转 90°",
  sdg: "S† 门 → 相位旋转 -90°",
  t: "T 门 → 相位旋转 45°",
  tdg: "T† 门 → 相位旋转 -45°",
  rz: "Rz 门 → 绕 Z 轴旋转（调节相位）",
  ry: "Ry 门 → 绕 Y 轴旋转（调节概率）",
  cx: "CX 门（CNOT）→ 纠缠：控制比特为 1 时翻转目标比特",
  cu1: "CU1 门 → 受控相位旋转",
  swap: "SWAP 门 → 交换两个比特的状态",
  ccx: "CCX 门（Toffoli）→ 双控制翻转",
  measure: "测量 → 观测比特，叠加态坍缩为确定值"
};
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
  document.getElementById("gateSec").style.display = "none";
  document.getElementById("histSec").style.display = "none";
  document.getElementById("fluctuation").style.display = "none";
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
        // 电路解读：去重后列出每个门的中文含义
        const seen = new Set();
        const lines = [];
        data.gates.forEach(g => {
          if (!seen.has(g.gate)) { seen.add(g.gate); const cn = GATE_CN[g.gate]; if (cn) lines.push(cn); }
        });
        if (lines.length) {
          document.getElementById("gateExplain").innerHTML = lines.map(l => '<div class="gate-line">' + l + '</div>').join("");
          document.getElementById("gateSec").style.display = "block";
        }
      }
      if (data.counts && Object.keys(data.counts).length) {
        // 前端推断理论期望：2 主峰都 >40% → 50/50；1 主峰 >90% → 100%
        const entries = Object.entries(data.counts);
        const total = entries.reduce((s,e)=>s+e[1],0);
        const sorted = entries.slice().sort((a,b)=>b[1]-a[1]);
        let expected = null;
        if (sorted.length >= 2 && sorted[0][1]/total > 0.4 && sorted[1][1]/total > 0.4) {
          expected = {}; expected[sorted[0][0]] = 0.5; expected[sorted[1][0]] = 0.5;
        } else if (sorted[0][1]/total > 0.9) {
          expected = {}; expected[sorted[0][0]] = 1.0;
        }
        histEl.innerHTML = renderHist(data.counts, expected);
        explainEl.textContent = data.explain || "";
        // 统计涨落解释
        const fl = document.getElementById("fluctuation");
        if (expected) {
          const keys = Object.keys(expected);
          if (keys.length >= 2) {
            fl.textContent = "量子测量本质是概率性的。理论上 Bell 态应该 50% 是 " + keys[0] + "、50% 是 " + keys[1] + "，但 1024 次采样会有统计涨落（±3% 以内都正常），所以你看到的数字可能不是正好 50%。";
          } else {
            fl.textContent = "量子测量本质是概率性的。理论上该态应该 100% 坍缩到 " + keys[0] + "，但 1024 次采样可能偶现极少量其他结果（属正常统计噪声）。";
          }
          fl.style.display = "block";
        } else {
          fl.style.display = "none";
        }
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
function renderHist(counts, expected){
  const items=Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,12);
  const total=items.reduce((s,e)=>s+e[1],0), max=Math.max(...items.map(e=>e[1]),1);
  const bw=46,n=items.length,w=Math.max(320,n*(bw+22)+80),h=240,base=h-40;
  let s=`<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:auto;max-width:640px"><rect x="0" y="0" width="${w}" height="${h}" rx="12" fill="#fff"/>`;
  // 理论期望：灰色虚线 + 标注
  if(expected){
    items.forEach((e,i)=>{
      const key=e[0];
      if(expected[key]!=null){
        const eh=(expected[key]*total)/max*(base-56);
        const ey=base-Math.max(4,eh);
        const x=40+i*(bw+22);
        s+=`<line x1="${x-6}" y1="${ey}" x2="${x+bw+6}" y2="${ey}" stroke="#888780" stroke-width="1.5" stroke-dasharray="4,3"/>`;
        s+=`<text x="${x+bw+10}" y="${ey+4}" text-anchor="start" font-size="11" fill="#888780">理论 ${(expected[key]*100).toFixed(0)}%</text>`;
      }
    });
  }
  items.forEach((e,i)=>{const [key,val]=e;const hh=Math.max(4,val/max*(base-56)),x=40+i*(bw+22),y=base-hh;
    s+=`<rect x="${x}" y="${y}" width="${bw}" height="${hh}" rx="5" fill="#378add"/><text x="${x+bw/2}" y="${y-6}" text-anchor="middle" font-size="12" fill="#444441">${(val/total*100).toFixed(1)}%</text><text x="${x+bw/2}" y="${base+18}" text-anchor="middle" font-size="12" fill="#5f5e5a">${key}</text>`;});
  return s+"</svg>";
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    _bind_host = os.environ.get("LOOMQ_BIND_HOST", "0.0.0.0")
    _port = int(os.environ.get("LOOMQ_PORT", "8765"))
    print(f"LoomQ Web Studio 启动: http://127.0.0.1:{_port}  (bind {_bind_host})")
    app.run(host=_bind_host, port=_port, debug=False)
