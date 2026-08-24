#!/usr/bin/env python3
"""LoomQ Web Studio - 用大白话指挥量子计算机.

A zero-CDN single-page web app built for the L2 interaction / inclusivity
experience. Users with no quantum background type plain language, the agent
(agent_chat) returns OpenQASM 2.0, and the page renders the circuit as an SVG
diagram plus a measurement histogram - all rendered client-side with no
external resources.

Layout: wide two-column (learn panel | workbench), guide always visible on
the left, line-by-line plain-language gate explanations, overlap-free
histogram with theoretical expectation markers.

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


# raw string：JS 正则有 \n \s \* 等反斜杠序列，普通字符串会把 \n 展开成
# 真实换行导致 JS 语法错误（页面脚本整体失效、例子按钮不渲染）
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoomQ 量子小工坊 · 用大白话指挥量子计算机</title>
<style>
:root{
  --bg:#f6f5f1; --card:#ffffff; --ink:#2b2b28; --muted:#5f5e5a; --faint:#8b8a83;
  --line:#e6e3d9; --accent:#0f6e56; --accent2:#185fa5; --soft:#e1f5ee; --soft2:#e6f1fb;
  --danger:#a32d2d; --chipbg:#f0eee6;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif;background:var(--bg);color:var(--ink);line-height:1.7;padding:20px 18px 50px}
.app{max-width:1500px;margin:0 auto;display:grid;grid-template-columns:minmax(330px,5fr) minmax(0,8fr);gap:18px;align-items:start}
@media (max-width:1000px){.app{grid-template-columns:1fr}}
.left{position:sticky;top:16px;display:flex;flex-direction:column;gap:16px;max-height:calc(100vh - 40px);overflow:auto;padding-right:2px}
@media (max-width:1000px){.left{position:static;max-height:none}}
.hero{background:linear-gradient(135deg,#0f6e56 0%,#0c447c 100%);border-radius:18px;padding:26px 26px 22px;color:#fff}
.hero h1{font-size:23px;font-weight:600;letter-spacing:.5px}
.hero .sub{font-size:13.5px;opacity:.92;margin-top:8px;line-height:1.75}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px}
.card h2{font-size:15px;font-weight:600;margin-bottom:12px;display:flex;align-items:center;gap:9px}
.step{display:inline-flex;width:22px;height:22px;border-radius:50%;background:var(--soft);color:var(--accent);font-size:12px;align-items:center;justify-content:center;flex:none;font-weight:600}
.guide{font-size:14px}
.guide details{background:#fff;border:1px solid var(--line);border-radius:12px;padding:11px 14px;margin-bottom:9px}
.guide details[open]{border-color:var(--accent)}
.guide summary{cursor:pointer;font-weight:600;font-size:13.5px;list-style:none;display:flex;align-items:center;gap:8px}
.guide summary::-webkit-details-marker{display:none}
.guide summary .caret{transition:transform .18s;font-size:10px;color:var(--faint)}
.guide details[open] summary .caret{transform:rotate(90deg)}
.guide p{color:var(--muted);font-size:13px;margin-top:7px;line-height:1.75}
.act-tag{display:inline-block;background:var(--soft);color:var(--accent);font-size:11px;border-radius:6px;padding:1px 7px;flex:none;font-weight:600}
.chips-hint{font-size:13px;color:var(--muted);margin-bottom:10px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{background:var(--chipbg);border:1px solid var(--line);color:var(--ink);border-radius:999px;padding:7px 13px;font-size:12.5px;cursor:pointer;transition:.15s;font-family:inherit;white-space:normal;text-align:left;line-height:1.5}
.chip:hover{background:var(--soft);border-color:var(--accent);color:var(--accent)}
textarea{width:100%;min-height:76px;border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:14px;font-family:inherit;resize:vertical;outline:none;background:#fff}
textarea:focus{border-color:var(--accent)}
.row{display:flex;gap:10px;align-items:flex-end}
button{background:var(--accent);color:#fff;border:0;border-radius:12px;padding:12px 24px;font-size:14px;cursor:pointer;font-family:inherit;white-space:nowrap}
button:disabled{opacity:.5;cursor:wait}
.spin{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;border-radius:50%;animation:r 1s linear infinite;vertical-align:-2px;margin-right:6px}
@keyframes r{to{transform:rotate(360deg)}}
.out{display:none;margin-top:16px}
.out.show{display:block}
.reply{background:#fbfaf6;border:1px solid var(--line);border-radius:12px;padding:13px 15px;font-size:13.5px;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow:auto;line-height:1.75}
.reply code{background:#f1efe8;border-radius:5px;padding:1px 6px;font-size:12px;font-family:Consolas,monospace}
.reply pre{background:#2b2b28;color:#e8e6df;border-radius:10px;padding:12px 14px;font-size:12.5px;overflow:auto;margin:8px 0;font-family:Consolas,monospace;white-space:pre;line-height:1.55}
.section{margin-top:18px}
.section h3{font-size:13px;font-weight:600;color:var(--muted);margin-bottom:9px;display:flex;align-items:center;gap:7px}
.err{background:#fcebeb;color:var(--danger);border-radius:10px;padding:10px 12px;font-size:13px;margin-top:12px}
.gate-line{display:flex;align-items:baseline;gap:9px;font-size:13px;color:var(--muted);padding:7px 2px;border-bottom:1px dashed var(--line);line-height:1.6}
.gate-line:last-child{border-bottom:0}
.gate-no{color:var(--faint);font-size:11px;flex:none;width:22px}
.gate-name{color:var(--accent);font-weight:600;flex:none}
.gate-where{color:var(--faint);font-size:12px;flex:none}
.gate-desc{flex:1}
.fluct{background:#fbfaf6;border:1px dashed var(--line);border-radius:10px;padding:9px 12px;font-size:12.5px;color:var(--muted);margin-top:10px;line-height:1.65}
.hist-legend{font-size:12px;color:var(--faint);margin-top:8px}
.muted{color:var(--muted);font-size:13px}
.howto{display:flex;flex-direction:column;gap:12px;margin-top:16px}
.howto-item{display:flex;gap:12px;align-items:flex-start}
.howto-no{display:inline-flex;width:24px;height:24px;border-radius:50%;background:var(--soft2);color:var(--accent2);font-size:13px;font-weight:600;align-items:center;justify-content:center;flex:none;margin-top:2px}
.howto-item b{font-size:13.5px;font-weight:600}
.howto-item p{font-size:12.5px;color:var(--muted);margin-top:2px;line-height:1.65}
code.inline{background:#f1efe8;border-radius:5px;padding:1px 6px;font-size:12px;font-family:Consolas,monospace}
.foot{text-align:center;color:var(--faint);font-size:12px;margin-top:22px;grid-column:1/-1}
</style>
</head>
<body>
<div class="app">

  <div class="left">
    <div class="hero">
      <h1>量子小工坊</h1>
      <div class="sub">不用学物理，不用写代码。用大白话告诉它你想要什么，它生成电路、运行模拟器，并用你能听懂的话解释每一步。</div>
    </div>

    <div class="card guide">
      <h2><span class="step">1</span> 量子计算是什么？（4 幕入门）</h2>
      <details open><summary><span class="act-tag">第 1 幕</span>经典 vs 量子<span class="caret">▶</span></summary>
      <p>经典比特只有 0 或 1，就像硬币只能正面或反面。量子比特在测量前处于叠加态——既有 0 的成分又有 1 的成分，就像一枚正在旋转的硬币。</p></details>
      <details open><summary><span class="act-tag">第 2 幕</span>叠加：同时是 0 和 1<span class="caret">▶</span></summary>
      <p>H 门让量子比特进入叠加态。2 个比特可以同时表示 00、01、10、11 四种状态——这就是量子并行计算的基础。</p></details>
      <details open><summary><span class="act-tag">第 3 幕</span>纠缠：远距离的神秘关联<span class="caret">▶</span></summary>
      <p>CX（CNOT）门让两个比特纠缠。测量其中一个，另一个立即"知道"结果——这就是 Bell 态、GHZ 态的基础。</p></details>
      <details open><summary><span class="act-tag">第 4 幕</span>测量：概率与统计涨落<span class="caret">▶</span></summary>
      <p>模拟器把电路重复运行 1024 次。理论上 Bell 态应得 50% 00 + 50% 11，但你看到的数字可能不是正好 50%，±3% 的偏差是正常涨落。</p></details>
    </div>

    <div class="card">
      <h2><span class="step">2</span> 不知道问什么？点这里</h2>
      <div class="chips-hint">每个按钮都是一句「人话」提问，点一下直接运行：</div>
      <div class="chips" id="chips"></div>
    </div>
  </div>

  <div class="right">
    <div class="card">
      <h2><span class="step">3</span> 说出你的量子实验</h2>
      <div class="row">
        <textarea id="input" placeholder="试试输入：生成一个 3 比特的 GHZ 态并测量"></textarea>
        <button id="go">运行</button>
      </div>
      <div class="howto" id="howto">
        <div class="howto-item"><span class="howto-no">1</span><div><b>在框里输入你的问题</b><p>用大白话就行，比如「让三个比特互相纠缠」。也可以点左边第 2 张卡里的例子按钮。</p></div></div>
        <div class="howto-item"><span class="howto-no">2</span><div><b>看「电路逐行解读」</b><p>每一行都用中文解释了是什么门、作用在哪个比特、有什么效果。</p></div></div>
        <div class="howto-item"><span class="howto-no">3</span><div><b>看测量结果</b><p>柱状图是模拟器跑 1024 次的统计；灰色虚线是理论上该出现的比例，±3% 以内的偏差都是正常的。</p></div></div>
      </div>
      <div class="out" id="out">
        <div class="section"><h3>智能体回答</h3><div class="reply" id="reply"></div></div>
        <div class="section" id="gateSec" style="display:none">
          <h3>电路逐行解读（小白友好）</h3>
          <div id="gateExplain"></div>
        </div>
        <div class="section" id="circuitSec" style="display:none"><h3>电路图</h3><div id="circuit"></div></div>
        <div class="section" id="histSec" style="display:none">
          <h3>测量结果</h3>
          <div id="hist"></div>
          <p class="hist-legend">蓝色柱 = 实际采样次数；灰色虚线 = 理论期望值。</p>
          <p class="muted" id="explain" style="margin-top:10px"></p>
          <div class="fluct" id="fluctuation" style="display:none"></div>
        </div>
        <div class="err" id="err" style="display:none"></div>
      </div>
    </div>
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

/* 转义 + 代码块/加粗的轻量渲染（零依赖） */
function escHtml(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function renderReply(text){
  let s = escHtml(text);
  s = s.replace(/```(\w*)\n([\s\S]*?)```/g, (m, lang, code) => "<pre>" + code + "</pre>");
  s = s.replace(/(^|\n)\*\*([^*\n]+)\*\*/g, "$1<strong>$2</strong>");
  s = s.replace(/`([^`\n]+)`/g, '<code class="inline">$1</code>');
  return s;
}

async function run() {
  const prompt = input.value.trim();
  if (!prompt) return;
  out.classList.add("show");
  document.getElementById("howto").style.display = "none";
  errEl.style.display = "none";
  replyEl.innerHTML = '<span class="spin"></span>智能体正在生成量子电路并自检…';
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
      replyEl.innerHTML = renderReply(data.reply || "");
      if (data.gates && data.gates.length) {
        circuitEl.innerHTML = renderCircuit(data.qubits, data.gates);
        document.getElementById("circuitSec").style.display = "block";
        /* 逐行解读：每个门 一行中文解释 + 作用在哪个比特 */
        const lines = data.gates.map((g, idx) => {
          const cn = GATE_CN[g.gate] || (g.gate.toUpperCase() + " 门");
          const where = g.gate === "measure"
            ? "测量 " + g.qubits.map(q => "q[" + q + "]").join("、")
            : "施加到 " + g.qubits.map(q => "第 " + (q + 1) + " 个比特（q[" + q + "]）").join("、");
          return '<div class="gate-line"><span class="gate-no">' + (idx + 1) + "</span>" +
                 '<span class="gate-name">' + escHtml(g.gate.toUpperCase()) + "</span>" +
                 '<span class="gate-where">' + escHtml(where) + "</span>" +
                 '<span class="gate-desc">' + escHtml(cn) + "</span></div>";
        });
        document.getElementById("gateExplain").innerHTML = lines.join("");
        document.getElementById("gateSec").style.display = "block";
      }
      if (data.counts && Object.keys(data.counts).length) {
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

/* ---- 电路 SVG（零依赖）---- */
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

/* ---- 柱状图：修复文字重叠（百分比在柱内白字，理论标注在虚线右端，最多 8 柱 + 其他）---- */
function renderHist(counts, expected){
  let items = Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  let restSum = 0;
  if (items.length > 8) {
    restSum = items.slice(8).reduce((s,e)=>s+e[1],0);
    items = items.slice(0,8);
    if (restSum > 0) items.push(["其他", restSum]);
  }
  const total = Object.values(counts).reduce((s,v)=>s+v,0) || 1;
  const max = Math.max(...items.map(e=>e[1]), 1);
  const bw = 58, gap = 30, n = items.length, h = 270, base = h - 52;
  const w = Math.max(440, 56 + n*(bw+gap) + 16);
  let s = `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:auto;max-width:720px"><rect x="0" y="0" width="${w}" height="${h}" rx="12" fill="#fff"/>`;
  items.forEach((e,i)=>{
    const key = e[0], val = e[1];
    const hh = Math.max(6, val/max*(base-72));
    const x = 46 + i*(bw+gap), y = base - hh;
    /* 理论期望：虚线画在柱顶对应高度，标签放在虚线右端（柱外），避免与百分比重叠 */
    if (expected && expected[key] != null) {
      const ey = base - Math.max(6, expected[key]*total/max*(base-72));
      const lx = x + bw + 7;
      s += `<line x1="${x-4}" y1="${ey}" x2="${lx+66}" y2="${ey}" stroke="#8b8a83" stroke-width="1.3" stroke-dasharray="4,3"/>`;
      s += `<text x="${lx}" y="${ey-3}" text-anchor="start" font-size="11" fill="#8b8a83">理论 ${Math.round(expected[key]*100)}%</text>`;
    }
    /* 柱 */
    s += `<rect x="${x}" y="${y}" width="${bw}" height="${hh}" rx="6" fill="#378add"/>`;
    /* 百分比：柱够高放柱内（白字），否则放柱顶上方 */
    const pct = (val/total*100).toFixed(1) + "%";
    if (hh > 26) {
      s += `<text x="${x+bw/2}" y="${y+14}" text-anchor="middle" font-size="12" fill="#fff" font-weight="600">${pct}</text>`;
    } else {
      s += `<text x="${x+bw/2}" y="${y-7}" text-anchor="middle" font-size="11.5" fill="#444441">${pct}</text>`;
    }
    /* 位串标签（柱下方） */
    s += `<text x="${x+bw/2}" y="${base+20}" text-anchor="middle" font-size="12.5" fill="#5f5e5a">${esc(key)}</text>`;
  });
  return s + "</svg>";
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
