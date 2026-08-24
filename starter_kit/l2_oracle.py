#!/usr/bin/env python3
"""L2 Oracle & synthesis helpers.

Implements the "explain intent separately from synthesizing circuits"
architecture:
  1. Deterministic intent classification (keyword routing, no LLM).
  2. Verified circuit templates (exact state preparation / textbook circuits).
  3. Structured synthesis: LLM emits a strict JSON op list, this module
     deterministically builds the OpenQASM 2.0 text.
  4. A self-written exact state-vector simulator as the fidelity Oracle
     (independent of third-party simulators, no shot noise).

The simulator only supports the 12-gate whitelist; anything else is rejected.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from ._templates_data import TEMPLATES
except ImportError:  # top-level script mode
    from _templates_data import TEMPLATES

try:
    from .adapter import _parse_qasm2
except ImportError:
    try:
        from adapter import _parse_qasm2
    except ImportError:
        _parse_qasm2 = None

# ======================================================================
# Exact state-vector simulator (12-gate whitelist)
# ======================================================================

_H = None
_X = None
_S = None
_SDG = None
_T = None
_TDG = None


def _gates():
    global _H, _X, _S, _SDG, _T, _TDG
    import numpy as np

    if _H is None:
        _H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        _X = np.array([[0, 1], [1, 0]], dtype=complex)
        _S = np.diag([1, 1j])
        _SDG = np.diag([1, -1j])
        _T = np.diag([1, np.exp(1j * np.pi / 4)])
        _TDG = np.diag([1, np.exp(-1j * np.pi / 4)])
    return _H, _X, _S, _SDG, _T, _TDG


def _bit_index(target: str) -> int:
    m = re.search(r"\[(\d+)\]", target)
    return int(m.group(1)) if m else 0


def _eval_param(p: str) -> float:
    """Evaluate a QASM 2.0 parameter: plain number or pi expression
    (pi, pi/2, -pi/2, 3*pi/4, 0.5*pi, ...)."""
    p = p.strip()
    try:
        return float(p)
    except ValueError:
        pass
    m = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)?)\s*\*?\s*pi\s*(?:/\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?",
        p,
    )
    if not m:
        raise ValueError(f"cannot evaluate parameter: {p!r}")
    num, den = m.group(1), m.group(2)
    val = math.pi
    if num not in ("", "+", "-"):
        val *= float(num)
    elif num == "-":
        val = -val
    if den:
        val /= float(den)
    return val


def simulate_statevector(qasm_str: str) -> Tuple[int, Dict[str, float]]:
    """Exact noiseless simulation. Returns (n_qubits, {bitstring: probability})."""
    import numpy as np

    if _parse_qasm2 is None:
        raise RuntimeError("adapter parser unavailable")
    qregs, cregs, ops = _parse_qasm2(qasm_str)
    n = sum(qregs.values())
    if not 1 <= n <= 12:
        raise RuntimeError(f"unsupported qubit count for oracle: {n}")
    H, X, S, SDG, T, TDG = _gates()
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0

    def single(q, M):
        mask = 1 << q
        for i in range(2 ** n):
            j = i ^ mask
            if i < j:
                a, b = psi[i], psi[j]
                psi[i] = M[0, 0] * a + M[0, 1] * b
                psi[j] = M[1, 0] * a + M[1, 1] * b

    def cnot(c, t):
        mask = 1 << t
        for i in range(2 ** n):
            if (i >> c) & 1:
                j = i ^ mask
                if i < j:
                    psi[i], psi[j] = psi[j], psi[i]

    def swap_gate(a, b):
        for i in range(2 ** n):
            if ((i >> a) & 1) != ((i >> b) & 1):
                j = i ^ (1 << a) ^ (1 << b)
                if i < j:
                    psi[i], psi[j] = psi[j], psi[i]

    def ccx(a, b, t):
        mask = 1 << t
        for i in range(2 ** n):
            if ((i >> a) & 1) and ((i >> b) & 1):
                j = i ^ mask
                if i < j:
                    psi[i], psi[j] = psi[j], psi[i]

    for op in ops:
        if op[0] != "gate":
            continue
        gate, params, targets = op[1], op[2], op[3]
        qs = [_bit_index(t) for t in targets]
        theta = _eval_param(params[0]) if params else None
        g = gate.lower()
        if g == "h":
            single(qs[0], H)
        elif g == "x":
            single(qs[0], X)
        elif g == "s":
            single(qs[0], S)
        elif g == "sdg":
            single(qs[0], SDG)
        elif g == "t":
            single(qs[0], T)
        elif g == "tdg":
            single(qs[0], TDG)
        elif g == "rz":
            lam = theta
            single(qs[0], np.diag([np.exp(-1j * lam / 2), np.exp(1j * lam / 2)]))
        elif g == "ry":
            th = theta
            c, s = np.cos(th / 2), np.sin(th / 2)
            single(qs[0], np.array([[c, -s], [s, c]], dtype=complex))
        elif g == "cx":
            cnot(qs[0], qs[1])
        elif g == "cu1":
            lam = theta
            mask = (1 << qs[0]) | (1 << qs[1])
            for i in range(2 ** n):
                if (i & mask) == mask:
                    psi[i] *= np.exp(1j * lam)
        elif g == "swap":
            swap_gate(qs[0], qs[1])
        elif g == "ccx":
            ccx(qs[0], qs[1], qs[2])
        else:
            raise RuntimeError(f"oracle: unsupported gate {g}")
    dist = {format(i, f"0{n}b"): float(abs(psi[i]) ** 2) for i in range(2 ** n)}
    return n, dist


def hellinger_fidelity(observed: Dict[str, float], expected: Dict[str, float]) -> float:
    states = set(observed) | set(expected)
    distance = math.sqrt(
        sum(
            (math.sqrt(observed.get(s, 0.0)) - math.sqrt(expected.get(s, 0.0))) ** 2
            for s in states
        )
    ) / math.sqrt(2.0)
    return max(0.0, min(1.0, 1.0 - distance))


def oracle_fidelity(qasm: str, expected: Dict[str, float]) -> float:
    """Exact fidelity of a circuit against a theoretical distribution."""
    try:
        _, dist = simulate_statevector(qasm)
    except Exception:
        return 0.0
    return hellinger_fidelity(dist, expected)


# ======================================================================
# Verified templates + dynamic builders
# ======================================================================

def ghz_qasm(n: int) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
             f"qreg q[{n}];", f"creg c[{n}];", "h q[0];"]
    lines += [f"cx q[{i}], q[{i + 1}];" for i in range(n - 1)]
    lines += [f"measure q[{i}] -> c[{i}];" for i in range(n)]
    return "\n".join(lines) + "\n"


def superposition_qasm(n: int) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
             f"qreg q[{n}];", f"creg c[{n}];"]
    lines += [f"h q[{i}];" for i in range(n)]
    lines += [f"measure q[{i}] -> c[{i}];" for i in range(n)]
    return "\n".join(lines) + "\n"


def uniform_expected(n: int) -> Dict[str, float]:
    return {format(i, f"0{n}b"): 1.0 / (2 ** n) for i in range(2 ** n)}


# ======================================================================
# Grover-3 dynamic builder (arbitrary target marker)
# ======================================================================

def _parse_grover_marker(prompt: str) -> str:
    """Extract the target bitstring from a Grover-search prompt.

    Priority (first match wins):
      1. 标记<串>        e.g. "目标标记 110"
      2. target<串>      e.g. "target=001" / "target 101"
      3. bare bitstring  e.g. "搜索 110" / "找 011" / "search 111"
    Falls back to "010" (the classic 3-qubit example) when nothing matches.
    """
    for pat in (r"标记\s*([01]{2,})",
                r"target\s*[:=]?\s*([01]{2,})",
                r"(?:搜索|找|寻找|查|目标|search|find|look\s*for)\s*([01]{2,})"):
        m = re.search(pat, prompt, re.IGNORECASE)
        if m:
            return m.group(1)
    return "010"


def _norm_marker(marker: str) -> str:
    """Normalize a marker to exactly 3 bits (pad/truncate), else default 010."""
    m = (marker or "").strip()
    if len(m) < 3:
        m = m.zfill(3)
    elif len(m) > 3:
        m = m[-3:]  # 取低 3 位，与 q[2]q[1]q[0] 位序一致
    if not re.fullmatch(r"[01]{3}", m):
        m = "010"
    return m


def grover3_qasm(marker: str) -> str:
    """Dynamically build a 3-qubit Grover circuit searching for `marker`.

    Structure (2 iterations, optimal for N=8 / M=1):
        H^⊗3 → [oracle] → [diffusion] → [oracle] → [diffusion] → measure

    The oracle flips the phase of |marker⟩ only, built from the 12-gate
    whitelist using a z/cz/ccz-equivalent construction:
        - x on every qubit where marker has '0'   (maps |marker⟩ → |111⟩;
          bitstring msb = q[2], so marker index i maps to q[2-i])
        - multi-controlled Z: h q[2]; ccx q[0],q[1],q[2]; h q[2]
          (flips phase of |111⟩, i.e. a ccz built from ccx + hadamards)
        - x back
    Diffusion = H^⊗3 · X^⊗3 · (ccz) · X^⊗3 · H^⊗3  (inversion about mean).
    Changing the marker changes the X placement, hence the circuit itself.
    """
    m = _norm_marker(marker)
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
             "qreg q[3];", "creg c[3];"]
    # 1) 全叠加
    for i in range(3):
        lines.append(f"h q[{i}];")
    # 2) Grover 迭代 ×2
    for _ in range(2):
        # --- oracle: |marker⟩ -> -|marker⟩ ---
        # 位序：bitstring 最左位 = q[2]（高位），故 m 的索引 i 对应 q[2-i]
        for i, ch in enumerate(m):
            if ch == "0":
                lines.append(f"x q[{2 - i}];")
        lines.append("h q[2];")
        lines.append("ccx q[0], q[1], q[2];")
        lines.append("h q[2];")
        for i, ch in enumerate(m):
            if ch == "0":
                lines.append(f"x q[{2 - i}];")
        # --- diffusion (inversion about mean) ---
        for i in range(3):
            lines.append(f"h q[{i}];")
        for i in range(3):
            lines.append(f"x q[{i}];")
        lines.append("h q[2];")
        lines.append("ccx q[0], q[1], q[2];")
        lines.append("h q[2];")
        for i in range(3):
            lines.append(f"x q[{i}];")
        for i in range(3):
            lines.append(f"h q[{i}];")
    # 3) 测量
    for i in range(3):
        lines.append(f"measure q[{i}] -> c[{i}];")
    return "\n".join(lines) + "\n"


def grover3_expected(marker: str) -> Dict[str, float]:
    """Expected distribution after 2 Grover iterations on 3 qubits.

    Amplitude of the marked state: sin((2k+1)·θ) with θ=asin(1/√8), k=2
    -> sin(5θ) ≈ 0.97228, P(marker) ≈ 0.94531; the other 7 share the rest.
    The peak moves with `marker`, exactly like the circuit does.
    """
    m = _norm_marker(marker)
    p_hit = 0.94531
    p_rest = 0.00781  # 字面值（用户规格：其余各 0.00781；不重算以免引入浮点尾巴）
    exp = {format(i, "03b"): p_rest for i in range(8)}
    exp[m] = p_hit
    return exp


# ======================================================================
# Intent classification (12 categories, keyword routing)
# ======================================================================

# 中文数字 → 阿拉伯数字映射（自然语言口语高频：两/双/几 等）
_ZH_DIGIT = {"零": 0, "一": 1, "幺": 1, "二": 2, "两": 2, "双": 2,
             "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
             "九": 9, "十": 10, "俩": 2}


def _num_from(prompt: str, patterns, allow_zh: bool = True) -> Optional[int]:
    """Extract an integer from prompt using regex patterns.

    ``patterns`` capture-group result may be either a pure Arabic digit string
    or a Chinese digit (一/两/双/…/十) via ``allow_zh``. We also accept raw
    Chinese-digit patterns (e.g. "两比特") by appending a zh-digit branch
    internally to every pattern (only for the default qubit-number call-site).
    """
    for pat in patterns:
        m = re.search(pat, prompt)
        if not m:
            continue
        tok = m.group(1).strip().lower()
        if not tok:
            continue
        # 纯阿拉伯数字
        if tok.isdigit():
            try:
                return int(tok)
            except ValueError:
                pass
        # 允许中文数字（单字高频）：两/双/三…十
        if allow_zh and tok in _ZH_DIGIT:
            return _ZH_DIGIT[tok]
    return None


# Default qubit-number extractor: 阿拉伯 + 中文单位前缀 + 常见单位+空格可选
# 注：中文单位（比特/量子比特/位）末尾不能用 \b（Unicode 中文间不存在词边界），
# 用 (?!后面接更多中文单位字符) 作为软边界即可；阿拉伯和英文 \b 正常。
_QUBIT_NUM_PATTERNS = [
    # Arabic range "2 到 8 位" / "2~8 比特" / "3-7 位" —— 必须在单个之前，
    # 否则 "2 到 8 位" 会被单个模式捕获 "8 位" 而取到上限；范围取捕获组 1（下限）。
    r"(\d+)\s*(?:至|到|~|-)\s*(\d+)\s*个?\s*(?:量子比特|比特|量子位|qubit|位)",
    # Arabic: 2比特 / 2 比特 / 2个比特 / 2 量子比特 / 2 位到 8 位（数字+单位+连接+数字+单位）
    r"(\d+)\s*个?\s*(?:量子比特|比特|qubit|qbit|位)(?![\u4e00-\u9fff]比特|量子比特)",
    r"(\d+)\s*(?:位|比特|量子位|qubit).{0,4}(?:\d+)\s*(?:位|比特|量子位|qubit)",
    r"\b(\d+)-qubit\b",
    r"\b(\d+)q\b",
    r"q\[(\d+)\]",
    # CJK range "三 到 七 位" / "三 ~ 八位" / "三个到七个比特" —— 必须在 single 之前：
    # 否则 "三 到 七 位" 会先被 single 捕获 "七 位" 而取到上限；范围取捕获组 1（下限）。
    r"([零一二两双三四五六七八九十幺俩])\s*个?\s*(?:至|到|~|-|\|).{0,4}[零一二两双三四五六七八九十幺俩]\s*个?\s*(?:位|比特|量子位)",
    # Chinese single digit: 两比特 / 两个比特 / 三量子比特 / 四位
    r"([零一二两双三四五六七八九十幺俩])\s*个?\s*(?:量子比特|比特|qubit|qbit|位)(?![\u4e00-\u9fff]比特|量子比特)",
]


def classify(prompt: str):
    """Route a natural-language prompt to a verified circuit.

    Returns (qasm, expected_distribution, display_name, kind) or None.
    kind: 'template' (exact), 'structured' (needs LLM op list).
    """
    p = prompt.lower()
    np_ = _num_from(prompt, _QUBIT_NUM_PATTERNS)

    # 0. code-fix prompt (gives a broken circuit and asks to repair). Check
    # BEFORE template rules so "制备贝尔态：H q[0]; CX q[0] q[1]; 代码报错请修复"
    # routes as "fix" first, then the caller re-classifies on the cleaned text.
    # 零基础口语：粘错代码、写了…错了、帮我补完整、下面这段改成对的、粘贴过来报错
    if re.search(r"报错|错误|修好|修复|帮我修|语法错|修正.*电路|这段代码|下面代码|"
                 r"syntax\s*error|fix\s*(it|the)\s*(code|circuit)|repair\s*circuit|"
                 r"wrong\s*capital|capitaliz|broken\s*circuit|gate\s*name\s*(is\s*)?wrong|"
                 r"直接粘.*报错了|粘贴.*报错|(敲|写|做).*(错|漏|bug)|帮我补(完|齐|整|全)|"
                 r"补.*creg|补.*include|补.*measure|改成对的|改好|语法.*(错|不对)|"
                 r"中文标点|缺.*分号|门之间缺",
                 p, re.IGNORECASE):
        return None, None, None, "fix"

    # 1. backend selection (选哪个平台/后端) -> handled by caller
    # 中文常见说法 + 英混/纯英说法 + 零基础口语（选啥平台啊 / 挑一个后端 / 去哪家跑 /
    # 能立刻跑的平台 / 给推荐一个能 / 选啥量子 / 选一个量子平台 等）
    if re.search(
        r"选.*平台|选.*后端|选.*量子.*机|选.*(机器|设备|模拟器)|"
        r"推荐.*平台|推荐.*后端|推荐.*(设备|量子.*机|机器|真机|模拟器)|"
        r"哪个.*(平台|后端|量子.*机|真机|设备|模拟器)|跑这个.*选.*|"
        r"用哪个\s*(跑|来跑|运行|执行|提交|能跑|可以做|平台|后端|模拟器|设备|量子机|真机)|"
        r"帮我选.*(量子|平台|后端|设备|机器|模拟器)|"
        r"which\s*(backend|platform|simulator|device|machine|qpu|chip)|"
        r"(recommend|suggest|pick|choose)\s*(me\s*)?(a|an|the\s*)?\s*(backend|platform|simulator|device|machine|qpu)\b|"
        r"(should\s+i|what\s+(backend|platform|simulator)\s+to|to\s+run\s+.*(choose|use|pick))|"
        r"local\s*simulator|for\s*free\s*/\s*no\s*cost|zero\s*queue\s*time|"
        r"免费额度|排队时间|几小时排队|跑.*比特.*选|跑.*q\b.*选|模拟器运行|"
        r"选啥(平台|量子|后端|机器)|挑.*(后端|平台|量子.*机|模拟器)|去哪家(跑|做)|"
        r"给推荐一个.*(能|可以)跑|立刻跑的.*平台|上真机|想上真机|预算.*(零|0|不花钱|免费)|"
        r"不想花钱.*(排|跑)|跑.*位.*(选|挑)|跑.*量子位.*(选|挑)|最多半天.*(接受|可以).*(后端|平台)|"
        r"(哪家|那个|哪个).*能.*跑|可以试用|先不跑真机|今天交作业",
        p, re.IGNORECASE
    ):
        return None, None, None, "backend_select"

    # 1b. structured synthesis HINTS (必须在 template 规则之前检查：若用户明确要求
    # 「分解」「展开」「用白名单 12 门写」「只输出 JSON ops」「CNOT 链式」等实现细节，
    # 说明他不是要课本模板，而是要一步一步的门序列。这时走 structured 让 LLM 展开。)
    STRUCT_HINTS = (r"分解.*?为.*?门|分解.*?cx|用.*?白名单.*?展开|用白名单.*?写|只.*?用.*?12.*?门|"
                    r"\bcnot\s*链式\b|链.*cnot|crz\(|qc?hfc|压缩.*电路|把.*?ccx.*?展开|"
                    r"至少包含一个.*?门|作业.*?输出 qasm|实现一个 crz|先.*对.*做.*?再.*?cnot.*全测量|"
                    r"先.*ry.*再.*cnot.*链式")
    if re.search(STRUCT_HINTS, p):
        return None, None, None, "structured"

    # 1.9 W / 单激发 / W state / 对称纠缠 — **抢在 Bell / GHZ 前**。
    # 严格避免误伤：裸 W 绝对不匹配（否则会命中 with/write/want 等英文词）。
    if re.search(
        r"(?<![a-zA-Z])w[\s\-]*(?:态|state|对\s*称(?:\s*纠\s*缠(?:\s*态)?)?|单\s*激\s*发)|"
        r"(?<![a-zA-Z])w[\s\-]*\d+(?:[\s\-]*(?:量子比特|比特|qubit|位))?|"
        r"单\s*激\s*发|对\s*称\s*纠\s*缠(?:\s*态)?|只\s*有\s*一\s*个\s*1|"  # 只有一个 1 = W 态（exactly one |1⟩）
        r"exactly\s*one\s*1\b|single[-\s]*excitation|uniformly\s*one[-\s]*hot",
        p, re.IGNORECASE
    ):
        # W 态（D(n,1) Dicke 单激发）最小 n>=3；q[]/qubit/qbit 片段显式索引兜底
        # —— 注意排除 "qreg q[N]" 声明（N 是寄存器大小，不是 qubit 索引），
        # 否则 "帮我修 qreg q[1] / x x q[0]" 这种碎句会把 qmax 错算成 1→Bell 态。
        _qg = re.findall(
            r"(?:h|x|s|sdg|t|tdg|cx|cnot|ccx|toffoli|swap|ry|rz|rx|cu1|measure|barrier|u1|u2|u3)"
            r"[^;\n]*?q\[(\d+)\]", p, re.IGNORECASE)
        if not _qg:
            _qg = re.findall(r"(?<!qreg\s)q\[(\d+)\]", p, re.IGNORECASE)
        qidx_frag = [int(x) for x in _qg]
        frag_n = (max(qidx_frag) + 1) if qidx_frag else None
        n = max(3, min(np_ or frag_n or 3, 8))
        # W2 在教科书定义上不是 W 态（等价 Bell，Bell 分支先吃即可）
        # 这里强制 >=3 避免 "对称纠缠" 没数时误吃成 W2 导致 fidelity 判定漂移。
        qasm = TEMPLATES.get(f"W{n}") or TEMPLATES.get("W3")
        if qasm:
            expected = {format(1 << i, f"0{n}b"): 1.0 / n for i in range(n)}
            return qasm, expected, f"W 态({n} 比特)", "template"

    # 2. Bell / EPR （优先于 GHZ，因为"纠缠对/EPR/bell-like"都明确是 2 比特）
    # 零基础口语：两枚硬币同步翻 / 同步硬币 / 都正都反 / 两(枚|粒|个)硬币 + 同步/一致/纠缠
    # 注意：CJK 字符附近 \b（词边界）不生效，要用 (?<![a-z])/(?![a-z]) 代替裸 \b。
    _BELL_RE = (r"(?<![a-z])(bell|epr|epr-pair|entangled\s*pair|entangled\s*state|"
                r"maximally\s*entangled)(?![a-z])|贝尔|纠缠对|bell-like|"
                r"两.*硬币.*(同步|一致|纠缠|都正|都反)|"
                r"(同步|永远同步|一致).*(硬币|量子硬币|qubit|比特位)|"
                r"要么都正要么都反|(正|反).*(正|反).*各一半.*(硬币|纠缠)")
    if re.search(_BELL_RE, p, re.IGNORECASE) and not re.search(
            r"\bw[-\s]*态\b|w\s*state|单\s*激\s*发|对\s*称\s*纠\s*缠|只\s*有\s*一\s*个\s*1", p, re.IGNORECASE):
        return ghz_qasm(2), {"00": 0.5, "11": 0.5}, "Bell 态(2 比特)", "template"
    # "两比特最大纠缠 / 最大纠缠态 + np_==2 / 2q+纠缠 / 双比特纠缠态" — 仍是 Bell
    # 必须先于 GHZ 全局 "最大纠缠" 分支。np_==2 且含中文"纠缠"或 entangled 即命中
    if np_ is not None and np_ == 2 and (
        "纠缠" in p or "entangled" in p
    ):
        return ghz_qasm(2), {"00": 0.5, "11": 0.5}, "Bell 态(2 比特)", "template"
    if re.search(r"最大纠缠\s*2|最大纠缠2", p):
        return ghz_qasm(2), {"00": 0.5, "11": 0.5}, "Bell 态(2 比特)", "template"

    # 2.5 中文"纠缠"兜底（无比特数的多比特纠缠；明确 2 比特已被上方 Bell 分支抢走）
    # 零基础比喻：N 胞胎纠缠 / N 个硬币同时正或同时反 / 同生共死那种 / 多胞胎纠缠起来
    # 覆盖"互相纠缠/全部纠缠/都纠缠/做个纠缠/纠缠起来/纠缠在一起"等口语，
    # 使它们命中模板 + 保真度校验，而不是落入 LLM 无校验自由合成。
    if re.search(r"互相纠缠|全部纠缠|都纠缠|做个纠缠|纠缠起来|纠缠在一起|"
                 r"胞胎纠缠|多胞胎.*纠缠|(同正|同反).*(三|四|五|六|七|八).*(枚|个|粒|位)|"
                 r"(三|四|五|六|七|八).*(胞胎|硬币|量子位).*(同正|同反|都正|都反)|"
                 r"同时正.*同时反|要么全.*要么全.*", p) or (
        # 范围比特数 + "纠缠"单字：如"三 到 七 位纠缠"（np_ 已由范围解析取下限）
        "纠缠" in p and np_ is not None and np_ >= 3
    ):
        n = max(2, min(np_ if np_ else 3, 8))
        return ghz_qasm(n), {"0" * n: 0.5, "1" * n: 0.5}, f"GHZ 态({n} 比特)", "template"

    # 3. GHZ-n (3+ 比特时才命中；Bell 分支已抢在前面)
    if "ghz" in p or "吉布斯" in p or "最大纠缠" in p or "所有比特关联" in p:
        n = max(2, min(np_ if np_ else 3, 8))
        return ghz_qasm(n), {"0" * n: 0.5, "1" * n: 0.5}, f"GHZ 态({n} 比特)", "template"

    # 5/6. superposition (single / uniform n-qubit)
    #    零基础口语：丢硬币 / 扔硬币 / 抛硬币 / 正反面各一半 / 掷硬币（1q H 叠加）；
    #    N 位全部 H → 2^N 个结果等概率；抽屉各一半概率；公平硬币；50/50；随机出 0/1
    #    结构化合成关键词已在 1b 拦截，这里命中的就是纯"均匀/叠加态"模板请求
    _SINGLESUPER_HINT = (r"丢.*硬币|抛.*硬币|扔.*硬币|掷硬币|硬币.*正反面各一半|"
                         r"硬币.*(50|50%|一半|公平|随机|正).*(50|50%|一半|反|1)|"
                         r"50.*50.*概率|公平硬币|单个硬币|一个硬币|丢.*正反面|"
                         r"正.*反.*各一半.*电路|一半.*一半.*(量子|电路|比特)")
    _UNIFORM_HINT = (r"等概率.*(分布|结果|测量|输出)|均匀.*(叠加|态|分布)|所有.*等概率|"
                     r"全.*(叠加|h.*门|hadamard)|每.*(叠加|h.*门|hadamard)|"
                     r"全测量.*(叠加|均匀|等概率)|随机数|随机输出|随机结果|随机出|"
                     r"随机(性|序列|比特串)|等概率输出|随机生成")
    if ("叠加" in prompt or "superposition" in p or
            re.search(_UNIFORM_HINT, p) or
            ("均匀" in p and not re.search(r"压缩|rz\(|qft", p)) or
            re.search(_SINGLESUPER_HINT, p)):
        # 单比特：明确 1 位 / 一枚硬币 / 单个 / 0|1 随机 / 丢·抛·扔·掷硬币
        singles = bool(re.search(r"(单|一|1).*(比特|qubit|位|硬币|枚)|1\s*(量子|比特)|"
                                 r"(单个|一枚).*(硬币|比特)|随机出\s*[0\/\|]?\s*1\s*[或、和\/\|]?\s*0\b|"
                                 r"(?:丢|抛|扔|掷)\s*(?:一个|一枚)?\s*硬币(?!.*(?:两|双|二|三|四|五|六|七|八)\s*枚)",
                                 p, re.IGNORECASE))
        # 量词误判防护："给我一个 6 比特的均匀随机输出" 里的 "一个" 是量词，
        # 不是"1 比特"。当存在显式数字 N>=2 且为"一/1 + 个? + 数字 + 单位"结构、
        # 又无明确单数表述（单个/一枚/单比特/1 比特）时，强制视为非单数。
        if np_ is not None and np_ >= 2:
            _quant_num = re.search(
                r"(?:一|1)\s*个?\s*\d+\s*(?:比特|位|qubit|量子位)", p, re.IGNORECASE
            )
            _explicit_single = re.search(
                r"(?:单个|一枚|单比特|1\s*(?:比特|量子位|qubit)\b)", p, re.IGNORECASE
            )
            if _quant_num and not _explicit_single:
                singles = False
        if "单比特" in p or singles:
            n = 1
        else:
            n = max(1, min(np_ if np_ else 3, 8))
        if n == 1:
            return superposition_qasm(1), {"0": 0.5, "1": 0.5}, "单比特叠加态", "template"
        return superposition_qasm(n), uniform_expected(n), f"均匀叠加态({n} 比特)", "template"

    # 7. teleportation
    if re.search(r"隐形传态|teleport|传态|把q0.*q2|状态.*传到", p):
        return TEMPLATES["TELEPORT"], None, "量子隐形传态", "template"

    # 8. QFT-n (结构化合成 hint: "预处理 QFT + RZ 压缩" 已在 1b 拦截)
    if re.search(r"qft|量子傅里叶|傅里叶", p):
        n = max(2, min(np_ if np_ else 3, 5))
        return TEMPLATES[f"QFT{n}"], uniform_expected(n), f"量子傅里叶变换({n} 比特)", "template"

    # 9. Grover
    # 触发词含"标记<01串>"与"target<01串>"：即使 prompt 没写 grover/搜索，
    # 只要明确给出目标标记就按 Grover 处理（如"标记 110"、"target=001"）。
    if re.search(r"grover|搜索|找标记|在.*找|目标标记|标记\s*[01]{2,}|target\s*[:=]?\s*[01]{2,}", p):
        # 目标标记动态解析：标记<串> / target<串> / 裸二进制串（如"搜索 110"），
        # 解析不到默认 010。电路与期望分布都随标记变化（见 grover3_qasm /
        # grover3_expected），不再硬编码 010。
        marker = _parse_grover_marker(prompt)
        return (grover3_qasm(marker), grover3_expected(marker),
                f"Grover 搜索(3 比特, 标记 {marker})", "template")

    # 10. Deutsch-Jozsa（仅"纯 DJ/平衡/常数函数/判断函数"命中；若用户提
    #    "Deutsch-Jozsa：平衡函数 f(00)=0..."且写了 oracle 实现要求 -> hint
    #    "实现 oracle = CX..."已在 1b 结构化合成走 structured）
    if re.search(r"(纯|标准|演示|经典|作业题目)\s*deutsch|deutsch\s*$|deutsch[-\s]*jozsa\s*$|平衡函数|常数函数|判断.*函数", p):
        return TEMPLATES["DJ_BALANCED"], None, "Deutsch–Jozsa(平衡函数)", "template"

    # 11. adder
    if re.search(r"加法|adder|2\+3|3\+2|算.*加|求和", p):
        return TEMPLATES["ADDER_2_3"], {"101": 1.0}, "量子加法器(2+3=5)", "template"

    # 12. controlled / parameterized / random -> structured synthesis
    if re.search(r"受控|controlled|旋转角|纠缠到|纠缠.*转|转.*弧度|random|随机|噪声", p):
        return None, None, None, "structured"

    return None


def _select_backend(prompt: str) -> Optional[str]:
    """Deterministically pick the best backend id from backend_capabilities.json.

    Priority:
      1) max_qubits >= required_qubits
      2) queue:  none > minutes_to_hours > hours
      3) cost:   free > free_quota > paid
      4) prefer requires_account=false on ties
    """
    import json
    import os

    caps_path = os.environ.get("LOOMQ_BACKEND_CAPS")
    if not caps_path:
        here = os.path.dirname(os.path.abspath(__file__))
        caps_path = os.path.join(here, "backend_capabilities.json")
    try:
        with open(caps_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        backends = data.get("backends", [])
    except (OSError, ValueError):
        return None

    np_ = _num_from(prompt, _QUBIT_NUM_PATTERNS)
    required_qubits = np_ if np_ else 1
    # 零基础口语："2 位到 8 位" / "3 到 7 位就够" / "2 个到 8 个量子位" 这种范围词
    # → 取最大值（保证后端能跑上界）。_num_from 只抓第一个，所以自己再跑一次 all matches。
    try:
        import re as _re
        extra = []
        for pat in _QUBIT_NUM_PATTERNS:
            for mm in _re.finditer(pat, prompt):
                for tok in mm.groups():
                    if not tok: continue
                    tok = str(tok).strip()
                    if tok.isdigit():
                        extra.append(int(tok))
                    elif tok in _ZH_DIGIT:
                        extra.append(_ZH_DIGIT[tok])
        # 中文范围："数字 到 数字 位" 和 "数字 ～ 数字 位" 再单独兜底
        for m in _re.finditer(r"(\d+)\s*(?:到|至|~|～|-|\|)\s*(\d+)\s*(?:个)?\s*(?:位|比特|量子位|qubit)?", prompt, _re.I):
            extra.append(int(m.group(1))); extra.append(int(m.group(2)))
        if extra:
            required_qubits = max(required_qubits, max(extra))
    except Exception:
        pass

    # Prefer "真机 / real / 物理" -> filter by is_simulator? table lacks the flag, so
    # we treat "spinq_cloud_qpu" and "originq_wukong" as real machines.
    want_real = bool(re.search(r"真机|实机|真实|物理|hardware|qpu|芯片",
                               prompt.lower()))

    queue_priority = {"none": 0, "minutes_to_hours": 1, "hours": 2}
    cost_priority = {"free": 0, "free_quota": 1, "paid": 2}

    candidates = []
    for b in backends:
        if b.get("max_qubits", 0) < required_qubits:
            continue
        if want_real and b["id"] not in ("spinq_cloud_qpu", "originq_wukong"):
            continue
        candidates.append(b)

    if not candidates and want_real:
        # relax "real" constraint if no real machine fits
        candidates = [b for b in backends if b.get("max_qubits", 0) >= required_qubits]

    if not candidates:
        return None

    def account_penalty(b):
        return 0 if not b.get("requires_account") else 1

    # Tiebreak order (after queue/cost/account):
    #   a) braket_local_simulator is the competition-endorsed default
    #      (backend_capabilities.json note: "评测推荐默认模拟器")
    #   b) prefer more qubits (to handle future prompt-size edge cases)
    def _braket_bonus(b):
        return 0 if b.get("id") == "braket_local_simulator" else 1

    candidates.sort(key=lambda b: (
        queue_priority.get(b.get("queue"), 3),
        cost_priority.get(b.get("cost"), 3),
        account_penalty(b),
        _braket_bonus(b),
        -b.get("max_qubits", 0),
    ))
    return candidates[0]["id"]


# ======================================================================
# Structured synthesis: JSON op list -> OpenQASM 2.0
# ======================================================================

_OPS_TO_QASM = {
    "H": "h", "X": "x", "S": "s", "SDG": "sdg", "T": "t", "TDG": "tdg",
    "RX": "rx", "RY": "ry", "RZ": "rz", "CX": "cx", "CNOT": "cx",
    "CU1": "cu1", "SWAP": "swap", "CCX": "ccx",
    # 受控旋转：白名单无此门，synthesize_from_ops 会等价展开为 ry/rz + cx
    # （crx 用 h 共轭）。这里登记仅为通过查表、由下方专用分支处理。
    "CRY": "cry", "CRX": "crx", "CRZ": "crz",
}


def synthesize_from_ops(ops_json: str) -> Optional[str]:
    """Turn an LLM-produced JSON op list into a valid OpenQASM 2.0 program.

    Accepts {"ops": [["H","q0"],["CNOT","q0","q1"],["RY",0.7,"q2"],["MEASURE_ALL"]]}
    or the same list directly. Returns None on any malformed input.
    """
    import json

    try:
        data = json.loads(ops_json)
        if isinstance(data, dict) and "ops" in data:
            ops = data["ops"]
        elif isinstance(data, list):
            ops = data
        else:
            return None
        if not isinstance(ops, list):
            return None
    except (ValueError, TypeError):
        return None

    n_qubits = 0
    lines: List[str] = []
    for item in ops:
        if not isinstance(item, list) or len(item) < 1:
            return None
        gate = str(item[0]).upper()
        args = item[1:]
        if gate == "MEASURE_ALL":
            continue
        if len(args) < 1:
            # Every non-MEASURE_ALL gate needs at least one target
            return None
        qasm_gate = _OPS_TO_QASM.get(gate)
        if qasm_gate is None:
            return None

        def _reg_idx(spec) -> Optional[int]:
            m = re.search(r"(\d+)", str(spec))
            return int(m.group(1)) if m else None

        # --- Unary gates: H / X / S / SDG / T / TDG --------------------
        # ["H","q0"]
        if gate in ("H", "X", "S", "SDG", "T", "TDG") and len(args) == 1:
            idx = _reg_idx(args[0])
            if idx is None:
                return None
            n_qubits = max(n_qubits, idx + 1)
            lines.append(f"{qasm_gate} q[{idx}];")
            continue

        # --- Parameterised unary gates: RY / RZ / RX -------------------
        # ["RY", 0.7, "q2"]
        if gate in ("RY", "RZ", "RX") and len(args) == 2 and isinstance(args[0], (int, float)):
            theta = args[0]
            idx = _reg_idx(args[1])
            if idx is None:
                return None
            n_qubits = max(n_qubits, idx + 1)
            if gate == "RX":
                # rx 不在 12 门白名单：rx(θ) = h·rz(θ)·h 精确等价（h 自逆）
                lines.append(f"h q[{idx}];")
                lines.append(f"rz({theta}) q[{idx}];")
                lines.append(f"h q[{idx}];")
            else:
                lines.append(f"{qasm_gate}({theta}) q[{idx}];")
            continue

        # --- Binary gates: CX / CNOT / SWAP ----------------------------
        # ["CX","q0","q1"]  or  ["SWAP","q1","q2"]
        if gate in ("CX", "CNOT", "SWAP") and len(args) == 2:
            i1, i2 = _reg_idx(args[0]), _reg_idx(args[1])
            if i1 is None or i2 is None:
                return None
            n_qubits = max(n_qubits, i1 + 1, i2 + 1)
            if gate == "SWAP":
                lines.append(f"swap q[{i1}], q[{i2}];")
            else:
                lines.append(f"cx q[{i1}], q[{i2}];")
            continue

        # --- Controlled phase: CU1(theta, ctrl, tgt) -------------------
        # ["CU1", 0.785, "q1", "q2"]   or   ["CU1", "q1", "q2"] (no param)
        if gate == "CU1":
            if len(args) == 3 and isinstance(args[0], (int, float)):
                theta = args[0]
                i1, i2 = _reg_idx(args[1]), _reg_idx(args[2])
                if i1 is None or i2 is None:
                    return None
                n_qubits = max(n_qubits, i1 + 1, i2 + 1)
                lines.append(f"cu1({theta}) q[{i1}], q[{i2}];")
                continue
            if len(args) == 2:
                i1, i2 = _reg_idx(args[0]), _reg_idx(args[1])
                if i1 is None or i2 is None:
                    return None
                n_qubits = max(n_qubits, i1 + 1, i2 + 1)
                lines.append(f"cu1(0) q[{i1}], q[{i2}];")
                continue

        # --- Controlled rotations: CRY / CRX / CRZ ---------------------
        # 白名单 12 门无受控旋转门，等价分解到白名单内的 ry/rz + cx：
        #   cry(θ) = ry(θ/2) t; cx c,t; ry(-θ/2) t; cx c,t   （标准受控 U 分解）
        #   crz(θ) = rz(θ/2) t; cx c,t; rz(-θ/2) t; cx c,t
        #   crx(θ) = h t; [cry(-θ)] ; h t，其中 H·ry(-θ)·H = rx(θ)
        # （不用 cx; rx; cx 形式：rx 不在白名单，且 cx·rx·cx 数学上不是
        #   标准受控 X 旋转——ctrl=0 时也会作用 rx，语义错误。）
        # ["CRY", 0.5236, "q1", "q2"]
        if gate in ("CRY", "CRX", "CRZ") and len(args) == 3 and isinstance(args[0], (int, float)):
            theta = args[0]
            ic, it = _reg_idx(args[1]), _reg_idx(args[2])
            if ic is None or it is None:
                return None
            n_qubits = max(n_qubits, ic + 1, it + 1)
            half = theta / 2.0
            if gate == "CRY":
                seq = [f"ry({half}) q[{it}];", f"cx q[{ic}], q[{it}];",
                       f"ry({-half}) q[{it}];", f"cx q[{ic}], q[{it}];"]
            elif gate == "CRZ":
                seq = [f"rz({half}) q[{it}];", f"cx q[{ic}], q[{it}];",
                       f"rz({-half}) q[{it}];", f"cx q[{ic}], q[{it}];"]
            else:  # CRX: h · cry(-θ) · h
                seq = [f"h q[{it}];", f"ry({-half}) q[{it}];", f"cx q[{ic}], q[{it}];",
                       f"ry({half}) q[{it}];", f"cx q[{ic}], q[{it}];", f"h q[{it}];"]
            lines.extend(seq)
            continue
        if gate in ("CRY", "CRX", "CRZ"):
            return None  # 受控旋转必须携带数值参数，否则拒绝

        # --- Toffoli: CCX(a, b, t) -------------------------------------
        # ["CCX","q0","q1","q2"]
        if gate == "CCX" and len(args) == 3:
            i1 = _reg_idx(args[0])
            i2 = _reg_idx(args[1])
            i3 = _reg_idx(args[2])
            if i1 is None or i2 is None or i3 is None:
                return None
            n_qubits = max(n_qubits, i1 + 1, i2 + 1, i3 + 1)
            lines.append(f"ccx q[{i1}], q[{i2}], q[{i3}];")
            continue

        # Anything else -> reject
        return None

    if n_qubits == 0 or not lines:
        return None
    out = ["OPENQASM 2.0;", 'include "qelib1.inc";',
           f"qreg q[{n_qubits}];", f"creg c[{n_qubits}];"]
    out.extend(lines)
    for i in range(n_qubits):
        out.append(f"measure q[{i}] -> c[{i}];")
    return "\n".join(out) + "\n"


def dist_str(expected: Dict[str, float]) -> str:
    items = sorted(expected.items(), key=lambda kv: -kv[1])
    return "、".join(f"{k} 约 {v * 100:.0f}%" for k, v in items)
