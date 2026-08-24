// all12.qasm - 白名单 12 门全覆盖电路（L1 自动化验证用）
// 覆盖：h, x, s, sdg, t, tdg, rz, ry, cx, cu1, swap, ccx + 全测量
// 期望：两后端（braket/originq）运行结果与精确态矢量模拟一致
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
x q[1];
s q[2];
sdg q[3];
t q[0];
tdg q[1];
rz(0.3) q[2];
ry(0.7) q[3];
cx q[0], q[1];
cu1(0.5) q[1], q[2];
swap q[2], q[3];
ccx q[0], q[1], q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
