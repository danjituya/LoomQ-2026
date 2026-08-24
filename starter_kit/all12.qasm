OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
// --- h ---
h q[0];
// --- x ---
x q[1];
// --- s / sdg / t / tdg (phase gates, all on q[0]) ---
s q[0];
sdg q[0];
t q[0];
tdg q[0];
// --- rz / ry (rotation gates) ---
rz(1.570796326795) q[2];
ry(0.785398163397) q[3];
// --- cx (entangle) ---
cx q[0], q[1];
// --- cu1 (controlled phase) ---
cu1(0.785398163397) q[2], q[3];
// --- swap ---
swap q[1], q[3];
// --- ccx (Toffoli) ---
ccx q[0], q[2], q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
