// LoomQ local stdgates.inc (Braket LocalSimulator compatible)
// Covers the 12-gate whitelist used by the competition circuits.
//
// IMPORTANT: phase gates (z/s/sdg/t/tdg/p/cp) are defined directly with
// U(0,0,lambda) = diag(1, e^{i*lambda}), which matches the standard qelib1
// phase-gate convention. We deliberately avoid `gphase`/`pow`/`inv` here:
// Braket's LocalSimulator was observed to mis-handle those primitives on
// multi-qubit interference circuits (e.g. Toffoli decomposition), which
// corrupts relative phases even though single-qubit measurements look fine.

gate z a { U(0, 0, pi) a; }
gate s a { U(0, 0, pi/2) a; }
gate sdg a { U(0, 0, -pi/2) a; }
gate t a { U(0, 0, pi/4) a; }
gate tdg a { U(0, 0, -pi/4) a; }
gate x a { U(pi, 0, pi) a; }
gate h a { U(pi/2, 0, pi) a; }
gate ry(theta) a { U(theta, 0, 0) a; }
gate rz(lambda) a { U(0, 0, lambda) a; }
gate p(lambda) a { U(0, 0, lambda) a; }
gate cnot a, b { ctrl @ x a, b; }
gate cx a, b { ctrl @ x a, b; }
gate cp(lambda) a, b { ctrl @ p(lambda) a, b; }
gate swap a, b { cnot a, b; cnot b, a; cnot a, b; }
gate ccx a, b, c { ctrl @ ctrl @ x a, b, c; }
