// LoomQ local stdgates.inc (Braket LocalSimulator compatible)
// Covers the 12-gate whitelist used by the competition circuits.
// Braket's parser treats U / gphase as builtins; the rest are defined here.

gate phaseshift(lambda) a { ctrl @ gphase(lambda) a; }
gate z a { phaseshift(pi) a; }
gate x a { U(pi, 0, pi) a; }
gate h a { U(pi/2, 0, pi) a; }
gate s a { pow(1/2) @ z a; }
gate sdg a { inv @ pow(1/2) @ z a; }
gate t a { pow(1/2) @ s a; }
gate tdg a { inv @ pow(1/2) @ s a; }
gate ry(theta) a { U(theta, 0, 0) a; }
gate rz(lambda) a { gphase(-lambda/2); U(0, 0, lambda) a; }
gate cnot a, b { ctrl @ x a, b; }
gate cx a, b { ctrl @ x a, b; }
gate cp(lambda) a, b { ctrl @ phaseshift(lambda) a, b; }
gate swap a, b { cnot a, b; cnot b, a; cnot a, b; }
gate ccx a, b, c { ctrl @ ctrl @ x a, b, c; }
