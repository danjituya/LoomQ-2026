"""L2 agent_chat mock test - validates the code path with a local mock API."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "starter_kit")

FENCE = chr(96) * 3  # ```

calls = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        calls.append(payload)
        content = (
            FENCE + "qasm\n"
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[3];\n"
            "creg c[3];\n"
            "h q[0];\n"
            "cx q[0], q[1];\n"
            "cx q[1], q[2];\n"
            "measure q -> c;\n" + FENCE
        )
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

os.environ["LOOMQ_LLM_BASE_URL"] = f"http://127.0.0.1:{server.server_port}"
os.environ["LOOMQ_LLM_API_KEY"] = "test-key"
os.environ["LOOMQ_LLM_MODEL"] = "deepseek-v4-flash"
os.environ["LOOMQ_LLM_TIMEOUT_SECONDS"] = "10"

import adapter

reply = adapter.agent_chat("生成一个 3 比特 GHZ 态并进行全测量")
qasm = adapter._extract_qasm_block(reply)
print("reply starts:", reply[:60].replace("\n", " "))
print("extracted qasm valid:", bool(qasm))
print("API calls made:", len(calls))
print("payload model:", calls[0]["model"], "| temperature:", calls[0]["temperature"])
print("thinking disabled:", calls[0].get("thinking"))
ok = bool(qasm) and len(calls) >= 1
print("L2 MOCK TEST:", "PASS" if ok else "FAIL")
server.shutdown()
server.server_close()
sys.exit(0 if ok else 1)
