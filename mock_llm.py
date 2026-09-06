"""mock_llm.py — 拟真测试用：模拟 llama-server(8080) 返回固定 LLM 回复
用法: python3 mock_llm.py   （然后 18771 桥正常转发到它）
改 REPLY 即可切换模拟的 LLM 回复内容（覆盖 我说我/我说你/你说我 各场景）
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 模拟 LLM 回复（可切换）
REPLY = "她问我了。我好害羞。[TOOL:回复] 我有点害羞啦[TOOL]"


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        data = json.dumps({"content": REPLY}, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)
    log_message = lambda *a: None


if __name__ == "__main__":
    print(f"[mock-llm] 8080 返回固定回复: {REPLY}")
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer(("127.0.0.1", 8080), H).serve_forever()
