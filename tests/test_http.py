from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from companion.llm import LLMConfig, OllamaLLM, OpenAILLM


class Handler(BaseHTTPRequestHandler):
    loaded = False

    def log_message(self, *args):  # noqa: ANN002
        pass

    def _json(self, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/tags":
            self._json({"models": [{"name": "qwen3:4b"}]})
        elif self.path == "/api/ps":
            models = []
            if Handler.loaded:
                models.append(
                    {
                        "name": "qwen3:4b",
                        "size": 3_600_000_000,
                        "size_vram": 0,
                        "context_length": 8192,
                        "expires_at": "2319-02-07T01:38:01+07:00",
                    }
                )
            self._json({"models": models})
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "qwen3:4b"}]})
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        payload = json.loads(body) if body else {}

        if self.path == "/api/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for chunk in ("при", "вет"):
                line = json.dumps(
                    {"message": {"content": chunk}, "done": False},
                    ensure_ascii=False,
                )
                self.wfile.write((line + "\n").encode())
                self.wfile.flush()
            final = json.dumps(
                {
                    "message": {"content": ""},
                    "done": True,
                    "eval_count": 2,
                    "eval_duration": 500_000_000,
                    "prompt_eval_count": 10,
                    "prompt_eval_duration": 1_000_000_000,
                }
            )
            self.wfile.write((final + "\n").encode())
        elif self.path == "/api/generate":
            Handler.loaded = payload.get("keep_alive") != 0
            self._json(
                {
                    "model": payload.get("model", "qwen3:4b"),
                    "done": True,
                }
            )
        elif self.path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in ("при", "вет"):
                event = json.dumps(
                    {"choices": [{"delta": {"content": chunk}}]},
                    ensure_ascii=False,
                )
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.send_error(404)


class HTTPBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        Handler.loaded = False

    def test_ollama_health_and_stream(self) -> None:
        cfg = LLMConfig(
            backend="ollama",
            base_url=f"http://127.0.0.1:{self.port}",
            model="qwen3:4b",
        )
        llm = OllamaLLM(cfg)
        self.assertIn("Ollama", llm.health())
        self.assertEqual(llm.list_models(), ["qwen3:4b"])
        self.assertIsNone(llm.warn_if_model_missing())
        text = llm.complete([{"role": "user", "content": "привет"}])
        self.assertEqual(text, "привет")
        stats = llm.last_stats
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.tokens, 2)
        self.assertAlmostEqual(stats.tokens_per_second, 4.0, places=2)
        self.assertEqual(stats.prompt_tokens, 10)
        self.assertFalse(stats.estimated)

    def test_ollama_load_and_unload(self) -> None:
        cfg = LLMConfig(
            backend="ollama",
            base_url=f"http://127.0.0.1:{self.port}",
            model="qwen3:4b",
        )
        llm = OllamaLLM(cfg)
        self.assertEqual(llm.loaded_models(), [])

        message = llm.load_model()
        self.assertIn("загружена", message)
        loaded = llm.loaded_models()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["name"], "qwen3:4b")
        self.assertEqual(loaded[0]["context_length"], 8192)

        message = llm.unload_model()
        self.assertIn("выгружена", message)
        self.assertEqual(llm.loaded_models(), [])

    def test_openai_health_and_stream(self) -> None:
        cfg = LLMConfig(
            backend="openai",
            base_url=f"http://127.0.0.1:{self.port}/v1",
            model="qwen3:4b",
        )
        llm = OpenAILLM(cfg)
        self.assertIn("OpenAI", llm.health())
        self.assertEqual(llm.list_models(), ["qwen3:4b"])
        text = llm.complete([{"role": "user", "content": "привет"}])
        self.assertEqual(text, "привет")
        stats = llm.last_stats
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertTrue(stats.estimated)
        self.assertGreater(stats.tokens_per_second, 0)


if __name__ == "__main__":
    unittest.main()
