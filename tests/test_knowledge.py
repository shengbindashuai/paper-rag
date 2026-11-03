"""读写边界、来源、任务生命周期与真实 MCP 传输；不调用付费 API。"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from knowledge.jobs import IngestJobs
from knowledge.schemas import (ReadRequest, DocumentIngestRequest, ConversationRequest, PaperIngestRequest)
from knowledge.sources import register_text, source_metadata, extract_document
from rag_core.io import atomic_json
from rag_core.storage import LibraryStore
from retrieval.config import RetrievalConfig
from retrieval.index import SnapshotIndex
from retrieval.tool_api import definitions, RetrievalToolAPI


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "tmp" / "knowledge_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def service(self):
        return SimpleNamespace(config=RetrievalConfig(self.root), lock=threading.RLock(),
                               reranker=SimpleNamespace(close=lambda: None))

    def test_source_identity_retries_and_conversation_roles(self):
        from ingestion.inventory import scan_library
        payload = {"source_type": "conversation", "title": "讨论", "conversation_id": "chat-1",
                   "messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "假设"}],
                   "result_path": "job-a"}
        first = register_text(self.root, payload)
        second = register_text(self.root, {**payload, "result_path": "job-b"})
        self.assertEqual(first, second)
        items, _ = scan_library(self.root)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "讨论")
        meta = source_metadata(self.root, first, items[0]["sha256"])
        self.assertEqual(meta["source_type"], "conversation")
        self.assertEqual([m["role"] for m in meta["messages"]], ["user", "assistant"])
        self.assertEqual(source_metadata(self.root, first, "wrong-sha"), {})

    def test_docx_paragraph_and_table_order(self):
        from docx import Document
        document = Document()
        document.add_paragraph("before")
        document.add_table(rows=1, cols=1).cell(0, 0).text = "inside"
        document.add_paragraph("after")
        path = self.root / "example.docx"
        document.save(path)
        text = extract_document(path)
        self.assertLess(text.index("before"), text.index("inside"))
        self.assertLess(text.index("inside"), text.index("after"))

    def test_strict_schema_rejects_ambiguous_sources_and_bad_ranges(self):
        for args in [{"title": "x"}, {"title": "x", "path": "a", "content": "b"},
                     {"title": "x", "content": " "}]:
            with self.assertRaises(ValidationError):
                DocumentIngestRequest(**args)
        with self.assertRaises(ValidationError):
            ReadRequest(doc_id="D", page_start=5, page_end=1)
        with self.assertRaises(ValidationError):
            ConversationRequest(title="x", messages=[{"role": "unknown", "content": "x"}])
        with self.assertRaises(ValueError):
            RetrievalToolAPI(self.service()).call("paper_search", [])
        self.assertEqual(len(definitions()), 8)

    def test_read_pagination_filters_alias_and_live_snapshot(self):
        directory = self.root / "RAG/library"
        store = LibraryStore(directory, {"test": 1})
        item = {"doc_id": "D", "source_file": "PDF/a.pdf", "sha256": "sha", "title": "paper"}
        chunks = [{"doc_id": "D", "chunk_id": f"D_{n}", "text": f"paragraph {n}", "pages": [n],
                   "section": "Methods" if n < 3 else "Results", "retrieval_eligible": True} for n in range(1, 4)]
        vectors = np.zeros((3, 1024), dtype=np.float32)
        vectors[:, 0] = 1
        store.commit_document(item, chunks, vectors)
        store.refresh_snapshot()
        store.record({**item, "doc_id": "alias"}, "duplicate", alias_of="D")
        index = SnapshotIndex(directory, root=self.root)
        try:
            result = index.read_document(ReadRequest(doc_id="alias", limit=2))
            self.assertEqual([c["text"] for c in result["chunks"]], ["paragraph 1", "paragraph 2"])
            self.assertEqual(result["next_offset"], 2)
            result = index.read_document(ReadRequest(doc_id="D", offset=2))
            self.assertFalse(result["has_more"])
            self.assertEqual(result["chunks"][0]["pages"], [3])
            filtered = index.read_document(ReadRequest(doc_id="D", section="methods", page_start=2))
            self.assertEqual(len(filtered["chunks"]), 1)
            old = index.status()["snapshot"]["revision"]
            chunks[0]["text"] = "updated"
            store.commit_document(item, chunks, vectors)
            store.refresh_snapshot()
            self.assertGreater(index.search(vectors[0], 1)[0]["snapshot_revision"], old)
            self.assertEqual(index.read_document(ReadRequest(doc_id="D"))["chunks"][0]["text"], "updated")
        finally:
            index.close()
            store.close()

    def test_queue_is_async_deduplicates_and_persists_completion(self):
        entered, release = threading.Event(), threading.Event()
        service = self.service()
        jobs = None
        def run(identifier):
            entered.set()
            release.wait(5)
            atomic_json(jobs.directory / f"{identifier}.result.json", {"doc_id": "N1", "indexed": True})
        jobs = IngestJobs(service, runner=run)
        try:
            request = DocumentIngestRequest(title="note", content="正文")
            job = jobs.submit("document", request)
            self.assertEqual(job["status"], "queued")
            self.assertTrue(entered.wait(5))
            self.assertEqual(jobs.status(job["job_id"])["status"], "running")
            self.assertEqual(jobs.submit("document", request)["job_id"], job["job_id"])
            release.set()
        finally:
            release.set()
            jobs.close()
        reopened = IngestJobs(service)
        try:
            result = reopened.status(job["job_id"])
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["result"]["indexed"])
        finally:
            reopened.close()

    def test_failure_not_reported_as_success_and_restart_interruption(self):
        def fail(identifier):
            raise RuntimeError("secret-value")
        jobs = IngestJobs(self.service(), runner=fail)
        job = jobs.submit("document", DocumentIngestRequest(title="x", content="body"))
        jobs.close()
        result = jobs.status(job["job_id"])
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("secret-value", json.dumps(result))
        path = jobs.directory / ("a" * 32 + ".job.json")
        atomic_json(path, {"job_id": "a" * 32, "identity": "test", "status": "running"})
        reopened = IngestJobs(self.service())
        try:
            self.assertEqual(reopened.status("a" * 32)["status"], "interrupted")
            with self.assertRaises(ValueError):
                reopened.status("../escape")
        finally:
            reopened.close()

    def test_path_allowlist(self):
        jobs = IngestJobs(self.service())
        try:
            with self.assertRaises(ValueError):
                jobs.validate_path(str(Path(__file__).resolve()), {".py"})
            path = self.root / "input.txt"
            path.write_text("text", encoding="utf-8")
            self.assertEqual(jobs.validate_path(str(path), {".txt"}), path)
            with self.assertRaises(ValueError):
                jobs.submit("paper", PaperIngestRequest(path=str(path)))
        finally:
            jobs.close()

    def test_worker_checks_actual_index_status_not_process_exit_only(self):
        from knowledge.worker import execute
        directory = self.root / "RAG/library"
        payload = {"source_type": "document", "title": "x", "content": "body",
                   "result_path": str(self.root / "result.json")}
        def rejected(root, **options):
            store = LibraryStore(directory)
            from ingestion.inventory import scan_library
            item = scan_library(root)[0][0]
            store.record(item, "quarantined", "empty")
            store.close()
            return {"final_status": {"revision": 0}}
        with patch("ingestion.pipeline.build_library", side_effect=rejected):
            with self.assertRaises(RuntimeError):
                execute(self.root, directory, payload)
        self.assertFalse(json.loads(Path(payload["result_path"]).read_text(encoding="utf-8"))["indexed"])

    def test_worker_published_document_is_readable_with_source_type(self):
        from knowledge.worker import execute
        from ingestion.inventory import scan_library
        directory = self.root / "RAG/library"
        payload = {"source_type": "document", "title": "实验记录", "content": "待验证的观察",
                   "source_url": "https://example.org/notes", "result_path": str(self.root / "result.json")}
        def build(root, **options):
            # 只替换模型编码；注册、任务确认、SQLite、快照和读取均使用真实实现。
            item = scan_library(root)[0][0]
            chunk = {"doc_id": item["doc_id"], "chunk_id": "C1", "text": "待验证的观察",
                     "source_sha256": item["sha256"], "source_file": item["source_file"],
                     "retrieval_eligible": True, "source_type": "reading_excerpt"}
            vector = np.zeros((1, 1024), dtype=np.float32)
            vector[0, 0] = 1
            store = LibraryStore(directory)
            try:
                store.commit_document(item, [chunk], vector)
                store.refresh_snapshot()
                return {"final_status": store.status()}
            finally:
                store.close()
        with patch("ingestion.pipeline.build_library", side_effect=build):
            result = execute(self.root, directory, payload)
        self.assertTrue(result["indexed"])
        index = SnapshotIndex(directory, root=self.root)
        try:
            chunk = index.read_document(ReadRequest(doc_id=result["doc_id"]))["chunks"][0]
            self.assertEqual(chunk["source_type"], "document")
            self.assertEqual(chunk["source_metadata"]["source_url"], payload["source_url"])
        finally:
            index.close()


@unittest.skipUnless(importlib.util.find_spec("mcp"), "安装 requirements-mcp.txt 后运行 MCP 传输测试")
class MCPTests(unittest.TestCase):
    def test_http_handshake_discovery_call_and_invalid_input(self):
        from starlette.testclient import TestClient
        from knowledge.mcp_server import build_server, build_http_app
        class API:
            def call(self, name, arguments):
                if name == "paper_read":
                    return {"doc_id": ReadRequest.model_validate(arguments).doc_id}
                return {"ok": True}
        with TestClient(build_http_app(build_server(API()), 8765), base_url="http://127.0.0.1:8765") as client:
            def rpc(method, params, identifier=1):
                return client.post("/mcp", json={"jsonrpc": "2.0", "id": identifier, "method": method, "params": params},
                                   headers={"Accept": "application/json, text/event-stream"})
            response = rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "1"}})
            self.assertEqual(response.status_code, 200)
            self.assertIn("instructions", response.json()["result"])
            tools = rpc("tools/list", {}).json()["result"]["tools"]
            self.assertEqual(len(tools), 8)
            self.assertTrue(next(t for t in tools if t["name"] == "paper_read")["annotations"]["readOnlyHint"])
            result = rpc("tools/call", {"name": "paper_read", "arguments": {"doc_id": "D"}}).json()["result"]
            self.assertEqual(result["structuredContent"]["doc_id"], "D")
            bad = rpc("tools/call", {"name": "paper_read", "arguments": {"doc_id": "D", "limit": 0}}).json()["result"]
            self.assertTrue(bad["isError"])
            blocked = client.post("/mcp", json={}, headers={"host": "evil.example", "Accept": "application/json, text/event-stream"})
            self.assertEqual(blocked.status_code, 421)

    def test_stdio_real_client_handshake_and_call(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        async def exercise():
            parameters = StdioServerParameters(command=sys.executable, args=["-m", "tests.mcp_fixture"],
                cwd=str(Path(__file__).resolve().parents[1]), env={**os.environ, "PYTHONUTF8": "1"})
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.assertEqual(len((await session.list_tools()).tools), 8)
                    result = await session.call_tool("paper_retrieval_status", {})
                    self.assertTrue(result.structuredContent["ok"])
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
