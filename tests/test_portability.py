"""First-run and relocated-workspace regressions; no models or remote APIs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ingestion.inventory import register_pdf
from knowledge.jobs import IngestJobs
from rag_core.config import CODE_ROOT, config_file, default_root
from retrieval.config import RetrievalConfig


class PortabilityTests(unittest.TestCase):
    def test_external_workspace_uses_bundled_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = RetrievalConfig.load(config_file(root, "retrieval.json"), root)
            self.assertEqual(config.path("gate_policy_file"), CODE_ROOT / "config/quality_gate_policy.json")
            self.assertEqual(config.path("embedding_dir"), root / "RAG/models/Qwen3-Embedding-0.6B")
            json.loads(config.path("gate_policy_file").read_text(encoding="utf-8"))

    def test_workspace_configuration_override_and_legacy_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            override = root / "config/quality_gate_policy.json"
            override.parent.mkdir()
            override.write_text('{"custom": true}', encoding="utf-8")
            self.assertEqual(RetrievalConfig(root).path("gate_policy_file"), override)
            config = RetrievalConfig(root, gate_policy_file="RAG_code/config/quality_gate_policy.json")
            self.assertEqual(config.path("gate_policy_file"), override)

    def test_workspace_environment_variable(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"RAG_WORKSPACE": folder}):
            self.assertEqual(default_root(), Path(folder).resolve())

    def test_first_pdf_registration_creates_archive_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "first.pdf"
            source.write_bytes(b"synthetic PDF registration input")
            doc_id = register_pdf(root, source)
            metadata = json.loads((root / "RAG/metadata" / (doc_id + ".json")).read_text(encoding="utf-8"))
            self.assertEqual((root / "RAG" / metadata["file"]).read_bytes(), source.read_bytes())
            self.assertEqual(register_pdf(root, source), doc_id)

    def test_ingest_subprocess_uses_code_location_instead_of_workspace_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = object.__new__(IngestJobs)
            jobs.root, jobs.index_dir, jobs.directory = root, root / "index", root
            (root / "sample.input.json").write_text("{}", encoding="utf-8")
            with patch("knowledge.jobs.subprocess.run") as run:
                jobs._run_process("sample")
            self.assertEqual(run.call_args.kwargs["cwd"], CODE_ROOT)
            self.assertIn(str(root), run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
