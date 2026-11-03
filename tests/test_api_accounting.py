"""请求字段最小化、精确缓存边界、持久化复用与用量记账。"""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import httpx

from rag_core.api_accounting import APILedger, quantify
from retrieval.quality_gate import DeepSeekQualityGate, GateError
from retrieval.config import RetrievalConfig
from tests.test_quality_gate import make_audit, api_response, hit


class APICacheTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / 'tmp/api_cache_tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'usage.sqlite3'
        self.options = RetrievalConfig(Path(__file__).resolve().parents[1]).deepseek
        self.calls = 0
        def handle(request):
            self.calls += 1
            body = json.loads(json.loads(request.content)['messages'][-1]['content'])
            return api_response(make_audit(body['evidence'], questions=body['fixed_subquestions']))
        self.transport = httpx.MockTransport(handle)

    @patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-only'})
    def test_exact_result_cache_reuses_persisted_audit_at_zero_new_usage(self):
        gate = DeepSeekQualityGate(self.options, self.transport, cache_path=self.path)
        first = gate.assess('question', [hit('a')])
        again = DeepSeekQualityGate(self.options, self.transport, cache_path=self.path).assess('question', [hit('a')])
        self.assertFalse(first['cache_hit'])
        self.assertTrue(again['cache_hit'])
        self.assertEqual(again['usage'], {})
        self.assertEqual(self.calls, 1)
        db = sqlite3.connect(self.path)
        try:
            self.assertEqual(db.execute('SELECT sum(cache_hit),count(*) FROM calls').fetchone(), (1, 2))
        finally:
            db.close()

    @patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-only'})
    def test_query_text_year_or_rubric_changes_miss_but_coordinates_do_not(self):
        gate = DeepSeekQualityGate(self.options, self.transport, cache_path=self.path)
        evidence = {**hit('a'), 'source_spans': [{'bbox': [1, 2, 3, 4]}], 'source_sha256': 'private'}
        gate.assess('q', [evidence])
        gate.assess('q', [{**evidence, 'source_spans': [], 'source_sha256': 'changed'}])
        self.assertEqual(self.calls, 1)
        gate.assess('different', [evidence])
        gate.assess('q', [{**evidence, 'text': 'new evidence'}])
        gate.assess('q', [{**evidence, 'year': 2026}])
        DeepSeekQualityGate(self.options, self.transport, rubric='different policy', cache_path=self.path).assess('q', [evidence])
        self.assertEqual(self.calls, 5)
        wire = gate.request_bytes('q', [evidence])
        for removed in [b'source_spans', b'bbox', b'source_sha256', b'source_file', b'snapshot_revision']:
            self.assertNotIn(removed, wire)

    @patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-only'})
    def test_invalid_output_is_not_cached_but_actual_tokens_are_counted(self):
        bad = make_audit([hit('a')])
        bad['assessments'][0]['support_spans'][0]['quote'] = 'invented'
        gate = DeepSeekQualityGate(self.options, httpx.MockTransport(lambda _: api_response(bad)), cache_path=self.path)
        for _ in range(2):
            with self.assertRaises(GateError):
                gate.assess('q', [hit('a')])
        db = sqlite3.connect(self.path)
        try:
            self.assertEqual(db.execute('SELECT count(*) FROM responses').fetchone()[0], 0)
            self.assertEqual(sum(json.loads(r[0])['total_tokens'] for r in db.execute('SELECT usage FROM calls')), 300)
        finally:
            db.close()

    def test_preflight_estimate_labels_tokenizer_and_output_ceiling(self):
        metrics = quantify({'messages': [{'content': 'hello'}], 'max_tokens': 123})
        self.assertEqual(metrics['max_output_tokens'], 123)
        self.assertIn('not exact', metrics['estimate_method'])

    def test_non_token_model_metadata_reuses_local_cache(self):
        from retrieval.prepare_reranker import cached_metadata
        calls = []
        def get(url, **kwargs):
            calls.append(url)
            return SimpleNamespace(status_code=200, headers={'ETag': 'revision'}, content=b'metadata',
                raise_for_status=lambda: None, json=lambda: {'sha': 'revision', 'siblings': []})
        session = SimpleNamespace(get=get)
        ledger = APILedger(self.path)
        path = self.path.parent / 'metadata.json'
        for _ in range(2):
            self.assertEqual(cached_metadata(session, 'https://test.invalid/meta', path, ledger)['sha'], 'revision')
        self.assertEqual(len(calls), 1)

    def test_supplementary_gate_round_preserves_existing_evidence_prefix(self):
        from os.path import commonprefix
        gate = DeepSeekQualityGate(self.options, self.transport)
        first = gate.build_request('q', [hit('a')])['messages'][-1]['content']
        second = gate.build_request('q', [hit('a'), hit('b')],
                                    [{'subquestion_id': 'q1', 'question': 'question'}])['messages'][-1]['content']
        prefix = commonprefix([first, second])
        self.assertIn('Evidence a', prefix)
        self.assertNotIn('fixed_subquestions', prefix)


if __name__ == '__main__':
    unittest.main()
