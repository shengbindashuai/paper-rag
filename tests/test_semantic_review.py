"""外部术语核验的来源边界、幻觉拒绝、断点缓存和三表发布。"""
import json
from pathlib import Path
from unittest.mock import patch
import unittest

import httpx

from tests import test_automation as fixtures
from terminology.maintenance import synchronize
from terminology.semantic_review import run, check_review, wire_items, decode_wire, decode_partial_wire


class SemanticReviewTests(unittest.TestCase):
    setUp = fixtures.AutomationTests.setUp
    tearDown = fixtures.AutomationTests.tearDown
    document = fixtures.AutomationTests.document
    open_current = fixtures.AutomationTests.open_current

    def review_options(self):
        return {'model': 'test-model', 'base_url': 'https://test.invalid', 'api_key_env': 'TERM_TEST_KEY',
                'batch_size': 2, 'concurrency': 2, 'evidence_per_pair': 2, 'context_chars': 900,
                'max_request_bytes': 90000, 'max_tokens': 8192, 'timeout_seconds': 2,
                'attempts': 1, 'auto_publish': True}

    def transport(self, mutate=None):
        self.calls = 0
        def handle(request):
            self.calls += 1
            payload = json.loads(request.content)
            self.assertTrue(any('json' in message['content'].lower() for message in payload['messages']))
            items = json.loads(payload['messages'][-1]['content'])['p']
            reviews = []
            for item in items:
                lf = 'proximal policy optimization' if item['s'] == 'PPO' else item['l']
                review = {'pair_id': str(item['i']), 'verdict': 'approve', 'long_form': lf,
                          'scope': 'machine_learning', 'reason': 'Explicit definition in source',
                          'evidence': [{'evidence_id': '0', 'quote': item['e']['0']}]}
                if mutate:
                    mutate(review)
                verdict = {'approve': 0, 'reject': 1, 'needs_review': 2}[review['verdict']]
                code = 1 if verdict == 0 and lf != item['l'] else 0 if verdict == 0 else 3 if verdict == 1 else 5
                proof = review['evidence'][0] if review['evidence'] else None
                index = int(proof['evidence_id']) if proof and proof['evidence_id'].isdigit() else 99 if proof else None
                reviews.append({'i': item['i'], 'v': verdict, 'l': review['long_form'], 's': review['scope'],
                                'c': code, 'e': index, 'q': proof['quote'] if proof else None})
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps({'r': reviews})}}], 'usage': {'total_tokens': 42}})
        return httpx.MockTransport(handle)

    def seed(self):
        self.document('P1', 'We use proximal policy optimization (PPO).')
        self.document('P2', 'We use large language model (LLM).')
        synchronize(self.root, self.index, self.output)

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_full_pipeline_corrects_and_publishes_with_source_lineage(self):
        self.seed()
        result = run(self.root, self.output, config=self.review_options(), transport=self.transport())
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['pair_results'], {'validated_approved': 2})
        self.assertEqual(result['publication']['active_terms'], 2)
        db = self.open_current()
        try:
            self.assertEqual(db.execute('SELECT count(*) FROM candidates').fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT long_form FROM candidates WHERE short_form='PPO'").fetchone()[0], 'policy optimization')
            self.assertEqual(db.execute("SELECT verified_long_form FROM verification WHERE short_form='PPO'").fetchone()[0], 'proximal policy optimization')
        finally:
            db.close()

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_invented_correction_fails_closed(self):
        self.seed()
        result = run(self.root, self.output, config=self.review_options(),
                     transport=self.transport(lambda r: r.update(long_form='invented terminology')))
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['pair_results'], {'error': 2})
        self.assertEqual(result['publication']['active_terms'], 0)

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_unknown_pair_or_evidence_cannot_be_published(self):
        self.seed()
        result = run(self.root, self.output, config=self.review_options(),
                     transport=self.transport(lambda r: r['evidence'][0].update(evidence_id='unknown')))
        self.assertEqual(result['pair_results'], {'error': 2})
        self.assertEqual(result['publication']['active_terms'], 0)

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_cache_skips_network_and_changed_source_reaudits(self):
        self.seed()
        transport = self.transport()
        run(self.root, self.output, config=self.review_options(), transport=transport)
        calls = self.calls
        second = run(self.root, self.output, config=self.review_options(), transport=transport)
        self.assertEqual(second['pending_pairs'], 0)
        self.assertEqual(self.calls, calls)
        self.document('P2', 'Different context: large language model (LLM).')
        synchronize(self.root, self.index, self.output)
        third = run(self.root, self.output, config=self.review_options(), transport=transport)
        self.assertEqual(third['pending_pairs'], 1)
        self.assertEqual(third['status'], 'completed')

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_changed_source_during_review_prevents_publication(self):
        self.seed()
        def change(review):
            pointer = json.loads((self.output / 'current.json').read_text(encoding="utf-8"))
            pointer['source_digest'] = 'different-revision'
            (self.output / 'current.json').write_text(json.dumps(pointer))
        result = run(self.root, self.output, config=self.review_options(), transport=self.transport(change))
        self.assertEqual(result['publication']['status'], 'stale_source')
        db = self.open_current()
        try:
            self.assertEqual(db.execute('SELECT count(*) FROM active_terms').fetchone()[0], 0)
        finally:
            db.close()

    def test_quote_must_include_both_full_form_and_short_form(self):
        item = {'pair_id': 'x', 'evidence': [{'evidence_id': 'e', 'text': 'large language model (LLM)',
                'text_start': 0, 'long_end': 20, 'short_start': 22, 'short_end': 25}]}
        review = {'pair_id': 'x', 'verdict': 'approve', 'long_form': 'large language model', 'scope': 'machine_learning',
                  'evidence': [{'evidence_id': 'e', 'quote': 'large language model'}]}
        with self.assertRaises(ValueError):
            check_review(review, item)

    def test_rejection_requires_source_proof(self):
        with self.assertRaises(ValueError):
            check_review({'pair_id': 'x', 'verdict': 'reject', 'evidence': []}, {'pair_id': 'x', 'evidence': []})

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_rejection_with_real_quote_does_not_activate(self):
        self.seed()
        result = run(self.root, self.output, config=self.review_options(),
                     transport=self.transport(lambda r: r.update(verdict='reject', long_form='', scope='')))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['pair_results'], {'validated_rejected': 2})
        self.assertEqual(result['publication']['active_terms'], 0)

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_repeated_invalid_proof_stops_retrying_without_approval(self):
        self.seed()
        transport = self.transport(lambda r: r.update(long_form='invented definition'))
        for _ in range(3):
            result = run(self.root, self.output, config=self.review_options(), transport=transport)
        self.assertEqual(result['pair_results'], {'blocked': 2})
        calls = self.calls
        result = run(self.root, self.output, config=self.review_options(), transport=transport)
        self.assertEqual(self.calls, calls)
        self.assertEqual(result['publication']['active_terms'], 0)

    def test_regular_build_runs_followup_outside_index_lock(self):
        from ingestion.pipeline import build_library
        from rag_core.storage import writer_lock
        def followup(root, directory):
            with writer_lock(directory):
                return {'status': 'test_completed'}
        with patch('ingestion.pipeline._build', return_value={'final_status': {}}), patch(
                'ingestion.pipeline.post_build_verification', side_effect=followup) as hook:
            result = build_library(self.root, index_dir=self.index)
            self.assertEqual(result['terminology']['status'], 'test_completed')
            self.assertEqual(hook.call_count, 1)

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_account_error_stops_retry_and_requires_explicit_resume(self):
        self.seed()
        config = self.review_options()
        config.update(attempts=3, concurrency=1, batch_size=1)
        calls = []
        def no_balance(request):
            calls.append(request.url.path)
            return httpx.Response(402, json={'error': {'code': 'insufficient_balance'}})
        result = run(self.root, self.output, config=config, transport=httpx.MockTransport(no_balance))
        self.assertEqual(result['status'], 'awaiting_account')
        self.assertEqual(result['http_error'], 402)
        self.assertEqual(len(calls), 1)
        result = run(self.root, self.output, config=config, transport=httpx.MockTransport(no_balance))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['publication']['active_terms'], 0)
        result = run(self.root, self.output, config=config, transport=self.transport(), resume_account=True)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['publication']['active_terms'], 2)

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_cached_only_publishes_valid_results_without_network(self):
        self.seed()
        transport = self.transport()
        run(self.root, self.output, config=self.review_options(), transport=transport, limit=1, publish=False)
        calls = self.calls
        result = run(self.root, self.output, config=self.review_options(), transport=transport,
                     cached_only=True, account_error=402)
        self.assertEqual(self.calls, calls)
        self.assertEqual(result['status'], 'awaiting_account')
        self.assertEqual(result['publication']['active_terms'], 1)

    def test_wire_omits_all_local_provenance_and_decodes_ids_locally(self):
        item = {'pair_id': 'local_pair', 'short_form': 'AI', 'extracted_long_form': 'artificial intelligence',
                'other_long_forms': [], 'evidence': [{'evidence_id': 'local_occurrence', 'doc_id': 'P1',
                    'source_sha256': 'secret_hash', 'segment_sha256': 'secret_segment_hash',
                    'long_start': 100, 'short_end': 150, 'bbox': [1, 2, 3, 4],
                    'text': 'artificial intelligence (AI)'}]}
        wire = wire_items([item])
        self.assertEqual(set(wire['p'][0]), {'i', 's', 'l', 'a', 'e'})
        serialized = json.dumps(wire)
        for value in ['sha256', 'local_pair', 'local_occurrence', 'bbox', 'long_start', 'doc_id']:
            self.assertNotIn(value, serialized)
        decoded = decode_wire({'r': [[0, 0, None, 'machine_learning', 0, 0, 'artificial intelligence (AI)']]}, [item])[0]
        self.assertEqual(decoded['pair_id'], 'local_pair')
        self.assertEqual(decoded['evidence'][0]['evidence_id'], 'local_occurrence')
        self.assertEqual(decoded['long_form'], 'artificial intelligence')

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_correction_binds_only_to_reviewed_occurrence_in_same_segment(self):
        self.document('P1', 'proximal policy optimization (PPO). Another policy optimization (PPO).')
        synchronize(self.root, self.index, self.output)
        result = run(self.root, self.output, config=self.review_options(), transport=self.transport())
        self.assertEqual(result['publication']['active_terms'], 1)

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_rejected_nfkc_variants_preserve_original_source_glyphs(self):
        self.document('P1', 'coefﬁcient training (CT).')
        self.document('P2', 'coefficient training (CT).')
        synchronize(self.root, self.index, self.output)
        result = run(self.root, self.output, config=self.review_options(),
                     transport=self.transport(lambda r: r.update(verdict='reject', long_form='', scope='')))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['publication']['active_terms'], 0)

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_multiple_corrected_pairs_merge_without_losing_provenance(self):
        self.document('P1', 'Signal-to-Noise Ratio (SNR).')
        self.document('P2', 'high-Signal-to-Noise Ratio (SNR).')
        synchronize(self.root, self.index, self.output)
        def correct(review):
            review.update(long_form='Signal-to-Noise Ratio')
        result = run(self.root, self.output, config=self.review_options(), transport=self.transport(correct))
        self.assertEqual(result['publication']['active_terms'], 1)
        db = self.open_current()
        try:
            self.assertEqual(db.execute('SELECT count(*) FROM active_term_pairs').fetchone()[0], 2)
            self.assertEqual(len(json.loads(db.execute('SELECT evidence_spans FROM active_terms').fetchone()[0])), 2)
        finally:
            db.close()

    def test_bad_row_does_not_discard_valid_rows_or_accept_duplicate_ids(self):
        items = [{'pair_id': 'p0', 'extracted_long_form': 'name', 'evidence': []},
                 {'pair_id': 'p1', 'extracted_long_form': 'other', 'evidence': []}]
        good = [0, 2, None, None, 5, None, None]
        reviews, errors = decode_partial_wire({'r': [good, [1, 'bad']]}, items)
        self.assertEqual([r['pair_id'] for r in reviews], ['p0'])
        self.assertIn('p1', errors)
        reviews, errors = decode_partial_wire({'r': [good, good]}, items)
        self.assertEqual(reviews, [])
        self.assertIn('p0', errors)

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_cache_quote_corruption_is_not_republished(self):
        self.seed()
        transport = self.transport()
        run(self.root, self.output, config=self.review_options(), transport=transport)
        import sqlite3
        audit = sqlite3.connect(self.output / 'semantic_audit.sqlite3')
        try:
            pair_id, raw = audit.execute('SELECT pair_id,review FROM pair_audits ORDER BY pair_id LIMIT 1').fetchone()
            review = json.loads(raw)
            review['evidence'][0]['quote'] = 'invented'
            audit.execute('UPDATE pair_audits SET review=? WHERE pair_id=?', (json.dumps(review), pair_id))
            audit.commit()
        finally:
            audit.close()
        calls = self.calls
        result = run(self.root, self.output, config=self.review_options(), transport=transport, cached_only=True)
        self.assertEqual(self.calls, calls)
        self.assertEqual(result['publication']['active_terms'], 1)
        self.assertEqual(result['pending_pairs'], 1)

    def test_named_short_output_allows_omitted_unused_fields(self):
        item = {'pair_id': 'p', 'extracted_long_form': 'artificial intelligence',
                'evidence': [{'evidence_id': 'e', 'text': 'artificial intelligence (AI)'}]}
        parsed = decode_wire({'r': [{'i': 0, 'v': 0, 's': 'machine_learning', 'e': '0',
                                     'q': 'artificial intelligence (AI)'}]}, [item])[0]
        self.assertEqual(parsed['long_form'], 'artificial intelligence')
        self.assertEqual(parsed['evidence'][0]['evidence_id'], 'e')
        parsed = decode_wire({'r': [{'i': 0, 'v': 2}]}, [item])[0]
        self.assertEqual(parsed['verdict'], 'needs_review')

    @patch.dict('os.environ', {'TERM_TEST_KEY': 'test-secret'})
    def test_publication_reuses_checked_segment_snapshot(self):
        from terminology.semantic_review import open_audit, publish_results
        self.seed()
        run(self.root, self.output, config=self.review_options(), transport=self.transport(), publish=False)
        pointer = json.loads((self.output / 'current.json').read_text(encoding="utf-8"))
        audit = open_audit(self.output / 'semantic_audit.sqlite3')
        try:
            versions = dict(audit.execute('SELECT pair_id,input_digest FROM pair_audits'))
            with patch('terminology.pilot.gzip.open', side_effect=AssertionError('source already checked')):
                result = publish_results(self.root, self.output, pointer, audit, versions)
            self.assertEqual(result['active_terms'], 2)
        finally:
            audit.close()


if __name__ == '__main__':
    unittest.main()
