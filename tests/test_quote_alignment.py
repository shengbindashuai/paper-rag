"""精确引用优先；仅唯一逐字匹配允许本地修正偏移，不调用真实 API。"""
import copy
import json
import os
import unittest
from unittest.mock import patch

import httpx

from retrieval.config import RetrievalConfig
from retrieval.gate_runner import QualityGateRunner
from retrieval.quality_gate import DeepSeekQualityGate, GateError, validate_audit
from retrieval.schemas import GatePolicy
from retrieval.service import select_context
from tests.test_quality_gate import ROOT, api_response, hit, make_audit


class QuoteAlignmentTests(unittest.TestCase):
    def audit_for(self, text, quote, start, end):
        hits = [hit('a', text)]
        audit = make_audit(hits)
        audit['assessments'][0]['support_spans'] = [{'start': start, 'end': end, 'quote': quote}]
        return hits, audit

    def test_correct_offsets_take_precedence_even_for_repeated_quotes(self):
        for start, end in [(0, 3), (4, 7)]:
            hits, audit = self.audit_for('PPO PPO', 'PPO', start, end)
            relocations = []
            self.assertEqual(validate_audit(audit, hits, quote_relocations=relocations), audit)
            self.assertEqual(relocations, [])

    def test_unique_quote_repairs_mismatch_and_invalid_integer_ranges(self):
        for start, end in [(0, 3), (4, 8), (4, 999), (-1, 0), (8, 4), (0, 0)]:
            with self.subTest(start=start, end=end):
                hits, audit = self.audit_for('abc PPO xyz', 'PPO', start, end)
                original = copy.deepcopy(audit)
                relocations = []
                result = validate_audit(audit, hits, quote_relocations=relocations)
                self.assertEqual(result['assessments'][0]['support_spans'][0], {'start': 4, 'end': 7, 'quote': 'PPO'})
                self.assertEqual(audit, original)
                self.assertEqual(relocations, [{'loc': ['assessments', 0, 'support_spans', 0],
                    'original_start': start, 'original_end': end, 'start': 4, 'end': 7}])
                second_relocations = []
                self.assertEqual(validate_audit(result, hits, quote_relocations=second_relocations), result)
                self.assertEqual(second_relocations, [])

    def test_unicode_utf8_and_utf16_offsets_relocate_to_python_characters(self):
        text, quote = '中文🙂 αβ 结束', 'αβ'
        start = text.index(quote)
        for bad_start in (len(text[:start].encode('utf-8')), len(text[:start].encode('utf-16-le')) // 2):
            hits, audit = self.audit_for(text, quote, bad_start, bad_start + len(quote))
            result = validate_audit(json.dumps(audit, ensure_ascii=False), hits)
            span = result['assessments'][0]['support_spans'][0]
            self.assertEqual((span['start'], span['end']), (start, start + len(quote)))

    def test_ambiguous_and_overlapping_matches_are_blocked(self):
        for text, quote, start, end in [('PPO PPO', 'PPO', 1, 4), ('aaa', 'aa', 0, 1),
                                       ('PPO PPO', 'PPO', 99, 100)]:
            with self.subTest(text=text):
                hits, audit = self.audit_for(text, quote, start, end)
                with self.assertRaises(GateError) as caught:
                    validate_audit(audit, hits)
                self.assertEqual(caught.exception.code, 'quote_ambiguous')
                self.assertFalse(caught.exception.retryable)

    def test_absent_quotes_do_not_use_normalization_or_fuzzy_matching(self):
        for text, quote in [('PPO clips', 'ppo clips'), ('alpha\nbeta', 'alpha beta'),
                            ('Ａ', 'A'), ('café', 'cafe\u0301'), ('alpha beta', 'alpha…beta')]:
            with self.subTest(text=text, quote=quote):
                hits, audit = self.audit_for(text, quote, 0, 1)
                with self.assertRaises(GateError) as caught:
                    validate_audit(audit, hits)
                self.assertEqual(caught.exception.code, 'quote_mismatch')

    def test_search_is_restricted_to_the_corresponding_passage(self):
        hits = [hit('a', 'alpha'), hit('b', 'unique PPO')]
        audit = make_audit(hits)
        audit['assessments'][0]['support_spans'] = [{'start': 0, 'end': 3, 'quote': 'PPO'}]
        with self.assertRaises(GateError) as caught:
            validate_audit(audit, hits)
        self.assertEqual(caught.exception.code, 'quote_mismatch')

    def test_alignment_does_not_bypass_schema_or_condition_checks(self):
        for start in ('4', 4.0, True, None):
            hits, audit = self.audit_for('abc PPO xyz', 'PPO', start, 7)
            with self.assertRaises(GateError) as caught:
                validate_audit(audit, hits)
            self.assertEqual(caught.exception.code, 'schema_invalid')
        hits, audit = self.audit_for('abc PPO xyz', 'PPO', 0, 3)
        audit['assessments'][0]['conditions'] = 'missing'
        with self.assertRaises(GateError) as caught:
            validate_audit(audit, hits)
        self.assertEqual(caught.exception.code, 'coverage_evidence_ineligible')

    def test_runner_preserves_local_relocation_through_double_validation(self):
        hits, audit = self.audit_for('prefix: 支撑机制的原文。', '支撑机制的原文。', 0, 1)
        gate = DeepSeekQualityGate(RetrievalConfig(ROOT).deepseek,
                                  httpx.MockTransport(lambda request: api_response(audit)))
        runner = QualityGateRunner(gate, GatePolicy())
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only'}):
            context, _, meta = runner.run('q', hits, lambda text: {'input_ids': list(text)}, 2048, 5, select_context)
        self.assertEqual(meta['status'], 'completed')
        self.assertTrue(meta['answer_allowed'])
        self.assertEqual(len(meta['quote_relocations']), 1)
        self.assertEqual(meta['quote_relocations'], meta['attempts'][0]['quote_relocations'])
        self.assertNotIn('支撑机制', json.dumps(meta['quote_relocations'], ensure_ascii=False))
        span = context[0]['quality_assessment']['support_spans'][0]
        self.assertEqual(hits[0]['text'][span['start']:span['end']], span['quote'])

    def test_runner_still_blocks_ambiguous_quotes_without_echoing_text(self):
        marker = 'private-exact-quote'
        hits, audit = self.audit_for(marker + ' ' + marker, marker, 1, 2)
        gate = DeepSeekQualityGate(RetrievalConfig(ROOT).deepseek,
                                  httpx.MockTransport(lambda request: api_response(audit)))
        runner = QualityGateRunner(gate, GatePolicy())
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only'}):
            context, _, meta = runner.run('q', hits, lambda text: {'input_ids': list(text)}, 2048, 5, select_context)
        self.assertEqual(context, [])
        self.assertEqual(meta['error_code'], 'quote_ambiguous')
        self.assertFalse(meta['answer_allowed'])
        self.assertEqual(len(meta['attempts']), 1)
        self.assertNotIn(marker, json.dumps(meta))


if __name__ == '__main__':
    unittest.main()
