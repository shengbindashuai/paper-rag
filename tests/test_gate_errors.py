"""分类错误码及诊断脱敏；使用 MockTransport，不发送真实密钥或论文。"""
import copy
import json
import os
import traceback
import unittest
from unittest.mock import patch

import httpx

from retrieval.config import RetrievalConfig
from retrieval.gate_runner import QualityGateRunner
from retrieval.quality_gate import DeepSeekQualityGate, GateError, validate_audit
from retrieval.schemas import GatePolicy
from retrieval.service import select_context
from tests.test_quality_gate import ROOT, api_response, hit, make_audit


SECRET = 'sk-PRIVATE-quote-and-key-0123456789'


class GateErrorTests(unittest.TestCase):
    def run_response(self, response, hits=None, gate=None, policy=None):
        hits = hits if hits is not None else [hit('a')]
        gate = gate or DeepSeekQualityGate(RetrievalConfig(ROOT).deepseek,
                                          httpx.MockTransport(lambda request: response))
        runner = QualityGateRunner(gate, policy or GatePolicy(retries_per_round=0))
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': SECRET}):
            return runner.run('q', hits, lambda text: {'input_ids': list(text)}, 2048, 5, select_context)

    def assert_failure(self, meta, code):
        self.assertEqual(meta['status'], 'failed')
        self.assertEqual(meta['error_code'], code)
        self.assertFalse(meta['answer_allowed'])
        self.assertFalse(meta['validated'])
        self.assertNotIn(SECRET, json.dumps(meta, ensure_ascii=False))

    def test_json_schema_and_response_envelope_are_distinct(self):
        cases = [
            (httpx.Response(200, content=b'{broken'), 'response_json_invalid'),
            (httpx.Response(200, content=b'{"choices":[],"choices":[]}'), 'response_json_duplicate_key'),
            (httpx.Response(200, json=[]), 'response_envelope_invalid'),
            (httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': 7}}]}), 'response_envelope_invalid')]
        for content, code in [('{broken ' + SECRET, 'json_invalid'),
                              ('{"assessments":[],"assessments":[]}', 'json_duplicate_key'),
                              ('{"assessments":NaN}', 'json_invalid'),
                              ('{}', 'schema_invalid')]:
            cases.append((httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                          'message': {'content': content}}]}), code))
        for response, code in cases:
            with self.subTest(code=code):
                context, _, meta = self.run_response(response)
                self.assertEqual(context, [])
                self.assert_failure(meta, code)
                self.assertEqual(len(meta['attempts']), 1)
                self.assertEqual(meta['attempts'][0]['error_code'], code)

    def test_schema_details_only_contain_locations_and_types(self):
        hits = [hit('a', SECRET)]
        audit = make_audit(hits)
        audit['assessments'][0]['relation'] = SECRET
        audit['assessments'][0]['support_spans'][0]['start'] = SECRET
        audit[SECRET] = {'Authorization': 'Bearer ' + SECRET}
        with self.assertRaises(GateError) as caught:
            validate_audit(audit, hits)
        error = caught.exception
        self.assertEqual(error.code, 'schema_invalid')
        self.assertTrue(error.__suppress_context__)
        self.assertNotIn(SECRET, str(error))
        self.assertNotIn(SECRET, ''.join(traceback.format_exception(error)))
        for detail in error.details:
            self.assertEqual(set(detail), {'loc', 'type'})
        self.assertIn({'loc': ['assessments', 0, 'relation'], 'type': 'literal_error'}, error.details)
        self.assertIn({'loc': ['assessments', 0, 'support_spans', 0, 'start'], 'type': 'int_type'}, error.details)
        self.assertIn({'loc': ['<extra>'], 'type': 'extra_forbidden'}, error.details)
        context, _, meta = self.run_response(api_response(audit), hits)
        self.assertEqual(context, [])
        self.assert_failure(meta, 'schema_invalid')
        self.assertEqual(meta['error_details'], error.details)
        self.assertEqual(meta['attempts'][0]['error_details'], error.details)

    def test_schema_diagnostics_are_bounded(self):
        audit = make_audit([hit('a')])
        for i in range(50): audit[SECRET + str(i)] = SECRET
        with self.assertRaises(GateError) as caught:
            validate_audit(audit, [hit('a')])
        self.assertEqual(len(caught.exception.details), 20)
        self.assertNotIn(SECRET, str(caught.exception))

    def test_id_quote_coverage_and_conflict_errors_have_specific_codes(self):
        hits = [hit('a'), hit('b')]
        good = make_audit(hits)
        cases = []
        def variant(code, mutate):
            audit = copy.deepcopy(good)
            mutate(audit)
            cases.append((audit, code))
        variant('evidence_id_unknown', lambda a: a['assessments'][0].update(evidence_id=SECRET))
        variant('evidence_id_duplicate', lambda a: a['assessments'].append(copy.deepcopy(a['assessments'][0])))
        variant('evidence_id_missing', lambda a: a['assessments'].pop())
        variant('subquestion_id_duplicate', lambda a: a['subquestions'].append(copy.deepcopy(a['subquestions'][0])))
        variant('coverage_subquestion_duplicate', lambda a: a['coverage'].append(copy.deepcopy(a['coverage'][0])))
        variant('coverage_subquestion_unknown', lambda a: a['coverage'][0].update(subquestion_id=SECRET))
        variant('coverage_subquestion_missing', lambda a: a.update(coverage=[]))
        variant('quote_mismatch', lambda a: a['assessments'][0]['support_spans'][0].update(quote=SECRET))
        variant('quote_out_of_bounds', lambda a: a['assessments'][0]['support_spans'][0].update(end=999, quote=SECRET))
        variant('quote_missing', lambda a: a['assessments'][0].update(support_spans=[]))
        variant('coverage_missing_inconsistent', lambda a: a['coverage'][0].update(missing=[SECRET]))
        variant('coverage_group_empty', lambda a: a['coverage'][0].update(support_sets=[[]]))
        variant('coverage_group_duplicate', lambda a: a['coverage'][0].update(support_sets=[['D:a', 'D:a']]))
        variant('coverage_evidence_duplicate', lambda a: a['coverage'][0].update(partial_evidence_ids=['D:a', 'D:a']))
        variant('coverage_evidence_unknown', lambda a: a['coverage'][0].update(support_sets=[[SECRET]]))
        variant('coverage_evidence_ineligible', lambda a: a['assessments'][0].update(conditions='missing'))
        variant('conflict_unreported', lambda a: a['assessments'][1].update(relation='contradicts'))
        for audit, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(GateError) as caught:
                    validate_audit(audit, hits)
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn(SECRET, str(caught.exception))
                _, _, meta = self.run_response(api_response(audit), hits)
                self.assert_failure(meta, code)
        with self.assertRaises(GateError) as caught:
            validate_audit(good, hits, [{'subquestion_id': 'q1', 'question': SECRET}])
        self.assertEqual(caught.exception.code, 'subquestions_changed')

    def test_conflict_reference_errors_are_distinct(self):
        hits = [hit('a'), hit('b')]
        base = make_audit(hits)
        base['assessments'][1]['relation'] = 'contradicts'
        base['conflicts'] = [{'subquestion_id': 'q1', 'evidence_ids': ['D:a', 'D:b'], 'explanation': 'conflict'}]
        cases = [('conflict_subquestion_unknown', lambda a: a['conflicts'][0].update(subquestion_id=SECRET)),
                 ('conflict_evidence_unknown', lambda a: a['conflicts'][0].update(evidence_ids=['D:a', SECRET])),
                 ('conflict_evidence_duplicate', lambda a: a['conflicts'][0].update(evidence_ids=['D:b', 'D:b'])),
                 ('conflict_evidence_ineligible', lambda a: a['assessments'][1].update(relation='background')),
                 ('conflict_contrary_missing', lambda a: a['assessments'][1].update(relation='partial_support'))]
        for code, mutate in cases:
            with self.subTest(code=code):
                audit = copy.deepcopy(base); mutate(audit)
                _, _, meta = self.run_response(api_response(audit), hits)
                self.assert_failure(meta, code)

    def test_later_schema_failure_clears_prior_review_text(self):
        hits = [hit('a', SECRET), hit('b')]
        good = make_audit(hits[:1], [], ['D:a'])
        bad = make_audit(hits); bad['assessments'][0]['relation'] = SECRET
        responses = iter([api_response(good), api_response(bad)])
        gate = DeepSeekQualityGate(RetrievalConfig(ROOT).deepseek,
                                  httpx.MockTransport(lambda request: next(responses)))
        context, _, meta = self.run_response(None, hits, gate, GatePolicy(batch_size=1, retries_per_round=0))
        self.assertEqual(context, [])
        self.assert_failure(meta, 'schema_invalid')
        self.assertEqual(meta['assessments'], [])
        self.assertEqual(meta['reviewed_count'], 1)

    def test_runner_revalidates_custom_adapter_without_leaking_pydantic_input(self):
        gate = DeepSeekQualityGate(RetrievalConfig(ROOT).deepseek)
        with patch.object(gate, 'assess', return_value={'assessments': SECRET}):
            context, _, meta = self.run_response(None, gate=gate)
        self.assertEqual(context, [])
        self.assert_failure(meta, 'schema_invalid')
        self.assertTrue(meta['error_details'])


if __name__ == '__main__':
    unittest.main()
