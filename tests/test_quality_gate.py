"""门控集成测试：MockTransport + 假检索设备，不联网、不读取真实论文。"""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from retrieval.config import RetrievalConfig
from retrieval.gate_runner import QualityGateRunner, evaluate_context
from retrieval.quality_gate import DeepSeekQualityGate, GateError, validate_audit
from retrieval.schemas import GatePolicy
from retrieval.service import RetrievalService, select_context


ROOT = Path(__file__).resolve().parents[1]


def hit(identifier, text=None):
    return {'doc_id': 'D', 'chunk_id': identifier, 'evidence_id': 'D:' + identifier,
            'text': text or 'Evidence ' + identifier, 'dense_score': 0.9,
            'source_type': 'reading_excerpt', 'source_file': 'notes.md', 'pages': []}


def make_audit(hits, groups=None, partial=None, questions=None):
    questions = questions or [{'subquestion_id': 'q1', 'question': '机制是什么？'}]
    assessments = []
    for h in hits:
        assessments.append({'evidence_id': h['evidence_id'], 'relation': 'partial_support',
            'applicability': 'supported', 'time_scope': 'not_applicable', 'conditions': 'supported',
            'support_spans': [{'start': 0, 'end': len(h['text']), 'quote': h['text']}],
            'reason': 'Supplied text supports the mechanism.'})
    groups = groups if groups is not None else [[hits[0]['evidence_id']]]
    return {'subquestions': questions, 'assessments': assessments,
            'coverage': [{'subquestion_id': q['subquestion_id'], 'support_sets': groups if i == 0 else [],
                          'partial_evidence_ids': partial or [],
                          'missing': [] if i == 0 and groups else ['缺少充分证据']}
                         for i, q in enumerate(questions)], 'conflicts': []}


def api_response(audit):
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
        'message': {'content': json.dumps(audit, ensure_ascii=False)}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}})


class GateContractTests(unittest.TestCase):
    def setUp(self):
        self.hits = [hit('a', '机制依赖条件 α。'), hit('b')]
        self.audit = make_audit(self.hits, [['D:a', 'D:b']])

    def test_unicode_spans_and_joint_partial_evidence_are_valid(self):
        self.assertEqual(validate_audit(self.audit, self.hits), self.audit)

    def test_forged_quotes_unknown_ids_and_inconsistent_coverage_fail(self):
        variants = []
        bad = copy.deepcopy(self.audit); bad['assessments'][0]['support_spans'][0]['quote'] = 'invented'; variants.append(bad)
        bad = copy.deepcopy(self.audit); bad['assessments'][0]['support_spans'] = []; variants.append(bad)
        bad = copy.deepcopy(self.audit); bad['assessments'].append(bad['assessments'][0]); variants.append(bad)
        bad = copy.deepcopy(self.audit); bad['assessments'].pop(); variants.append(bad)
        bad = copy.deepcopy(self.audit); bad['coverage'][0]['support_sets'] = [['unknown']]; variants.append(bad)
        bad = copy.deepcopy(self.audit); bad['coverage'][0]['support_sets'] = [[]]; variants.append(bad)
        bad = copy.deepcopy(self.audit); bad['coverage'][0]['missing'] = ['条件缺失']; variants.append(bad)
        bad = copy.deepcopy(self.audit); bad['coverage'] = []; variants.append(bad)
        for dimension in ('applicability', 'time_scope', 'conditions'):
            for value in ('missing', 'uncertain'):
                bad = copy.deepcopy(self.audit); bad['assessments'][0][dimension] = value; variants.append(bad)
        bad = copy.deepcopy(self.audit); bad['assessments'][0]['relation'] = 'background'; variants.append(bad)
        bad = copy.deepcopy(self.audit); bad['answer_allowed'] = True; variants.append(bad)
        for audit in variants:
            with self.subTest(audit=audit), self.assertRaises(GateError):
                validate_audit(audit, self.hits)

    def test_fixed_question_plan_cannot_shrink_or_change(self):
        questions = self.audit['subquestions'] + [{'subquestion_id': 'q2', 'question': '有什么限制？'}]
        with self.assertRaises(GateError):
            validate_audit(self.audit, self.hits, questions)

    def test_contrary_passage_requires_explicit_conflict(self):
        self.audit['assessments'][1]['relation'] = 'contradicts'
        self.audit['coverage'][0]['support_sets'] = [['D:a']]
        with self.assertRaises(GateError): validate_audit(self.audit, self.hits)
        self.audit['conflicts'] = [{'subquestion_id': 'q1', 'evidence_ids': ['D:a', 'D:b'], 'explanation': '相反结果'}]
        validated = validate_audit(self.audit, self.hits)
        outcome = evaluate_context(validated, [self.hits[0]], GatePolicy())
        self.assertEqual(outcome['decision'], 'conflict')
        self.assertFalse(outcome['answer_allowed'])


class GateFlowTests(unittest.TestCase):
    def setUp(self):
        self.key_patch = patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only'})
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.options = RetrievalConfig(ROOT).deepseek
        self.tokenizer = lambda text: {'input_ids': list(text)}

    def run_gate(self, handler, hits, policy=None, budget=2048, top_k=5):
        gate = DeepSeekQualityGate(self.options, httpx.MockTransport(handler))
        runner = QualityGateRunner(gate, policy or GatePolicy(retry_backoff_seconds=0.0))
        return runner.run('机制与限制', hits, self.tokenizer, budget, top_k, select_context)

    def test_cumulative_refill_finds_joint_evidence_after_first_batch(self):
        counts = []
        def handler(request):
            payload = json.loads(request.content)
            data = json.loads(payload['messages'][1]['content'])
            evidence = data['evidence']; counts.append(len(evidence))
            self.assertEqual(evidence[0]['metadata']['source_type'], 'reading_excerpt')
            if len(evidence) == 1:
                return api_response(make_audit(evidence, [], ['D:a']))
            self.assertIsNotNone(data['fixed_subquestions'])
            return api_response(make_audit(evidence, [['D:a', 'D:b']]))
        context, _, meta = self.run_gate(handler, [hit('a'), hit('b')], GatePolicy(batch_size=1))
        self.assertEqual(counts, [1, 2])
        self.assertEqual(len(context), 2)
        self.assertEqual(meta['decision'], 'sufficient')
        self.assertTrue(meta['validated'])
        self.assertEqual(meta['usage']['total_tokens'], 300)

    def test_joint_group_cannot_pass_after_context_or_topk_trimming(self):
        hits = [hit('a'), hit('b', 'b' * 1000)]
        handler = lambda request: api_response(make_audit(hits, [['D:a', 'D:b']]))
        for budget, top_k in [(100, 5), (2048, 1)]:
            with self.subTest(budget=budget, top_k=top_k):
                context, _, meta = self.run_gate(handler, hits, budget=budget, top_k=top_k)
                self.assertEqual([h['evidence_id'] for h in context], ['D:a'])
                self.assertEqual(meta['decision'], 'partial')
                self.assertFalse(meta['answer_allowed'])
                self.assertFalse(meta['validated'])
                self.assertTrue(meta['gaps'])

    def test_limited_answer_only_for_fully_covered_subquestions(self):
        hits = [hit('a')]
        questions = [{'subquestion_id': 'q1', 'question': '机制？'}, {'subquestion_id': 'q2', 'question': '限制？'}]
        handler = lambda request: api_response(make_audit(hits, questions=questions))
        for allow in (True, False):
            _, _, meta = self.run_gate(handler, hits, GatePolicy(allow_partial_answers=allow))
            self.assertEqual(meta['decision'], 'partial')
            self.assertEqual(meta['answer_allowed'], allow)
            self.assertEqual(meta['answer_scope'], ['q1'] if allow else [])

    def test_conflict_is_preserved_even_when_topk_only_fits_one_side(self):
        hits = [hit('a'), hit('b')]
        audit = make_audit(hits)
        audit['assessments'][1]['relation'] = 'contradicts'
        audit['conflicts'] = [{'subquestion_id': 'q1', 'evidence_ids': ['D:a', 'D:b'], 'explanation': '实验结论相反'}]
        context, _, meta = self.run_gate(lambda request: api_response(audit), hits, top_k=1)
        self.assertEqual(len(context), 1)
        self.assertEqual(meta['decision'], 'conflict')
        self.assertFalse(meta['answer_allowed'])
        self.assertEqual(len(meta['observed_conflicts']), 1)

    def test_conflict_cannot_disappear_in_later_round(self):
        hits = [hit('a'), hit('b'), hit('c')]
        def handler(request):
            data = json.loads(json.loads(request.content)['messages'][1]['content'])
            evidence = data['evidence']
            audit = make_audit(evidence)
            if len(evidence) == 2:
                audit['assessments'][1]['relation'] = 'contradicts'
                audit['conflicts'] = [{'subquestion_id': 'q1', 'evidence_ids': ['D:a', 'D:b'], 'explanation': '冲突'}]
            return api_response(audit)
        _, _, meta = self.run_gate(handler, hits, GatePolicy(batch_size=2))
        self.assertEqual(meta['decision'], 'conflict')
        self.assertFalse(meta['answer_allowed'])

    def test_retryable_error_is_bounded_and_401_is_not_retried(self):
        hits = [hit('a')]
        responses = iter([httpx.Response(429), api_response(make_audit(hits))])
        _, _, meta = self.run_gate(lambda request: next(responses), hits)
        self.assertEqual(len(meta['attempts']), 2)
        self.assertTrue(meta['validated'])
        context, _, meta = self.run_gate(lambda request: httpx.Response(401, text='SECRET'), hits)
        self.assertEqual(len(meta['attempts']), 1)
        self.assertEqual(meta['status'], 'failed')
        self.assertEqual(meta['error_code'], 'http_401')
        self.assertEqual(context, [])
        self.assertNotIn('SECRET', json.dumps(meta))

    def test_empty_truncated_malformed_and_forged_responses_never_pass(self):
        invalid_audit = make_audit([hit('a')])
        invalid_audit['assessments'][0]['support_spans'][0]['quote'] = 'forged'
        responses = [api_response(invalid_audit), httpx.Response(200, json=[]),
            httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '   '}}]}),
            httpx.Response(200, json={'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]})]
        for response in responses:
            with self.subTest(response=response):
                context, _, meta = self.run_gate(lambda request: response, [hit('a')])
                self.assertEqual(context, [])
                self.assertEqual(meta['status'], 'failed')
                self.assertFalse(meta['answer_allowed'])
                self.assertLessEqual(len(meta['attempts']), 2)

    def test_budget_stops_do_not_claim_whole_scope_has_no_evidence(self):
        hits = [hit(str(i)) for i in range(4)]
        def handler(request):
            data = json.loads(json.loads(request.content)['messages'][1]['content'])
            return api_response(make_audit(data['evidence'], [], [data['evidence'][0]['evidence_id']]))
        policies = [GatePolicy(batch_size=1, max_rounds=1), GatePolicy(batch_size=1, max_candidates=1),
                    GatePolicy(batch_size=1, max_api_attempts=1),
                    GatePolicy(batch_size=1, max_total_output_tokens=4096)]
        for policy in policies:
            with self.subTest(policy=policy):
                _, _, meta = self.run_gate(handler, hits, policy)
                self.assertEqual(meta['reviewed_count'], 1)
                self.assertEqual(meta['unreviewed_count'], 3)
                self.assertFalse(meta['candidate_scope_exhausted'])
                self.assertFalse(meta['answer_allowed'])

    def test_request_and_output_budgets_stop_before_network(self):
        def forbidden(request): self.fail('不应发起网络请求')
        for policy in [GatePolicy(max_request_bytes=1024), GatePolicy(max_total_request_bytes=1024),
                       GatePolicy(max_total_output_tokens=1)]:
            context, _, meta = self.run_gate(forbidden, [hit('a')], policy)
            self.assertEqual(context, [])
            self.assertEqual(meta['status'], 'budget_exhausted')
            self.assertFalse(meta['schema_validated'])

    def test_retry_cannot_overrun_global_attempt_or_output_budget(self):
        for policy in [GatePolicy(max_api_attempts=1, retry_backoff_seconds=0.0),
                       GatePolicy(max_total_output_tokens=4096, retry_backoff_seconds=0.0)]:
            context, _, meta = self.run_gate(lambda request: httpx.Response(429), [hit('a')], policy)
            self.assertEqual(len(meta['attempts']), 1)
            self.assertEqual(context, [])
            self.assertEqual(meta['status'], 'failed')
            self.assertEqual(meta['error_code'], 'retry_budget_exhausted')

    def test_smaller_complete_alternative_group_can_fit_context(self):
        hits = [hit('large', 'x' * 2000), hit('small')]
        audit = make_audit(hits, [['D:large'], ['D:small']])
        context, _, meta = self.run_gate(lambda request: api_response(audit), hits, budget=100)
        self.assertEqual([h['evidence_id'] for h in context], ['D:small'])
        self.assertEqual(meta['decision'], 'sufficient')

    def test_schema_rejects_unknown_policy_fields_and_wrong_types(self):
        for data in [{'allow_conflicts': True}, {'max_rounds': '4'}, {'max_rounds': 0}]:
            with self.assertRaises(ValueError): GatePolicy.model_validate(data)

    def test_failure_in_later_round_does_not_release_previous_partial_result(self):
        hits = [hit('a'), hit('b')]
        responses = iter([api_response(make_audit(hits[:1], [], ['D:a'])), httpx.Response(401)])
        context, _, meta = self.run_gate(lambda request: next(responses), hits, GatePolicy(batch_size=1))
        self.assertEqual(context, [])
        self.assertEqual(meta['decision'], 'review_failed')
        self.assertFalse(meta['answer_allowed'])

    def test_deadline_expired_response_does_not_pass(self):
        now = [0.0]
        def handler(request):
            now[0] = 10.0
            return api_response(make_audit([hit('a')]))
        with patch('retrieval.gate_runner.time.monotonic', side_effect=lambda: now[0]):
            context, _, meta = self.run_gate(handler, [hit('a')], GatePolicy(total_timeout_seconds=1.0))
        self.assertEqual(context, [])
        self.assertEqual(meta['error_code'], 'deadline_exceeded')

    def test_duplicate_blank_and_ineligible_candidates_are_prefiltered(self):
        hits = [hit('a'), hit('duplicate', 'Evidence a'), hit('blank', ' '), {**hit('bad'), 'retrieval_eligible': False}]
        context, _, meta = self.run_gate(lambda request: api_response(make_audit(hits[:1])), hits)
        self.assertEqual(len(context), 1)
        self.assertEqual(meta['prefiltered_count'], 3)

    def test_no_candidates_needs_no_api_call(self):
        context, _, meta = self.run_gate(lambda request: self.fail('不应调用 API'), [])
        self.assertEqual(context, [])
        self.assertEqual(meta['status'], 'completed')
        self.assertEqual(meta['decision'], 'insufficient')
        self.assertFalse(meta['validated'])

    def test_service_preserves_ungated_search_and_exposes_gate_decision(self):
        hits = [hit('a')]
        class Router:
            model_sha256 = 'test'
            tokenizer = staticmethod(lambda text: {'input_ids': list(text)})
            def token_length(self, query): return 5
            def encode(self, query): return np.ones(1024), {}
            def status(self): return {}
            def close(self): pass
        class Index:
            signature = {'model_sha256': 'test'}
            def search(self, *args): return copy.deepcopy(hits)
            def status(self): return {'kind': 'test'}
            def close(self): pass
        class Reranker:
            model = None
            def close(self): pass
        gate = DeepSeekQualityGate(self.options, httpx.MockTransport(lambda request: api_response(make_audit(hits))))
        service = RetrievalService(RetrievalConfig(ROOT), router=Router(), index=Index(), reranker=Reranker(), gate=gate)
        self.addCleanup(service.close)
        ordinary = service.search({'query': 'q', 'rerank': False, 'quality_gate': False})
        self.assertEqual(ordinary['quality_gate']['status'], 'not_requested')
        self.assertFalse(ordinary['results'][0]['quality_validated'])
        gated = service.search({'query': 'q', 'rerank': False})
        self.assertEqual(gated['schema_version'], 2)
        self.assertTrue(gated['quality_gate']['answer_allowed'])
        self.assertEqual(gated['context'][0]['rank'], 1)
        self.assertTrue(gated['results'][0]['quality_validated'])
        self.assertTrue(service.status()['quality_gate']['default_enabled'])
        with patch.dict(os.environ, {}, clear=True), patch.object(service.router, 'encode') as encode:
            with self.assertRaises(GateError) as raised:
                service.search({'query': 'q', 'rerank': False})
            self.assertEqual(raised.exception.code, 'not_configured')
            encode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
