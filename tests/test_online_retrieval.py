"""设备边界、模型释放、索引一致性及 API 返回校验；不联网、不加载真实权重。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from rag_core.storage import LibraryStore
from retrieval.config import RetrievalConfig
from retrieval.embedding_router import QueryEmbeddingRouter, choose_device
from retrieval.index import SnapshotIndex
from retrieval.quality_gate import DeepSeekQualityGate, assessment_passes
from retrieval.schemas import SearchRequest
from retrieval.service import select_context


class RouterTests(unittest.TestCase):
    def test_boundaries_and_no_truncation(self):
        for count, expected in [(128,128),(129,256),(256,256),(257,512),(513,1024),(1024,1024)]:
            self.assertEqual(choose_device(count), ("NPU",expected))
        self.assertEqual(choose_device(1025), ("GPU",None))
        self.assertEqual(choose_device(32768), ("GPU",None))
        with self.assertRaises(ValueError): choose_device(32769)

    def test_128_pinned_and_only_one_optional_tier(self):
        released = []
        class FakeModel:
            def __init__(self, length): self.length = length
            def encode_one(self, text): return np.ones(1024), {}
            def close(self): released.append(self.length)
        router = QueryEmbeddingRouter(RetrievalConfig(Path('.')), tokenizer=lambda text: {'input_ids':[0]},
                                      npu_factory=FakeModel, model_sha256='test')
        router.start()
        router._get_npu(256)
        pinned = router.models[128]
        router._get_npu(512)
        self.assertEqual(released, [256])
        self.assertEqual(sorted(router.models), [128,512])
        self.assertIs(router.models[128], pinned)
        router._get_npu(1024)
        self.assertEqual(released, [256,512])
        router.close()
        self.assertIn(128,released)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.options = RetrievalConfig(Path('.')).deepseek
        self.hits = [{'evidence_id':'D:C','text':'Evidence','title':'Paper','pages':[1]}]

    def response(self, identifier='D:C', finish='stop'):
        return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps({
            'subquestions':[{'subquestion_id':'q1','question':'q'}], 'assessments':[
            {'evidence_id':identifier,'relation':'direct_support','applicability':'not_applicable',
             'time_scope':'not_applicable','conditions':'supported','reason':'Direct evidence',
             'support_spans':[{'start':0,'end':8,'quote':'Evidence'}]}],
            'coverage':[{'subquestion_id':'q1','support_sets':[[identifier]],'partial_evidence_ids':[], 'missing':[]}],
            'conflicts':[]})}}]}

    def test_missing_key_fails_before_network(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError): DeepSeekQualityGate(self.options).assess('q',self.hits)

    def test_valid_json_request_and_response(self):
        def handler(request):
            payload=json.loads(request.content)
            self.assertEqual(payload['response_format'],{'type':'json_object'})
            self.assertEqual(request.url.path,'/chat/completions')
            return httpx.Response(200,json=self.response())
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test-only'}):
            result=DeepSeekQualityGate(self.options,httpx.MockTransport(handler)).assess('q',self.hits)
        self.assertTrue(assessment_passes(result['assessments'][0]))

    def test_unknown_ids_and_truncated_responses_rejected(self):
        for payload in [self.response('other'),self.response(finish='length')]:
            with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test-only'}):
                gate=DeepSeekQualityGate(self.options,httpx.MockTransport(lambda request:httpx.Response(200,json=payload)))
                with self.assertRaises(RuntimeError): gate.assess('q',self.hits)

    def test_missing_condition_blocks_even_if_model_says_keep(self):
        item=json.loads(self.response()['choices'][0]['message']['content'])['assessments'][0]
        item['conditions']='missing'
        self.assertFalse(assessment_passes(item))


class IndexTests(unittest.TestCase):
    def test_read_only_snapshot_filter_and_stale_revision(self):
        temporary_root=Path(__file__).resolve().parents[1]/'tmp'/'retrieval_tests'
        temporary_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            directory=Path(temporary)
            writer=LibraryStore(directory,{'model_sha256':'test'})
            for i in range(2):
                vector=np.zeros((1,1024),dtype=np.float32);vector[0,i]=1
                writer.commit_document({'doc_id':str(i),'source_file':'PDF/a.pdf','sha256':'test','title':str(i)},
                    [{'doc_id':str(i),'chunk_id':'c','text':'Evidence','retrieval_eligible':True}],vector)
            writer.refresh_snapshot()
            reader=SnapshotIndex(directory)
            vector=np.zeros(1024,dtype=np.float32);vector[0]=1
            self.assertEqual(reader.search(vector,2)[0]['doc_id'],'0')
            self.assertEqual(reader.search(vector,2,['1'])[0]['doc_id'],'1')
            self.assertEqual(reader.get_chunk('0','c')['text'],'Evidence')
            writer.record({'doc_id':'0','source_file':'PDF/a.pdf','sha256':'x','title':'0'},'failed')
            with self.assertRaises(RuntimeError):reader.search(vector,2)
            reader.close();writer.close()

    def test_context_budget_preserves_whole_evidence(self):
        hits=[{'evidence_id':'a','text':'a'*1000},{'evidence_id':'b','text':'short'}]
        tokenizer=lambda text:{'input_ids':list(text)}
        selected,used=select_context(hits,tokenizer,100)
        self.assertEqual([x['evidence_id'] for x in selected],['b'])
        self.assertLessEqual(used,100)

    def test_request_validation(self):
        with self.assertRaises(ValueError):SearchRequest(query='  ')
        with self.assertRaises(ValueError):SearchRequest(query='q',candidate_k=2,top_k=5)


if __name__=='__main__':unittest.main()
