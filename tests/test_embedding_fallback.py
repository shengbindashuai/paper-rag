"""模拟 NPU 缺失/崩溃/恢复，验证 GPU 回退、冷却和显存释放顺序。"""
from pathlib import Path
import unittest
import numpy as np
from retrieval.config import RetrievalConfig
from retrieval.embedding_router import QueryEmbeddingRouter


class Model:
    def __init__(self): self.closed = False; self.fail = False
    def encode_one(self, query):
        if self.fail: raise RuntimeError('NPU inference failed')
        return np.ones(1024, dtype=np.float32)/32, {}
    def close(self): self.closed = True


class FallbackTests(unittest.TestCase):
    def setUp(self): self.events = []; self.count = 80
    def make_router(self, factory):
        events = self.events
        class GPU:
            def encode(self, texts, **options):
                events.append(('encode', options))
                return np.ones((1,1024),dtype=np.float32)/32, {}
            def close(self): events.append('gpu_closed')
        def load_gpu(): events.append('gpu_loaded'); return GPU()
        router = QueryEmbeddingRouter(RetrievalConfig(Path('.')), model_sha256='same',
            tokenizer=lambda text: {'input_ids':[0]*self.count}, npu_factory=factory, gpu_factory=load_gpu)
        router.before_gpu = lambda: events.append('reranker_released')
        self.addCleanup(router.close)
        return router

    def test_unavailable_at_start_and_cooldown(self):
        attempts=[]
        def unavailable(length): attempts.append(length); raise RuntimeError('No NPU')
        router=self.make_router(unavailable)
        router.start()  # 服务在无 NPU 的机器上也能启动。
        _,route=router.encode('q')
        router.encode('q again')
        self.assertEqual(attempts,[128])
        self.assertEqual(route['reason'],'npu_unavailable')
        self.assertEqual(route['device'],'GPU')
        self.assertEqual(route['requested_npu_length'],128)
        self.assertLess(self.events.index('reranker_released'),self.events.index('gpu_loaded'))
        self.assertIn('gpu_closed',self.events)

    def test_inference_failure_releases_worker(self):
        model=Model();router=self.make_router(lambda length:model)
        router.start();model.fail=True
        _,route=router.encode('q')
        self.assertTrue(model.closed)
        self.assertEqual(router.status()['resident_npu_lengths'],[])
        self.assertEqual(route['device'],'GPU')

    def test_optional_tier_failure_keeps_128_usable(self):
        pinned=Model()
        def load(length):
            if length==128:return pinned
            raise OSError('Driver failed loading 256')
        router=self.make_router(load)
        self.count=200
        self.assertEqual(router.encode('longer')[1]['device'],'GPU')
        self.count=80
        self.assertEqual(router.encode('short')[1]['device'],'NPU')
        self.assertFalse(pinned.closed)

    def test_recovery_after_cooldown(self):
        available=[False]
        def load(length):
            if not available[0]:raise RuntimeError('NPU offline')
            return Model()
        router=self.make_router(load)
        router.encode('first')
        available[0]=True
        router.npu_failures[128]['retry_at']=0
        self.assertEqual(router.encode('retry')[1]['device'],'NPU')

    def test_model_mismatch_and_over_limit_do_not_fallback(self):
        def mismatch(length):raise ValueError('Source model SHA mismatch')
        router=self.make_router(mismatch)
        with self.assertRaises(ValueError):router.encode('q')
        self.assertNotIn('gpu_loaded',self.events)
        self.count=32769
        with self.assertRaises(ValueError):router.encode('oversized')
        self.assertNotIn('gpu_loaded',self.events)

    def test_gpu_failure_propagates(self):
        def unavailable(length):raise RuntimeError('No NPU')
        router=self.make_router(unavailable)
        def gpu_failed():raise RuntimeError('GPU unavailable')
        router.gpu_factory=gpu_failed
        with self.assertRaisesRegex(RuntimeError,'GPU unavailable'):router.encode('q')


if __name__=='__main__':unittest.main()
