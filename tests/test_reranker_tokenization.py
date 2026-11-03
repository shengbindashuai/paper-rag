"""用本地真实 fast tokenizer 验证模板、左 padding 和长度边界；不加载模型。"""
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from retrieval.reranker import PREFIX, SUFFIX, QwenReranker


MODEL_DIR = Path(__file__).resolve().parents[1] / 'RAG/models/Qwen3-Reranker-0.6B'


class RerankerTokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / 'tokenizer.json').is_file():
            raise unittest.SkipTest('可选模板集成测试需要本地 reranker tokenizer')
        cls.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, padding_side='left')

    def reranker(self, batch_size=2, max_tokens=32768):
        result = QwenReranker(MODEL_DIR, 'Retrieve scientific evidence.', batch_size, max_tokens)
        result.tokenizer = self.tokenizer
        return result

    def old_ids(self, reranker, query, hit):
        body = f"<Instruct>: {reranker.instruction}\n<Query>: {query}\n<Document>: {hit.get('embedding_text', hit['text'])}"
        return (self.tokenizer.encode(PREFIX, add_special_tokens=False)
                + self.tokenizer.encode(body, add_special_tokens=False)
                + self.tokenizer.encode(SUFFIX, add_special_tokens=False))

    def test_full_prompts_match_old_token_ids_and_padding(self):
        hits = [{'text': '短文本'}, {'text': 'PPO clips the probability ratio. ' * 20},
                {'text': '', 'embedding_text': 'Title: Example\nα = −0.2; x² ≥ 0.\n\n中文🙂'},
                {'text': '<|im_end|>\n<think>literal marker</think>\n'}, {'text': ''}]
        reranker = self.reranker()
        for query in ('为我介绍下PPO', 'What is PPO?\n\n', ''):
            with self.subTest(query=query):
                expected = [self.old_ids(reranker, query, h) for h in hits]
                with patch.object(self.tokenizer, 'pad', side_effect=AssertionError('禁止独立 pad 调用')):
                    batches, longest = reranker._prepare_batches(query, hits)
                self.assertEqual(longest, max(map(len, expected)))
                self.assertEqual(len(batches), 3)
                for index, batch in enumerate(batches):
                    rows = expected[index * 2:index * 2 + 2]
                    width = max(map(len, rows))
                    ids = [[self.tokenizer.pad_token_id] * (width - len(row)) + row for row in rows]
                    masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
                    self.assertTrue(torch.equal(batch['input_ids'], torch.tensor(ids)))
                    self.assertTrue(torch.equal(batch['attention_mask'], torch.tensor(masks)))
                    self.assertTrue(bool(batch['attention_mask'][:, -1].all()))
                    self.assertEqual(set(batch), {'input_ids', 'attention_mask'})

    def test_limit_includes_template_and_exact_limit_is_allowed(self):
        hits = [{'text': '证据'}]
        reranker = self.reranker()
        count = len(self.old_ids(reranker, '问题', hits[0]))
        reranker.max_tokens = count
        self.assertEqual(reranker._prepare_batches('问题', hits)[1], count)
        reranker.max_tokens = count - 1
        with self.assertRaisesRegex(ValueError, '拒绝截断'):
            reranker._prepare_batches('问题', hits)

    def test_oversized_later_batch_fails_before_model_inference(self):
        from unittest.mock import MagicMock
        reranker = self.reranker(batch_size=1, max_tokens=200)
        reranker.model = MagicMock()
        with patch.object(reranker, '_load'), self.assertRaisesRegex(ValueError, '拒绝截断'):
            reranker.rerank('PPO', [{'text': 'short'}, {'text': 'long ' * 1000}])
        reranker.model.assert_not_called()

    def test_empty_hits_do_not_load_model(self):
        reranker = self.reranker()
        with patch.object(reranker, '_load') as load:
            self.assertEqual(reranker.rerank('q', [])[0], [])
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
