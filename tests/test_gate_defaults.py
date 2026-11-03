"""验证各查询入口默认启用门控，显式关闭仍有效；不加载模型或调用网络。"""
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import rag_library
from retrieval import cli
from retrieval.config import RetrievalConfig
from retrieval.schemas import SearchRequest
from retrieval.tool_api import RetrievalToolAPI, definitions


ROOT = Path(__file__).resolve().parents[1]


class GateDefaultTests(unittest.TestCase):
    def test_request_and_tool_schema_default_to_enabled(self):
        self.assertTrue(SearchRequest(query='q').quality_gate)
        self.assertFalse(SearchRequest(query='q', quality_gate=False).quality_gate)
        self.assertTrue(definitions()[0]['inputSchema']['properties']['quality_gate']['default'])

    def test_tool_dispatch_honors_default_and_explicit_disable(self):
        service = MagicMock()
        api = RetrievalToolAPI(service)
        api.call('paper_search', {'query': 'q'})
        self.assertTrue(service.search.call_args.args[0].quality_gate)
        api.call('paper_search', {'query': 'q', 'quality_gate': False})
        self.assertFalse(service.search.call_args.args[0].quality_gate)

    def test_retrieval_cli_default_and_both_flags(self):
        for flags, enabled in [([], True), (['--quality-gate'], True), (['--no-quality-gate'], False)]:
            with self.subTest(flags=flags), patch('sys.argv', ['cli', 'search', '--query', 'q', *flags]), \
                    patch('sys.stdout'), patch('retrieval.cli.RetrievalConfig.load', return_value=RetrievalConfig(ROOT)), \
                    patch('retrieval.service.RetrievalService') as factory:
                factory.return_value.search.return_value = {}
                cli.main()
                self.assertEqual(factory.return_value.search.call_args.args[0].quality_gate, enabled)
                self.assertEqual(factory.return_value.search.call_args.args[0].candidate_k, 200)

    def test_library_cli_default_and_both_flags(self):
        for flags, enabled in [([], True), (['--quality-gate'], True), (['--no-quality-gate'], False)]:
            with self.subTest(flags=flags), patch('sys.argv', ['rag_library', 'search', '--query', 'q', *flags]), \
                    patch('sys.stdout'), patch.object(Path, 'is_file', return_value=True), \
                    patch('rag_core.storage.LibraryStore'), patch('retrieval.query.PaperRetriever') as factory:
                factory.return_value.search.return_value = {}
                rag_library.main()
                self.assertEqual(factory.return_value.search.call_args.kwargs['quality_gate'], enabled)
                self.assertEqual(factory.return_value.search.call_args.kwargs['candidate_k'], 200)


if __name__ == '__main__':
    unittest.main()
