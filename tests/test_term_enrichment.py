"""仅批准关系生效；语境不明不展开；频率不产生同义词或改写原问题。"""
import json
from pathlib import Path
import sqlite3
import unittest
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np

from retrieval.config import RetrievalConfig
from retrieval.term_enrichment import TermEnrichment

ROOT=Path(__file__).resolve().parents[1]


class TermTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:');self.db.row_factory=sqlite3.Row
        self.db.executescript('''
          CREATE TABLE active_terms(term_id TEXT,canonical TEXT,aliases TEXT,scope TEXT,requires_scope_match INT,
            requires_disambiguation INT,relation TEXT,evidence_spans TEXT);
          CREATE TABLE candidates(short_form TEXT,long_form TEXT);
          CREATE TABLE term_statistics(term TEXT,idf REAL,is_stopword INT);
          CREATE TABLE document_statistics(doc_id TEXT,source_sha256 TEXT,term_counts TEXT);
        ''')
        self.tool=TermEnrichment(RetrievalConfig(ROOT))
    def tearDown(self):self.db.close()
    def add(self,canonical='Proximal Policy Optimization',alias='PPO',scope='reinforcement_learning',ambiguous=0,doc='P0011',identifier='t1'):
        self.db.execute('INSERT INTO active_terms VALUES (?,?,?,?,?,?,?,?)',(identifier,canonical,json.dumps([alias]),scope,1,ambiguous,
            'abbreviation_of',json.dumps([{'doc_id':doc}])))
    def test_scope_matched_expansion_retains_original(self):
        self.add()
        original='强化学习中PPO如何更新策略？'
        expanded,meta=self.tool.expand(self.db,original)
        self.assertTrue(expanded.startswith(original))
        self.assertIn('Proximal Policy Optimization',expanded)
        self.assertEqual(meta['applied'][0]['term_id'],'t1')
    def test_pending_candidates_never_expand(self):
        self.db.execute("INSERT INTO candidates VALUES ('PPO','unapproved invented expansion')")
        self.assertEqual(self.tool.expand(self.db,'强化学习 PPO')[0],'强化学习 PPO')
    def test_scope_mismatch_boundary_and_missing_context(self):
        self.add(ambiguous=1)
        for query in ['医学 PPO','强化学习 APPO','PPO是什么？','强化学习 PPO']:
            self.assertEqual(self.tool.expand(self.db,query)[0],query)
    def test_ambiguity_resolved_by_doc_evidence_not_domain_alone(self):
        self.add(ambiguous=1)
        self.add(canonical='Particle Physics Observatory',scope='reinforcement_learning',ambiguous=1,identifier='t2',doc='P9999')
        self.assertFalse(self.tool.expand(self.db,'强化学习 PPO')[1]['applied'])
        expanded,meta=self.tool.expand(self.db,'PPO',doc_ids=['P0011'])
        self.assertIn('Proximal Policy Optimization',expanded)
        self.assertEqual(len(meta['applied']),1)
    def test_full_form_context_selects_among_same_scope(self):
        self.add(canonical='proximal policy optimization',ambiguous=1)
        self.add(canonical='particle physics observatory',ambiguous=1,identifier='t2')
        _,meta=self.tool.expand(self.db,'强化学习 proximal policy optimization 的 PPO')
        self.assertEqual(meta['applied'][0]['term_id'],'t1')
    def test_query_related_keywords_and_stale_source(self):
        for term,idf in [('policy',1),('gradient',4),('popular',1),('the',1)]:
            self.db.execute('INSERT INTO term_statistics VALUES (?,?,?)',(term,idf,int(term=='the')))
        counts={'policy':4,'gradient':2,'popular':100000,'the':99999}
        self.db.execute('INSERT INTO document_statistics VALUES (?,?,?)',('d','sha',json.dumps(counts)))
        hits=[{'doc_id':'d','chunk_id':'a','source_sha256':'sha','text':'policy popular popular','rerank_score':.9},
              {'doc_id':'d','chunk_id':'b','source_sha256':'sha','text':'gradient','rerank_score':.8},
              {'doc_id':'d','chunk_id':'c','source_sha256':'old','text':'gradient','rerank_score':.7}]
        ordered,meta=self.tool.weight(self.db,'policy gradient',hits)
        self.assertEqual({x['term'] for x in meta['keywords']},{'policy','gradient'})
        self.assertNotIn('popular',{x['term'] for x in meta['keywords']})
        by_id={h['chunk_id']:h for h in ordered}
        self.assertGreater(by_id['b']['term_tfidf_score'],by_id['a']['term_tfidf_score'])
        self.assertEqual(by_id['c']['term_tfidf_score'],0)
        self.assertEqual(hits[0]['rerank_score'],.9)
        self.assertNotIn('term_tfidf_score',hits[0])
    def test_missing_statistics_keeps_order_and_query(self):
        hits=[{'doc_id':'d','chunk_id':'a','source_sha256':'x','text':'unknown'}]
        ordered,meta=self.tool.weight(self.db,'unknown',hits)
        self.assertEqual(meta['status'],'no_query_matches')
        self.assertNotIn('term_rerank_score',ordered[0])
        self.assertEqual(self.tool.expand(None,'raw')[0],'raw')

    def test_service_uses_expansion_only_for_recall(self):
        from retrieval.service import RetrievalService
        calls={}
        class Terms:
            @contextmanager
            def snapshot(self):yield None,{'status':'test'}
            def expand(self,db,query,scopes,doc_ids):return query+'\nApproved Full Form',{'applied':[]}
            def weight(self,db,query,hits):calls['frequency']=query;return hits,{'status':'test'}
        class Router:
            model_sha256='test'
            tokenizer=staticmethod(lambda text:{'input_ids':[1]})
            def token_length(self,q):return 10
            def encode(self,q):calls['embedding']=q;return np.ones(1024),{}
        class Index:
            signature={'model_sha256':'test'}
            def hybrid_search(self,v,q,*args):
                calls['recall']=q
                return [{'doc_id':'d','chunk_id':'c','text':'evidence','dense_score':.5}],{}
            def status(self):return {}
        class Reranker:
            def close(self):pass
            def rerank(self,q,hits):calls['rerank']=q;return hits,{}
        class Gate:
            def require_configured(self):pass
        class Runner:
            def run(self,q,hits,*args):
                calls['gate']=q
                return [{**hits[0],'context_text':'evidence'}],1,{'status':'completed','validated':False,'schema_validated':True}
        c=RetrievalConfig(ROOT,mmr_enabled=False)
        with patch('retrieval.service.TermEnrichment',return_value=Terms()):
            service=RetrievalService(c,router=Router(),index=Index(),reranker=Reranker(),gate=Gate())
        service.gate_runner=Runner()
        result=service.search({'query':'original'})
        self.assertEqual(calls['embedding'],'original\nApproved Full Form')
        self.assertEqual(calls['recall'],calls['embedding'])
        for key in ['rerank','gate','frequency']:self.assertEqual(calls[key],'original')
        self.assertEqual(result['query'],'original')


if __name__=='__main__':unittest.main()
