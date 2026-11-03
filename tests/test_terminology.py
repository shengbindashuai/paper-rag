"""缩写字符对齐、来源区间、三表发布边界和文档级统计。"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from terminology.schwartz_hearst import extract_pairs, align_long_form
from terminology.pilot import collect, verify, export
from terminology.statistics import CorpusStatistics
from rag_core.storage import LibraryStore


ROOT = Path(__file__).resolve().parents[1]


class ExtractionTests(unittest.TestCase):
    def check(self,text,expected):
        pairs=extract_pairs(text)
        self.assertEqual([(p['short_form'],p['long_form']) for p in pairs],expected)
        for pair in pairs:
            self.assertEqual(text[pair['short_start']:pair['short_end']],pair['short_form'])
            self.assertEqual(text[pair['long_start']:pair['long_end']],pair['long_form'])
            for match in pair['alignment']:
                self.assertEqual(pair['short_form'][match['short_index']].lower(),text[match['segment_index']].lower())
        return pairs

    def test_forward_reverse_and_multiple_pairs(self):
        self.check('We use proximal policy optimization (PPO) and trust region policy optimization (TRPO).',
                   [('PPO','policy optimization'),('TRPO','trust region policy optimization')])
        self.check('PPO (proximal policy optimization) is used.',[('PPO','policy optimization')])
        self.check('AI (artificial intelligence) is used.',[('AI','artificial intelligence')])

    def test_word_boundary_internal_letters_and_hyphenated_full_word(self):
        self.check('We examine Heat shock transcription factor (HSF).',[('HSF','Heat shock transcription factor')])
        self.check('We use Gcn5-related N-acetyltransferase (GNAT).',[('GNAT','Gcn5-related N-acetyltransferase')])
        self.assertIsNone(align_long_form('AB','xalpha beta'))

    def test_repeated_sources_are_not_collapsed(self):
        pairs=self.check('proximal policy optimization (PPO). Again proximal policy optimization (PPO).',
                         [('PPO','policy optimization'),('PPO','policy optimization')])
        self.assertNotEqual(pairs[0]['short_start'],pairs[1]['short_start'])

    def test_noise_and_non_definitions_do_not_force_pairing(self):
        for text in ['Results (2026).','Equation (x).','Numbers (1, 2).','No definition (XYZ).','See Figure (ABC).',
                     'proximal policy optimization. New sentence (PPO).','proximal policy\n\noptimization (PPO).']:
            self.check(text,[])

    def test_fullwidth_parentheses_single_line_break_and_case_preservation(self):
        self.check('proximal policy\noptimization（PPO）',[('PPO','policy\noptimization')])
        self.check('large language model (llm)',[('llm','large language model')])


class WorkflowTests(unittest.TestCase):
    def test_all_occurrences_pending_default_scoped_publication_and_conflicts(self):
        base=ROOT/'tmp'/'terminology_tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            root=Path(folder);library=root/'RAG/library'
            store=LibraryStore(library)
            for doc_id,text in [('P1','proximal policy optimization (PPO). proximal policy optimization (PPO).'),
                                ('P2','preferred provider organization (PPO).')]:
                item={'doc_id':doc_id,'source_file':f'PDF/{doc_id}.pdf','sha256':doc_id,'title':doc_id}
                store.record(item,'indexed')
                destination=library/'processed'/doc_id;destination.mkdir(parents=True)
                cleaned={'source_sha256':doc_id,'segments':[{'segment_id':doc_id+':s1','kind':'paragraph',
                         'text':text,'retrieval_eligible':True,'section_path':['Method'],
                         'sources':[{'page':1,'segment_char_start':0,'segment_char_end':len(text)}]}]}
                with gzip.open(destination/'cleaned.json.gz','wt',encoding='utf-8') as f:json.dump(cleaned,f)
            store.close()
            db,extra=collect(root,root/'output')
            try:
                self.assertEqual(db.execute('SELECT count(*) FROM candidates').fetchone()[0],3)
                verify(db,[])
                self.assertEqual(db.execute('SELECT count(*) FROM active_terms').fetchone()[0],0)
                self.assertEqual(db.execute("SELECT count(*) FROM verification WHERE conflict_status='multiple_long_forms'").fetchone()[0],2)
                db.execute('DELETE FROM verification');db.commit()
                verify(db,[{'short_form':'PPO','extracted_long_form':'policy optimization','long_form':'proximal policy optimization','decision':'approved',
                            'scope':'reinforcement_learning','reviewer':'test','reason':'explicit source definition',
                            'evidence':[{'doc_id':'P1','segment_id':'P1:s1'}]}],root)
                active=dict(db.execute('SELECT * FROM active_terms').fetchone())
                self.assertEqual(active['requires_scope_match'],1)
                self.assertEqual(active['requires_disambiguation'],1)
                self.assertEqual(active['canonical'],'proximal policy optimization')
                self.assertEqual(db.execute("SELECT decision FROM verification WHERE verified_long_form IS NOT NULL").fetchone()[0],'approved_corrected')
                summary=export(db,root/'output',extra)
                self.assertFalse(summary['production_query_expansion_changed'])
                self.assertEqual(summary['counts']['candidates'],3)
            finally:db.close()


class StatisticsTests(unittest.TestCase):
    def test_raw_tf_df_tfidf_stopwords_and_duplicate_source(self):
        stats=CorpusStatistics()
        def add(doc,sha,text):
            stats.add({'doc_id':doc,'source_file':doc+'.pdf','source_sha256':sha,'status':'indexed'},
                      [{'kind':'paragraph','retrieval_eligible':True,'section_path':['Method'],'text':text},
                       {'kind':'paragraph','retrieval_eligible':True,'section_path':['References'],'text':'spurious spurious'}])
        add('A','sha1','PPO PPO policy the')
        add('B','sha2','TRPO policy the')
        add('alias','sha1','PPO PPO policy the')
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
        try:
            summary=stats.publish(db)
            ppo=dict(db.execute("SELECT * FROM term_statistics WHERE term='ppo'").fetchone())
            self.assertEqual((ppo['raw_frequency'],ppo['document_frequency']),(2,1))
            self.assertAlmostEqual(ppo['idf'],math.log(3/2)+1)
            self.assertEqual(db.execute("SELECT document_frequency FROM term_statistics WHERE term='policy'").fetchone()[0],2)
            self.assertEqual(db.execute("SELECT raw_frequency,is_stopword FROM term_statistics WHERE term='the'").fetchone()[:],(2,1))
            self.assertEqual(db.execute("SELECT count(*) FROM document_keywords WHERE term='the'").fetchone()[0],0)
            self.assertEqual(db.execute("SELECT count(*) FROM term_statistics WHERE term='spurious'").fetchone()[0],0)
            self.assertEqual(summary['duplicate_sources_skipped'],1)
            self.assertEqual(summary['document_count'],2)
            for row in db.execute('SELECT doc_id,term_counts FROM document_statistics'):
                counts=json.loads(row['term_counts']);scores=[]
                for term,count in counts.items():
                    if term=='the':continue
                    idf=db.execute('SELECT idf FROM term_statistics WHERE term=?',(term,)).fetchone()[0]
                    scores.append(((1+math.log(count))*idf)**2)
                norm=math.sqrt(sum(scores))
                for keyword in db.execute('SELECT * FROM document_keywords WHERE doc_id=?',(row['doc_id'],)):
                    expected=(1+math.log(counts[keyword['term']]))*keyword['idf']/norm
                    self.assertAlmostEqual(keyword['tfidf'],expected)
                self.assertAlmostEqual(db.execute('SELECT sum(tfidf*tfidf) FROM document_keywords WHERE doc_id=?',
                                                (row['doc_id'],)).fetchone()[0],1.0)
        finally:db.close()


if __name__=='__main__':unittest.main()
