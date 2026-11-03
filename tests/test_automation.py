"""目录稳定性、失败恢复、事务回滚及新增文献后的全库统计/核验保留。"""
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from knowledge.automation import LibraryAutomation
from knowledge.jobs import IngestJobs
from rag_core.io import atomic_json
from rag_core.storage import LibraryStore
from retrieval.config import RetrievalConfig
from terminology.maintenance import synchronize


class AutomationTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / 'tmp/automation_tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)
        self.index = self.root / 'RAG/library'
        self.output = self.root / 'RAG/terminology/auto'
        self.config = {'watch_directories': ['RAG/inbox', 'RAG/PDF'], 'poll_seconds': 1,
                       'settle_seconds': 2, 'terminology_debounce_seconds': 2,
                       'terminology_directory': 'RAG/terminology/auto'}

    def tearDown(self):
        self.temp.cleanup()

    def document(self, doc_id, text):
        sha = hashlib.sha256(text.encode()).hexdigest()
        store = LibraryStore(self.index)
        store.record({'doc_id': doc_id, 'source_file': f'PDF/{doc_id}.pdf', 'sha256': sha, 'title': doc_id}, 'indexed')
        store.close()
        path = self.index / 'processed' / doc_id / 'cleaned.json.gz'
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump({'source_sha256': sha, 'segments': [{'segment_id': doc_id + ':s1', 'kind': 'paragraph',
                'text': text, 'retrieval_eligible': True, 'section_path': ['Method'],
                'sources': [{'page': 1, 'segment_char_start': 0, 'segment_char_end': len(text)}]}]}, stream)

    def decisions(self):
        path = self.root / 'reviews.json'
        atomic_json(path, {'decisions': [{'short_form': 'PPO', 'extracted_long_form': 'policy optimization',
            'long_form': 'proximal policy optimization', 'decision': 'approved', 'scope': 'reinforcement_learning',
            'reviewer': 'test', 'reason': 'explicit source', 'evidence': [{'doc_id': 'P1', 'segment_id': 'P1:s1'}]}]})
        return path

    def open_current(self):
        current = json.loads((self.output / 'current.json').read_text(encoding='utf-8'))
        db = sqlite3.connect(current['database'])
        db.row_factory = sqlite3.Row
        return db

    def test_incremental_extraction_global_idf_and_review_preservation(self):
        self.document('P1', 'proximal policy optimization (PPO). PPO policy.')
        synchronize(self.root, self.index, self.output, self.decisions())
        self.document('P2', 'large language model (LLM). policy training.')
        with patch('terminology.maintenance.collect', wraps=__import__('terminology.pilot', fromlist=['collect']).collect) as collect:
            result = synchronize(self.root, self.index, self.output, self.decisions())
        self.assertEqual(collect.call_args.args[2], ['P2'])
        self.assertEqual(result['summary']['document_statistics'], 2)
        db = self.open_current()
        try:
            self.assertEqual(db.execute('SELECT canonical FROM active_terms').fetchone()[0], 'proximal policy optimization')
            self.assertEqual(db.execute("SELECT decision FROM verification WHERE short_form='LLM'").fetchone()[0], 'pending')
            self.assertEqual(db.execute("SELECT document_frequency FROM term_statistics WHERE term='policy'").fetchone()[0], 2)
            self.assertAlmostEqual(db.execute("SELECT idf FROM term_statistics WHERE term='policy'").fetchone()[0], 1.0)
        finally:
            db.close()

    def test_changed_evidence_demotes_approval_and_keeps_history(self):
        self.document('P1', 'proximal policy optimization (PPO).')
        synchronize(self.root, self.index, self.output, self.decisions())
        self.document('P1', 'Changed context. proximal policy optimization (PPO).')
        synchronize(self.root, self.index, self.output, self.decisions())
        db = self.open_current()
        try:
            self.assertEqual(db.execute('SELECT count(*) FROM active_terms').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT decision FROM verification').fetchone()[0], 'pending')
        finally:
            db.close()
        self.assertTrue(list(self.output.glob('live_*/invalidated_reviews_*.json')))

    def test_update_failure_rolls_back_and_retry_recomputes(self):
        self.document('P1', 'large language model (LLM).')
        original = synchronize(self.root, self.index, self.output)
        self.document('P2', 'reinforcement learning (RL).')
        with patch('terminology.maintenance.verify', side_effect=RuntimeError('simulated failure')):
            with self.assertRaises(RuntimeError):
                synchronize(self.root, self.index, self.output)
        db = self.open_current()
        try:
            self.assertEqual(db.execute('SELECT count(*) FROM document_statistics').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT count(DISTINCT doc_id) FROM candidates').fetchone()[0], 1)
        finally:
            db.close()
        self.assertEqual(json.loads((self.output / 'current.json').read_text(encoding="utf-8"))['source_digest'], original['source_digest'])
        self.assertEqual(synchronize(self.root, self.index, self.output)['summary']['document_statistics'], 2)

    def test_export_failure_recovers_committed_statistics(self):
        self.document('P1', 'large language model (LLM).')
        synchronize(self.root, self.index, self.output)
        self.document('P2', 'reinforcement learning (RL).')
        with patch('terminology.maintenance.export', side_effect=OSError('simulated full disk')):
            with self.assertRaises(OSError):
                synchronize(self.root, self.index, self.output)
        synchronize(self.root, self.index, self.output)
        current = json.loads((self.output / 'current.json').read_text(encoding="utf-8"))
        summary = json.loads((Path(current['database']).parent / 'summary.json').read_text(encoding="utf-8"))
        self.assertEqual(summary['statistics']['document_count'], 2)

    def test_watch_waits_for_stability_deduplicates_and_recovers_restart(self):
        service = SimpleNamespace(config=RetrievalConfig(self.root), lock=threading.RLock(),
                                  reranker=SimpleNamespace(close=lambda: None))
        ids = []
        jobs = None
        def runner(identifier):
            ids.append(identifier)
            from knowledge.sources import register_text
            payload = json.loads((jobs.directory / f'{identifier}.input.json').read_text(encoding='utf-8'))
            doc_id = register_text(self.root, payload)
            atomic_json(jobs.directory / f'{identifier}.result.json', {'doc_id': doc_id, 'indexed': True})
        jobs = IngestJobs(service, runner=runner)
        auto = LibraryAutomation(jobs, self.config)
        path = self.root / 'RAG/inbox/note.txt'
        path.write_text('first content', encoding='utf-8')
        try:
            auto.scan(100)
            self.assertFalse(ids)
            auto.scan(101)
            self.assertFalse(ids)
            path.write_text('changed content', encoding='utf-8')
            auto.scan(102)
            auto.scan(103)
            self.assertFalse(ids)
            auto.scan(104)
            auto.scan(105)
        finally:
            auto.close()
            jobs.close()
        self.assertEqual(len(ids), 1)
        jobs = IngestJobs(service, runner=runner)
        auto = LibraryAutomation(jobs, self.config)
        try:
            auto.scan(110)
            self.assertFalse(path.exists())
            archived = list((self.root / 'RAG/PDF/文档原件').glob('*.txt'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_text(encoding='utf-8'), 'changed content')
        finally:
            auto.close()
            jobs.close()
        self.assertEqual(len(ids), 1)

    def test_pdf_cleanup_requires_completed_matching_archive_and_unchanged_source(self):
        import shutil
        service = SimpleNamespace(config=RetrievalConfig(self.root), lock=threading.RLock(),
                                  reranker=SimpleNamespace(close=lambda: None))
        jobs = IngestJobs(service, runner=lambda identifier: None)
        auto = LibraryAutomation(jobs, self.config)
        path = self.root / 'RAG/inbox/paper.pdf'
        archive = self.root / 'RAG/PDF/P1__paper.pdf'
        path.write_bytes(b'paper source')
        shutil.copyfile(path, archive)
        identifier = 'a' * 32
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        atomic_json(jobs.directory / f'{identifier}.input.json', {'input_sha256': expected})
        job = {'job_id': identifier, 'status': 'failed', 'result': {'indexed': True,
                    'document': {'source_file': 'PDF/P1__paper.pdf'}}}
        try:
            self.assertFalse(auto.archive_completed(path, {}, job))
            self.assertTrue(path.exists())
            job['status'] = 'completed'
            path.write_bytes(b'new content after submission')
            with self.assertRaises(ValueError):
                auto.archive_completed(path, {}, job)
            self.assertTrue(path.exists())
            path.write_bytes(b'paper source')
            archive.write_bytes(b'broken archive')
            with self.assertRaises(ValueError):
                auto.archive_completed(path, {}, job)
            self.assertTrue(path.exists())
            archive.write_bytes(b'paper source')
            self.assertTrue(auto.archive_completed(path, {}, job))
            self.assertFalse(path.exists())
            self.assertEqual(archive.read_bytes(), b'paper source')
        finally:
            auto.close()
            jobs.close()

    def test_native_event_observer_and_no_idle_automation_ticks(self):
        from knowledge.file_events import DirectoryEvents
        changed = threading.Event()
        observer = DirectoryEvents([(self.root, 'files')], lambda reason: changed.set())
        observer.start()
        try:
            (self.root / 'notify.txt').write_text('new file', encoding='utf-8')
            self.assertTrue(changed.wait(3))
        finally:
            observer.close()
        self.assertFalse(observer.thread.is_alive())
        service = SimpleNamespace(config=RetrievalConfig(self.root), lock=threading.RLock(),
                                  reranker=SimpleNamespace(close=lambda: None))
        jobs = IngestJobs(service, runner=lambda identifier: None)
        auto = LibraryAutomation(jobs, self.config)
        first, again = threading.Event(), threading.Event()
        def tick(*args, **kwargs):
            (again if first.is_set() else first).set()
        try:
            with patch.object(auto, 'tick', side_effect=tick) as action:
                auto.signal('startup')
                auto.thread.start()
                self.assertTrue(first.wait(3))
                self.assertFalse(again.wait(0.2))
                self.assertEqual(action.call_count, 1)
        finally:
            auto.close()
            jobs.close()

    def test_registered_pdf_is_not_resubmitted_and_old_queue_cache_clears(self):
        service = SimpleNamespace(config=RetrievalConfig(self.root), lock=threading.RLock(),
                                  reranker=SimpleNamespace(close=lambda: None))
        store = LibraryStore(self.index)
        store.close()
        jobs = IngestJobs(service, runner=lambda identifier: None)
        auto = LibraryAutomation(jobs, self.config)
        path = self.root / 'RAG/PDF/P2__pending.pdf'
        path.write_bytes(b'pdf content')
        atomic_json(self.root / 'RAG/metadata/P2.json', {'file': 'PDF/P2__pending.pdf'})
        identifier = 'b' * 32
        atomic_json(jobs.directory / f'{identifier}.job.json', {'job_id': identifier, 'status': 'completed'})
        auto.files[str(path.resolve())] = {'signature': [], 'status': 'queued', 'job_id': identifier}
        try:
            with patch.object(jobs, 'submit') as submit:
                auto.scan(100)
            submit.assert_not_called()
            self.assertFalse(auto.files)
            self.assertTrue(path.exists())
        finally:
            auto.close()
            jobs.close()

    def test_invalid_input_is_reported_once_until_file_changes(self):
        service = SimpleNamespace(config=RetrievalConfig(self.root), lock=threading.RLock(),
                                  reranker=SimpleNamespace(close=lambda: None))
        jobs = IngestJobs(service, runner=lambda identifier: None)
        auto = LibraryAutomation(jobs, self.config)
        path = self.root / 'RAG/inbox/bad.txt'
        path.write_bytes(b'\xff\xfe\xfd')
        try:
            auto.scan(100)
            with patch.object(jobs, 'submit', wraps=jobs.submit) as submit, self.assertLogs(level='ERROR'):
                auto.scan(102)
                auto.scan(104)
            self.assertEqual(submit.call_count, 1)
            self.assertEqual(auto.files[str(path.resolve())]['status'], 'failed')
            self.assertEqual(len(auto.status['file_errors']), 1)
            path.write_text('corrected', encoding='utf-8')
            auto.scan(105)
            self.assertEqual(auto.files[str(path.resolve())]['status'], 'settling')
        finally:
            auto.close()
            jobs.close()


if __name__ == '__main__':
    unittest.main()
