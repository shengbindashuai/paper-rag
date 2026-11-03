"""本地目录监听和术语统计维护，与 MCP 共用生命周期与串行入库队列。"""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import sqlite3
from threading import Condition, Event, Thread
import time

from rag_core.config import config_file, resource_path
from knowledge.schemas import DocumentIngestRequest, PaperIngestRequest
from rag_core.io import atomic_json, utc_now, file_sha256
from terminology.maintenance import source_state, state_digest, synchronize


class LibraryAutomation:
    def __init__(self, jobs, config):
        self.jobs, self.root, self.index_dir = jobs, jobs.root, jobs.index_dir
        self.config = config
        self.watch = [(self.root / p).resolve() for p in config['watch_directories']]
        self.directory = (self.root / config['terminology_directory']).resolve()
        if not self.directory.is_relative_to(self.root) or any(not p.is_relative_to(self.root) for p in self.watch):
            raise ValueError('自动维护路径必须位于工作区内')
        for path in self.watch:
            path.mkdir(parents=True, exist_ok=True)
        self.state_path = self.index_dir / 'automation_watch.json'
        self.status_path = self.index_dir / 'automation_status.json'
        self.files = json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.is_file() else {}
        self.status = {'enabled': True, 'watch_directories': [str(p) for p in self.watch],
                       'watch_mode': 'filesystem_events', 'idle_polling': False,
                       'settle_seconds': config['settle_seconds'],
                       'terminology': {'status': 'starting'}}
        self.stop_event = Event()
        self.condition, self.reasons = Condition(), set()
        self.next_terms_check = None
        self.observer = None
        self.unsubscribe = jobs.subscribe(self.signal)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='terminology-maintenance')
        self.future = None
        self.pending_digest, self.pending_since = None, 0
        self.last_digest = None
        pointer = self.directory / 'current.json'
        if pointer.is_file():
            current = json.loads(pointer.read_text(encoding='utf-8'))
            self.last_digest = current['source_digest']
            self.status['terminology'] = {'status': 'completed', **current}
        self.retry_after = 0
        self.thread = Thread(target=self.run, name='knowledge-automation', daemon=True)

    @classmethod
    def from_config(cls, jobs, path=None):
        path = path or config_file(jobs.root, "knowledge_automation.json")
        if not path.is_file():
            return None
        config = json.loads(path.read_text(encoding='utf-8'))
        if not config.get('enabled', False):
            return None
        for key in ['settle_seconds', 'terminology_debounce_seconds']:
            if not isinstance(config[key], (int, float)) or config[key] < 1:
                raise ValueError(f'{key} 必须至少为 1 秒')
        return cls(jobs, config)

    def start(self):
        from knowledge.file_events import DirectoryEvents
        snapshots = self.index_dir / 'snapshots'
        snapshots.mkdir(parents=True, exist_ok=True)
        self.observer = DirectoryEvents([*((p, 'files') for p in self.watch), (snapshots, 'index')], self.signal)
        self.observer.start()
        self.signal('startup')
        self.thread.start()

    def signal(self, reason):
        with self.condition:
            self.reasons.add(reason)
            self.condition.notify()

    def _known_archived(self):
        path = self.index_dir / 'library.sqlite3'
        if not path.is_file():
            return set()
        db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        try:
            known = {(self.root / 'RAG' / row[0]).resolve() for row in db.execute('SELECT source_file FROM documents')}
            # 注册已完成、编码尚在进行的 PDF 也是归档文件，不能再次排队。
            for metadata in (self.root / 'RAG/metadata').glob('P*.json'):
                if '.processing' not in metadata.name:
                    data = json.loads(metadata.read_text(encoding='utf-8'))
                    if data.get('file'):
                        known.add((self.root / 'RAG' / data['file']).resolve())
            return known
        finally:
            db.close()

    def scan(self, now):
        known = self._known_archived()
        archived = (self.root / 'RAG/PDF').resolve()
        observed, dirty = set(), False
        for folder in self.watch:
            for original in folder.rglob('*'):
                if original.name.startswith(('.', '~$')) or original.suffix.lower() not in {'.pdf', '.md', '.txt', '.docx'}:
                    continue
                try:
                    path = original.resolve()
                    if not path.is_relative_to(folder) or not path.is_file():
                        continue
                    if path.is_relative_to(archived) and path.suffix.lower() != '.pdf':
                        continue
                    if path.is_relative_to(archived) and path in known:
                        continue  # 不自动重试旧隔离文档，也不改写已归档版本。
                    stat = path.stat()
                    signature = [stat.st_size, stat.st_mtime_ns]
                    key = str(path)
                    observed.add(key)
                    entry = self.files.get(key)
                    if not entry or entry['signature'] != signature:
                        self.files[key] = {'signature': signature, 'stable_since': now, 'status': 'settling'}
                        dirty = True
                        continue
                    if entry.get('job_id'):
                        job = self.jobs.status(entry['job_id'])
                        if job['status'] != entry['status']:
                            entry.update(status=job['status'], updated_at=utc_now())
                            dirty = True
                        if job['status'] == 'completed':
                            if self.archive_completed(path, entry, job):
                                dirty = True
                        if job['status'] != 'interrupted':
                            continue
                        entry.pop('job_id')
                    if entry['status'] == 'failed' or now - entry['stable_since'] < self.config['settle_seconds']:
                        continue
                    if path.suffix.lower() == '.pdf':
                        request, kind = PaperIngestRequest(path=key), 'paper'
                    else:
                        request, kind = DocumentIngestRequest(path=key, title=path.stem[:300]), 'document'
                    job = self.jobs.submit(kind, request)
                    entry.update(job_id=job['job_id'], status=job['status'], updated_at=utc_now())
                    logging.info('Auto ingest submitted: %s; job=%s', path.name, job['job_id'])
                    dirty = True
                except ValueError:
                    if str(path) in self.files:
                        self.files[str(path)].update(status='failed', error='输入校验失败；请检查格式、编码、大小和路径')
                        dirty = True
                    logging.exception('Auto ingest input validation failed: %s', original.name)
                except (OSError, KeyError):
                    logging.exception('Auto ingest could not read a watched file: %s', original.name)
                except RuntimeError:
                    if str(path) in self.files:
                        self.files[str(path)]['status'] = 'waiting_queue'
                        dirty = True
                    logging.info('Auto ingest queue busy; retrying when a job completes')
        # 删除缓存不会删除源文件或知识库条目；重放同一内容由入库队列去重。
        for key in list(self.files):
            if key not in observed:
                entry = self.files[key]
                if entry.get('job_id'):
                    try:
                        entry['status'] = self.jobs.status(entry['job_id'])['status']
                    except KeyError:
                        entry['status'] = 'interrupted'
                if entry.get('status') not in {'queued', 'running'}:
                    del self.files[key]
                    dirty = True
        if dirty:
            atomic_json(self.state_path, self.files)
        self.status['watch_files'] = {state: sum(e['status'] == state for e in self.files.values())
                                      for state in {e['status'] for e in self.files.values()}}
        self.status['recent_jobs'] = [{'path': path, 'job_id': entry['job_id'], 'status': entry['status']}
                                      for path, entry in sorted(self.files.items(),
                                          key=lambda pair: pair[1].get('updated_at', ''), reverse=True)
                                      if entry.get('job_id')][:10]
        self.status['file_errors'] = [{'path': path, 'error': entry['error']}
                                     for path, entry in self.files.items() if entry.get('error')][:10]

    def archive_completed(self, path, entry, job):
        """成功且原件/归档都通过哈希核对后，才清理投递文件。"""
        inbox = (self.root / 'RAG/inbox').resolve()
        if job['status'] != 'completed' or not path.is_relative_to(inbox) or not path.is_file():
            return False
        payload = json.loads((self.jobs.directory / (job['job_id'] + '.input.json')).read_text(encoding='utf-8'))
        expected = payload.get('input_sha256')
        if not expected or not job.get('result', {}).get('indexed'):
            return False
        before = path.stat()
        if file_sha256(path) != expected:
            raise ValueError('投递文件在提交后发生变化，保留原件')
        pdf_root = (self.root / 'RAG/PDF').resolve()
        if path.suffix.lower() == '.pdf':
            document = job['result'].get('document') or {}
            destination = (self.root / 'RAG' / document.get('source_file', '')).resolve()
            if not destination.is_relative_to(pdf_root) or not destination.is_file() or file_sha256(destination) != expected:
                raise ValueError('PDF 归档缺失或哈希不符，保留投递原件')
        else:
            # 文档保持原格式，放入 PDF 目录的独立原件子目录，不伪装为 PDF。
            metadata = self.root / 'RAG/knowledge_sources' / (job['result']['doc_id'] + '.json')
            source = json.loads(metadata.read_text(encoding='utf-8'))
            backup = (self.root / 'RAG' / source.get('original_archive', '')).resolve()
            if not backup.is_relative_to((self.root / 'RAG').resolve()) or not backup.is_file() or file_sha256(backup) != expected:
                raise ValueError('文档原件备份缺失或不符，保留投递原件')
            destination = pdf_root / '文档原件' / (job['job_id'] + '__' + path.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and file_sha256(destination) != expected:
                raise FileExistsError('归档位置已有不同内容，不覆盖')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('归档期间投递文件发生变化，保留原件')
        record = {'job_id': job['job_id'], 'source': str(path), 'destination': str(destination),
                  'sha256': expected, 'at': utc_now()}
        # 先记清理意图，重启后仍可根据任务和哈希安全重试。
        with (self.index_dir / 'automation_archives.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        if path.suffix.lower() != '.pdf' and not destination.exists():
            path.rename(destination)
        else:
            path.unlink()
        entry.update(archive_status='archived', archive_path=str(destination))
        logging.info('Inbox source archived: %s -> %s', path.name, destination.name)
        return True

    def tick(self, now=None, *, scan_files=True):
        now = time.time() if now is None else now
        if scan_files:
            self.scan(now)
        if self.future and self.future.done():
            try:
                current = self.future.result()
                self.last_digest = current['source_digest']
                self.status['terminology'] = {'status': 'completed', **current}
                if current.get('semantic_verification', {}).get('status') in {'partial', 'failed'}:
                    self.retry_after = now + 60
            except Exception:
                logging.exception('Automatic terminology maintenance failed; previous database remains available')
                self.status['terminology'] = {'status': 'failed', 'updated_at': utc_now(),
                                             'error': '查看本地服务日志；上次发布结果保留'}
                self.retry_after = now + 60
            self.future = None
        digest = state_digest(source_state(self.root, self.index_dir))
        self.next_terms_check = None
        from terminology.semantic_review import needs_review
        semantic_pending = needs_review(self.root, self.directory, digest)
        if digest != self.last_digest or semantic_pending:
            if digest != self.pending_digest:
                self.pending_digest, self.pending_since = digest, now
            with self.jobs.lock:
                busy = any(json.loads(p.read_text(encoding='utf-8'))['status'] in {'queued', 'running'}
                           for p in self.jobs.directory.glob('*.job.json'))
            if not busy and not self.future and now >= self.retry_after and now - self.pending_since >= self.config['terminology_debounce_seconds']:
                decisions = resource_path(self.root, self.config['initial_decisions']) if self.config.get('initial_decisions') else None
                self.future = self.pool.submit(self._synchronize, decisions)
                self.future.add_done_callback(lambda _: self.signal('terms_completed'))
                self.status['terminology'] = {'status': 'running', 'started_at': utc_now()}
            elif not busy and not self.future:
                self.next_terms_check = max(self.retry_after, self.pending_since + self.config['terminology_debounce_seconds'])
        self.status['checked_at'] = utc_now()
        atomic_json(self.status_path, self.status)

    def _synchronize(self, decisions):
        # 避免维护持有索引写锁时，刚开始的入库任务被非阻塞写锁拒绝。
        with self.jobs.service.lock:
            result = synchronize(self.root, self.index_dir, self.directory, decisions)
        # 外部审查不占用检索/GPU 锁；发布前重新检查语料版本。
        from terminology.semantic_review import run
        result['semantic_verification'] = run(self.root, self.directory, stop_event=self.stop_event)
        return result

    def run(self):
        scan_due = None
        while not self.stop_event.is_set():
            with self.condition:
                while not self.reasons and not self.stop_event.is_set():
                    deadlines = [t for t in (scan_due, self.next_terms_check) if t is not None]
                    delay = max(0, min(deadlines) - time.time()) if deadlines else None
                    if delay == 0:
                        break
                    self.condition.wait(delay)
                reasons, self.reasons = self.reasons, set()
            if self.stop_event.is_set():
                break
            now = time.time()
            try:
                if 'watcher_error' in reasons:
                    self.status['watch_error'] = str(self.observer.error)
                    logging.error('Native filesystem notifications stopped: %s', self.observer.error)
                scan_files = bool(reasons & {'startup', 'files', 'jobs'}) or (scan_due is not None and now >= scan_due)
                self.tick(now, scan_files=scan_files)
                deadlines = [entry['stable_since'] + self.config['settle_seconds'] for entry in self.files.values()
                             if entry['status'] == 'settling' and entry['stable_since'] + self.config['settle_seconds'] > now]
                scan_due = min(deadlines) if deadlines else None
            except Exception:
                logging.exception('Knowledge automation event failed')
                self.next_terms_check = time.time() + 60  # 仅错误时重试，不做空闲轮询。

    def close(self):
        self.stop_event.set()
        self.signal('stop')
        self.unsubscribe()
        if self.observer:
            self.observer.close()
        if self.thread.is_alive():
            self.thread.join()
        self.pool.shutdown(wait=True)
        self.status.update(enabled=False, stopped_at=utc_now())
        atomic_json(self.status_path, self.status)
