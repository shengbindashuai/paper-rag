"""持续维护候选、核验和文档统计；事务提交后发布导出和状态。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from rag_core.io import atomic_json, utc_now
from rag_core.storage import writer_lock
from terminology.pilot import collect, export, verify
from terminology.statistics import CorpusStatistics


def source_state(root, index_dir):
    """文档元信息和清洗文件版本；读取不加载向量或 GPU 模型。"""
    path = index_dir / "library.sqlite3"
    if not path.is_file():
        return {}
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        documents = [dict(r) for r in db.execute("SELECT * FROM documents ORDER BY doc_id")]
    finally:
        db.close()
    aliases = {}
    for row in documents:
        if row['alias_of']:
            aliases.setdefault(row['alias_of'], []).append(row)
    result = {}
    for row in documents:
        path = index_dir / 'processed' / row['doc_id'] / 'cleaned.json.gz'
        stat = path.stat() if path.is_file() else None
        result[row['doc_id']] = json.dumps(
            [row, aliases.get(row['doc_id'], []), [stat.st_size, stat.st_mtime_ns] if stat else None],
            sort_keys=True, ensure_ascii=False)
    return result


def state_digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def saved_decisions(db):
    """保留所有明确决定；来源版本改变的批准退回待核验。"""
    decisions, invalidated = [], []
    for row in db.execute("SELECT * FROM verification WHERE decision!='pending'"):
        evidence = json.loads(row['verified_evidence'])
        valid = bool(evidence) or row['decision'] == 'rejected'
        for item in evidence:
            current = db.execute('SELECT source_sha256,segment_sha256,flags,source_spans FROM candidates WHERE occurrence_id=?',
                                 (item['occurrence_id'],)).fetchone()
            if not current or current['source_sha256'] != item['source_sha256']:
                valid = False
            elif item.get('segment_sha256') and current['segment_sha256'] != item['segment_sha256']:
                valid = False
            elif not json.loads(current['source_spans']) or set(json.loads(current['flags'])).intersection(
                    {'note_not_paper', 'reference_section', 'document_not_indexed', 'non_paragraph_source', 'retrieval_ineligible_source'}):
                valid = False
        if not db.execute('SELECT 1 FROM candidates WHERE pair_id=?', (row['pair_id'],)).fetchone():
            valid = False
        if not valid:
            invalidated.append(dict(row))
            continue
        decisions.append({'short_form': row['short_form'], 'extracted_long_form': row['long_form'],
                          'long_form': row['verified_long_form'] or row['long_form'],
                          'decision': 'rejected' if row['decision'] == 'rejected' else 'approved',
                          'scope': row['scope'], 'reviewer': row['reviewer'], 'reason': row['review_reason'],
                          'evidence': [{'doc_id': item['doc_id'], 'segment_id': item['segment_id'],
                                        'occurrence_id': item['occurrence_id']} for item in evidence]})
    return decisions, invalidated


def _sync_metadata(db, state, segments, statistics):
    db.execute('CREATE TABLE IF NOT EXISTS sync_documents(doc_id TEXT PRIMARY KEY,state TEXT,segment_count INTEGER)')
    db.execute('DELETE FROM sync_documents')
    db.executemany('INSERT INTO sync_documents VALUES (?,?,?)',
                   [(doc_id, data, segments.get(doc_id, 0)) for doc_id, data in state.items()])
    db.execute('CREATE TABLE IF NOT EXISTS sync_settings(key TEXT PRIMARY KEY,value TEXT)')
    db.execute('INSERT OR REPLACE INTO sync_settings VALUES (?,?)', ('statistics', json.dumps(statistics, ensure_ascii=False)))


def synchronize(root, index_dir, directory, initial_decisions=None):
    """只有新增/变化文档重提取，全库 IDF 与关键词重新计算。"""
    root, index_dir, directory = root.resolve(), index_dir.resolve(), directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # 与建库共用索引写锁，避免读取半写入的清洗文件；不持有查询模型锁。
    with writer_lock(directory), writer_lock(index_dir):
        state = source_state(root, index_dir)
        pointer = directory / 'current.json'
        previous = json.loads(pointer.read_text(encoding='utf-8')) if pointer.is_file() else None
        if previous:
            output = directory / previous['directory']
            if not output.resolve().is_relative_to(directory):
                raise ValueError('术语输出目录越界')
            db = sqlite3.connect(output / 'terminology.sqlite3')
            db.row_factory = sqlite3.Row
            old = {r['doc_id']: r for r in db.execute('SELECT * FROM sync_documents')}
            changed = [doc_id for doc_id, data in state.items() if doc_id not in old or old[doc_id]['state'] != data]
            removed = set(old) - set(state)
            extra = {'statistics': json.loads(db.execute("SELECT value FROM sync_settings WHERE key='statistics'").fetchone()[0])}
            segments = {doc_id: row['segment_count'] for doc_id, row in old.items() if doc_id in state}
            try:
                if changed or removed:
                    history = {'at': utc_now(), 'verification': [dict(r) for r in db.execute(
                        "SELECT * FROM verification WHERE decision!='pending'")],
                        'active_terms': [dict(r) for r in db.execute('SELECT * FROM active_terms')]}
                    atomic_json(output / ('review_history_' + utc_now().replace(':', '-') + '.json'), history)
                    db.execute('BEGIN IMMEDIATE')
                    with tempfile.TemporaryDirectory(prefix='delta_', dir=directory) as delta:
                        partial = None
                        try:
                            if changed:
                                partial, partial_extra = collect(root, Path(delta), changed, index_dir=index_dir)
                                if partial.execute("SELECT 1 FROM run_documents WHERE status='error'").fetchone():
                                    raise RuntimeError('增量提取失败，保留上一版术语与统计')
                                segments.update({doc_id: partial_extra['document_segments'].get(doc_id, 0) for doc_id in changed})
                            for doc_id in set(changed) | removed:
                                for table in ['candidates', 'run_documents', 'document_statistics']:
                                    db.execute(f'DELETE FROM {table} WHERE doc_id=?', (doc_id,))
                            if partial:
                                for table in ['candidates', 'run_documents', 'document_statistics']:
                                    for row in partial.execute(f'SELECT * FROM {table}'):
                                        db.execute(f'INSERT INTO {table} VALUES (' + ','.join('?' for _ in row) + ')', tuple(row))
                        finally:
                            if partial:
                                partial.close()
                    decisions, invalidated = saved_decisions(db)
                    if invalidated:
                        atomic_json(output / ('invalidated_reviews_' + utc_now().replace(':', '-') + '.json'), invalidated)
                    stats = CorpusStatistics()
                    for row in db.execute('SELECT * FROM document_statistics'):
                        stats.documents.append((dict(row), Counter(json.loads(row['term_counts']))))
                    for table in ['document_statistics', 'term_statistics', 'document_keywords']:
                        db.execute(f'DROP TABLE {table}')
                    stats_summary = stats.publish(db, commit=False)
                    stats_summary['alias_documents_not_counted'] = db.execute(
                        "SELECT count(*) FROM run_documents WHERE status='alias_preserved_on_canonical'").fetchone()[0]
                    db.execute('DELETE FROM active_terms')
                    db.execute('DELETE FROM verification')
                    snapshot_db = None
                    try:
                        audit_path = directory / 'semantic_audit.sqlite3'
                        loader = None
                        if audit_path.is_file():
                            from terminology.semantic_review import snapshot_loader
                            snapshot_db = sqlite3.connect(audit_path.as_uri() + '?mode=ro', uri=True)
                            snapshot_db.row_factory = sqlite3.Row
                            loader = snapshot_loader(root, snapshot_db)
                        verify(db, decisions, root, commit=False, source_loader=loader)
                    finally:
                        if snapshot_db:
                            snapshot_db.close()
                    _sync_metadata(db, state, segments, stats_summary)
                    db.commit()
                    extra = {'statistics': stats_summary}
                # 导出失败后可从已提交数据库重试，不重复提取或丢失核验决定。
                extra = {'documents_selected': len(state), 'segments_scanned': sum(segments.values()),
                         'statistics': extra['statistics'], 'source_digest': state_digest(state),
                         'automatic_maintenance': True}
                summary = export(db, output, extra)
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()
        else:
            output = Path(tempfile.mkdtemp(prefix='live_', dir=directory))
            db = None
            try:
                db, extra = collect(root, output, index_dir=index_dir)
                if db.execute("SELECT 1 FROM run_documents WHERE status='error'").fetchone():
                    raise RuntimeError('首次术语扫描失败，未发布自动维护版本')
                decisions = json.loads(initial_decisions.read_text(encoding='utf-8'))['decisions'] if initial_decisions and initial_decisions.is_file() else []
                verify(db, decisions, root)
                _sync_metadata(db, state, extra.pop('document_segments'), extra['statistics'])
                db.commit()
                summary = export(db, output, {**extra, 'source_digest': state_digest(state), 'automatic_maintenance': True})
            finally:
                if db:
                    db.close()
        result = {'directory': output.name, 'source_digest': state_digest(state), 'updated_at': utc_now(),
                  'database': str(output / 'terminology.sqlite3'), 'summary': summary['counts']}
        atomic_json(pointer, result)
        return result
