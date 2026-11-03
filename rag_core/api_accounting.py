"""所有外部模型调用共用的本地用量台账与已验证结果缓存。"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from contextlib import contextmanager


def cache_key(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=2)
def tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def quantify(payload, tokenizer_file=None):
    messages = [m.get('content', '') for m in payload.get('messages', [])]
    chars = sum(map(len, messages))
    if tokenizer_file and Path(tokenizer_file).is_file():
        count = sum(len(tokenizer(str(tokenizer_file)).encode(text).ids) for text in messages) + 12
        method = 'Qwen3 local tokenizer proxy; not exact DeepSeek tokens'
    else:
        count = math.ceil(chars / 3) + 12
        method = 'character heuristic; not exact provider tokens'
    return {'request_bytes': len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()),
            'message_characters': chars, 'estimated_input_tokens': count, 'estimate_method': method,
            'max_output_tokens': payload.get('max_tokens', 0)}


class APILedger:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.memory = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connection() as db:
                db.executescript('''
                  PRAGMA journal_mode=WAL;
                  CREATE TABLE IF NOT EXISTS responses(key TEXT PRIMARY KEY,value TEXT,created REAL);
                  CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY,at REAL,kind TEXT,model TEXT,
                    request_key TEXT,cache_hit INTEGER,request_metrics TEXT,usage TEXT,error TEXT);
                ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, key, ttl=86400):
        if self.path:
            with self.connection() as db:
                row = db.execute('SELECT value,created FROM responses WHERE key=?', (key,)).fetchone()
        else:
            row = self.memory.get(key)
        return json.loads(row[0]) if row and time.time() - row[1] <= ttl else None

    def put(self, key, value):
        row = (json.dumps(value, ensure_ascii=False), time.time())
        if self.path:
            with self.connection() as db:
                db.execute('INSERT OR REPLACE INTO responses VALUES (?,?,?)', (key, *row))
        else:
            self.memory[key] = row

    def record(self, kind, model, key, metrics, usage=None, *, hit=False, error=''):
        if self.path:
            with self.connection() as db:
                db.execute('INSERT INTO calls(at,kind,model,request_key,cache_hit,request_metrics,usage,error) VALUES (?,?,?,?,?,?,?,?)',
                           (time.time(), kind, model, key, int(hit), json.dumps(metrics), json.dumps(usage or {}), error))

    def summary(self):
        totals = {}
        if not self.path:
            return totals
        with self.connection() as db:
            for kind, hit, raw_usage, error in db.execute('SELECT kind,cache_hit,usage,error FROM calls'):
                item = totals.setdefault(kind, {'logical_calls': 0, 'local_cache_hits': 0, 'external_calls': 0,
                    'errors': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'provider_cache_hit_tokens': 0})
                item['logical_calls'] += 1
                item['local_cache_hits'] += hit
                item['external_calls'] += int(not hit)
                item['errors'] += int(bool(error))
                usage = json.loads(raw_usage)
                if not hit:
                    for target, source in [('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'completion_tokens'),
                                           ('provider_cache_hit_tokens', 'prompt_cache_hit_tokens')]:
                        value = usage.get(source, 0)
                        if type(value) is int:
                            item[target] += value
        return totals
