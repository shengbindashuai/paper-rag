"""只读已发布生效表和词频表：扩展只帮助召回，TF-IDF 只作辅助排序。

不读取 candidates 来构造别名；不按全库高频词生成查询。语境判断是可审查
的规则，不冒充语义分类模型：范围、原文过滤、全称词项不足以消歧时不展开。
"""
from collections import Counter
from contextlib import contextmanager
import json
import math
import re
import sqlite3

from terminology.statistics import tokens, STOPWORDS


def alias_pattern(alias):
    # 英文/数字边界，不把 PPO 命中 APPO；中文相邻字符允许直接提问。
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])")


class TermEnrichment:
    def __init__(self, config):
        self.config = config

    @contextmanager
    def snapshot(self):
        db, metadata = None, {"status": "disabled"}
        try:
            if self.config.term_enrichment_enabled:
                pointer_path = self.config.path("terminology_pointer")
                if not pointer_path.is_file():
                    metadata = {"status": "not_published"}
                else:
                    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                    path = (pointer_path.parent / pointer["directory"] / "terminology.sqlite3").resolve()
                    if not path.is_relative_to(pointer_path.parent.resolve()):
                        raise ValueError("术语数据库路径越界")
                    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=.25)
                    db.row_factory = sqlite3.Row
                    db.execute("BEGIN")
                    db.execute("SELECT count(*) FROM active_terms").fetchone()
                    metadata = {"status": "ready", "database": str(path),
                                "source_digest": pointer.get("source_digest"), "verified_at": pointer.get("verified_at")}
        except (OSError, sqlite3.Error) as exc:
            if db is not None: db.close()
            db = None
            metadata = {"status": "unavailable", "error_type": type(exc).__name__,
                        "reason": "database_locked" if "locked" in str(exc).lower() else "resource_unavailable"}
        try:
            yield db, metadata
        finally:
            if db is not None: db.close()

    def expand(self, db, query, scopes=None, doc_ids=None):
        matches, skipped = [], []
        if db is None:
            return query, {"applied": [], "skipped": []}
        cues = json.loads(self.config.path("term_scope_cues_file").read_text(encoding="utf-8"))
        query_words = set(tokens(query)) - STOPWORDS
        groups = {}
        for row in db.execute("SELECT term_id,canonical,aliases,scope,requires_scope_match,requires_disambiguation FROM active_terms WHERE relation='abbreviation_of' ORDER BY term_id"):
            for alias in json.loads(row["aliases"]):
                if alias_pattern(alias).search(query):
                    groups.setdefault(alias, []).append(dict(row))
        for alias, rows in groups.items():
            eligible = []
            for row in rows:
                evidence = db.execute("SELECT evidence_spans FROM active_terms WHERE term_id=?", (row['term_id'],)).fetchone()
                evidence = json.loads(evidence[0])
                if not evidence:
                    continue
                doc_match = bool(doc_ids and set(doc_ids) & {x['doc_id'] for x in evidence})
                scope = row['scope']
                scope_cues = [scope.replace('_', ' ')] + cues.get(scope, [])
                scope_match = scope in (scopes or []) or doc_match or any(
                    (cue.casefold() in query.casefold() if not cue.isascii() else
                     re.search(r'(?<![a-z0-9_])'+re.escape(cue)+r'(?![a-z0-9_])', query, re.I)) for cue in scope_cues)
                if row['requires_scope_match'] and not scope_match:
                    continue
                canonical_tokens = set(tokens(row['canonical'])) - STOPWORDS - set(tokens(alias))
                overlap = len(query_words & canonical_tokens)
                exact = row['canonical'].casefold() in query.casefold()
                eligible.append((100 if exact else 10 if doc_match else overlap, row, evidence, exact or doc_match or overlap >= 2))
            eligible.sort(key=lambda x: (-x[0], x[1]['term_id']))
            # 严格处理 requires_disambiguation：仅“同一领域”不足以证明唯一含义。
            chosen = eligible[0] if eligible else None
            ambiguous = chosen and (chosen[1]['requires_disambiguation'] or len({x[1]['canonical'].casefold() for x in eligible}) > 1)
            if not chosen or (ambiguous and (not chosen[3] or len(eligible)>1 and eligible[0][0] == eligible[1][0])):
                skipped.append({"alias": alias, "reason": "scope_or_context_insufficient",
                                "active_options": [{k:r[k] for k in ['term_id','canonical','scope']} for r in rows]})
                continue
            _, row, evidence, _ = chosen
            if len(matches) >= self.config.term_max_expansions:
                skipped.append({"alias":alias,"reason":"expansion_limit"}); continue
            matches.append({"alias":alias,"canonical":row['canonical'],"term_id":row['term_id'],
                            "scope":row['scope'],"evidence_doc_ids":sorted({x['doc_id'] for x in evidence})})
        expansions = list(dict.fromkeys(m['canonical'] for m in matches if m['canonical'].casefold() not in query.casefold()))
        expanded = query + ("\n" + "; ".join(expansions) if expansions else "")
        return expanded, {"applied": matches, "skipped": skipped}

    def weight(self, db, query, hits):
        if self.config.term_weight == 0:
            return hits, {"status":"disabled", "weight":0}
        if db is None or not hits:
            return hits, {"status":"unavailable" if db is None else "empty"}
        idfs = {r['term']:r['idf'] for r in db.execute("SELECT term,idf FROM term_statistics WHERE is_stopword=0")}
        query_counts = Counter(t for t in tokens(query) if t not in STOPWORDS and t in idfs)
        weighted = {t:(1+math.log(n))*idfs[t] for t,n in query_counts.items()}
        selected = sorted(weighted, key=lambda t:(-weighted[t],t))[:self.config.term_max_keywords]
        q = {t:weighted[t] for t in selected}
        q_norm = math.sqrt(sum(v*v for v in q.values()))
        def similarity(counts):
            vector={t:(1+math.log(n))*idfs[t] for t,n in counts.items() if n>0 and t in idfs}
            norm=math.sqrt(sum(v*v for v in vector.values()))
            return sum(v*vector.get(t,0) for t,v in q.items())/(q_norm*norm) if norm and q_norm else 0.0
        output, documents = [], {}
        for hit in hits:
            doc_id=hit['doc_id']
            if doc_id not in documents:
                documents[doc_id]=db.execute("SELECT source_sha256,term_counts FROM document_statistics WHERE doc_id=?",(doc_id,)).fetchone()
            document=documents[doc_id]
            valid=document is not None and document['source_sha256']==hit.get('source_sha256')
            chunk_score=similarity(Counter(tokens(hit['text']))) if valid else 0.0
            doc_score=similarity(json.loads(document['term_counts'])) if valid else 0.0
            output.append({**hit,"term_tfidf_score":.8*chunk_score+.2*doc_score,
                           "term_statistics_status":"matched" if valid else "missing_or_stale_source"})
        if not q or not any(h['term_tfidf_score']>0 for h in output):
            return output,{"status":"no_query_matches","keywords":selected}
        lexical=sorted(output,key=lambda h:-h['term_tfidf_score'])
        ranks={}
        for rank,h in enumerate(lexical,1):ranks.setdefault(h['term_tfidf_score'],rank)
        w=self.config.term_weight
        for rank,h in enumerate(output,1):
            h['term_rerank_score']=(1-w)/(60+rank)+w/(60+ranks[h['term_tfidf_score']])
        output.sort(key=lambda h:-h['term_rerank_score'])
        return output,{"status":"applied","keywords":[{"term":t,"weight":q[t]} for t in selected],
                      "weight":w,"method":"weighted_rank_fusion; lexical=0.8*chunk_tfidf_cosine+0.2*document_tfidf_cosine"}
