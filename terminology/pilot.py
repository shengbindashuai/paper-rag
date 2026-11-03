"""本地三表测试流程：候选保留全部出现；核验逐对管理；测试生效需显式决定。"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import unicodedata

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.config import default_root
from rag_core.io import atomic_json, utc_now
from terminology.schwartz_hearst import extract_pairs
from terminology.statistics import CorpusStatistics


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:24]


def create_tables(db):
    db.executescript("""
        CREATE TABLE candidates(
          occurrence_id TEXT PRIMARY KEY,pair_id TEXT NOT NULL,short_form TEXT,long_form TEXT,
          short_key TEXT,long_key TEXT,doc_id TEXT,segment_id TEXT,source_file TEXT,source_sha256 TEXT,
          cleaned_file TEXT,cleaned_sha256 TEXT,segment_sha256 TEXT,short_start INTEGER,short_end INTEGER,
          long_start INTEGER,long_end INTEGER,context TEXT,source_spans TEXT,pattern TEXT,flags TEXT,
          alignment TEXT,document_status TEXT,aliases TEXT);
        CREATE INDEX candidate_pair ON candidates(pair_id);
        CREATE INDEX candidate_short ON candidates(short_key);
        CREATE TABLE verification(
          pair_id TEXT PRIMARY KEY,short_form TEXT,long_form TEXT,occurrence_count INTEGER,
          document_count INTEGER,distinct_content_count INTEGER,proposed_relation TEXT,
          scope TEXT,conflict_status TEXT,alternative_pair_ids TEXT,flags TEXT,
          decision TEXT,reviewer TEXT,review_reason TEXT,verified_occurrence_ids TEXT,
          verified_long_form TEXT,verified_evidence TEXT);
        CREATE TABLE active_terms(
          term_id TEXT PRIMARY KEY,pair_id TEXT UNIQUE,canonical TEXT,aliases TEXT,relation TEXT,
          scope TEXT,evidence_occurrence_ids TEXT,requires_scope_match INTEGER,
          requires_disambiguation INTEGER,remaining_alternative_pair_ids TEXT,
          reviewer TEXT,review_reason TEXT,published_at TEXT,evidence_spans TEXT);
        CREATE TABLE run_documents(doc_id TEXT PRIMARY KEY,status TEXT,candidate_count INTEGER,error TEXT);
    """)


def collect(root, output, doc_ids=None, *, index_dir=None):
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "terminology.sqlite3"
    if destination.exists():
        raise FileExistsError("测试输出已存在；请指定新的 --output，避免覆盖人工核验结果")
    db = sqlite3.connect(destination)
    db.row_factory = sqlite3.Row
    create_tables(db)
    directory = index_dir or root / "RAG/library"
    source = sqlite3.connect((directory / "library.sqlite3").as_uri()+"?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    # 固定一个来源清单快照；不向主库写入，也不把重叠 chunk 重复当作定义来源。
    documents = list(source.execute("SELECT * FROM documents ORDER BY doc_id"))
    aliases = {}
    for row in documents:
        if row["alias_of"]:
            aliases.setdefault(row["alias_of"], []).append({"doc_id": row["doc_id"], "source_file": row["source_file"], "source_sha256": row["source_sha256"]})
    source.close()
    segment_count = 0
    document_segments = {}
    statistics = CorpusStatistics()
    selected = [d for d in documents if doc_ids is None or d["doc_id"] in doc_ids]
    if doc_ids and set(doc_ids) - {d["doc_id"] for d in selected}:
        raise ValueError("指定文档不存在")
    for number, document in enumerate(selected, 1):
        doc_id = document["doc_id"]
        path = directory / "processed" / doc_id / "cleaned.json.gz"
        count = 0
        if document["alias_of"]:
            db.execute("INSERT INTO run_documents VALUES (?,?,?,?)", (doc_id, "alias_preserved_on_canonical", 0, None))
            continue
        if not path.is_file():
            db.execute("INSERT INTO run_documents VALUES (?,?,?,?)", (doc_id, "missing_cleaned_source", 0, None))
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                cleaned = json.load(stream)
            if cleaned["source_sha256"] != document["source_sha256"]:
                raise ValueError("缓存与主库来源哈希不一致")
            cleaned_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            for segment in cleaned["segments"]:
                text = segment.get("text", "")
                if not text:
                    continue
                segment_count += 1
                document_segments[doc_id] = document_segments.get(doc_id, 0) + 1
                for pair in extract_pairs(text):
                    assert text[pair["short_start"]:pair["short_end"]] == pair["short_form"]
                    assert text[pair["long_start"]:pair["long_end"]] == pair["long_form"]
                    short_key, long_key = normalized(pair["short_form"]), normalized(pair["long_form"])
                    pair_id = "pair_" + digest([short_key, long_key])
                    occurrence = "occ_" + digest([doc_id, segment["segment_id"], pair["short_start"], pair["long_start"], pair_id])
                    flags = set(pair["flags"])
                    if segment.get("kind") != "paragraph": flags.add("non_paragraph_source")
                    if not segment.get("retrieval_eligible", True): flags.add("retrieval_ineligible_source")
                    if document["status"] != "indexed": flags.add("document_not_indexed")
                    if not str(document["source_file"]).lower().endswith(".pdf"): flags.add("note_not_paper")
                    if any("reference" in s.casefold() or "bibliograph" in s.casefold() for s in segment.get("section_path", [])):
                        flags.add("reference_section")
                    flags.update(segment.get("quality_flags", []))
                    left, right = min(pair["short_start"], pair["long_start"]), max(pair["short_end"], pair["long_end"])
                    anchors = [a for a in segment.get("sources", []) if a.get("segment_char_end", len(text)) > left and a.get("segment_char_start", 0) < right]
                    values = [occurrence, pair_id, pair["short_form"], pair["long_form"], short_key, long_key,
                              doc_id, segment["segment_id"], document["source_file"], cleaned["source_sha256"],
                              str(path.relative_to(root)), cleaned_sha, hashlib.sha256(text.encode()).hexdigest(),
                              pair["short_start"], pair["short_end"], pair["long_start"], pair["long_end"],
                              text[max(0,left-140):min(len(text),right+140)], json.dumps(anchors,ensure_ascii=False),
                              pair["pattern"], json.dumps(sorted(flags),ensure_ascii=False),
                              json.dumps(pair["alignment"]), document["status"],
                              json.dumps(aliases.get(doc_id, []),ensure_ascii=False)]
                    db.execute("INSERT INTO candidates VALUES ("+",".join("?" for _ in values)+")", values)
                    count += 1
            statistics.add(document,cleaned["segments"])
            db.execute("INSERT INTO run_documents VALUES (?,?,?,?)", (doc_id, "scanned", count, None))
        except Exception as exc:
            # 单文档解析失败可审计，继续其余文档；不会标记为扫描成功。
            db.execute("INSERT OR REPLACE INTO run_documents VALUES (?,?,?,?)", (doc_id, "error", count, type(exc).__name__))
        if number % 100 == 0:
            db.commit()
            print(f"Scanned {number}/{len(selected)} documents", flush=True)
    db.commit()
    stats_summary = statistics.publish(db)
    stats_summary["alias_documents_not_counted"] = sum(bool(d["alias_of"]) for d in selected)
    return db, {"documents_selected": len(selected), "segments_scanned": segment_count,
                "document_segments": document_segments, "statistics":stats_summary}


def verify(db, decisions, root=None, *, commit=True, source_loader=None):
    db.execute('CREATE TABLE IF NOT EXISTS active_term_pairs(term_id TEXT,pair_id TEXT PRIMARY KEY)')
    db.execute('DELETE FROM active_term_pairs')
    grouped = list(db.execute("SELECT pair_id,short_key,long_key,min(short_form) short_form,min(long_form) long_form,count(*) occurrence_count,count(DISTINCT doc_id) document_count,count(DISTINCT source_sha256) distinct_content_count FROM candidates GROUP BY pair_id ORDER BY short_key,long_key"))
    by_short = {}
    for row in grouped:
        by_short.setdefault(row["short_key"], []).append(row["pair_id"])
    unused = list(decisions)
    wanted_segments, source_cache, resolved_paths = {}, {}, {}
    root_real = root.resolve() if root is not None else None
    for item in decisions:
        for evidence in item.get('evidence', []):
            wanted_segments.setdefault(evidence['doc_id'], set()).add(evidence['segment_id'])
    for group in grouped:
        alternatives = [p for p in by_short[group["short_key"]] if p != group["pair_id"]]
        flags = set()
        for row in db.execute("SELECT flags FROM candidates WHERE pair_id=?", (group["pair_id"],)):
            flags.update(json.loads(row[0]))
        decision, reviewer, reason, scope, verified = "pending", None, None, "unassigned", []
        verified_long_form, verified_evidence = None, []
        for item in decisions:
            extracted = item.get("extracted_long_form",item["long_form"])
            if normalized(item["short_form"]) != group["short_key"] or normalized(extracted) != group["long_key"]:
                continue
            if decision != "pending":
                raise ValueError("同一配对有多个核验决定")
            decision, reviewer, reason, scope = item["decision"], item["reviewer"], item["reason"], item["scope"]
            if decision not in {"approved", "rejected"} or not all([reviewer, reason, scope]):
                raise ValueError("核验决定缺少状态、审核者、理由或适用范围")
            for evidence in item["evidence"]:
                rows = list(db.execute("SELECT * FROM candidates WHERE pair_id=? AND doc_id=? AND segment_id=?", (group["pair_id"], evidence["doc_id"], evidence["segment_id"])))
                if evidence.get('occurrence_id'):
                    rows = [row for row in rows if row['occurrence_id'] == evidence['occurrence_id']]
                for row in rows:
                    issues = set(json.loads(row["flags"]))
                    if json.loads(row["source_spans"]) and not issues.intersection({"note_not_paper", "reference_section", "document_not_indexed", "non_paragraph_source", "retrieval_ineligible_source"}):
                        if root is None:
                            raise ValueError("发布核验结果需要 root 以重新验证原始来源")
                        path = resolved_paths.get(row['cleaned_file'])
                        if path is None:
                            path = (root_real / row['cleaned_file']).resolve()
                            if not path.is_relative_to(root_real):
                                raise ValueError("核验来源文件已变化或越界")
                            resolved_paths[row['cleaned_file']] = path
                        if path not in source_cache:
                            loaded = source_loader(row, wanted_segments.get(row['doc_id'], set())) if source_loader else None
                            if loaded is None:
                                stat = path.stat()
                                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                                with gzip.open(path,"rt",encoding="utf-8") as stream: cleaned=json.load(stream)
                                selected_segments = {s['segment_id']: s for s in cleaned['segments']
                                                     if s['segment_id'] in wanted_segments.get(row['doc_id'], set())}
                                loaded = (actual_hash, (stat.st_size, stat.st_mtime_ns), selected_segments)
                            source_cache[path] = loaded
                            stat = path.stat()
                            if (stat.st_size, stat.st_mtime_ns) != loaded[1]:
                                raise ValueError('核验来源文件已变化')
                        actual_hash, original_stat, selected_segments = source_cache[path]
                        if actual_hash != row['cleaned_sha256']:
                            raise ValueError('核验来源文件已变化')
                        segment=selected_segments[row['segment_id']]
                        text=segment["text"]
                        if hashlib.sha256(text.encode()).hexdigest()!=row["segment_sha256"]:
                            raise ValueError("核验段落已变化")
                        if text[row["short_start"]:row["short_end"]]!=row["short_form"] or text[row["long_start"]:row["long_end"]]!=row["long_form"]:
                            raise ValueError("候选原文区间无效")
                        import re
                        # 拒绝的是原配对，不发布规范全称；来源引用用该次出现的原字形，避免 NFKC 分组代表替换数学字形。
                        evidence_form = row['long_form'] if decision == 'rejected' else item['long_form']
                        expression=r"\s+".join(re.escape(w) for w in evidence_form.split())
                        matches=[m for m in re.finditer(expression,text,re.IGNORECASE) if m.end()==row["long_end"]]
                        if len(matches)!=1:
                            raise ValueError("核验后的全称必须在同一原段落中逐字出现并与原配对位置对应")
                        match=matches[0]
                        verified_evidence.append({"occurrence_id":row["occurrence_id"],"doc_id":row["doc_id"],
                            "segment_id":row["segment_id"],"source_file":row["source_file"],"source_sha256":row["source_sha256"],
                            "segment_sha256":row["segment_sha256"],"cleaned_sha256":row["cleaned_sha256"],
                            "long_start":match.start(),"long_end":match.end(),"quote":match.group(),
                            "short_start":row["short_start"],"short_end":row["short_end"],
                            "source_spans":json.loads(row["source_spans"])})
                        verified.append(row["occurrence_id"])
            if decision == "approved" and not verified:
                raise ValueError("批准条目必须能定位到已索引论文的正文来源")
            verified_long_form=item["long_form"]
            if decision=="approved" and normalized(item["long_form"])!=group["long_key"]:
                decision="approved_corrected"
            unused.remove(item)
        db.execute("INSERT INTO verification VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            group["pair_id"], group["short_form"], group["long_form"], group["occurrence_count"], group["document_count"], group["distinct_content_count"],
            "abbreviation_of", scope, "multiple_long_forms" if alternatives else "none_observed",
            json.dumps(alternatives), json.dumps(sorted(flags),ensure_ascii=False), decision, reviewer, reason, json.dumps(verified),
            verified_long_form,json.dumps(verified_evidence,ensure_ascii=False)))
        if decision in {"approved","approved_corrected"}:
            verified_aliases = sorted({row[0] for occurrence in verified for row in db.execute(
                "SELECT short_form FROM candidates WHERE occurrence_id=?", (occurrence,))})
            term_id = "term_"+digest([scope,group["short_key"],normalized(verified_long_form)])
            existing = db.execute('SELECT * FROM active_terms WHERE term_id=?', (term_id,)).fetchone()
            if existing:
                # 不同噪声前缀/截短可核验为同一规范映射；合并映射，保留所有配对与证据。
                merged_evidence = {e['occurrence_id']: e for e in json.loads(existing['evidence_spans']) + verified_evidence}
                db.execute('UPDATE active_terms SET aliases=?,evidence_occurrence_ids=?,requires_disambiguation=?,remaining_alternative_pair_ids=?,evidence_spans=? WHERE term_id=?', (
                    json.dumps(sorted(set(json.loads(existing['aliases'])) | set(verified_aliases)), ensure_ascii=False),
                    json.dumps(sorted(set(json.loads(existing['evidence_occurrence_ids'])) | set(verified))),
                    int(existing['requires_disambiguation'] or bool(alternatives)),
                    json.dumps(sorted(set(json.loads(existing['remaining_alternative_pair_ids'])) | set(alternatives))),
                    json.dumps(list(merged_evidence.values()), ensure_ascii=False), term_id))
            else:
                db.execute("INSERT INTO active_terms VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                term_id, group["pair_id"], verified_long_form,
                json.dumps(verified_aliases,ensure_ascii=False),
                "abbreviation_of", scope, json.dumps(verified), 1, int(bool(alternatives)), json.dumps(alternatives), reviewer, reason, utc_now(),
                json.dumps(verified_evidence,ensure_ascii=False)))
            db.execute('INSERT INTO active_term_pairs VALUES (?,?)', (term_id, group['pair_id']))
    if unused:
        raise ValueError("有核验决定未匹配到提取候选，拒绝静默发布")
    for path, (_, original_stat, _) in source_cache.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != original_stat:
            raise ValueError('发布前来源文件发生变化')
    if commit:
        db.commit()


def export(db, output, extra):
    counts = {}
    for table in ["candidates", "verification", "active_terms", "active_term_pairs", "run_documents", "document_statistics", "term_statistics", "document_keywords"]:
        counts[table] = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        with (output / f"{table}.jsonl").open("w",encoding="utf-8") as stream:
            for row in db.execute(f"SELECT * FROM {table}"):
                data = dict(row)
                for key in ["source_spans","flags","alignment","aliases","alternative_pair_ids","verified_occurrence_ids","evidence_occurrence_ids","remaining_alternative_pair_ids","verified_evidence","evidence_spans"]:
                    if key in data: data[key] = json.loads(data[key])
                if "term_counts" in data: data["term_counts"] = json.loads(data["term_counts"])
                stream.write(json.dumps(data,ensure_ascii=False)+"\n")
    summary = {"created_at":utc_now(), "algorithm":"Schwartz-Hearst 2003, local implementation",
               "production_query_expansion_changed":False, "test_publication_only":True, **extra, "counts":counts,
               "document_scan_states":dict(db.execute("SELECT status,count(*) FROM run_documents GROUP BY status")),
               "verification_states":dict(db.execute("SELECT decision,count(*) FROM verification GROUP BY decision")),
               "conflicting_short_forms":db.execute("SELECT count(DISTINCT c.short_key) FROM candidates c JOIN verification v USING(pair_id) WHERE v.conflict_status='multiple_long_forms'").fetchone()[0]}
    atomic_json(output / "summary.json",summary)
    write_report(db,output,summary)
    return summary


def write_report(db,output,summary):
    def safe(value):return str(value).replace("|","\\|").replace("\n"," ")
    lines=["# Schwartz–Hearst 三表与词项统计测试", "",
           "这是本地测试快照，未连接在线查询展开。来源为连续清洗段落，不读取重叠 chunk 作为重复定义。",
           "", "| 项目 | 数量 |", "|---|---:|"]
    for key,label in [("candidates","候选出现/来源记录"),("verification","归并配对/核验记录"),
                      ("active_terms","测试生效条目"),("document_statistics","统计文档"),
                      ("term_statistics","词项"),("document_keywords","文档关键词记录")]:
        lines.append(f"| {label} | {summary['counts'][key]} |")
    lines += ["",f"扫描文档状态：`{json.dumps(summary['document_scan_states'],ensure_ascii=False)}`。",
              f"有多个候选全称的缩写分组：{summary['conflicting_short_forms']}。这包含截短、拼写变体、噪声和真正歧义，不能直接等同于语义冲突数量。",
              "", "## 测试生效表", "",
              "仅包含显式核验决定。PPO 的原始最短匹配是 policy optimization，依据原段落补全 proximal；原始候选没有被覆盖。",
              "核验者 codex_source_check 表示 AI 本地原文核对，不是人工专家审定。条目必须匹配适用领域；有其他候选含义时还需消歧。",
              "", "| 规范全称 | 别名 | 适用领域 | 需要消歧 |", "|---|---|---|---|"]
    for row in db.execute("SELECT * FROM active_terms ORDER BY canonical"):
        lines.append(f"| {safe(row['canonical'])} | {safe(row['aliases'])} | {safe(row['scope'])} | {bool(row['requires_disambiguation'])} |")
    lines += ["", "## 高频待核验配对（不是已确认术语）", "",
              "按文档覆盖数排序，仅用于安排核验顺序；高频不是含义正确或允许展开的证据。",
              "", "| 缩写 | 原始提取全称 | 出现次数 | 文档数 |", "|---|---|---:|---:|"]
    for row in db.execute("SELECT * FROM verification WHERE decision='pending' ORDER BY document_count DESC,pair_id LIMIT 30"):
        lines.append(f"| {safe(row['short_form'])} | {safe(row['long_form'])} | {row['occurrence_count']} | {row['document_count']} |")
    lines += ["", "## 高频词项（预览排除停用词，完整词频表保留停用词）", "",
              "| 词项 | 原始词频 | 文档频率 DF | IDF | 进入文档 Top20 的次数 |", "|---|---:|---:|---:|---:|"]
    for row in db.execute("SELECT * FROM term_statistics WHERE is_stopword=0 ORDER BY raw_frequency DESC,term LIMIT 25"):
        lines.append(f"| {safe(row['term'])} | {row['raw_frequency']} | {row['document_frequency']} | {row['idf']:.4f} | {row['keyword_document_count']} |")
    lines += ["", "## 原始 PPO 论文的 TF-IDF 关键词预览", "",
              "| 排名 | 词项 | TF | 文档内 L2 归一化 TF-IDF |", "|---|---|---:|---:|"]
    for row in db.execute("SELECT * FROM document_keywords WHERE doc_id='P0011' ORDER BY rank"):
        lines.append(f"| {row['rank']} | {safe(row['term'])} | {row['raw_tf']} | {row['tfidf']:.6f} |")
    lines += ["", "TF-IDF 不判断术语语义；例如引用键、公式变量、提取残留也可能高分，不能直接转成别名。",
              "", "## 数据与边界", "",
              "- terminology.sqlite3 是完整三表及统计表；同名 JSONL 是逐行导出。",
              "- candidates 保留每次出现的原短语、清洗段落字符区间、来源文件/页码或行号、块坐标、哈希、别名文档和质量标记。",
              "- verification 保留原始匹配与核验后的全称、核验理由、适用范围、其他候选和对应证据。",
              "- active_terms 仅来自明确核验决定，并要求领域匹配/必要时消歧；本次不自动用于生产查询。",
              "- 统计单位是文档，按源内容 SHA256 去重；仅用已索引、可检索正文段落，排除参考文献章节。",
              "- 词频保留停用词；TF-IDF 在非停用词空间内计算。`tf=1+ln(raw_tf)`，`idf=1+ln((N+1)/(df+1))`，随后对每文档做 L2 归一化。",
              "- 英文是词/缩写，中文是重叠二元词；不是中文语义分词。完整停用词清单在 summary.json。",
              "- Schwartz–Hearst 允许内部字母匹配，可能截短全称。无括号定义、非字母对应的译名、跨段定义可能漏检。",
              "- 没有标注完整语料金标准，不能把单元测试通过解释为全库精确率或召回率。",
              "", "方法来源：https://psb.stanford.edu/psb-online/proceedings/psb03/schwartz.pdf", ""]
    (output/"核验摘要.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=default_root())
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--doc-id",nargs="+")
    parser.add_argument("--decisions",type=Path)
    parser.add_argument("--refresh-review",action="store_true",help="保留候选和统计，仅按决定文件刷新核验/测试生效表；旧核验保存为历史")
    args = parser.parse_args()
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))["decisions"] if args.decisions else []
    if args.refresh_review:
        if args.decisions is None:parser.error("刷新核验必须提供 --decisions")
        output=args.output.resolve()
        extra=json.loads((output/"summary.json").read_text(encoding="utf-8"))
        db=sqlite3.connect(output/"terminology.sqlite3");db.row_factory=sqlite3.Row
        history={"verification":[dict(r) for r in db.execute("SELECT * FROM verification WHERE decision!='pending'")],
                 "active_terms":[dict(r) for r in db.execute("SELECT * FROM active_terms")]}
        atomic_json(output/("review_history_"+utc_now().replace(":","-")+".json"),history)
        db.execute("DELETE FROM active_terms");db.execute("DELETE FROM verification")
        extra={k:extra[k] for k in ["documents_selected","segments_scanned","statistics"]}
        extra["statistics"]["alias_documents_not_counted"] = db.execute(
            "SELECT count(*) FROM run_documents WHERE status='alias_preserved_on_canonical'").fetchone()[0]
    else:
        db, extra = collect(args.root.resolve(),args.output.resolve(),args.doc_id)
    try:
        verify(db,decisions,args.root.resolve())
        summary = export(db,args.output.resolve(),extra)
        display={**summary,"statistics":{k:v for k,v in summary["statistics"].items() if k!="stopwords"}}
        print(json.dumps(display,ensure_ascii=False,indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
