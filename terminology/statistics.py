"""文档级 TF/DF/TF-IDF 与关键词统计；不把重叠 chunk 当作独立文档。"""
from collections import Counter
import json
import math
import re
import unicodedata


STOPWORDS = set("a an the of to and or in on for with by from as at is are was were be been being we our us this that these those it its their they them there here such can could may might will would should have has had do does did not also which who when where what how than then both each all any only more most other into over under using used use based given let respectively namely et al between through within without same while however therefore figure table section appendix eq equation reference references theorem lemma proof corollary assumption algorithm algorithms".split())
STOPWORDS.update({"如何", "怎样", "什么", "进行", "通过", "使用", "一个", "哪些", "是否"})


def tokens(text):
    for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", unicodedata.normalize("NFKC", text).casefold()):
        if "\u3400" <= token[0] <= "\u9fff":
            yield from ([token] if len(token) <= 2 else (token[i:i+2] for i in range(len(token)-1)))
        elif len(token) >= 2 and any(c.isalpha() for c in token):
            yield token


class CorpusStatistics:
    def __init__(self):
        self.documents, self.seen_sources = [], set()
        self.skipped_duplicate_sources = 0

    def add(self, document, segments):
        if document["status"] != "indexed":
            return
        if document["source_sha256"] in self.seen_sources:
            self.skipped_duplicate_sources += 1
            return
        counts = Counter()
        for segment in segments:
            if segment.get("kind") != "paragraph" or not segment.get("retrieval_eligible", True):
                continue
            if any("reference" in s.casefold() or "bibliograph" in s.casefold() for s in segment.get("section_path", [])):
                continue
            counts.update(tokens(segment.get("text", "")))
        if counts:
            self.seen_sources.add(document["source_sha256"])
            self.documents.append((dict(document), counts))

    def publish(self, db, top_k=20, *, commit=True):
        schema = """
          CREATE TABLE document_statistics(doc_id TEXT PRIMARY KEY,source_file TEXT,source_sha256 TEXT,
             token_count INTEGER,unique_term_count INTEGER,term_counts TEXT);
          CREATE TABLE term_statistics(term TEXT PRIMARY KEY,raw_frequency INTEGER,document_frequency INTEGER,
             idf REAL,tfidf_sum REAL,tfidf_max REAL,keyword_document_count INTEGER,is_stopword INTEGER);
          CREATE TABLE document_keywords(doc_id TEXT,rank INTEGER,term TEXT,raw_tf INTEGER,idf REAL,tfidf REAL,
             PRIMARY KEY(doc_id,rank));
        """
        for statement in schema.split(";"):
            if statement.strip():
                db.execute(statement)
        frequency, df = Counter(), Counter()
        for _, counts in self.documents:
            frequency.update(counts)
            df.update(counts.keys())
        number = len(self.documents)
        idf = {term: math.log((1+number)/(1+count))+1 for term,count in df.items()}
        sums, maxima, keyword_docs = Counter(), {}, Counter()
        for document, counts in self.documents:
            db.execute("INSERT INTO document_statistics VALUES (?,?,?,?,?,?)", (
                document["doc_id"],document["source_file"],document["source_sha256"],sum(counts.values()),len(counts),
                json.dumps(counts,ensure_ascii=False,sort_keys=True)))
            # 词频表保留停用词；TF-IDF 与关键词排名在去除停用词的词项空间内计算。
            scores = {term:(1+math.log(count))*idf[term] for term,count in counts.items() if term not in STOPWORDS}
            norm = math.sqrt(sum(value*value for value in scores.values()))
            scores = {term:value/norm for term,value in scores.items()} if norm else {}
            for term, value in scores.items():
                sums[term] += value
                maxima[term] = max(maxima.get(term,0),value)
            keywords = sorted(scores,key=lambda term:(-scores[term],term))[:top_k]
            for rank, term in enumerate(keywords,1):
                db.execute("INSERT INTO document_keywords VALUES (?,?,?,?,?,?)", (document["doc_id"],rank,term,counts[term],idf[term],scores[term]))
                keyword_docs[term] += 1
        db.executemany("INSERT INTO term_statistics VALUES (?,?,?,?,?,?,?,?)", [
            (term,count,df[term],idf[term],sums.get(term,0),maxima.get(term,0),keyword_docs.get(term,0),int(term in STOPWORDS))
            for term,count in sorted(frequency.items())])
        if commit:
            db.commit()
        return {"document_count":number,"vocabulary_size":len(frequency),"total_tokens":sum(frequency.values()),
                "duplicate_sources_skipped":self.skipped_duplicate_sources,"keyword_top_k":top_k,
                "tokenizer":"NFKC+casefold; English alphanumeric words (>=2, containing letters), Chinese overlapping bigrams",
                "source_scope":"indexed documents; eligible paragraph segments; reference sections excluded; deduplicate source SHA256",
                "tfidf":"tf=1+ln(raw_tf); idf=1+ln((N+1)/(df+1)); L2 normalize over non-stopword terms per document",
                "stopwords":sorted(STOPWORDS),"candidate_term_status":"statistical keywords are not verified aliases"}
