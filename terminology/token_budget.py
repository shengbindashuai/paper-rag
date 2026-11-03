"""离线对照用量与压缩请求；不调用模型 API。"""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import copy

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.config import default_root, config_file
from rag_core.io import atomic_json, utc_now
from rag_core.api_accounting import quantify, tokenizer
from terminology.semantic_review import make_payload, normalized


def quantify_saved(root):
    root = root.resolve()
    directory = root / 'RAG/terminology/auto'
    db = sqlite3.connect((directory / 'semantic_audit.sqlite3').as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    requests = {r['request_id']: dict(r) for r in db.execute("SELECT * FROM api_requests WHERE status='completed'")
                if not json.loads(r['usage']).get('wire_protocol')}
    grouped = defaultdict(list)
    eligible = []
    for row in db.execute('SELECT * FROM pair_audits'):
        usage = json.loads(row['usage'])
        if usage.get('request_id') in requests:
            grouped[usage['request_id']].append(dict(row))
        if row['status'] in {'validated_approved', 'validated_rejected', 'needs_review'}:
            eligible.append(dict(row))
    usage = defaultdict(int)
    for row in requests.values():
        for key, value in json.loads(row['usage']).items():
            if type(value) is int:
                usage[key] += value
    token_path = root / 'RAG/models/Qwen3-Embedding-0.6B/tokenizer.json'
    local = tokenizer(str(token_path))
    before_input, after_input, sample_actual, sample_pairs = 0, 0, 0, 0
    rows_compared = 0
    config = {'model': 'deepseek-flash', 'max_tokens': 4096}
    for request_id, rows in grouped.items():
        if len(rows) != requests[request_id]['pair_count']:
            continue
        items = [json.loads(row['request']) for row in rows]
        before = make_payload(items, {**config, 'max_tokens': 8192}, legacy=True)
        after = make_payload(items, config)
        before_input += quantify(before, token_path)['estimated_input_tokens']
        after_input += quantify(after, token_path)['estimated_input_tokens']
        sample_actual += json.loads(requests[request_id]['usage']).get('prompt_tokens', 0)
        sample_pairs += len(items)
        rows_compared += 1
    before_output, after_output = 0, 0
    for row in eligible:
        item, review = json.loads(row['request']), json.loads(row['review'])
        old = json.dumps({'reviews': [review]}, ensure_ascii=False)
        verdict = {'approve': 0, 'reject': 1, 'needs_review': 2}[review['verdict']]
        lf = None if normalized(review['long_form']) == normalized(item['extracted_long_form']) else review['long_form']
        proof = review['evidence'][0] if review['evidence'] else None
        code = (0 if lf is None else 1) if verdict == 0 else 3 if verdict == 1 else 5
        compact = {'i': 0, 'v': verdict, 'c': code}
        if verdict == 0:
            compact['s'] = review['scope']
            if lf is not None:
                compact['l'] = lf
        if proof:
            compact.update(e=0, q=proof['quote'])
        new = json.dumps({'r': [compact]}, ensure_ascii=False, separators=(',', ':'))
        before_output += len(local.encode(old).ids)
        after_output += len(local.encode(new).ids)
    total_pairs = sum(r['pair_count'] for r in requests.values())
    data = {'created_at': utc_now(), 'external_requests_made': 0,
        'historical_actual': {'successful_requests': len(requests), 'pairs_in_successful_requests': total_pairs,
            **usage, 'mean_input_tokens_per_pair': usage['prompt_tokens'] / total_pairs,
            'mean_output_tokens_per_pair': usage['completion_tokens'] / total_pairs},
        'paired_local_estimate': {'tokenizer': 'Qwen3 proxy, not exact DeepSeek tokenizer',
            'matched_complete_batches': rows_compared, 'matched_pairs': sample_pairs,
            'input_before': before_input, 'input_after': after_input,
            'input_reduction_percent': 100 * (1 - after_input / before_input),
            'output_pairs': len(eligible), 'output_before': before_output, 'output_after': after_output,
            'output_reduction_percent': 100 * (1 - after_output / before_output),
            'calibrated_new_input_per_pair': sample_actual * after_input / before_input / sample_pairs,
            'note': '输出对照由已保存审查结果重建，不是新模型实际输出；新协议以随后 API usage 为准'},
        'cache_policy': {'terminology': 'semantic input + model + policy; revalidate source locally; reuse legacy validated results',
            'quality_gate': 'exact query + evidence + semantic metadata + model + schema/rubric + date; persistent validated audit cache',
            'provider_prompt_cache': 'stable system prefix; monitor provider hit/miss counters; no promised hit rate'}}
    # 门控另取本地 16 个真实片段作离线对照，不运行检索或发出 API 请求。
    from retrieval.quality_gate import DeepSeekQualityGate
    from retrieval.config import RetrievalConfig
    from retrieval.schemas import GatePayload
    source = sqlite3.connect((root / 'RAG/library/library.sqlite3').as_uri() + '?mode=ro', uri=True)
    hits = []
    for doc_id in ['P0011', 'P0022']:
        title = source.execute('SELECT title FROM documents WHERE doc_id=?', (doc_id,)).fetchone()[0]
        for row in source.execute('SELECT data FROM chunks WHERE doc_id=? ORDER BY id LIMIT 8', (doc_id,)):
            hit = json.loads(row[0])
            hit.update(evidence_id=hit['doc_id'] + ':' + hit['chunk_id'], title=title)
            hits.append(hit)
    source.close()
    gate = DeepSeekQualityGate(RetrievalConfig.load(config_file(root, "retrieval.json"), root).deepseek)
    query = 'PPO 如何重复利用采样数据？'
    new_gate = gate.build_request(query, hits)
    old_gate = copy.deepcopy(new_gate)
    old_gate['messages'][0]['content'] = gate.rubric + '\nJSON schema:\n' + json.dumps(GatePayload.model_json_schema(), ensure_ascii=False)
    old_gate['messages'][1]['content'] = json.dumps({'query': query, 'fixed_subquestions': None, 'evidence': [
        {'evidence_id': hit['evidence_id'], 'text': hit['text'], 'metadata': {key: hit[key] for key in (
            'title', 'section', 'pages', 'source_file', 'source_type', 'source_kind', 'line_start', 'line_end',
            'source_spans', 'year', 'version', 'snapshot_revision') if key in hit}} for hit in hits]}, ensure_ascii=False)
    old_metrics, new_metrics = quantify(old_gate, token_path), quantify(new_gate, token_path)
    data['quality_gate_local_example'] = {'chunks': len(hits), 'actual_api_calls': 0,
        'input_before_proxy': old_metrics['estimated_input_tokens'], 'input_after_proxy': new_metrics['estimated_input_tokens'],
        'input_reduction_percent': 100 * (1 - new_metrics['estimated_input_tokens'] / old_metrics['estimated_input_tokens']),
        'note': '本地片段示例，不是实际检索批次；不是 DeepSeek 实测 token'}
    atomic_json(directory / 'token量化.json', data)
    actual, estimate = data['historical_actual'], data['paired_local_estimate']
    lines = ['# API token 量化与缓存优化', '', '本报告离线生成，没有发起模型请求。', '',
        '## 已发生的真实用量（DeepSeek 返回）', '',
        f"成功请求 {actual['successful_requests']} 批，涵盖 {total_pairs} 个批次内配对记录（包含重试，非去重配对数）。",
        f"输入 {actual['prompt_tokens']:,} token；输出 {actual['completion_tokens']:,} token；合计 {actual['total_tokens']:,} token。",
        f"每配对平均输入 {actual['mean_input_tokens_per_pair']:.1f}、输出 {actual['mean_output_tokens_per_pair']:.1f} token。",
        f"服务端前缀缓存命中 {actual.get('prompt_cache_hit_tokens', 0):,} token，命中率 {100 * actual.get('prompt_cache_hit_tokens', 0) / max(actual['prompt_tokens'], 1):.2f}%。",
        '', '## 同样本离线对照（估算）', '',
        f"输入对照：{rows_compared} 个完整历史批次、{sample_pairs} 个配对。Qwen3 代理计数估算输入减少 {estimate['input_reduction_percent']:.1f}%。",
        f"输出对照：{len(eligible)} 个已保存结论。改为短键对象、枚举理由和单条充分引用，代理计数估算输出减少 {estimate['output_reduction_percent']:.1f}%。",
        f"按历史实际输入量校准，新输入约 {estimate['calibrated_new_input_per_pair']:.1f} token/配对，仅为预测。",
        '本地 tokenizer 不是 DeepSeek 精确 tokenizer，输出为重建对照；新接口返回的 usage 才是实测值。',
        '', '## 两处模型 API 均处理', '',
        '- 术语核验：外发仅有短序号、缩写、全称、其他含义及原文；来源哈希、ID、坐标和区间留在本地。短键对象允许省略未用字段，避免固定数组缺列造成重试。',
        '- 质量门控：保留语义相关的标题、章节、来源类型、年份/版本；移除路径、坐标和页码等定位信息。',
        '- 术语缓存按语义输入命中，重新绑定当前来源并校验；不因无关元数据变化重复付费。',
        '- 门控缓存仅保存通过本地校验的审查，命中后仍重新校验当前证据，实际新增 API token 为零。',
        f"- 门控另取 {len(hits)} 个本地片段离线对照：代理输入 {old_metrics['estimated_input_tokens']:,} → {new_metrics['estimated_input_tokens']:,} token（估算，不是线上实测）。",
        '- 固定系统前缀、紧凑 JSON，台账记录实际用量、服务端缓存命中/未命中与本地缓存命中。',
        '- 模型文件下载不是 token API，沿用文件哈希校验与断点缓存，按下载字节统计。',
        '', '完整数据见 token量化.json；后续调用台账见 RAG/library/api_accounting.sqlite3。', '']
    (directory / 'token量化.md').write_text('\n'.join(lines), encoding='utf-8')
    db.close()
    return data


def update_live_measurement(root):
    """在固定离线基线后追加新协议实测，不把预测冒充实测。"""
    directory = root / 'RAG/terminology/auto'
    data = json.loads((directory / 'token量化.json').read_text(encoding='utf-8'))
    db = sqlite3.connect(root / 'RAG/library/api_accounting.sqlite3')
    totals, successful, failed = defaultdict(int), 0, 0
    for raw_metrics, raw_usage, error in db.execute("SELECT request_metrics,usage,error FROM calls WHERE kind='terminology' AND cache_hit=0"):
        metrics, usage = json.loads(raw_metrics), json.loads(raw_usage)
        if not metrics.get('wire_protocol', '').startswith('compact_v'):
            continue
        if error:
            failed += 1
        else:
            successful += 1
        for key, value in usage.items():
            if type(value) is int:
                totals[key] += value
    db.close()
    actual = {'recorded_successful_calls': successful, 'recorded_failed_calls': failed,
              **totals, 'includes_retries': True,
              'provider_cache_hit_percent': 100 * totals['prompt_cache_hit_tokens'] / max(totals['prompt_tokens'], 1)}
    data['optimized_actual'] = actual
    atomic_json(directory / 'token量化.json', data)
    path = directory / 'token量化.md'
    text = path.read_text(encoding='utf-8').split('\n## 精简协议实测')[0]
    text += (f"\n## 精简协议实测\n\n已记录成功 {successful} 次、失败 {failed} 次调用。"
             f"输入 {totals['prompt_tokens']:,}、输出 {totals['completion_tokens']:,} token；包含格式重试，未隐藏额外用量。\n\n"
             f"服务端前缀命中 {totals['prompt_cache_hit_tokens']:,} token，占已记录输入 {actual['provider_cache_hit_percent']:.2f}%。"
             '这不是未来保证命中率；本地结果缓存命中不产生新模型 token。\n')
    path.write_text(text, encoding='utf-8')
    return actual


if __name__ == '__main__':
    print(json.dumps(quantify_saved(default_root()), ensure_ascii=False, indent=2))
