"""DeepSeek 只判断证据语义；结构、原文引用及最终放行由本地代码校验。"""
import hashlib
import json
import os
from pathlib import Path
import re
from datetime import date

import httpx
from pydantic import ValidationError

from .schemas import GatePayload
from rag_core.api_accounting import APILedger, cache_key, quantify


def _schema_field_names(schema):
    """仅允许契约中的字段名进入诊断；未知字段名本身也可能包含敏感内容。"""
    fields = set()
    if isinstance(schema, dict):
        fields.update(schema.get("properties", {}))
        for value in schema.values():
            fields.update(_schema_field_names(value))
    elif isinstance(schema, list):
        for value in schema:
            fields.update(_schema_field_names(value))
    return fields


_SAFE_FIELDS = _schema_field_names(GatePayload.model_json_schema())


def _safe_details(details):
    # 只保留 loc/type，不输出 Pydantic 的 input、msg、ctx、url 等内容。
    return [{"loc": [part if type(part) is int or part in _SAFE_FIELDS else "<extra>"
                     for part in item.get("loc", ())],
             "type": item["type"] if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", item.get("type", ""))
                     else "validation_error"} for item in details[:20]]


class GateError(RuntimeError):
    """可安全返回给调用者的错误，不包含响应、密钥、URL 或原文。"""

    def __init__(self, code, retryable=False, details=()):
        self.code, self.retryable = code, retryable
        self.details = _safe_details(details)
        suffix = " " + json.dumps(self.details, ensure_ascii=False) if self.details else ""
        super().__init__(f"DeepSeek 门控失败（{code}）；没有将证据标记为通过{suffix}")

    def diagnostics(self):
        return {"error_code": self.code, "error_details": self.details}


def _fail(code, *loc):
    raise GateError(code, details=[{"loc": loc, "type": code}])


def _parse_json(data, prefix="json"):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise GateError(prefix + "_duplicate_key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise GateError(prefix + "_invalid")

    try:
        return json.loads(data, object_pairs_hook=pairs_hook, parse_constant=invalid_constant)
    except (ValueError, UnicodeDecodeError, TypeError):
        raise GateError(prefix + "_invalid") from None


def assessment_passes(item):
    """可用于支持回答的片段；部分支持可参与联合证据组。"""
    return (item["relation"] in {"direct_support", "partial_support"}
            and bool(item["support_spans"])
            and all(item[k] in {"supported", "not_applicable"}
                    for k in ("applicability", "time_scope", "conditions")))


def _align_quote(span, text, loc, relocations):
    """只修正唯一逐字匹配的偏移；不归一化文本，不选择最近的重复匹配。"""
    start, end, quote = span["start"], span["end"], span["quote"]
    in_bounds = 0 <= start < end <= len(text)
    if in_bounds and text[start:end] == quote:
        return
    found = text.find(quote)
    if found < 0:
        _fail("quote_mismatch" if in_bounds else "quote_out_of_bounds", *loc, "quote")
    # 从 found+1 搜索，重叠出现（例如 aaa 中的 aa）也算多处匹配。
    if text.find(quote, found + 1) >= 0:
        _fail("quote_ambiguous", *loc, "quote")
    span.update(start=found, end=found + len(quote))
    if relocations is not None:
        relocations.append({"loc": list(loc), "original_start": start, "original_end": end,
                            "start": span["start"], "end": span["end"]})


def validate_audit(data, hits, fixed_subquestions=None, *, quote_relocations=None):
    """跨记录校验。原文位置正确并不意味着模型的语义判断一定正确。"""
    if isinstance(data, (str, bytes, bytearray)):
        data = _parse_json(data)
    try:
        parsed = GatePayload.model_validate(data)
    except ValidationError as exc:
        raise GateError("schema_invalid", details=exc.errors(
            include_url=False, include_context=False, include_input=False)) from None
    audit = parsed.model_dump()
    expected = {h["evidence_id"]: h for h in hits}
    ids = [a["evidence_id"] for a in audit["assessments"]]
    if len(expected) != len(hits):
        _fail("input_id_duplicate")
    if len(set(ids)) != len(ids):
        _fail("evidence_id_duplicate", "assessments")
    if set(ids) - set(expected):
        _fail("evidence_id_unknown", "assessments")
    if set(expected) - set(ids):
        _fail("evidence_id_missing", "assessments")
    questions = audit["subquestions"]
    qids = {q["subquestion_id"] for q in questions}
    if len(qids) != len(questions):
        _fail("subquestion_id_duplicate", "subquestions")
    if fixed_subquestions is not None and questions != fixed_subquestions:
        _fail("subquestions_changed", "subquestions")
    coverage_ids = [c["subquestion_id"] for c in audit["coverage"]]
    if len(set(coverage_ids)) != len(coverage_ids):
        _fail("coverage_subquestion_duplicate", "coverage")
    if set(coverage_ids) - qids:
        _fail("coverage_subquestion_unknown", "coverage")
    if qids - set(coverage_ids):
        _fail("coverage_subquestion_missing", "coverage")
    assessments = {a["evidence_id"]: a for a in audit["assessments"]}
    for index, item in enumerate(audit["assessments"]):
        text = expected[item["evidence_id"]]["text"]
        for span_index, span in enumerate(item["support_spans"]):
            _align_quote(span, text, ("assessments", index, "support_spans", span_index), quote_relocations)
        if item["relation"] in {"direct_support", "partial_support", "contradicts"} and not item["support_spans"]:
            _fail("quote_missing", "assessments", index, "support_spans")
    for index, coverage in enumerate(audit["coverage"]):
        groups = coverage["support_sets"]
        if bool(groups) == bool(coverage["missing"]):
            _fail("coverage_missing_inconsistent", "coverage", index)
        references = list(coverage["partial_evidence_ids"])
        if len(set(references)) != len(references):
            _fail("coverage_evidence_duplicate", "coverage", index, "partial_evidence_ids")
        for group_index, group in enumerate(groups):
            if not group:
                _fail("coverage_group_empty", "coverage", index, "support_sets", group_index)
            if len(set(group)) != len(group):
                _fail("coverage_group_duplicate", "coverage", index, "support_sets", group_index)
            references.extend(group)
        if any(i not in assessments for i in references):
            _fail("coverage_evidence_unknown", "coverage", index)
        if any(not assessment_passes(assessments[i]) for i in references):
            _fail("coverage_evidence_ineligible", "coverage", index)
    contrary = set()
    for index, conflict in enumerate(audit["conflicts"]):
        refs = conflict["evidence_ids"]
        if conflict["subquestion_id"] not in qids:
            _fail("conflict_subquestion_unknown", "conflicts", index, "subquestion_id")
        if len(set(refs)) != len(refs):
            _fail("conflict_evidence_duplicate", "conflicts", index, "evidence_ids")
        if not set(refs) <= set(expected):
            _fail("conflict_evidence_unknown", "conflicts", index, "evidence_ids")
        if any(assessments[i]["relation"] not in {"direct_support", "partial_support", "contradicts"} for i in refs):
            _fail("conflict_evidence_ineligible", "conflicts", index, "evidence_ids")
        found = {i for i in refs if assessments[i]["relation"] == "contradicts"}
        if not found:
            _fail("conflict_contrary_missing", "conflicts", index, "evidence_ids")
        contrary.update(found)
    if contrary != {i for i, a in assessments.items() if a["relation"] == "contradicts"}:
        _fail("conflict_unreported", "conflicts")
    return audit


class DeepSeekQualityGate:
    def __init__(self, options, transport=None, rubric=None, *, cache_path=None, tokenizer_file=None):
        self.options, self.transport = options, transport
        self.ledger = APILedger(cache_path)
        self.tokenizer_file = tokenizer_file
        self.rubric = rubric if rubric is not None else (
            Path(__file__).resolve().parents[1] / "config/quality_gate_rubric.md").read_text(encoding="utf-8")
        self.rubric_sha256 = hashlib.sha256(self.rubric.encode("utf-8")).hexdigest()
        if type(options["max_tokens"]) is not int or options["max_tokens"] < 1 or options["timeout_seconds"] <= 0:
            raise ValueError("门控输出预算和超时必须为正数")

    def require_configured(self):
        if not os.environ.get(self.options["api_key_env"]):
            raise GateError("not_configured")

    def build_request(self, query, hits, fixed_subquestions=None):
        def compact_schema(value):
            if isinstance(value, dict):
                return {k: compact_schema(v) for k, v in value.items() if k != 'title'}
            if isinstance(value, list):
                return [compact_schema(v) for v in value]
            return value
        evidence = []
        for hit in hits:
            metadata = {key: hit[key] for key in (
                "title", "section", "source_type", "source_kind", "year", "version") if key in hit}
            evidence.append({"evidence_id": hit["evidence_id"], "text": hit["text"], "metadata": metadata})
        return {"model": self.options["model"], "stream": False, "temperature": 0,
                "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
                "max_tokens": self.options["max_tokens"], "messages": [
                    {"role": "system", "content": self.rubric +
                        '\nUse compact JSON only. Keep reasons short; quote the shortest sufficient verbatim spans.\nJSON schema:\n' + json.dumps(
                        compact_schema(GatePayload.model_json_schema()), ensure_ascii=False, separators=(',', ':'))},
                    {"role": "user", "content": json.dumps({"query": query,
                        "evidence": evidence, "fixed_subquestions": fixed_subquestions}, ensure_ascii=False, separators=(',', ':'))}]}

    def request_bytes(self, query, hits, fixed_subquestions=None):
        # 与实际发送相同的序列化方式；字节上限不是 DeepSeek token 计数。
        return self._encode(self.build_request(query, hits, fixed_subquestions))

    @staticmethod
    def _encode(payload):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def assess(self, query, hits, fixed_subquestions=None, timeout_seconds=None):
        self.require_configured()
        encoded = self.request_bytes(query, hits, fixed_subquestions)
        payload = json.loads(encoded)
        key = cache_key([self.options['base_url'], payload, date.today().isoformat()])
        metrics = quantify(payload, self.tokenizer_file)
        metrics['wire_protocol'] = 'quality_gate_v2'
        metrics['evidence_count'] = len(hits)
        cached = self.ledger.get(key, self.options.get('cache_ttl_seconds', 86400))
        if cached:
            audit = validate_audit(cached['audit'], hits, fixed_subquestions)
            self.ledger.record('quality_gate', self.options['model'], key, metrics, hit=True)
            return {'status': 'completed', 'provider': 'deepseek', 'model': self.options['model'],
                    **audit, 'quote_relocations': [], 'usage': {}, 'cache_hit': True, 'token_metrics': metrics}
        usage_for_accounting = {}
        try:
            timeout = min(self.options["timeout_seconds"], timeout_seconds) if timeout_seconds is not None else self.options["timeout_seconds"]
            with httpx.Client(timeout=timeout, transport=self.transport) as client:
                response = client.post(self.options["base_url"].rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + os.environ[self.options["api_key_env"]],
                             "Content-Type": "application/json"}, content=encoded)
                if response.status_code == 429 or response.status_code >= 500:
                    raise GateError(f"http_{response.status_code}", retryable=True)
                response.raise_for_status()
                body = _parse_json(response.content, prefix="response_json")
                if isinstance(body, dict):
                    usage_for_accounting = body.get('usage', {})
            if (not isinstance(body, dict) or not isinstance(body.get("choices"), list)
                    or not body["choices"] or not isinstance(body["choices"][0], dict)):
                raise GateError("response_envelope_invalid")
            choice = body["choices"][0]
            if not isinstance(choice.get("finish_reason"), str):
                raise GateError("response_envelope_invalid")
            if choice.get("finish_reason") != "stop":
                raise GateError("incomplete_response", retryable=True)
            if not isinstance(choice.get("message"), dict) or not isinstance(choice["message"].get("content"), str):
                raise GateError("response_envelope_invalid")
            content = choice["message"]["content"]
            if not content.strip():
                raise GateError("empty_response", retryable=True)
            relocations = []
            audit = validate_audit(content, hits, fixed_subquestions, quote_relocations=relocations)
            self.ledger.put(key, {'audit': audit})
            self.ledger.record('quality_gate', self.options['model'], key, metrics, body.get('usage', {}))
            return {"status": "completed", "provider": "deepseek", "model": self.options["model"],
                    **audit, "quote_relocations": relocations, "usage": body.get("usage", {}),
                    'cache_hit': False, 'token_metrics': metrics}
        except GateError:
            self.ledger.record('quality_gate', self.options['model'], key, metrics, usage_for_accounting, error='response_validation_failed')
            raise
        except httpx.TimeoutException:
            self.ledger.record('quality_gate', self.options['model'], key, metrics, error='timeout')
            raise GateError("timeout", retryable=True) from None
        except httpx.HTTPStatusError as exc:
            self.ledger.record('quality_gate', self.options['model'], key, metrics, error='http_' + str(exc.response.status_code))
            raise GateError(f"http_{exc.response.status_code}") from None
        except httpx.TransportError:
            self.ledger.record('quality_gate', self.options['model'], key, metrics, error='transport_error')
            raise GateError("transport_error", retryable=True) from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise GateError("response_envelope_invalid") from None
