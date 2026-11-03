"""有界补审、联合证据组装、最终上下文放行；模型不能修改本地策略。"""
import hashlib
import json
import time

from .quality_gate import GateError, assessment_passes, validate_audit


def evaluate_context(audit, context, policy, observed_conflicts=()):
    retained = {h["evidence_id"] for h in context}
    covered, gaps = [], []
    for coverage in audit["coverage"]:
        if any(set(group) <= retained for group in coverage["support_sets"]):
            covered.append(coverage["subquestion_id"])
        else:
            gaps.append({"subquestion_id": coverage["subquestion_id"],
                         "missing": coverage["missing"] or ["充分证据组未完整进入最终上下文"]})
    support_ids = {a["evidence_id"] for a in audit["assessments"] if assessment_passes(a)}
    conflicts = list(observed_conflicts) or audit["conflicts"]
    if conflicts:
        decision = "conflict"
    elif audit["subquestions"] and len(covered) == len(audit["subquestions"]):
        decision = "sufficient"
    elif retained & support_ids:
        decision = "partial"
    else:
        decision = "insufficient"
    allowed = decision == "sufficient" or (decision == "partial" and policy.allow_partial_answers and bool(covered))
    return {"decision": decision, "answer_allowed": allowed,
            "answer_scope": covered if allowed else [], "covered_subquestions": covered,
            "gaps": gaps, "context_evidence_ids": [h["evidence_id"] for h in context],
            "context_complete": bool(audit["subquestions"]) and not gaps and not conflicts}


def select_gate_context(hits, audit, tokenizer, budget, top_k, selector, observed_conflicts=()):
    """优先放入完整联合证据组；保留冲突双方，绝不截断片段。"""
    assessments = {a["evidence_id"]: a for a in audit["assessments"]}
    conflicts = list(observed_conflicts) or audit["conflicts"]
    conflict_ids = {i for c in conflicts for i in c["evidence_ids"]}
    eligible = {h["evidence_id"]: {**h, "quality_assessment": assessments[h["evidence_id"]]}
                for h in hits if assessment_passes(assessments[h["evidence_id"]]) or h["evidence_id"] in conflict_ids}
    selected, count = [], 0

    def add_group(ids):
        nonlocal selected, count
        chosen = {h["evidence_id"] for h in selected}
        if any(i not in eligible for i in ids):
            return False
        trial = selected + [eligible[i] for i in ids if i not in chosen]
        if len(trial) > top_k:
            return False
        packed, tokens = selector(trial, tokenizer, budget)
        if len(packed) != len(trial):
            return False
        selected, count = packed, tokens
        return True

    for conflict in conflicts:
        add_group(conflict["evidence_ids"])
    for coverage in audit["coverage"]:
        chosen = {h["evidence_id"] for h in selected}
        groups = sorted(coverage["support_sets"], key=lambda group: len(set(group) - chosen))
        for group in groups:
            if add_group(group):
                break
    for hit in hits:
        if hit["evidence_id"] in eligible:
            add_group([hit["evidence_id"]])
    return selected, count


class QualityGateRunner:
    def __init__(self, gate, policy):
        self.gate, self.policy = gate, policy

    def run(self, query, hits, tokenizer, context_budget, top_k, selector):
        started = time.monotonic()
        deadline = started + self.policy.total_timeout_seconds
        policy = self.policy
        meta = {"status": "completed", "provider": "deepseek", "model": self.gate.options["model"],
                "validated": False, "schema_validated": False, "decision": "not_evaluated",
                "answer_allowed": False, "answer_scope": [], "policy_version": policy.version,
                "policy_sha256": hashlib.sha256(json.dumps(policy.model_dump(), sort_keys=True).encode()).hexdigest(),
                "rubric_sha256": self.gate.rubric_sha256, "attempts": [], "rounds": 0,
                "request_bytes": 0, "reserved_output_tokens": 0, "usage": {},
                "assessments": [], "subquestions": [], "coverage": [], "conflicts": [],
                "observed_conflicts": [], "reviewed_count": 0, "eligible_count": 0,
                "context_dropped_ids": [], "observed_conflict_assessments": {}, "quote_relocations": []}
        unique, seen_ids, seen_text = [], set(), set()
        for hit in hits:
            text = hit.get("text", "")
            if not text.strip() or hit.get("retrieval_eligible") is False:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if hit["evidence_id"] in seen_ids or digest in seen_text:
                continue
            seen_ids.add(hit["evidence_id"])
            seen_text.add(digest)
            unique.append(hit)
        meta["prefiltered_count"] = len(hits) - len(unique)
        pool, audit, questions = [], None, None
        context, count, stop_reason = [], 0, "no_candidates"
        limit = min(len(unique), policy.max_candidates)
        try:
            while len(pool) < limit and meta["rounds"] < policy.max_rounds:
                if time.monotonic() >= deadline:
                    stop_reason = "time_budget"
                    break
                next_pool = list(pool)
                stop_reason = "request_byte_budget"
                for hit in unique[len(pool):min(limit, len(pool) + policy.batch_size)]:
                    proposed = next_pool + [hit]
                    size = len(self.gate.request_bytes(query, proposed, questions))
                    if size > policy.max_request_bytes or meta["request_bytes"] + size > policy.max_total_request_bytes:
                        break
                    next_pool = proposed
                if len(next_pool) == len(pool):
                    break
                size = len(self.gate.request_bytes(query, next_pool, questions))
                response = None
                for retry in range(policy.retries_per_round + 1):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        stop_reason = "time_budget"
                        break
                    if len(meta["attempts"]) >= policy.max_api_attempts:
                        stop_reason = "attempt_budget"
                        break
                    if meta["reserved_output_tokens"] + self.gate.options["max_tokens"] > policy.max_total_output_tokens:
                        stop_reason = "output_token_budget"
                        break
                    if meta["request_bytes"] + size > policy.max_total_request_bytes:
                        stop_reason = "request_byte_budget"
                        break
                    attempt = {"round": meta["rounds"] + 1, "candidate_count": len(next_pool), "status": "started"}
                    meta["attempts"].append(attempt)
                    meta["request_bytes"] += size
                    meta["reserved_output_tokens"] += self.gate.options["max_tokens"]
                    attempt_started = time.monotonic()
                    try:
                        response = self.gate.assess(query, next_pool, fixed_subquestions=questions, timeout_seconds=remaining)
                        if time.monotonic() > deadline:
                            raise GateError("deadline_exceeded")
                        # 在本地边界再次验证，防止自定义适配器绕过契约。
                        if not isinstance(response, dict):
                            raise GateError("response_envelope_invalid")
                        relocations = []
                        audit_data = validate_audit({key: response[key] for key in
                            ("subquestions", "assessments", "coverage", "conflicts") if key in response},
                            next_pool, questions, quote_relocations=relocations)
                        response = {**response, **audit_data,
                                    "quote_relocations": response.get("quote_relocations", []) + relocations}
                        attempt["quote_relocations"] = response["quote_relocations"]
                        attempt['cache_hit'] = response.get('cache_hit', False)
                        attempt['token_metrics'] = response.get('token_metrics', {})
                        attempt["status"] = "completed"
                        break
                    except GateError as exc:
                        response = None
                        attempt.update(status="failed", **exc.diagnostics())
                        if not exc.retryable or retry == policy.retries_per_round:
                            raise
                        delay = policy.retry_backoff_seconds * (2 ** retry)
                        if time.monotonic() + delay >= deadline:
                            raise GateError("deadline_exceeded") from None
                        time.sleep(delay)
                    finally:
                        attempt["elapsed_ms"] = (time.monotonic() - attempt_started) * 1000
                if response is None:
                    # 已开始审查但未能完成重试，不能回退到上一轮放行。
                    if meta["attempts"] and meta["attempts"][-1]["status"] == "failed":
                        raise GateError("retry_budget_exhausted")
                    break
                pool, audit = next_pool, response
                questions = audit["subquestions"]
                meta["rounds"] += 1
                for key in ("subquestions", "assessments", "coverage", "conflicts", "quote_relocations"):
                    meta[key] = audit[key]
                for conflict in audit["conflicts"]:
                    if conflict not in meta["observed_conflicts"]:
                        meta["observed_conflicts"].append(conflict)
                    for item in audit["assessments"]:
                        if item["evidence_id"] in conflict["evidence_ids"]:
                            meta["observed_conflict_assessments"].setdefault(item["evidence_id"], item)
                usage = audit.get("usage")
                for key, value in (usage if isinstance(usage, dict) else {}).items():
                    if type(value) is int and value >= 0:
                        meta["usage"][key] = meta["usage"].get(key, 0) + value
                meta.update(schema_validated=True, reviewed_count=len(pool))
                conflict_ids = {i for c in meta["observed_conflicts"] for i in c["evidence_ids"]}
                meta["eligible_count"] = sum(assessment_passes(a) or a["evidence_id"] in conflict_ids
                                             for a in audit["assessments"])
                context, count = select_gate_context(pool, audit, tokenizer, context_budget, top_k, selector,
                                                     meta["observed_conflicts"])
                meta.update(evaluate_context(audit, context, policy, meta["observed_conflicts"]))
                if time.monotonic() > deadline:
                    raise GateError("deadline_exceeded")
                if meta["decision"] == "sufficient":
                    stop_reason = "sufficient_context"
                    break
                stop_reason = "round_budget"
            else:
                stop_reason = ("candidates_exhausted" if len(pool) == len(unique) else
                               "candidate_budget" if len(pool) >= limit else "round_budget")
        except (GateError, ValueError, KeyError, TypeError) as exc:
            diagnostics = exc.diagnostics() if isinstance(exc, GateError) else {
                "error_code": "internal_validation_error", "error_details": []}
            meta.update(status="failed", decision="review_failed", answer_allowed=False, answer_scope=[],
                        context_complete=False, context_evidence_ids=[], validated=False,
                        assessments=[], subquestions=[], coverage=[], conflicts=[], gaps=[],
                        observed_conflicts=[], observed_conflict_assessments={}, quote_relocations=[], **diagnostics)
            context, count, stop_reason = [], 0, "review_failed"
        if not unique:
            meta.update(decision="insufficient", context_complete=False, context_evidence_ids=[], gaps=[])
        elif audit is None and meta["status"] != "failed":
            meta.update(status="budget_exhausted", context_complete=False, context_evidence_ids=[], gaps=[])
        meta["validated"] = meta["status"] == "completed" and meta["decision"] == "sufficient"
        meta["stop_reason"] = stop_reason
        meta["candidate_scope_exhausted"] = len(pool) == len(unique)
        meta["unreviewed_count"] = len(unique) - len(pool)
        retained = {h["evidence_id"] for h in context}
        meta["context_dropped_ids"] = [h["evidence_id"] for h in pool if h["evidence_id"] not in retained]
        meta["elapsed_ms"] = (time.monotonic() - started) * 1000
        return context, count, meta
