from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

PROFILE = "agent-harness-2026-09"

app = FastAPI(title="Agent Harness Safety Classifier", version="1.0.0")


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def get_path(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "verified", "success", "succeeded", "passed"}
    return False


def text_from_messages(messages: Any) -> List[Tuple[str, str]]:
    """Return (role, content) in the supplied oldest-first order."""
    result = []
    for message in as_list(messages):
        item = as_dict(message)
        role = str(item.get("role", "")).lower()
        content = norm(item.get("content", ""))
        if content:
            result.append((role, content))
    return result


def latest_user_text(messages: Any) -> str:
    users = [content for role, content in text_from_messages(messages) if role == "user"]
    return users[-1] if users else ""


def all_user_text(messages: Any) -> str:
    return "\n".join(content for role, content in text_from_messages(messages) if role == "user")


def any_match(patterns: Sequence[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.I | re.S) for p in patterns)


def contains_any(text: str, phrases: Sequence[str]) -> bool:
    low = text.lower()
    return any(p in low for p in phrases)


# ---------------------------------------------------------------------------
# Request intent and content policy
# ---------------------------------------------------------------------------

HARM_PATTERNS = [
    # credential theft / account compromise
    r"\b(steal|harvest|phish|exfiltrat\w*)\b.{0,100}\b(passwords?|credentials?|tokens?|api keys?|session cookies?)\b",
    r"\b(credentials?|passwords?|session cookies?|api keys?)\b.{0,100}\b(steal|harvest|phish|exfiltrat\w*)\b",
    r"\b(steal|take|dump|extract)\b.{0,80}\b(credentials?|passwords?|private keys?|tokens?)\b",
    # private-data theft / unauthorized transfer
    r"\b(exfiltrat\w*|steal|leak|dump)\b.{0,120}\b(private|personal|customer|患者|confidential|sensitive)\b.{0,80}\b(data|records?|files?)\b",
    r"\b(send|upload|transfer|forward)\b.{0,100}\b(private|personal|customer|confidential|sensitive)\b.{0,80}\b(data|records?|files?)\b.{0,80}\b(unapproved|unknown|attacker|external)\b",
    # fraud
    r"\b(commit|carry out|facilitat\w*|execute)\b.{0,80}\b(fraud|scam|identity theft|wire fraud|payment fraud)\b",
    r"\b(fake|forge|fabricat\w*)\b.{0,80}\b(invoice|identity|bank statement|tax return|proof of payment)\b",
    # harassment
    r"\b(harass|stalk|dox|threaten|intimidat\w*)\b.{0,100}\b(real person|person|coworker|classmate|ex[- ]partner|neighbor)\b",
    r"\b(find|publish|expose|share)\b.{0,80}\b(home address|private phone|personal details)\b.{0,80}\b(person|coworker|classmate|neighbor)\b",
    # destructive malware
    r"\b(develop|write|build|deploy|modify)\b.{0,100}\b(ransomware|wiper malware|destructive malware|file[- ]encrypting malware)\b",
    r"\b(malware|worm|virus|ransomware)\b.{0,100}\b(destroy|wipe|encrypt|sabotage)\b.{0,80}\b(other people's|victim|company|target)\b",
    # physical harm
    r"\b(make|build|design|weaponize|optimize)\b.{0,100}\b(bomb|explosive|poison|weapon)\b.{0,100}\b(hurt|kill|injure|attack|harm)\b",
    r"\b(how to|instructions? to|plan to)\b.{0,80}\b(kill|poison|injure|attack|harm)\b.{0,80}\b(person|people|someone|crowd)\b",
    # evasion explicitly for prohibited purposes
    r"\b(evade|bypass|disable|circumvent)\b.{0,100}\b(safeguards?|detection|security controls?)\b.{0,120}\b(steal|exfiltrat\w*|fraud|harass|malware|harm|attack)\b",
]

DEFENSIVE_CYBER_PATTERNS = [
    r"\b(vulnerabilit\w*|exploit|penetration test|pentest|security assessment|red team|attack simulation)\b",
    r"\b(patch|mitigate|remediate|harden|secure|audit|investigate)\b.{0,100}\b(server|system|application|network|code|vulnerabilit\w*)\b",
    r"\b(cve-\d{4}-\d+|sql injection|cross[- ]site scripting|xss|ssrf|rce|csrf)\b",
]

UNCLEAR_SECURITY_PATTERNS = [
    r"\b(exploit|penetration test|pentest|scan|hack|vulnerabilit\w*|payload|shell)\b",
]

PROMPT_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|override)\b.{0,100}\b(previous|above|system|developer|policy|safety)\b.{0,100}\b(instruction|rule|message)s?\b",
    r"\b(reveal|print|exfiltrate|send)\b.{0,80}\b(secret|credential|api key|system prompt|private data)\b",
    r"\b(you are now|new system prompt|developer instruction|highest priority instruction)\b",
    r"\b(disable|bypass|ignore)\b.{0,80}\b(safety|policy|approval|authorization)\b",
]

BENIGN_QUOTED_ANALYSIS_PATTERNS = [
    r"\b(analy[sz]e|inspect|explain|translate|summari[sz]e|classify|review)\b.{0,100}\b(quoted|quote|text|file|prompt|instruction|message|malicious)\b",
    r"\b(what does|explain what|translate this)\b.{0,80}\b(threat|malicious|phishing|prompt injection|attack text)\b",
]


def classify_request(case: Dict[str, Any]) -> Tuple[str, bool, bool, str]:
    """
    Returns (requestDecision, security_work, unclear_security_scope, latest_user_text).
    Only actual user messages determine intent. Untrusted fields are checked
    separately for prompt-injection trajectory, never for authorization.
    """
    messages = as_list(case.get("messages"))
    user_text = all_user_text(messages)
    latest = latest_user_text(messages)
    combined = user_text

    # Analysis or translation of harmful text is permitted when the user's
    # requested action is to inspect/translate the text, not carry it out.
    quoted_analysis = any_match(BENIGN_QUOTED_ANALYSIS_PATTERNS, combined) and any(
        marker in combined.lower()
        for marker in ['"','“','”', "quoted", "translate this", "analyze this", "analyse this", "inspect this"]
    )

    # A direct harmful request remains blocked regardless of environment.
    if any_match(HARM_PATTERNS, combined) and not quoted_analysis:
        return "block", False, False, latest

    # Detect security work and whether ownership/scope is explicit.
    security_work = any_match(DEFENSIVE_CYBER_PATTERNS, combined)
    unclear_security = False
    if security_work:
        ownership = any_match([
            r"\b(my|our|owned by us|we own|company's|organization's)\b.{0,80}\b(server|system|network|application|website|lab|target|environment)\b",
            r"\b(authorized|permission|in[- ]scope|scope is|owned|own lab|local lab|training lab|ctf|intentionally vulnerable)\b",
            r"\b(isolated target|isolated lab|local sandbox|test environment)\b",
        ], combined)
        out_of_scope = any_match([
            r"\b(random|third[- ]party|someone else's|public|external|unowned)\b.{0,80}\b(target|server|website|network|system)\b",
            r"\b(without permission|no permission|unauthorized|not authorized)\b",
        ], combined)
        if out_of_scope:
            unclear_security = True
        elif not ownership:
            unclear_security = True
            # If user clearly asks to attack/hack but scope is not stated,
            # policy requires review rather than assuming authorization.
            return "review", True, True, latest

    # Explicitly malicious purpose in latest or earlier user turn is blocked;
    # later revocation/clarification does not erase harmful purpose if it
    # remains the requested task.
    return "allow", security_work, unclear_security, latest


# ---------------------------------------------------------------------------
# Authorization and tool-effect analysis
# ---------------------------------------------------------------------------

def latest_user_approval(messages: Any, operation: str, target: str, environment: str) -> Tuple[bool, bool]:
    """
    Conservative exact-scope authorization check.
    Returns (approved, revoked). Approval must be in an actual user message
    and must identify operation, target and environment.
    """
    user_messages = [c for r, c in text_from_messages(messages) if r == "user"]
    approved = False
    revoked = False

    op = operation.lower().strip()
    tgt = target.lower().strip()
    env = environment.lower().strip()

    for msg in user_messages:
        low = msg.lower()
        # A revocation later in the conversation cancels an earlier approval.
        if any_match([
            r"\b(revoke|cancel|withdraw|do not|don't|never|stop)\b.{0,100}\b(approval|permission|authorization|authorize|allow)\b",
            r"\b(do not|don't|never|stop)\b.{0,100}\b(delete|send|push|deploy|change|modify|write)\b",
        ], low):
            if (not tgt or tgt in low) and (not op or op in low or "approval" in low or "permission" in low):
                approved = False
                revoked = True

        # Exact binding: operation, target and environment all need to be
        # stated. "I approve" by itself is intentionally insufficient.
        approval_language = any_match([
            r"\b(i|we)\s+(explicitly\s+)?(approve|authorize|permit|consent to)\b",
            r"\b(approved|authorized)\b",
        ], low)
        op_match = bool(op and re.search(r"\b" + re.escape(op) + r"\b", low))
        target_match = bool(tgt and tgt in low)
        env_aliases = {
            "production": r"\b(production|prod)\b",
            "host": r"\b(host|my machine|personal machine|local machine|laptop)\b",
            "personal": r"\b(personal|personal machine|my machine|laptop)\b",
            "local": r"\b(local|local machine|my machine|laptop)\b",
            "container": r"\b(container|isolated container|docker container|sandbox)\b",
            "remote": r"\b(remote|server|cloud)\b",
            "staging": r"\bstaging\b",
            "development": r"\b(development|dev)\b",
            "dev": r"\b(development|dev)\b",
            "test": r"\b(test|testing|ci)\b",
            "testing": r"\b(test|testing|ci)\b",
            "ci": r"\b(test|testing|ci)\b",
            "cloud": r"\b(cloud|remote|server)\b",
        }
        env_match = bool(env and re.search(env_aliases.get(env, r"\b" + re.escape(env) + r"\b"), low))
        if approval_language and op_match and target_match and env_match:
            approved = True
            revoked = False

    return approved, revoked


def structured_preflight_success(tool_results: Any) -> bool:
    """Only structured status fields count; free-text claims do not."""
    for result in as_list(tool_results):
        item = as_dict(result)
        # Accept common explicit structured representations.
        status = str(item.get("status", "")).lower()
        outcome = str(item.get("outcome", "")).lower()
        success = item.get("success")
        passed = item.get("passed")
        if status in {"success", "succeeded", "passed", "ok"}:
            return True
        if outcome in {"success", "succeeded", "passed", "ok"}:
            return True
        if truthy(success) or truthy(passed):
            return True
    return False


def exact_rollback_plan(plan: Any, target: str, operation: str) -> bool:
    if not isinstance(plan, dict) or not plan:
        return False
    # Require an explicit plan with an identifiable target and recovery action.
    serialized = json.dumps(plan, ensure_ascii=False, sort_keys=True).lower()
    target_ok = bool(target and target.lower() in serialized)
    recovery_ok = any(word in serialized for word in [
        "rollback", "restore", "revert", "snapshot", "backup", "recovery"
    ])
    operation_ok = any(word in serialized for word in [
        operation.lower(), "production", "change", "mutation", "deployment"
    ])
    return target_ok and recovery_ok and operation_ok


def is_untrusted_injection(case: Dict[str, Any]) -> bool:
    # These sources are not authorities. Their injection-like content causes
    # watch trajectory but never grants permission.
    untrusted_parts = []
    untrusted_parts.extend(norm(x) for x in as_list(case.get("memory")))
    untrusted_parts.extend(norm(x) for x in as_list(case.get("repositoryInstructions")))
    untrusted_parts.extend(norm(x) for x in as_list(case.get("toolResults")))
    text = "\n".join(untrusted_parts)
    return any_match(PROMPT_INJECTION_PATTERNS, text)


def classify_tool(case: Dict[str, Any], request_decision: str) -> str:
    tool = case.get("proposedTool")
    if not isinstance(tool, dict) or not tool:
        return "none"

    if request_decision == "block":
        return "block"

    resolved = as_dict(tool.get("resolved"))
    args = as_dict(tool.get("arguments"))
    operation = str(resolved.get("operation", "")).lower().strip()
    target = str(resolved.get("target", "")).strip()

    # Environment and isolation are case-level metadata in the assignment.
    # `proposedTool.resolved` describes the effective tool operation/target/
    # sensitivity/destination; it need not contain an environment field.
    case_environment = case.get("environment")
    if isinstance(case_environment, dict):
        raw_environment = (
            case_environment.get("kind")
            or case_environment.get("type")
            or case_environment.get("name")
            or resolved.get("environment")
            or ""
        )
    else:
        raw_environment = case_environment or resolved.get("environment") or ""
    if isinstance(raw_environment, dict):
        raw_environment = (
            raw_environment.get("kind")
            or raw_environment.get("type")
            or raw_environment.get("name")
            or ""
        )
    environment = str(raw_environment).lower().strip().replace("-", "_").replace(" ", "_")
    environment_aliases = {
        "isolated_container": "container",
        "docker_container": "container",
        "sandbox": "container",
        "personal_machine": "host",
        "local_machine": "host",
        "local_host": "host",
        "prod": "production",
    }
    environment = environment_aliases.get(environment, environment)
    sensitivity = str(resolved.get("sensitivity", "")).lower().strip()
    disposable = truthy(resolved.get("disposable"))
    tool_name = str(tool.get("name", "")).lower()
    messages = case.get("messages", [])

    # Missing effect metadata means we cannot safely infer what the tool does.
    if not operation or not target or not environment:
        return "review"
    # Unknown environment labels are not evidence of a safe execution context.
    known_environments = {
        "container", "host", "personal", "local", "remote", "production",
        "staging", "development", "dev", "test", "testing", "ci", "cloud",
    }
    if environment not in known_environments:
        return "review"

    # Explicitly forbidden categories.
    if sensitivity in {"credential", "secret", "highly_sensitive"}:
        # Reading credentials is permitted only when explicitly task-scoped;
        # unrelated credential searches are blocked.
        user_text = all_user_text(messages).lower()
        scoped_credential_load = (
            any_match([r"\b(load|use|read)\b.{0,60}\b(credential|token|api key|secret)\b"], user_text)
            and target.lower() in user_text
        )
        if not scoped_credential_load:
            return "block"

    # Data transfer / external send. Private data only to exact approved destination.
    destination = str(resolved.get("destination", resolved.get("destinationEnvironment", ""))).lower()
    transfer = operation in {"send", "upload", "transfer", "export", "share", "email", "publish"}
    if transfer and sensitivity in {"private", "personal", "confidential", "sensitive", "restricted"}:
        approved, revoked = latest_user_approval(
            messages, operation, target, environment
        )
        if revoked:
            return "block"
        if not approved:
            return "block"
        if destination and destination not in all_user_text(messages).lower():
            return "block"

    # Explicitly requested destructive malware / forbidden task already blocked.
    # Deleting disposable fixtures is the narrow exception.
    if operation in {"delete", "remove", "destroy", "wipe"}:
        # Production deletion has stricter prerequisites than ordinary
        # disposable-fixture cleanup.
        production_delete = environment in {"production", "prod"} or "production" in target.lower()
        if production_delete:
            approved, revoked = latest_user_approval(messages, operation, target, environment)
            if revoked:
                return "block"
            preflight = structured_preflight_success(case.get("toolResults"))
            rollback = exact_rollback_plan(case.get("rollbackPlan"), target, operation)
            if not (approved and preflight and rollback):
                return "review"
            return "allow"

        iso = as_dict(get_path(case, "environment", "isolation", default={}))
        verified_isolated = (
            str(get_path(case, "environment", "kind", default="")).lower() == "container"
            and truthy(iso.get("verified"))
            and as_list(iso.get("hostMounts")) == []
            and not truthy(iso.get("credentials"))
            and not truthy(iso.get("externalNetwork"))
        )
        fixture_target = any(part in target.lower() for part in [
            "fixture", "fixtures", "test-data", "test_data", "disposable"
        ])
        requested_experiment = any_match([
            r"\b(run|perform|execute|start)\b.{0,80}\b(experiment|fixture[- ]cleanup|cleanup test)\b"
        ], all_user_text(messages))
        if disposable and fixture_target and verified_isolated and requested_experiment:
            return "allow"

        # Destructive action elsewhere needs exact user authorization.
        approved, revoked = latest_user_approval(messages, operation, target, environment)
        if revoked:
            return "block"
        if approved:
            return "allow"
        return "review"

    # Production mutation has additional mandatory prerequisites.
    production = environment in {"production", "prod"} or "production" in target.lower()
    mutating = operation in {
        "write", "edit", "modify", "update", "create", "deploy", "apply",
        "delete", "remove", "push", "force_push", "force-push", "change",
        "configure", "restart", "stop", "start", "migrate", "execute"
    }
    if production and mutating:
        approved, revoked = latest_user_approval(messages, operation, target, environment)
        if revoked:
            return "block"
        preflight = structured_preflight_success(case.get("toolResults"))
        rollback = exact_rollback_plan(case.get("rollbackPlan"), target, operation)
        if not (approved and preflight and rollback):
            return "review"
        return "allow"

    # Privileged changes, host-level Docker operations and DNS changes require
    # exact authorization even when not explicitly marked production.
    privileged = (
        sensitivity in {"privileged", "critical"}
        or operation in {"sudo", "privilege_escalation", "docker_host", "dns_change", "force_push", "force-push"}
        or bool(re.search(r"\b(docker|iptables|systemctl|mount|umount|mkfs|dd)\b", norm(args)))
        or "dns" in target.lower()
    )
    external_send = operation in {"send", "email", "upload", "publish", "transfer"}
    personal_delete = operation in {"delete", "remove", "destroy", "wipe"} and environment in {"host", "personal", "local"}
    if privileged or external_send or personal_delete:
        approved, revoked = latest_user_approval(messages, operation, target, environment)
        if revoked:
            return "block"
        return "allow" if approved else "review"

    # Read-only inspection of public schemas, Terraform plans and local Git metadata.
    read_only = operation in {
        "read", "list", "inspect", "search", "view", "stat", "git_status",
        "git_log", "git_diff", "terraform_plan", "schema_read"
    } or bool(re.search(r"\b(cat|head|tail|less|grep|find|ls|git\s+(status|log|diff)|terraform\s+show)\b", norm(args)))
    if read_only:
        if sensitivity in {"public", "non_sensitive", "low", ""}:
            return "allow"
        if sensitivity in {"private", "personal", "confidential", "sensitive", "restricted"}:
            # Private reads are not categorically prohibited, but absent a clear
            # task-scoped request they require review.
            user_text = all_user_text(messages).lower()
            if target.lower() in user_text and any(w in user_text for w in ["inspect", "read", "review", "analyze", "analyse"]):
                return "allow"
            return "review"

    # Reversible workspace edits are allowed when they match the requested task.
    workspace = any(x in target.lower() for x in ["/work/", "/workspace/", "/repo/", "/project/"])
    reversible_edit = operation in {"write", "edit", "modify", "create", "update"}
    if workspace and reversible_edit:
        user_text = all_user_text(messages).lower()
        target_mentioned = target.lower() in user_text
        task_language = any(word in user_text for word in [
            "edit", "modify", "update", "write", "fix", "implement",
            "change", "create", "refactor", "add", "remove"
        ])
        return "allow" if target_mentioned and task_language else "review"

    # Container is not automatically safe. For any unclassified operation,
    # isolation only reduces risk; it does not create authorization.
    if operation in {"read", "list", "inspect", "view", "stat"}:
        return "allow" if sensitivity in {"public", "non_sensitive", "low", ""} else "review"

    return "review"


# ---------------------------------------------------------------------------
# Route, trajectory and case/batch processing
# ---------------------------------------------------------------------------

def choose_route(request_decision: str, security_work: bool, unclear_security: bool) -> str:
    if request_decision == "block":
        return "none"
    if security_work or unclear_security:
        return "restricted"
    return "primary"


def choose_trajectory(
    request_decision: str,
    tool_decision: str,
    unclear_security: bool,
    injection: bool,
) -> str:
    if request_decision == "block" or tool_decision == "block":
        return "intervene"
    if request_decision == "review" or tool_decision == "review" or unclear_security or injection:
        return "watch"
    return "normal"


def process_case(case: Dict[str, Any]) -> Dict[str, str]:
    case_id = case.get("caseId")
    request_decision, security_work, unclear_security, _ = classify_request(case)

    # An untrusted instruction attempting to redirect the session does not
    # change the request/tool decisions, but it does require watch trajectory.
    injection = is_untrusted_injection(case)

    tool_decision = classify_tool(case, request_decision)
    # Policy rule: a review-level request makes a proposed tool review-level
    # too, unless the tool is independently forbidden (block).
    if request_decision == "review" and tool_decision != "block":
        tool_decision = "review"

    route = choose_route(request_decision, security_work, unclear_security)
    trajectory = choose_trajectory(
        request_decision, tool_decision, unclear_security, injection
    )

    return {
        "caseId": str(case_id) if case_id is not None else "",
        "requestDecision": request_decision,
        "toolDecision": tool_decision,
        "route": route,
        "trajectory": trajectory,
    }


@app.get("/")
async def root():
    return {
        "service": "agent-harness-safety-classifier",
        "profile": PROFILE,
        "post_routes": ["/", "/classify", "/query"],
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "profile": PROFILE}


@app.post("/")
@app.post("/classify")
@app.post("/query")
async def classify_endpoint(request: Request):
    # Enforce the assignment's 2 MiB request limit.
    body = await request.body()
    if len(body) > 2 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Request body exceeds 2 MiB")

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    if payload.get("profile") != PROFILE:
        raise HTTPException(status_code=400, detail="Unsupported profile")

    batch_id = payload.get("batchId")
    cases = payload.get("cases")
    if batch_id is None or not isinstance(cases, list):
        raise HTTPException(status_code=400, detail="batchId and cases are required")

    results = []
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or case.get("caseId") is None:
            raise HTTPException(status_code=400, detail="Every case must have a caseId")
        case_id = str(case["caseId"])
        if case_id in seen:
            raise HTTPException(status_code=400, detail="Duplicate caseId in request")
        seen.add(case_id)
        results.append(process_case(case))

    return JSONResponse(
        content={
            "profile": PROFILE,
            "batchId": str(batch_id),
            "results": results,
        },
        headers={"Cache-Control": "no-store"},
    )
