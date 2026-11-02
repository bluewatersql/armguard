"""t9 Genie API operations (space upsert, CAN_RUN grant, eval run) and their pure helpers."""
import re, time

TERMINAL = {"DONE", "EVALUATION_FAILED", "EVALUATION_CANCELLED", "EVALUATION_TIMEOUT"}
_MASK = ((r"\S+@\S+", "<principal>"), (r"https?://\S+", "<url>"), (r"[\w.-]+\.databricks\.com\b", "<host>"))


def redact(x, *hide):
    """Error (first line) or text with emails, URLs, hosts and `hide` masked."""
    s = x if isinstance(x, str) else (str(x).splitlines() or [type(x).__name__])[0][:300]
    for h in filter(None, hide):
        s = s.replace(h, "<redacted>")
    for p, r in _MASK:
        s = re.sub(p, r, s)
    return s


def poll(get, status, limit=1200, every=10, clock=time.monotonic, sleep=time.sleep):
    """get() until status(result) is terminal or limit s pass; unknown states = running."""
    t0 = clock()
    while status(r := get()) not in TERMINAL and clock() - t0 < limit:
        sleep(every)
    return r


def passed(n_ok, n):
    return n == 12 and n_ok >= 10


def by_title(w, title):
    """Live (non-trashed) space id with this exact title, else None."""
    tok = None
    while True:
        p = w.genie.list_spaces(page_token=tok)
        hit = next((x.space_id for x in p.spaces or [] if x.title == title), None)
        if hit or not (tok := p.next_page_token):
            return hit


def upsert(w, sp, c, wh, ser, title):
    """Update the recorded space, else the live one with this title, else create; record the id."""
    from databricks.sdk.errors import InvalidParameterValue, NotFound, PermissionDenied
    A = c.fq("armguard_ops", "app_config")
    sp.sql(f"CREATE TABLE IF NOT EXISTS {A} (key STRING, value STRING)")
    rec = dict(sp.sql(f"SELECT key, value FROM {A}").collect()).get("genie_space_id")
    kw = {"title": title, "serialized_space": ser, "description": "Cascadia Ospreys arm-care: risk queue, workload, briefs, cases, league IL."}
    sid, verb, tried = None, "created", {None}
    for find in (lambda: rec, lambda: by_title(w, title)):
        if (cand := find()) not in tried:
            tried.add(cand)
            try:  # trashed: get_space works, update_space fails
                sid, verb = w.genie.update_space(cand, warehouse_id=wh, **kw).space_id or cand, "updated"
                break
            except (InvalidParameterValue, NotFound, PermissionDenied) as e:
                print("update failed, next:", redact(e, c.catalog))
    if not sid:
        parent = f"/Workspace/Users/{w.current_user.me().user_name}/armguard"  # redacted on error
        w.workspace.mkdirs(parent)
        sid = w.genie.create_space(wh, parent_path=parent, **kw).space_id
    sp.sql(f"MERGE INTO {A} t USING (SELECT 'genie_space_id' key, :v value) s ON t.key = s.key "
           "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *", args={"v": sid})
    return sid, verb


def grant(w, sid):
    """CAN_RUN for the app SP (permissions API) -> (log, ok); WAIT while the app is missing."""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
    try:
        app_sp = w.apps.get("armguard").service_principal_client_id
    except NotFound:
        return "WAIT app 'armguard' not found (re-run t9 after the app is deployed)", True
    except Exception as e:
        return f"FAIL app lookup: {redact(e)}", False
    try:
        acl = [AccessControlRequest(service_principal_name=app_sp, permission_level=PermissionLevel.CAN_RUN)]
        w.permissions.update("genie", sid, access_control_list=acl)
        return "OK app SP: CAN_RUN", True
    except Exception as e:
        return f"FAIL grant: {redact(e)}", False


def evaluate(w, sid, g):
    """Genie eval-run API, <= 20 min -> (run, {question_id: (assessment, reasons)})."""
    G, val = w.genie, lambda x: getattr(x, "value", x)
    rid = G.genie_create_eval_run(sid).eval_run_id
    r = poll(lambda: G.genie_get_eval_run(sid, rid), lambda x: val(x.eval_run_status))
    if val(r.eval_run_status) != "DONE":
        raise SystemExit(f"eval run {rid} (space {sid}): {val(r.eval_run_status)}; app grant: {g}")
    res, tok, out = [], None, {}
    while True:
        page = G.genie_list_eval_results(sid, rid, page_token=tok)
        res, tok = res + (page.eval_results or []), page.next_page_token
        if not tok:
            break
    for x in res:
        d = G.genie_get_eval_result_details(sid, rid, x.result_id)
        out[x.benchmark_question_id] = (val(d.assessment), [val(a) for a in d.assessment_reasons or []])
        print("bench", x.benchmark_question_id, out[x.benchmark_question_id], [a.response for a in d.actual_response or []])
    return r, out


def try_evaluate(w, sid, g, *hide):
    """evaluate(), fail-soft (the Beta eval run may fail or time out) -> (run, results, None) or (None, {}, failure text)."""
    try:
        return (*evaluate(w, sid, g), None)
    except (Exception, SystemExit) as e:
        return None, {}, f"eval run FAILED/timeout: {redact(e, *hide)}"
