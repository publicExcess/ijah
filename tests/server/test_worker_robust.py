"""The engine thread survives bugs and says so when it cannot (lab-fixsrv, 17 Oct). CPU, mock engine, real Worker.

    $PY tests/server/test_worker_robust.py

An exception in admission or in the scheduler outside a request's own try used to kill the thread while /health kept
saying ok and every request waited forever. Now: the request being admitted (or those in flight) fail with an error and
the loop goes on; MAX_ERRORS in a row end the thread, /health answers 503 error, queued requests are failed, and the
process exits for its supervisor.
"""
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
from tokenizers import Tokenizer  # noqa: E402

import backend as B  # noqa: E402
import chat  # noqa: E402
import ne_server as S  # noqa: E402

PACK = ijah_paths.pack_or_skip("tests/server/test_worker_robust.py", need=("tokenizer.json", "chat_template.jinja"))
TOK = Tokenizer.from_file(str(Path(PACK) / "tokenizer.json"))
FAILS = []
S.log = lambda *a: None   # (the errors below are on purpose)


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


class Quiet:
    """stderr for the expected tracebacks"""
    def __enter__(self):
        self.e = sys.stderr
        sys.stderr = open("/dev/null", "w")

    def __exit__(self, *a):
        sys.stderr.close()
        sys.stderr = self.e


def make(slots=1):
    sess = B.Session([B.MockBackend(TOK, ctx=8192, delay=0.0) for _ in range(slots)], 8192, max_ckpts=2,
                     mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    tpl = chat.Template(PACK, TOK, "xhigh")
    w = S.Worker([sess], tpl.im_end, 2)
    return sess, tpl, w


def job(sess, tpl, text, max_new=40):
    ids, _ = tpl.render_ids({"messages": [{"role": "user", "content": text}], "reasoning_effort": "none"})
    j = S.Job({}, ids, False, max_new, (0.0, 1.0, 1, 0), [], "t")
    j.sess, j.model = sess, None
    return j


def outcome(j, timeout=10):
    """-> "done" | "error: msg" (the first terminal event of a job)"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            ev = j.events.get(timeout=0.1)
        except queue.Empty:
            continue
        if ev[0] == "done":
            return "done"
        if ev[0] == "error":
            return "error: " + str(ev[1])
    return "timeout"


def main():
    sess, tpl, w = make()
    w.start()
    a = job(sess, tpl, "first")
    w.q.put(a)
    check(outcome(a) == "done", "a normal request runs")

    # 1. a bug while admitting one job fails that job; the next one runs
    orig = sess.choose_mtp
    boom = [1]

    def choose(*x, **k):
        if boom:
            boom.pop()
            raise RuntimeError("bug in choose_mtp")
        return orig(*x, **k)
    sess.choose_mtp = choose
    b, c = job(sess, tpl, "second"), job(sess, tpl, "third")
    with Quiet():
        w.q.put(b)
        ob = outcome(b)
        w.q.put(c)
        oc = outcome(c)
    sess.choose_mtp = orig
    check(ob.startswith("error: internal error") and oc == "done" and w.is_alive() and not any(sl.busy for sl in sess.slots),
          f"exception while admitting one job: that job fails ({ob!r}), the next runs ({oc}), thread alive, slot free")

    # 2. a bug in the scheduler outside any try: requests in flight fail, the loop keeps serving
    slow = B.MockBackend(TOK, ctx=8192, delay=0.02)
    sess2 = B.Session([slow], 8192, max_ckpts=2, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    w2 = S.Worker([sess2], tpl.im_end, 2)
    w2.start()
    d = job(sess2, tpl, "LONGTEST in flight", max_new=300)
    w2.q.put(d)
    time.sleep(0.3)
    orig_admit = w2._admit
    left = [1]

    def admit():
        if left:
            left.pop()
            raise RuntimeError("bug in _admit")
        return orig_admit()
    w2._admit = admit
    e = job(sess2, tpl, "after")
    with Quiet():
        w2.q.put(e)   # wakes nothing: the loop is busy with d; the next pass raises
        od = outcome(d)
        oe = outcome(e)
    check(od.startswith("error: internal error") and oe == "done" and w2.is_alive() and w2.dead is None
          and not slow.dirty,
          f"scheduler exception: the request in flight fails ({od!r}), the loop serves on ({oe})")
    check(w2.health()[0] and w2.errors == 0, "health ok again after a good pass")

    # 3. /health over HTTP: ok -> 503 error once the thread is dead; queued requests are failed, not left waiting
    class FakeApp:
        pass
    app = FakeApp()
    app.worker, app.models, app.ctx, app.created = w2, {}, 8192, 0
    app.a = type("A", (), {"effort": "xhigh"})()
    app.vision = False
    app.authorized = lambda *a, **k: (True, "x")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(app))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/health"

    def get():
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as ex:
            return ex.code, json.loads(ex.read())
    st, h = get()
    check(st == 200 and h["status"] == "ok", f"/health ok while the engine thread runs ({st})")
    fatal = []
    w2.on_fatal = lambda: fatal.append(1)
    w2._admit = lambda: (_ for _ in ()).throw(RuntimeError("persistent bug"))
    f = job(sess2, tpl, "doomed")
    with Quiet():
        w2.q.put(f)
        of = outcome(f, 30)
        for _ in range(100):
            if not w2.is_alive():
                break
            time.sleep(0.1)
    st, h = get()
    check(not w2.is_alive() and w2.dead and fatal == [1] and of.startswith("error"),
          f"{w2.MAX_ERRORS} scheduler errors in a row end the thread ({w2.dead}); on_fatal called; the request failed ({of!r})")
    check(st == 503 and h["status"] == "error" and "died" in h["reason"], f"/health -> {st} {h.get('status')}: {h.get('reason')}")
    check(w2.health()[0] is False, "worker.health() is false for a dead thread (an HTTP thread polls it while it waits)")

    # 4. a stalled loop (a pass that does not return) is reported by /health after STALL_S
    sess3, _, w3 = make()
    w3.start()
    gate = threading.Event()
    orig_adm3 = w3._admit
    w3._admit = lambda: gate.wait(30)
    j3 = job(sess3, tpl, "stall")
    w3.q.put(j3)
    time.sleep(0.4)
    check(w3.health(stall_s=0.2)[0] is False and w3.health(stall_s=100)[0] is True,
          "a pass that does not return for longer than STALL_S is a stall: " + w3.health(stall_s=0.2)[1])
    gate.set()
    w3._admit = orig_adm3
    check(outcome(j3) == "done" and w3.health(stall_s=0.2)[0], "released: the request completes, health ok")
    w3.q.put(None)
    srv.shutdown()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
