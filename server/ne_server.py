"""OpenAI-compatible HTTP server for the native engine (lab-server). docs/server.md.

    $PY server/ne_server.py [--model-id ID | --pack DIR] [--port 8140] [--ctx 262144] [--mtp 2] [--mtp-policy auto|always|never]
        [--effort xhigh] [--key-file ~/.config/ijah/api-key] [--lib build/libengine.dylib] [--shim PATH]
        [--mock [--mock-delay S]]

Endpoints: POST /v1/chat/completions (stream and not), GET /v1/models, GET /health. Binds 127.0.0.1 only. Every
/v1 request needs `Authorization: Bearer <key>` (the key file's content; compared in constant time; logs show only
sha256(key)[:8]). One generation at a time: a single engine worker thread, requests queue in arrival order.
($PY = .venv/bin/python -I -B, or $IJAH_PY; start / stop with tools/serve.sh, which also runs the RAM watchdog and,
when IJAH_LOCK_SCRIPT is set, takes the machine lock.)
"""
import argparse
import collections
import hashlib
import hmac
import json
import os
import queue
import secrets
import select
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
from tokenizers import Tokenizer  # noqa: E402

import backend as B  # noqa: E402
import chat  # noqa: E402
import sampling as SMP  # noqa: E402
import anthropic_api  # noqa: E402
from prefix_cache import parse_bytes  # noqa: E402
import prefix_cache as PC  # noqa: E402
import grammar as G  # noqa: E402
import vision as V  # noqa: E402

KEY_FILE = Path.home() / ".config/ijah/api-key"
POLL_S = 0.25   # a request thread looks at its client socket (and its job) this often while no token arrives
STALL_S = float(os.environ.get("NE_SRV_STALL_S", 900))   # a pass of the engine loop this long is a hang (0 = no check)
MAX_BODY = int(float(os.environ.get("NE_SRV_MAX_BODY_MB", 64)) * (1 << 20))   # request bodies above this: 413
SOCKET_TIMEOUT = float(os.environ.get("NE_SRV_SOCKET_TIMEOUT", 120))   # a client that stalls a read / write this long is dropped
KEEPALIVE_S = float(os.environ.get("NE_SRV_KEEPALIVE_S", 10))   # a stream silent this long gets an SSE keepalive
# lab-pfpath: GPU keep-warm while the engine thread idles: a ping (Backend.ping) every GPU_WARM_EVERY s for up to
# GPU_WARM_S s after the last request (0 = off). Without it the first command buffer after ~1 s of idle waits ~0.6 s
# and each recording's first replay ~0.1 s more (docs/prefill.md "server path")
GPU_WARM_S = float(os.environ.get("NE_GPU_WARM_S", 900))
GPU_WARM_EVERY = float(os.environ.get("NE_GPU_WARM_EVERY", 0.5))
# lab-snap (docs/server.md 3f): text whose end inside the first message is a shared-prefix boundary: the pack template's
# tools block ends with '</IMPORTANT>' before the system text (which may differ per session: date, directory, ...)
SHARED_MARKERS = ("</IMPORTANT>",)
_T0 = time.perf_counter()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def key_hash(k):
    return hashlib.sha256(k.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, req, prompt, thinking, max_new, sampler, stops, rid, constraint=None, extras=None, index=0):
        self.req, self.prompt, self.thinking, self.max_new, self.sampler = req, prompt, thinking, max_new, sampler
        self.stops = stops
        self.constraint = constraint   # grammar.Constraint (response_format / forced tool_choice) or None
        self.extras = extras or SMP.Extras()   # logit_bias / penalties / min_p / logprobs / seed (sampling.py)
        self.extra = self.extras.engine_extra()   # None = the sampler as before (MTP-capable)
        self.index = index             # choice index (n > 1: sequential jobs over one prompt)
        self.images = {}   # lab-vision: 40-bit digest -> vision.VisionImage of the images in the prompt
        self.id = rid
        self.model = None   # the Served model (lab-srvperf: --model-id A,B serves several from one process)
        self.events = queue.Queue()   # ("toks", [ids]) | ("done", stats) | ("error", msg)
        self.cancel = threading.Event()
        self.t_queued = time.perf_counter()


class Worker(threading.Thread):
    """the one engine thread: requests take a free slot (model instance) in arrival order and run interleaved,
    one prefill piece / decode token per turn (lab-srvcache); a request waits while every slot is busy"""
    MAX_ERRORS = 20   # scheduler errors in a row before the thread gives up

    def __init__(self, sessions, im_end, depth, seed_mtp=False):
        """sessions: one Session per served model (its slots + prefix cache); a job runs on job.sess"""
        super().__init__(daemon=True, name="engine")
        self.seed_mtp = seed_mtp   # a request with a seed may take the MTP path (best-effort reproducibility)
        self.sessions = list(sessions) if isinstance(sessions, (list, tuple)) else [sessions]
        self.s, self.im_end, self.depth = self.sessions[0], im_end, depth
        self.q = queue.Queue()
        self.waiting = collections.deque()
        self.active = []  # [job, generator, slot]
        self.done = 0
        self._stopping = False   # (not `_stop`: threading.Thread has a method of that name, is_alive() calls it)
        self.dead = None         # why the thread ended unexpectedly (/health says error; the process then exits for the supervisor)
        self.errors = 0          # scheduler errors since the last good pass of the loop
        self.beat = time.monotonic()   # last sign of life of the loop (a generator step finished, a job taken)
        self.idle = False        # blocked waiting for a job (no beat expected)
        self.on_fatal = None     # called once if the thread dies

    @property
    def busy(self):
        return self.active[0][0] if self.active else None

    def queued(self):
        return self.q.qsize() + len(self.waiting)

    def contended(self, sl):
        """another request needs the engine thread: one runs on another slot, or one waits (or just arrived) while
        another slot of its model is free for it. Idle resident conversations do not count; with one slot (per model)
        nothing can preempt. Called on the engine thread (also from inside an MTP run's per-round callback)."""
        if len(self.active) > 1 or (self.active and self.active[0][2] is not sl):
            return True
        if not self.waiting and self.q.empty():
            return False
        self._pull(False)   # the engine thread is the queue's only consumer
        return any(not s.busy and s is not sl for j in self.waiting for s in j.sess.slots)

    def _warm(self):
        """keep the GPU warm while idle (GPU_WARM_S after the last sign of life); -> the next timeout (None = block)"""
        if GPU_WARM_S <= 0 or time.monotonic() - self.beat > GPU_WARM_S:
            return None
        for S in self.sessions:
            for sl in S.slots:
                ping = getattr(sl.b, "ping", None)
                if ping is not None:
                    try:
                        ping()
                    except Exception as e:   # noqa: BLE001 (a keep-warm failure is never a request's)
                        log(f"GPU keep-warm failed: {e!r}")
                        return None
        return GPU_WARM_EVERY

    def _pull(self, block):
        tmo = GPU_WARM_EVERY if block and GPU_WARM_S > 0 else None
        while True:
            if not block and self.q.empty():   # (no queue.Empty exception per decode token)
                return
            try:
                j = self.q.get(block=block, timeout=tmo if block else None)
            except queue.Empty:
                if block:   # idle: keep the GPU warm, wait again
                    tmo = self._warm()
                    continue
                return
            block = False
            if j is None:
                self._stopping = True
                return
            self.waiting.append(j)

    def _admit(self):
        """start waiting jobs on free slots of their model, in arrival order per model (a job whose model has no free
        slot does not hold up another model's job)"""
        blocked = set()
        i = 0
        while i < len(self.waiting):
            job = self.waiting[i]
            if job.cancel.is_set():
                del self.waiting[i]
                job.events.put(("error", "cancelled"))
                continue
            S = job.sess
            if id(S) in blocked:
                i += 1
                continue
            reuse = job.req.get("ne_cache", True) is not False  # ne_cache false: tests
            try:
                sl, cand = S.pick(job.prompt, reuse)
            except Exception as e:   # (a bug choosing a slot fails this job, not the engine thread)
                log(f"{job.id}: choosing a slot failed: {e!r}")
                traceback.print_exc(file=sys.stderr)
                del self.waiting[i]
                job.events.put(("error", f"internal error: {e}"))
                continue
            if sl is None:
                if not any(x.busy for x in S.slots):  # fits no slot even with all idle (make_job checks the window)
                    del self.waiting[i]
                    job.events.put(("error", "prompt does not fit any slot"))
                    continue
                blocked.add(id(S))
                i += 1
                continue
            del self.waiting[i]
            try:
                n = len(job.prompt)
                k = cand[0] if cand else 0
                if hasattr(sl.b, "prepare"):  # lab-vision: the image features + the prompt's positions on this slot's model
                    try:
                        sl.b.prepare(job.prompt, job.images)
                    except Exception as e:
                        log(f"{job.id}: image preparation failed: {e}")
                        job.events.put(("error", str(e)))
                        continue
                temp, top_p, top_k, _ = job.sampler
                if job.extra is not None and not getattr(sl.b, "can_extra", False):
                    if not job.extras.neutral:   # an engine without the sampler functions cannot honour a knob: say so
                        log(f"{job.id}: the engine library lacks the lab-sampler functions (rebuild it)")
                        job.events.put(("error", "this engine build does not support logit_bias / penalties / min_p / "
                                                 "logprobs (rebuild libengine)"))
                        continue
                    job.extra = None   # a seed alone: ignored (the draws are random per request, as before)
                # stop strings need an abortable loop (the per-round callback: the HTTP thread's cancel ends the run after
                # the round; the state stays = fed + output[:-1]). MTP-continuation engines run MTP whenever no other
                # request needs the engine thread and yield at a round boundary when one does (Session.run); the older
                # library's MTP call holds the engine thread for the whole reply, so only when nothing else runs or waits
                can = (temp < 0.01 or top_k <= 32) and (not job.stops or getattr(sl.b, "mtp_abortable", False)) \
                    and (getattr(sl.b, "mtp_cont", False) or not self.contended(sl))
                can = can and job.constraint is None   # the MTP loop has no token mask: constrained requests step
                # image prompts: MTP from the current state (lab-mtpimg: the feeds set the image rows, the head runs at
                # the reference's positions); the older library's fresh MTP call has no image rows: step there
                can = can and (not job.images or getattr(sl.b, "mtp_cont", False))
                force = job.req.get("ne_path")  # benchmarks: "step" | "mtp" overrides the policy
                # lab-sampler: the MTP loop applies logit_bias, penalties and min_p exactly (the verify rows get the sampler's
                # transform, the accept compares the same distributions) but gives no logprobs: those requests step. A seed
                # alone steps too (reproducible: one path, the RNG counter = the token's index), unless the server runs
                # --seed-mtp or the request says ne_path "mtp" (best effort: MTP rounds differ from steps when another
                # request interrupts)
                if job.extra is not None:
                    can = can and not job.extras.logprobs and (not job.extras.seeded or self.seed_mtp or force == "mtp") \
                        and (getattr(sl.b, "mtp_cont", False) or job.extras.neutral)
                if force == "step" or (force == "mtp" and can and S.mtp_policy != "never"):
                    use = force == "mtp"
                else:
                    use = S.choose_mtp(n, k, job.max_new, can)
                job.max_new = min(job.max_new, sl.ctx - n)  # a smaller slot: its window is the reply cap
                job.ctx = sl.ctx
                job.t_start = time.perf_counter()
                B.trace(f"{job.id}: admitted after {(job.t_start - job.t_queued) * 1e3:.1f} ms queued, mtp {use}")
                gen = S.run(sl, cand, job.prompt, job.max_new, job.sampler, use,
                                 lambda t, lp=None, job=job: job.events.put(("toks", t, lp)), job.cancel.is_set,
                                 self.im_end, self.depth, others=lambda: len(self.active) > 1,
                                 constraint=job.constraint, contended=lambda sl=sl: self.contended(sl),
                                 extra=job.extra, capture=job.req.get("ne_shared", True) is not False)
                self.active.append([job, gen, sl])
            except Exception as e:   # a bug admitting ONE job fails that job, not the engine thread
                log(f"{job.id}: admitting the request failed: {e!r}")
                traceback.print_exc(file=sys.stderr)
                job.events.put(("error", f"internal error: {e}"))
                try:
                    S.fail(sl)
                except Exception:   # noqa: BLE001
                    sl.busy = False

    def _step(self):
        """one pass of the engine thread's loop: take new jobs, start the ones that fit a slot, advance every active
        generator by one prefill piece / decode token. -> True when the thread should end"""
        blocking = not self.active and not self.waiting
        self.idle = blocking
        self._pull(block=blocking)
        self.idle = False
        self.beat = time.monotonic()
        if self._stopping and not self.active:
            return True
        self._admit()
        for ent in list(self.active):
            job, gen, sl = ent
            try:
                next(gen)
                continue
            except StopIteration as e:
                st = e.value
                st["queue_ms"] = (job.t_start - job.t_queued) * 1e3
                if job.extra is not None:
                    st["sampling"] = job.extras.names()
                if job.constraint is not None:
                    st["grammar"] = job.constraint.kind
                job.events.put(("done", st))
                c = job.sess.cache
                dec = max(st["completion_tokens"] - 1, 0) / max(st["decode_ms"], 1e-3) * 1e3
                own = st.get("decode_own_ms")
                own = f" (own steps {(st['completion_tokens'] - 1) / own * 1e3:.1f})" if own and dec < 0.9 * (
                    (st["completion_tokens"] - 1) / own * 1e3) else ""
                mid = f"{job.model.id} " if job.model is not None and len(self.sessions) > 1 else ""
                log(f"{job.id}: {mid}{st['path']} slot {st['slot']} prompt {len(job.prompt)} (cached "
                    f"{st['cached_tokens']}) -> {st['completion_tokens']} tokens; prefill {st['prefill_ms']:.0f} ms "
                    f"(restore {st.get('restore_ms', 0):.0f}), decode {dec:.1f} tok/s{own}; "
                    f"sampling T={job.sampler[0]:g} p={job.sampler[1]:g} k={job.sampler[2]}"
                    f"{'' if any(job.req.get(x) is not None for x in ('temperature', 'top_p', 'top_k')) else ' (defaults)'}; "
                    f"{c.summary()}")
            except Exception as e:  # engine errors end this request, not the server
                log(f"{job.id}: engine error {e}")
                traceback.print_exc(file=sys.stderr)
                job.events.put(("error", str(e)))
                job.sess.fail(sl)
            self.active.remove(ent)
            self.done += 1
            self.beat = time.monotonic()
        self.errors = 0
        return False

    def _fail_all(self, msg):
        """fail every request in flight (the scheduler itself broke: their state is not to be trusted) and reset their
        slots; waiting / queued requests stay queued"""
        for job, gen, sl in list(self.active):
            job.events.put(("error", msg))
            try:
                gen.close()   # (its finally clauses: the slot's mask / extra sampling parameters)
            except Exception:   # noqa: BLE001
                pass
            try:
                job.sess.fail(sl)
            except Exception:   # noqa: BLE001
                sl.busy = False
        self.active.clear()

    def run(self):
        """the engine thread. An exception outside a request's own try (the scheduler, admission) fails the requests in
        flight and the loop goes on serving; MAX_ERRORS in a row without a good pass end the thread (/health then says
        error and the process exits for its supervisor)."""
        try:
            while True:
                try:
                    if self._step():
                        return
                except Exception as e:   # noqa: BLE001
                    self.errors += 1
                    log(f"engine thread: scheduler error {self.errors}/{self.MAX_ERRORS}: {e!r}")
                    traceback.print_exc(file=sys.stderr)
                    self._fail_all(f"internal error: {e}")
                    if self.errors >= self.MAX_ERRORS:
                        raise RuntimeError(f"{self.errors} scheduler errors in a row; the last: {e!r}")
                    time.sleep(min(0.05 * self.errors, 1.0))
        except BaseException as e:   # noqa: BLE001
            self.dead = f"{type(e).__name__}: {e}"
            log(f"engine thread died: {self.dead}")
            self._fail_all(f"the engine thread died: {e}")
            while True:   # nothing will serve what is queued
                try:
                    j = self.q.get_nowait()
                except queue.Empty:
                    break
                if j is not None:
                    j.events.put(("error", f"the engine thread died: {e}"))
            for j in self.waiting:
                j.events.put(("error", f"the engine thread died: {e}"))
            self.waiting.clear()
            if self.on_fatal is not None:
                self.on_fatal()

    def health(self, stall_s=None):
        """-> (ok, reason): the thread is running, and a pass of its loop is not stuck"""
        if self.dead:
            return False, f"engine thread died: {self.dead}"
        if not self.is_alive():
            return False, "engine thread is not running"
        stall_s = STALL_S if stall_s is None else stall_s
        if stall_s and not self.idle and time.monotonic() - self.beat > stall_s:
            return False, f"engine thread stalled for {time.monotonic() - self.beat:.1f} s"
        return True, ""


# ------------------------------------------------------------------------------------------------- server
class Served:
    """one model id of this process: its pack, tokenizer + template, Session (slots = model instances, prefix cache)"""

    def __init__(self, mid, pack, tok, tpl, session, kit=None):
        self.id, self.pack, self.tok, self.tpl, self.session, self.kit = mid, pack, tok, tpl, session, kit
        self.texts = SMP.TokenText(tok)   # token id -> (text, bytes) for logprobs

    @property
    def backend(self):
        return self.session.slots[0].b


class App:
    def __init__(self, a):
        self.a = a
        self.key = self._key(Path(a.key_file).expanduser())
        served = getattr(a, "served", None) or [(a.model_id, a.pack)]
        slot_ctx = [a.ctx] + [min(a.ctx, a.slot_ctx)] * (a.slots - 1)
        self.ctx = a.ctx
        if str(a.cache_disk).lower() in ("on", "yes", "true"):
            a.cache_disk = "20GB"
        disk = parse_bytes(a.cache_disk)
        nm = len(served)
        self.models = {}   # id -> Served, in --model-id order (the first is the default)
        first = None
        kits = {}
        t_load = 0.0
        share = bool(getattr(a, "share_state", False)) and nm > 1
        phys = [B.Phys() for _ in slot_ctx] if share else None   # --share-state: slot j of every model = one state
        base_backs = None
        for mid, pack in served:
            tok = Tokenizer.from_file(str(Path(pack) / "tokenizer.json"))
            tpl = chat.Template(pack, tok, a.effort)
            t0 = time.perf_counter()
            sf = (lambda j: base_backs[j]) if share and base_backs else (lambda j: None)
            if a.mock:
                backs = [B.MockBackend(tok, c, a.mock_delay, prefill_rate=a.mock_prefill_rate or None, mtp_depth=a.mtp,
                                       name=mid if nm > 1 else None, share_state=sf(j)) for j, c in enumerate(slot_ctx)]
            else:
                # further models open their pack on the same engine: files they share with an open pack (same inode:
                # Swift-1.5 shares the base's experts) are one mapping + one set of buffers (core/pack.mm)
                backs = [B.RealBackend(ROOT, pack, a.ctx, a.mtp, Path(a.lib).resolve(), None, log, engine=first,
                                       state_from=sf(0))]
                for j, c in enumerate(slot_ctx[1:], 1):
                    backs.append(B.RealBackend(ROOT, pack, c, a.mtp, Path(a.lib).resolve(), None, log, share=backs[0],
                                               state_from=sf(j)))
                first = first or backs[0]
            base_backs = base_backs or backs
            t_load += time.perf_counter() - t0
            # the prefix cache budgets are split between the models (each has its own cache: keys never cross models)
            # SSD tier (lab-persist, kept across restarts): one directory per model id (one model: --cache-dir itself),
            # entries keyed by the engine build, that model's pack and id, and the state layout
            cdir = a.cache_dir if nm == 1 else str(Path(a.cache_dir) / mid)
            ident = PC.identity(None if a.mock else Path(a.lib).resolve(), pack, mid, backs[0].cache_layout()) \
                if disk else None
            sess = B.Session(backs, self.ctx, a.ckpts, a.mtp_policy, log=log, cache_ram=parse_bytes(a.cache_ram) // nm,
                             disk_dir=cdir if disk else None, disk_bytes=disk // nm, pf_cap=a.pf_cap, phys=phys,
                             ident=ident, markers=[tok.encode(m, add_special_tokens=False).ids
                                                   for m in SHARED_MARKERS])
            kit = None   # grammar-constrained decoding: response_format, tool_choice required / a named function
            if not getattr(backs[0], "can_mask", False):
                log("structured output off: the engine library has no ne_fn_model_set_mask (rebuild it)")
            elif not a.no_grammar:
                tkey = hashlib.sha256((Path(pack) / "tokenizer.json").read_bytes()).hexdigest()
                if tkey not in kits:
                    try:
                        t1 = time.perf_counter()
                        kits[tkey] = G.GrammarKit(tok, backs[0].VOCAB, B.RealBackend.STOPS)
                        log(f"structured output on (xgrammar; {time.perf_counter() - t1:.1f} s)")
                    except G.Unavailable as e:
                        kits[tkey] = None
                        log(f"structured output off: {e}")
                kit = kits[tkey]
            self.models[mid] = Served(mid, pack, tok, tpl, sess, kit)
            if nm > 1:
                log(f"model {mid}: {len(backs)} slot(s) ready ({time.perf_counter() - t0:.1f} s)")
        d = next(iter(self.models.values()))
        self.default = d
        allb = [sl.b for m in self.models.values() for sl in m.session.slots]
        if not a.mock and len(allb) > 1 and os.environ.get("NE_SRV_SHARE_PF", "1") != "0":
            # every model instance (slots, models) prefills on the engine thread, one call at a time: one set of chunk
            # buffers for all of them (lab-srvperf; NE_SRV_SHARE_PF=0: one set each)
            ok = [b.share_prefill(allb[0]) for b in allb[1:]]
            log(f"chunked-prefill buffers shared by {1 + sum(ok)} of {len(allb)} model instances")
        # the default model's views (single-model callers)
        self.model_id, self.tok, self.tpl, self.session, self.backend, self.kit = d.id, d.tok, d.tpl, d.session, \
            d.backend, d.kit
        self.worker = Worker([m.session for m in self.models.values()], self.tpl.im_end, a.mtp, a.seed_mtp)
        self.worker.start()
        self.created = int(time.time())
        self.vision = False   # lab-vision: set True once image parts are accepted (advertised as 'vision')
        log(f"engine ready ({'mock' if a.mock else 'native'}; load {t_load:.1f} s; models {', '.join(self.models)}; "
            f"slots {len(slot_ctx)} per model (ctx {', '.join(str(c) for c in slot_ctx)}); MTP "
            f"{'off' if not a.mtp else f'depth {a.mtp}, policy {self.session.mtp_policy}'}; prefix cache "
            f"{'on' if self.backend.kvio else 'off (live state only)'}: RAM {a.cache_ram}, disk {a.cache_disk}"
            f"{f' (split over {nm} models)' if nm > 1 else ''}, {a.ckpts} checkpoints per conversation"
            f"{'; models share each slot (--share-state)' if share else ''}; key sha256 "
            f"{key_hash(self.key)})")

    def close(self, timeout=30):
        """every served model's prefix cache: the queued SSD writes land (up to timeout s each) -> models whose writes did not"""
        late = []
        for m in self.models.values():
            try:
                if not m.session.cache.close(timeout=timeout):
                    late.append(m.id)
            except Exception as e:   # noqa: BLE001
                log(f"prefix cache of {m.id}: closing failed: {e!r}")
                late.append(m.id)
        return late

    def model_for(self, name):
        """the Served model a request names: exact id, else case-insensitive; no name -> the default. An unknown
        name is an error when several models are served (one model: ignored, as before)."""
        if not name:
            return self.default
        m = self.models.get(name)
        if m is None:
            m = next((x for k, x in self.models.items() if k.lower() == str(name).lower()), None)
        if m is None:
            if len(self.models) == 1:
                return self.default
            raise chat.RequestError(f"unknown model {name!r}; this server has {', '.join(self.models)}")
        return m

    @staticmethod
    def _key(path):
        if not path.exists():
            raise SystemExit(f"no API key at {path}: tools/serve.sh keygen")
        if path.stat().st_mode & 0o077:
            raise SystemExit(f"{path} must be chmod 600")
        k = path.read_text().strip()
        if len(k) < 16:   # some clients' shared provider keys are short; loopback only
            raise SystemExit(f"{path}: key too short")
        return k

    def authorized(self, header, x_api_key=None):
        """Authorization: Bearer KEY (OpenAI) or x-api-key: KEY (Anthropic)"""
        if header and header.startswith("Bearer "):
            k = header[7:].strip()
        elif x_api_key:
            k = x_api_key.strip()
        else:
            return False, "missing"
        return hmac.compare_digest(k.encode(), self.key.encode()), key_hash(k)

    def capabilities(self, m=None):
        """/v1/models `capabilities` (Sushi's names); json_schema only while the grammar kit is loaded"""
        caps = ["chat", "tool_use", "streaming", "reasoning"]
        if (m or self.default).kit is not None:
            caps.append("json_schema")
        if self.vision:
            caps.append("vision")
        return caps

    # -- request -> jobs
    def make_jobs(self, req, rid, raw=None):
        """-> [Job]: n choices x prompts. raw: token-id prompts of /v1/completions (no chat template, no reasoning /
        tool parsing); None: render req['messages'] with the pack's template. The choices of a request run one after
        the other (the prefix cache makes every choice after the first nearly free)."""
        M = self.model_for(req.get("model"))
        kit = M.kit
        extras = SMP.Extras.parse(req, M.backend.VOCAB, completions=raw is not None)
        n_choices = extras.n
        constraint_of = lambda: None   # noqa: E731
        images = []
        if raw is None:
            rf = G.GrammarKit.response_format(req)   # validates; None = plain text
            tm = G.GrammarKit.tool_mode(req)
            if rf is not None and kit is None:
                raise chat.RequestError("response_format json_object / json_schema is not available on this server")
            if tm is not None and kit is None:
                log("tool_choice required / a named function treated as auto: structured output is off")
                tm = None
            if rf is not None and tm is None:
                req = G.with_format_hint(req, rf)
            srcs = []
            # (special-token text inside the messages stays text; only the template's markup is special, so the image
            # placeholders in `prompt` are exactly those of the real image parts in `srcs`)
            prompt, thinking = M.tpl.render_ids(req, srcs)
            if kit is not None and (rf or tm):
                constraint_of = lambda: kit.for_request(req, thinking)   # noqa: E731 (a matcher per choice)
            if srcs or V.IMAGE_TOKEN in prompt:  # lab-vision: CPU preprocessing here, the tower runs on the engine thread
                try:
                    for src in srcs:
                        img, _ = V.load_image(src)
                        images.append(V.VisionImage(img))
                    prompt, _, _ = V.expand_prompt(prompt, images)
                except V.ImageError as e:
                    raise chat.RequestError(str(e))
                cap = int(os.environ.get("NE_VISION_ROWS", 16384))
                if sum(im.n_tokens for im in images) > cap:
                    raise chat.RequestError(f"images need {sum(im.n_tokens for im in images)} tokens > {cap} "
                                            "(NE_VISION_ROWS)")
            prompts = [prompt]
        else:
            if G.GrammarKit.response_format(req) is not None or G.GrammarKit.tool_mode(req) is not None:
                raise chat.RequestError("response_format / tool_choice are chat completions features")
            prompts, thinking = raw, False
        temp = float(req.get("temperature", 1.0) if req.get("temperature") is not None else 1.0)
        top_p = float(req.get("top_p", 0.95) if req.get("top_p") is not None else 0.95)
        top_k = req.get("top_k", 20)
        top_k = 20 if top_k is None else int(top_k)
        if top_k <= 0 or top_k > 64:
            top_k = 64  # the engine's sampler keeps at most 64 candidates
        if temp < 0.01:
            temp, top_p, top_k = 0.0, 1.0, 1
        seed0 = req.get("seed")
        mt = req.get("max_completion_tokens", req.get("max_tokens"))
        jobs = []
        for prompt in prompts:
            n = len(prompt)
            if n + 1 > self.ctx:
                raise chat.RequestError(f"prompt is {n} tokens; the context window is {self.ctx}")
            room = self.ctx - n
            max_new = room if mt is None else max(1, min(int(mt), room))  # reply cap = the context window
            for i in range(n_choices):
                idx = len(jobs)
                seed = secrets.randbits(32) if seed0 is None else (int(seed0) + i) & 0xFFFFFFFF
                stops = chat.StopFilter(req.get("stop"))
                job = Job(req, prompt, thinking, max_new, (temp, min(max(top_p, 1e-3), 1.0), top_k, seed), stops.stops,
                          rid, constraint_of(), extras, idx)
                job.images = {im.d40: im for im in images}
                job.model, job.sess = M, M.session
                jobs.append(job)
        return jobs


# --------------------------------------------------------------------------------------------- wire formats
def _sse(obj):
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


class ChatFmt:
    """/v1/chat/completions"""
    raw = False
    done_marker = True

    def __init__(self, app, rid, req, model_id):
        self.app, self.rid, self.req, self.model_id = app, rid, req, model_id
        self.created = chat.now()
        self.base = {"id": rid, "object": "chat.completion.chunk", "created": self.created, "model": model_id}

    wire = staticmethod(_sse)
    keepalive = b": keepalive\n\n"   # an SSE comment: clients ignore it, a dead socket makes the write fail

    def begin(self, jobs):
        return []

    def start_choice(self, job):
        return [self._chunk(job.index, {"role": "assistant", "content": ""})]

    def _chunk(self, idx, delta, fr=None, lps=None):
        ch = {"index": idx, "delta": delta, "finish_reason": fr}
        if lps:
            ch["logprobs"] = {"content": lps, "refusal": None}
        return dict(self.base, choices=[ch])

    def deltas(self, job, ds, lps, force=False):
        msgs = []
        for d in ds:
            if d[0] == "content":
                delta = {"content": d[1]}
            elif d[0] == "reasoning":
                delta = {"reasoning_content": d[1]}
            elif d[0] == "tool_start":
                delta = {"tool_calls": [{"index": d[1], "id": d[2], "type": "function",
                                         "function": {"name": d[3], "arguments": ""}}]}
            elif d[0] == "tool_args":
                delta = {"tool_calls": [{"index": d[1], "function": {"arguments": d[2]}}]}
            else:
                continue
            msgs.append(self._chunk(job.index, delta))
        if lps:
            if msgs:
                msgs[0]["choices"][0]["logprobs"] = {"content": lps, "refusal": None}
            elif force:
                msgs.append(self._chunk(job.index, {}, lps=lps))
            else:
                return None   # the tokens made no text yet (a partial character, held text): sent with the next chunk
        return msgs

    def end_choice(self, job, r):
        return [self._chunk(job.index, {}, r["finish"])]

    def end(self, h, results):
        if (self.req.get("stream_options") or {}).get("include_usage"):
            return [dict(self.base, choices=[], usage=h._usage(results),
                         timings=h._timings(results[0]["stats"], results[0]["state"]))]
        return []

    def error(self, msg):
        return [{"error": {"message": msg, "type": "server_error"}}]

    def full(self, h, results):
        choices = []
        for r in results:
            msg = {"role": "assistant", "content": r["content"] if (r["content"] or not r["calls"]) else None}
            if r["reasoning"] or r["job"].thinking:
                msg["reasoning_content"] = r["reasoning"]
            if r["calls"]:
                msg["tool_calls"] = r["calls"]
            ch = {"index": r["job"].index, "message": msg, "finish_reason": r["finish"]}
            if r["job"].extras.logprobs:
                ch["logprobs"] = {"content": r["lps"], "refusal": None}
            choices.append(ch)
        return {"id": self.rid, "object": "chat.completion", "created": self.created, "model": self.model_id,
                "choices": choices, "usage": h._usage(results),
                "timings": h._timings(results[0]["stats"], results[0]["state"])}


class CompFmt(ChatFmt):
    """/v1/completions (raw prompt: the text verbatim; `logprobs` = the number of alternatives)"""
    raw = True

    def __init__(self, app, rid, req, model_id):
        super().__init__(app, rid, req, model_id)
        self.base = {"id": rid, "object": "text_completion", "created": self.created, "model": model_id}

    def start_choice(self, job):
        return []

    def _chunk(self, idx, text, fr=None, lps=None):
        return dict(self.base, choices=[{"index": idx, "text": text, "logprobs": SMP.legacy_logprobs(lps) if lps else None,
                                         "finish_reason": fr}])

    def deltas(self, job, ds, lps, force=False):
        text = "".join(d[1] for d in ds if d[0] == "content")
        if not text and lps and not force:
            return None
        if not text and not lps:
            return []
        return [self._chunk(job.index, text, None, lps)]

    def end_choice(self, job, r):
        return [self._chunk(job.index, "", r["finish"])]

    def full(self, h, results):
        choices = []
        for r in results:
            lp = SMP.legacy_logprobs(r["lps"]) if r["job"].extras.logprobs else None
            choices.append({"index": r["job"].index, "text": r["content"], "logprobs": lp,
                            "finish_reason": "length" if r["finish"] == "length" else "stop"})
        return {"id": self.rid, "object": "text_completion", "created": self.created, "model": self.model_id,
                "choices": choices, "usage": h._usage(results),
                "timings": h._timings(results[0]["stats"], results[0]["state"])}


class AnthropicFmt:
    """/v1/messages (one message; `req` is the converted chat request)"""
    raw = False
    done_marker = False

    def __init__(self, app, rid, req, model_id):
        self.app, self.rid, self.req, self.model_id = app, rid, req, model_id
        self.stream = None

    @staticmethod
    def wire(m):
        return b"event: " + m[0].encode() + b"\ndata: " + json.dumps(m[1], ensure_ascii=False).encode() + b"\n\n"

    keepalive = b'event: ping\ndata: {"type": "ping"}\n\n'   # Anthropic's own keepalive event

    def begin(self, jobs):
        self.stream = anthropic_api.Stream(self.rid, self.model_id, len(jobs[0].prompt))
        return self.stream.start()

    def start_choice(self, job):
        return []

    def deltas(self, job, ds, lps, force=False):
        return self.stream.deltas(ds)

    def end_choice(self, job, r):
        return []

    def _stop(self, r):
        return anthropic_api.stop_reason(r["finish"], r["stop_string"])

    def end(self, h, results):
        r = results[0]
        reason, ss = self._stop(r)
        u = anthropic_api.usage(r["stats"]["prompt_tokens"], r["stats"]["cached_tokens"], r["state"]["n"])
        return self.stream.finish(reason, ss, r["state"]["n"], {k: u[k] for k in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")})

    def error(self, msg):
        return [("error", {"type": "error", "error": {"type": "api_error", "message": msg}})]

    def full(self, h, results):
        r = results[0]
        reason, ss = self._stop(r)
        return anthropic_api.message(
            self.rid, self.model_id, anthropic_api.blocks(r["content"], r["reasoning"], r["calls"], r["job"].thinking),
            reason, ss, anthropic_api.usage(r["stats"]["prompt_tokens"], r["stats"]["cached_tokens"], r["state"]["n"]))


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ijah"
        sys_version = ""
        timeout = SOCKET_TIMEOUT   # socket timeout: an idle keep-alive connection, a half-sent request, a client that
        #                            stopped reading a stream are dropped instead of holding a thread for ever

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj, extra=None):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code, msg, typ="invalid_request_error", extra=None):
            if self.path.split("?")[0].startswith("/v1/messages"):   # Anthropic's error body
                at = {"authentication_error": "authentication_error", "server_error": "api_error"}.get(typ, typ)
                if code == 404:
                    at = "not_found_error"
                return self._json(code, {"type": "error", "error": {"type": at, "message": msg}}, extra)
            self._json(code, {"error": {"message": msg, "type": typ, "code": code}}, extra)

        def _auth(self):
            ok, h = app.authorized(self.headers.get("Authorization"), self.headers.get("x-api-key"))
            if not ok:
                log(f"401 {self.command} {self.path} from {self.client_address[0]} (key {h})")
                self._err(401, "invalid or missing API key", "authentication_error", {"WWW-Authenticate": "Bearer"})
            return ok

        def do_GET(self):
            p = self.path.split("?")[0].rstrip("/")
            if p == "/health":
                w = app.worker
                ok, why = w.health()
                h = {"status": "ok" if ok else "error", "busy": w.busy is not None, "active": len(w.active),
                     "queued": w.queued()}
                if not ok:
                    h["reason"] = why
                return self._json(200 if ok else 503, h)
            if not self._auth():
                return
            if p == "/v1/models":
                return self._json(200, {"object": "list", "data": [{
                    "id": m.id, "object": "model", "created": app.created, "owned_by": "ijah",
                    "context_length": app.ctx, "max_model_len": app.ctx, "capabilities": app.capabilities(m),
                    "input_modalities": ["text", "image"] if app.vision else ["text"],
                    "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"],
                    "default_reasoning_effort": app.a.effort} for m in app.models.values()]})
            self._err(404, f"no route {self.command} {p}")

        def handle_expect_100(self):
            """curl sends 'Expect: 100-continue' for big bodies: refuse an oversize one before the client sends it"""
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = 0
            if n > MAX_BODY:
                self.close_connection = True
                self._err(413, f"request body of {n} bytes exceeds the limit of {MAX_BODY} (NE_SRV_MAX_BODY_MB)")
                return False
            return super().handle_expect_100()

        def _drain(self, n, limit=16 << 20):
            """read and drop (up to limit bytes) a body we refuse, so the client sees our reply, not a reset"""
            try:
                left = min(n, limit)
                while left > 0:
                    d = self.rfile.read(min(left, 1 << 16))
                    if not d:
                        break
                    left -= len(d)
            except (OSError, ValueError):
                pass

        def _body(self):
            """-> the request body (bytes) or None after the error reply. Content-Length is required, a non-negative
            integer, at most MAX_BODY; the body is read in full before anything else is answered, so the connection stays
            in step for the next request (keep-alive) whatever the reply is."""
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                self.close_connection = True
                self._err(411, "chunked request bodies are not supported: send a Content-Length")
                return None
            cl = self.headers.get("Content-Length")
            if cl is None:
                self.close_connection = True
                self._err(411, "Content-Length required")
                return None
            try:
                n = int(cl)
                if n < 0 or not cl.strip().isdigit():
                    raise ValueError
            except ValueError:
                self.close_connection = True
                self._err(400, f"bad Content-Length {cl[:40]!r}")
                return None
            if n > MAX_BODY:
                self.close_connection = True   # (the stream is abandoned after the reply)
                self._err(413, f"request body of {n} bytes exceeds the limit of {MAX_BODY} (NE_SRV_MAX_BODY_MB)")
                self._drain(n)
                return None
            try:
                data = self.rfile.read(n)
            except (OSError, ValueError):   # timeout / reset while the client was sending
                self.close_connection = True
                return None
            if len(data) < n:   # the client closed early
                self.close_connection = True
                return None
            return data

        def do_POST(self):
            p = self.path.split("?")[0].rstrip("/")
            B.trace(f"POST {p}")
            data = self._body()
            B.trace(f"body read: {len(data or b'')} bytes")
            if data is None:
                return
            if p not in ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/messages/count_tokens"):
                if not self._auth():
                    return
                return self._err(404, f"no route {self.command} {p}")
            if not self._auth():
                return
            try:
                req = json.loads(data or b"{}")
                if not isinstance(req, dict):
                    raise chat.RequestError("body must be a JSON object")
                if p == "/v1/completions":
                    rid = "cmpl-" + secrets.token_hex(12)
                    jobs = app.make_jobs(req, rid, raw=self._raw_prompts(req))
                    fmt = CompFmt(app, rid, req, jobs[0].model.id)
                elif p == "/v1/messages":
                    rid = "msg_" + secrets.token_hex(12)
                    oreq = anthropic_api.to_openai(req)
                    jobs = app.make_jobs(oreq, rid)
                    fmt = AnthropicFmt(app, rid, oreq, jobs[0].model.id)
                    req = oreq
                elif p == "/v1/messages/count_tokens":
                    oreq = anthropic_api.to_openai(dict(req, max_tokens=req.get("max_tokens") or 1))
                    M = app.model_for(req.get("model"))
                    ids, _ = M.tpl.render_ids(oreq)
                    return self._json(200, {"input_tokens": len(ids)})
                else:
                    rid = "chatcmpl-" + secrets.token_hex(12)
                    jobs = app.make_jobs(req, rid)
                    B.trace(f"rendered + tokenized: {len(jobs[0].prompt)} tokens")
                    fmt = ChatFmt(app, rid, req, jobs[0].model.id)
            except (chat.RequestError, ValueError, TypeError, RecursionError) as e:
                return self._err(400, str(e) if not isinstance(e, RecursionError) else "request is nested too deeply")
            if req.get("stream"):
                self._stream(jobs, req, fmt)
            else:
                self._complete(jobs, req, fmt)

        def _raw_prompts(self, req):
            """/v1/completions `prompt`: a string, a list of strings, a token-id list or a list of token-id lists
            -> token-id lists (no chat template)"""
            M = app.model_for(req.get("model"))
            pr = req.get("prompt")
            if isinstance(pr, str):
                pr = [pr]
            elif isinstance(pr, list) and pr and all(isinstance(x, int) and not isinstance(x, bool) for x in pr):
                pr = [pr]
            if not isinstance(pr, list) or not pr or len(pr) > 64:
                raise chat.RequestError("prompt must be a string, a list of strings, token ids or lists of token ids")
            for k in ("echo", "suffix"):
                if req.get(k):
                    raise chat.RequestError(f"{k} is not supported")
            if (req.get("best_of") or 1) != 1:
                raise chat.RequestError("best_of is not supported")
            out = []
            for x in pr:
                if isinstance(x, str):
                    ids = M.tpl.encode(x)
                elif isinstance(x, list) and x and all(isinstance(t, int) and not isinstance(t, bool) and
                                                       0 <= t < M.backend.VOCAB for t in x):
                    ids = list(x)
                else:
                    raise chat.RequestError("prompt must be a string, a list of strings, token ids or lists of token ids")
                if not ids:
                    raise chat.RequestError("empty prompt")
                out.append(ids)
            return out

        # -- generation output
        def _pipeline(self, job, req, raw=False):
            detok = chat.Detok(job.model.tok)
            stop = chat.StopFilter(req.get("stop"))
            tools_on = bool(req.get("tools")) and req.get("tool_choice") != "none" and not raw
            parser = chat.RawParser() if raw else chat.OutputParser(job.thinking, chat.tool_schemas(req), tools_on)
            state = {"n": 0, "reasoning_tokens": 0, "eos": False, "stop_hit": False, "lps": [], "stop": stop}
            stops = set(B.RealBackend.STOPS)
            want_lp = job.extras.logprobs
            top_n = job.extras.top_logprobs

            def feed(toks, lps=None):
                out = []
                for t in toks:
                    if state["eos"] or state["stop_hit"]:
                        break
                    state["n"] += 1
                    if t in stops:
                        state["eos"] = True
                        break
                    if parser.mode == "reasoning":
                        state["reasoning_tokens"] += 1
                    if want_lp and lps is not None:
                        state["lps"].append(SMP.lp_entry(job.model.texts, t, lps, top_n))
                    s = stop.feed(detok(t))
                    if s:
                        out += parser.feed(s)
                    if stop.hit:
                        state["stop_hit"] = True
                        job.cancel.set()
                return out

            def finish():
                out = []
                s = stop.flush()
                if s:
                    out += parser.feed(s)
                return out + parser.finish()

            return feed, finish, state, parser

        def _peer_closed(self):
            """the client's socket is closed or reset. The request body is already read, so a readable socket with
            nothing to read (select + MSG_PEEK -> b"") is the peer's FIN; data waiting is a pipelined request (alive)."""
            try:
                sock = self.connection
                if not select.select([sock], [], [], 0)[0]:
                    return False
                return sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
            except (BlockingIOError, InterruptedError):
                return False
            except (OSError, ValueError):   # reset, or the socket is already closed
                return True

        def _wait_events(self, job, on_toks, on_idle=None):
            """-> ("done", stats) | ("error", msg) | ("gone", None): the client closed its connection while we waited
            (checked every POLL_S, also while no token arrives: prefill, held tool arguments, a queue). on_idle(now)
            runs on the same beat (a stream's keepalive)."""
            t_chk = time.monotonic()
            while True:
                try:
                    ev = job.events.get(timeout=POLL_S)
                except queue.Empty:
                    ev = None
                if ev is not None:
                    if ev[0] != "toks":
                        return ev
                    on_toks(ev[1], ev[2] if len(ev) > 2 else None)
                now = time.monotonic()
                if ev is None or now - t_chk >= POLL_S:
                    t_chk = now
                    if self._peer_closed():
                        return ("gone", None)
                    ok, why = app.worker.health()
                    if not ok:   # nobody will answer this request
                        return ("error", why)
                if on_idle is not None:
                    on_idle(now)

        def _finish_reason(self, state, parser, stats, job):
            if parser.finished_calls and not parser.cut:   # a call cut off mid-parameter is no tool call: stop / length
                return "tool_calls"
            if state["eos"] or state["stop_hit"]:
                return "stop"
            return "length" if stats.get("completion_tokens", 0) >= job.max_new or \
                stats["prompt_tokens"] + stats.get("completion_tokens", 0) >= getattr(job, "ctx", app.ctx) else "stop"

        def _usage(self, results):
            """OpenAI usage: the prompt once, the completions summed (n > 1)"""
            seen, ntok, rtok, ptok, ctok = set(), 0, 0, 0, 0
            for r in results:
                ntok += r["state"]["n"]
                rtok += r["state"]["reasoning_tokens"]
                if id(r["job"].prompt) not in seen:   # n choices over one prompt: the prompt counts once
                    seen.add(id(r["job"].prompt))
                    ptok += r["stats"]["prompt_tokens"]
                    ctok += r["stats"]["cached_tokens"]
            return {"prompt_tokens": ptok, "completion_tokens": ntok, "total_tokens": ptok + ntok,
                    "prompt_tokens_details": {"cached_tokens": ctok},
                    "completion_tokens_details": {"reasoning_tokens": rtok}}

        def _timings(self, stats, state):
            dec = stats.get("decode_ms") or 0
            t = {"path": stats["path"], "slot": stats.get("slot"), "prefill_ms": round(stats.get("prefill_ms", 0), 1),
                 "decode_ms": round(dec, 1), "queue_ms": round(stats.get("queue_ms", 0), 1),
                 "decode_tok_s": round((state["n"] - 1) / (dec / 1e3), 2) if dec > 0 and state["n"] > 1 else None}
            if stats.get("host_us_per_token") is not None:
                t["host_us_per_token"] = round(stats["host_us_per_token"], 1)   # step path, alone: the server's cost
            if stats.get("grammar"):
                t["grammar"] = stats["grammar"]   # object | schema | required | forced: the step path with a token mask
            if stats.get("sampling"):
                t["sampling"] = stats["sampling"]  # the request's logit_bias / penalties / min_p / logprobs / seed
            for k in ("restore_ms", "persist_ms"):  # prefix restore (RAM / SSD) and the SSD write-through's engine time
                if stats.get(k) is not None:
                    t[k] = round(stats[k], 1)
            if stats.get("cache_src"):
                t["cache"] = stats["cache_src"]   # live | ram | ssd | none (where the reused prefix came from)
            if stats.get("shared_src"):
                t["shared_prefix"] = True   # lab-snap: restored from a shared-prefix snapshot (docs/server.md 3f)
            if stats.get("shared_made"):
                t["shared_made"] = stats["shared_made"]   # shared-prefix snapshots this request captured
            return t

        def _result(self, job, ev, deltas, state, parser):
            content, reasoning, calls = chat.assemble(deltas)
            stats = ev[1]
            return {"job": job, "content": content, "reasoning": reasoning, "calls": calls, "stats": stats,
                    "state": state, "finish": self._finish_reason(state, parser, stats, job),
                    "stop_string": state["stop"].matched, "lps": state["lps"]}

        def _complete(self, jobs, req, fmt):
            """the choices run one after the other (each is submitted when the previous one is done). However this
            returns (reply, error, client gone, an exception) the jobs are cancelled: a finished one ignores it, a
            running one stops after its current step and frees its slot."""
            results = []
            try:
                for job in jobs:
                    feed, finish, state, parser = self._pipeline(job, req, fmt.raw)
                    deltas = []
                    app.worker.q.put(job)
                    ev = self._wait_events(job, lambda t, lp=None: deltas.extend(feed(t, lp)))
                    if ev[0] == "gone":
                        self.close_connection = True
                        log(f"{jobs[0].id}: client disconnected")
                        return
                    if ev[0] == "error":
                        return self._err(500, f"engine error: {ev[1]}", "server_error")
                    deltas += finish()
                    results.append(self._result(job, ev, deltas, state, parser))
                self._json(200, fmt.full(self, results))
            finally:
                for j in jobs:
                    j.cancel.set()

        def _stream(self, jobs, req, fmt):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            gone = [False]

            def hangup():
                if not gone[0]:
                    gone[0] = True
                    for j in jobs:
                        j.cancel.set()   # client went away: stop the generation
                    log(f"{jobs[0].id}: client disconnected")

            last = [time.monotonic()]   # the last write: a stream silent for KEEPALIVE_S gets a keepalive

            def send(msgs):
                if gone[0]:
                    return
                try:
                    for m in msgs:
                        self.wfile.write(fmt.wire(m))
                    self.wfile.flush()
                    last[0] = time.monotonic()
                except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                    hangup()

            def idle(now):
                """nothing sent for KEEPALIVE_S (a long prefill, a tool call whose arguments are held until each
                parameter is complete, a reasoning pause): an SSE comment keeps proxies and clients from timing out and
                makes a dead socket fail the write"""
                if not gone[0] and now - last[0] >= KEEPALIVE_S:
                    try:
                        self.wfile.write(fmt.keepalive)
                        self.wfile.flush()
                        last[0] = now
                    except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                        hangup()

            results, failed = [], None
            try:
                send(fmt.begin(jobs))
                for job in jobs:
                    if gone[0]:
                        break
                    feed, finish, state, parser = self._pipeline(job, req, fmt.raw)
                    app.worker.q.put(job)
                    send(fmt.start_choice(job))
                    pending = []   # logprob entries of tokens that produced no output yet (partial characters, held text)

                    def emit(deltas):
                        lps = pending + state["lps"]
                        pending.clear()
                        state["lps"].clear()
                        msgs = fmt.deltas(job, deltas, lps)
                        if msgs is None:   # nothing sendable yet: keep the logprobs for the next chunk
                            pending.extend(lps)
                        else:
                            send(msgs)
                            if B.TRACE and not state.get("_traced") and msgs:
                                state["_traced"] = True
                                B.trace("first delta written")

                    ev = self._wait_events(job, lambda t, lp=None: emit(feed(t, lp)), idle)
                    if ev[0] == "gone":
                        hangup()
                        break
                    if ev[0] == "error":
                        failed = ev[1]
                        break
                    emit(finish())
                    results.append(self._result(job, ev, [], state, parser))
                    if pending:
                        send(fmt.deltas(job, [], list(pending), force=True))
                    send(fmt.end_choice(job, results[-1]))
                if failed is not None:
                    send(fmt.error(f"engine error: {failed}"))
                elif not gone[0]:
                    send(fmt.end(self, results))
                if not gone[0] and fmt.done_marker:
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except OSError:
                        pass
            finally:
                for j in jobs:   # on any exit (done, error, client gone, an exception here): free the engine
                    j.cancel.set()

    return H


MODELS = HERE / "models.json"


def pack_complete(d):
    """A pack directory is servable: config + index + tokenizer present, and a T6 pack (trunk6 metadata) has all its
    layers converted plus the bf16 mixer (a partial T6 pack is a valid but mixed model: not served by id)."""
    d = Path(d)
    if not all((d / f).exists() for f in ("config.json", "model.safetensors.index.json", "tokenizer.json")):
        return False
    t6 = json.loads((d / "model.safetensors.index.json").read_text()).get("metadata", {}).get("trunk6")
    if t6 is None:
        return True
    n = json.loads((d / "config.json").read_text()).get("text_config", {}).get("num_hidden_layers", 48)
    return len(t6.get("layers", [])) == n and bool(t6.get("mixer_bf16"))


def resolve_model(model_id, pack=None, registry=MODELS):
    """(model id, pack dir, note). --pack wins (the id is then only the name reported); else the registry's first
    complete pack for the id ('@IJAH_PACKS/' = the packs directory, IJAH_PACKS, default ~/.cache/ijah/packs); unknown
    id without --pack -> SystemExit listing the known ids."""
    reg = json.loads(Path(registry).read_text())
    model_id = model_id or reg["default"]
    if pack:
        return model_id, str(Path(pack).expanduser()), "--pack"
    if model_id not in reg["models"]:
        raise SystemExit(f"unknown --model-id {model_id!r} (known: {', '.join(reg['models'])}; or pass --pack DIR)")
    tried = []
    for c in reg["models"][model_id]["packs"]:
        d = str(ijah_paths.packs_dir() / c[len("@IJAH_PACKS/"):]) if c.startswith("@IJAH_PACKS/") else str(Path(c).expanduser())
        if pack_complete(d):
            return model_id, d, f"registry ({'first choice' if not tried else 'fallback; incomplete or missing: ' + ', '.join(tried)})"
        tried.append(d)
    hint = "".join(f"; download: hf download {ijah_paths.HF_ORG}/{Path(t).name} --local-dir {t}" for t in tried[:1])
    raise SystemExit(f"--model-id {model_id}: no complete pack among {tried}{hint} (or pass --pack DIR)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--pack", default=None, help="pack directory (default: the --model-id's pack in server/models.json)")
    ap.add_argument("--ctx", type=int, default=262144, help="context window = KV size (tmax); reply cap = this")
    ap.add_argument("--mtp", type=int, default=2, help="MTP draft depth (0 = off; > 2: greedy requests still draft 2 "
                    "deep, docs/mtp.md 11)")
    ap.add_argument("--mtp-policy", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--effort", default="xhigh", choices=["xhigh", "medium", "low"],
                    help="reasoning effort when a request names none")
    ap.add_argument("--ckpts", type=int, default=4,
                    help="checkpoints kept per conversation (~113 MB each): the system prompt's + the newest")
    ap.add_argument("--slots", type=int, default=1, help="model instances = requests running at once (each extra slot holds a full --ctx KV: ~+7 GB at 262K)")
    ap.add_argument("--slot-ctx", type=int, default=0, help="context window of slots 2.. (0 = --ctx). User rule: never below the advertised window — a smaller slot silently caps replies")
    ap.add_argument("--pf-cap", type=int, default=0,
                    help="largest prefill piece while another request decodes (e.g. 1025; 0 = canonical pieces, "
                         "outputs independent of concurrency)")
    ap.add_argument("--cache-ram", default="8GB", help="prefix cache RAM budget (parked KV rows + checkpoints)")
    ap.add_argument("--cache-disk", default="0", help="prefix cache SSD budget (e.g. 20GB; on = 20GB; 0 = off); "
                    "the SSD tier persists across restarts")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache/ijah/prefix-cache"),
                    help="SSD tier directory (kept across restarts; entries keyed by build + pack + model)")
    ap.add_argument("--key-file", default=str(KEY_FILE))
    ap.add_argument("--model-id", default=None, help="model id reported and, without --pack, looked up in "
                    "server/models.json (default: its 'default', Qwen3.8-Flash-Next-native); "
                    "ID1,ID2: serve both from one process (shared expert buffers; requests routed by `model`; "
                    "the first is the default)")
    ap.add_argument("--share-state", action="store_true", help="several models: slot j of every model uses ONE KV + "
                    "recurrent state (one set of buffers: -7.9 GB per extra model at a 262K window); requests to "
                    "different models then run one after another, a switch parks the other model's conversation")
    ap.add_argument("--models", default=None, help="'all': every registry id with a complete pack (= --model-id "
                    "of all of them, the registry default first)")
    ap.add_argument("--list-models", action="store_true", help="print the registry with each id's resolved pack")
    ap.add_argument("--lib", default=os.environ.get("NE_LIB", str(ROOT / "build/libengine.dylib")))
    ap.add_argument("--shim", default=None, help="snapshot shim dylib (tools/serve.sh build); none = live reuse only")
    ap.add_argument("--pid-file", default="")
    ap.add_argument("--no-grammar", action="store_true", help="no structured output (response_format, forced tools)")
    ap.add_argument("--seed-mtp", action="store_true", help="a request with a seed may take the MTP path (default: "
                    "seeded requests step, which reproduces for a seed; MTP rounds are only best effort)")
    ap.add_argument("--prefill-int8", action="store_true", help="opt-in: int8 activations for the T6 trunk Linears in "
                    "prefill (in >= 1024; NE_T6_A8=128): ~+3-4%% cold prefill, KLD A n.s. / B +2%% (CI touches 0) / C "
                    "better (docs/prefill.md 4f); off by default to keep quality")
    ap.add_argument("--mock", action="store_true", help="CPU mock engine (tests)")
    ap.add_argument("--mock-delay", type=float, default=0.0, help="mock seconds per token")
    ap.add_argument("--mock-prefill-rate", type=float, default=0.0, help="mock prefill tokens per second (0 = instant)")
    a = ap.parse_args()
    if a.prefill_int8:   # read by the engine library (kernels/trunk6) when the model loads, so set before loading
        os.environ.setdefault("NE_T6_A8", "128")
    if os.environ.get("NE_T6_A8", "0") not in ("", "0"):
        log(f"prefill int8 trunk activations ON (NE_T6_A8={os.environ['NE_T6_A8']})")
    if a.list_models:
        for mid in json.loads(MODELS.read_text())["models"]:
            try:
                print(f"{mid}: {resolve_model(mid)[1]}  [{resolve_model(mid)[2]}]")
            except SystemExit as e:
                print(f"{mid}: {e}")
        return
    if a.models:
        if a.models != "all":
            raise SystemExit("--models takes 'all' (or use --model-id ID1,ID2)")
        reg = json.loads(MODELS.read_text())
        ids = [reg["default"]] + [m for m in reg["models"] if m != reg["default"]]
        ok = []
        for mid in ids:
            try:
                resolve_model(mid)
                ok.append(mid)
            except SystemExit as e:
                log(f"--models all: skipping {mid} ({e})")
        a.model_id = ",".join(ok)
    ids = [x.strip() for x in (a.model_id or "").split(",") if x.strip()] or [None]
    if len(ids) > 1 and a.pack:
        raise SystemExit("--pack serves one model; with several --model-id ids every id needs a registry pack")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"--model-id lists an id twice: {a.model_id}")
    a.served = []
    for mid in ids:
        mid, pack, how = resolve_model(mid, a.pack)
        log(f"model {mid}: pack {pack} ({how})")
        a.served.append((mid, pack))
    a.model_id, a.pack = a.served[0]
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("bind to loopback only")
    a.ctx = (a.ctx + 63) // 64 * 64
    if a.slot_ctx <= 0:
        a.slot_ctx = a.ctx
    if a.slot_ctx < a.ctx:
        raise SystemExit(f"--slot-ctx {a.slot_ctx} < --ctx {a.ctx}: every slot must hold the full context window (no reply caps)")
    a.slot_ctx = (a.slot_ctx + 63) // 64 * 64
    a.slots = max(1, a.slots)
    import signal

    def on_term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    if a.pid_file:
        Path(a.pid_file).write_text(str(os.getpid()))
    app = App(a)
    fatal = []

    def on_fatal():   # the engine thread died: stop serving a dead engine, exit non-zero so the supervisor restarts us
        fatal.append(1)
        os.kill(os.getpid(), signal.SIGTERM)

    app.worker.on_fatal = on_fatal
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    srv.daemon_threads = True
    log(f"listening on http://{a.host}:{a.port} (pid {os.getpid()})")
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        late = app.close(timeout=30)
        if late:
            log(f"prefix cache: some SSD writes did not land within 30 s ({', '.join(late)})")
        if a.pid_file:
            Path(a.pid_file).unlink(missing_ok=True)
        log("stopped")
    if fatal:
        sys.exit(70)


if __name__ == "__main__":
    main()
