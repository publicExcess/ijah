"""Multi-conversation prefix cache of the Ijah server (lab-srvcache, lab-persist; docs/server.md section 3b).

A conversation (Conv) = the token sequence `toks` whose KV rows it describes + GDN/PLE checkpoints (Snap: the whole
recurrent state at a position p, a prefix of toks) + host copies of its KV rows (segments). A conversation is either
RESIDENT in a slot (an engine model instance: its KV buffer holds rows [0, len(toks)) and the segments, if any, are
copies of the first rows) or PARKED (only the host copies: rows [0, saved_n), toks cut to the newest checkpoint).
Restoring a parked conversation at checkpoint p = copy KV rows [0, p) back into a slot + load the checkpoint's state.

KV is O(T) per conversation (~28 KB per token), the recurrent state O(1) (113 MB): parking copies the KV rows
(~45 GB/s, 50K tokens in ~30 ms) instead of re-prefilling them (~1500 tok/s, 50K tokens in ~35 s).

Forks share buffers: a conversation started from another one's checkpoint references the same Snap objects and the
same segment buffers (immutable numpy arrays); bytes are counted once per buffer. The RAM budget evicts whole
parked conversations, least recently used first.

SSD tier (lab-persist): a durable mirror of conversations that survives server restarts and crashes. Every finished
request writes its slot's conversation through (only what changed: the KV rows past the last write + new
checkpoints), a RAM eviction completes the mirror, and a hit reads it back. Entries live under
<dir>/<fingerprint>/<entry>/ (fingerprint = sha256 of the format, the engine library + metallib contents, the numerics
env, the pack's files (name, size, mtime), the model id and the state / KV layout), so an entry is only ever restored
by the build and pack that wrote it. Each entry = raw .bin files (tokens, checkpoints, KV segments) + manifest.json
(written last, tmp + rename) with the size and crc32 of every file; at start the directory is indexed (manifests,
token files checked), damaged entries and stray files are deleted, and a hit verifies every file it reads (a mismatch
drops the entry and the request prefills cold). Other fingerprints' directories are kept (base vs Swift pack) and are
the first to go when the budget is exceeded; then entries least recently used (manifest mtime) first.
"""
import errno
import fcntl
import hashlib
import json
import os
import queue
import shutil
import threading
import time
import zlib
from pathlib import Path

import numpy as np

FORMAT = 1
MAGIC = "native-engine prefix cache"


def parse_bytes(s):
    """'8GB' / '512MB' / '0' / '1.5G' -> bytes (binary units)"""
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().upper().rstrip("B").rstrip("I")
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(float(s or 0))


def lcp(a, b, block=4096):
    """length of the common prefix of two int lists (slices compare in C). pfast: forward in blocks, then within the
    first block that differs -- O(prefix) element copies; the old whole-list compare + binary search copied up to
    ~2 x 18 slices of the full length on every mismatch (10.8 ms vs 0.65 ms at 190K tokens; every request whose
    previous reply re-renders differently paid it per candidate, ~26 ms queued at 188K, docs/prefill.md 4i)"""
    n = min(len(a), len(b))
    i = 0
    while i + block <= n and a[i:i + block] == b[i:i + block]:
        i += block
    hi = min(n, i + block)
    if a[i:hi] == b[i:hi]:
        return hi
    while a[i] == b[i]:   # (a difference lies in [i, hi))
        i += 1
    return i


class Snap:
    """recurrent state at position pos (buf: backend-defined numpy array, meta: backend-defined)"""
    __slots__ = ("pos", "buf", "meta")

    def __init__(self, pos, buf, meta):
        self.pos, self.buf, self.meta = pos, buf, meta


class Conv:
    _next = 0

    def __init__(self):
        Conv._next += 1
        self.id = Conv._next
        self.toks = []
        self.snaps = {}     # pos -> Snap (pos <= len(toks))
        # [a, b, buf, cap, ha, hb, hcap]: host copy of KV rows [a, b) (buf laid out for cap rows) and of the MTP
        # head's KV rows [ha, hb) (hcap rows, behind the main part); the segments tile [0, saved_n) and [0, head_n)
        self.segs = []
        self.slot = None    # index of the slot whose KV buffer holds rows [0, len(toks)), or None (parked)
        self.last = time.monotonic()
        self.disk = None    # DiskEntry mirroring this conversation (SSD tier), or None
        self.dvalid = 0     # rows [0, dvalid) of that mirror (and its checkpoints up to dvalid) are this conversation's
        # lab-snap: a shared-prefix snapshot (docs/server.md 3f): toks = a token prefix several sessions share, one
        # checkpoint at its end, its KV rows; never resident, never extended: a request restoring it gets a fork
        self.shared = False

    @property
    def saved_n(self):
        return self.segs[-1][1] if self.segs else 0

    @property
    def head_n(self):
        return self.segs[-1][5] if self.segs else 0

    @staticmethod
    def _cut(segs, p):
        """segments for the prefix p: rows < p; head rows < p - 1 (head row j pairs with token j + 1)"""
        h = max(0, p - 1)
        return [[a, min(b, p), buf, cap, min(ha, h), min(hb, h), hcap] for a, b, buf, cap, ha, hb, hcap in segs
                if a < p]

    def touch(self):
        self.last = time.monotonic()

    def best_snap(self, limit):
        """largest checkpoint position <= limit usable now (parked: also <= saved_n)"""
        lim = limit if self.slot is not None else min(limit, self.saved_n)
        ps = [p for p in self.snaps if p <= lim]
        return max(ps) if ps else 0

    def truncate(self, p):
        self.toks = self.toks[:p]
        for q in [q for q in self.snaps if q > p]:
            del self.snaps[q]
        self.segs = self._cut(self.segs, p)
        self.dvalid = min(self.dvalid, p)  # rows past p will be recomputed: the mirror's are another history

    def fork(self, p):
        c = Conv()
        c.toks = self.toks[:p]
        c.snaps = {q: s for q, s in self.snaps.items() if q <= p}
        c.segs = self._cut(self.segs, p)
        return c

    def thin(self, keep):
        """keep at most `keep` checkpoints: the first (system prompt, shared by conversations) and the newest"""
        ps = sorted(self.snaps)
        if len(ps) > keep:
            for q in ps[1:len(ps) - keep + 1]:
                del self.snaps[q]

    def buffers(self):
        for s in self.snaps.values():
            yield s.buf
        for s in self.segs:
            yield s[2]


# ---------------------------------------------------------------------------------------------- SSD tier identity
_ENV_SKIP = {"NE_NO_LOCK", "NE_LIB", "NE_METALLIB", "NE_SERVE_PORT", "NE_MODEL", "NE_VISION_ROWS", "NE_MTP_DEBUG",
             "NE_PACK"}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def metallib_of(lib):
    """the metallib the engine library loads (core/pipelines.mm find_metallib)"""
    e = os.environ.get("NE_METALLIB")
    if e:
        return Path(e)
    cand = Path(lib).parent / "engine.metallib"
    return cand if cand.exists() else Path("build/engine.metallib")


def identity(lib, pack, model_id, layout):
    """-> dict naming everything a stored KV row / checkpoint depends on (its sha256 is the entry namespace).
    Contents of the engine library and metallib (any rebuild changes them), the numerics env (NE_* knobs), the pack's
    files by name / size / mtime (weights are 81 GB: statted, not hashed), the model id, the backend's state layout."""
    ident = {"format": FORMAT, "model_id": model_id, "layout": layout,
             "env": {k: v for k, v in sorted(os.environ.items())
                     if k.startswith("NE_") and k not in _ENV_SKIP and not k.startswith("NE_PROF")}}
    if lib is not None:
        lib = Path(lib).resolve()
        ml = metallib_of(lib)
        ident["lib"] = {"path": str(lib), "sha256": _sha256_file(lib)}
        ident["metallib"] = {"path": str(ml), "sha256": _sha256_file(ml) if ml.exists() else None}
    if pack is not None:
        pk = Path(pack).expanduser().resolve()
        h = hashlib.sha256()
        for f in sorted(pk.rglob("*")):
            try:
                st = f.stat()  # through symlinks (hard / soft links to shared expert files)
            except OSError:
                continue
            if f.is_file():
                h.update(f"{f.relative_to(pk)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
        ident["pack"] = {"path": str(pk), "files": h.hexdigest()}
    return ident


def fingerprint(ident):
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------- SSD tier
class DiskEntry:
    """one conversation on disk. The engine thread owns the record (toks, n, snaps, segs, sizes, tokf, bytes, conv,
    serial); the writer owns crc (file -> (size, crc32) of every file it landed) and the manifest it last wrote."""
    __slots__ = ("name", "dir", "toks", "n", "snaps", "segs", "sizes", "tokf", "bytes", "last", "conv", "ram",
                 "pending", "idle", "crc", "bad", "dead", "serial", "manifest", "shared")

    def __init__(self, name, d):
        self.name, self.dir = name, d
        self.toks, self.n = [], 0
        self.snaps = {}     # pos -> (file, meta list)
        self.segs = []      # [a, b, cap, ha, hb, hcap, file] tiling [0, n) and the head rows [0, max hb)
        self.sizes = {}     # file -> bytes (every file of the record, written or queued)
        self.tokf = None
        self.bytes = 0
        self.last = time.time()
        self.conv = None    # the RAM conversation this entry mirrors (lookups use it instead), or None
        self.ram = None     # an evicted conversation kept until its writes have landed (a hit takes it back)
        self.pending = 0
        self.idle = threading.Event()
        self.idle.set()
        self.crc = {}
        self.bad = False    # a write failed: the entry is gone
        self.dead = False   # evicted: queued writes are skipped
        self.serial = 0
        self.manifest = None
        self.shared = False  # lab-snap: the mirror of a shared-prefix snapshot (manifest "shared": true)

    @property
    def snap_pos(self):
        return sorted(self.snaps)

    def head_n(self):
        return max([s[4] for s in self.segs] + [0])


class DiskTier:
    """the SSD mirror of conversations (see the module doc). One live tier per directory: an flock on <dir>/.lock;
    a second server on the same directory gets a private, non-persistent subdirectory."""

    def __init__(self, root, budget, ident=None, log=print, shared_cap=32):
        self.base = Path(root).expanduser()
        self.budget = budget
        self.log = log
        self.ident = ident or {"format": FORMAT, "anonymous": True}
        self.shared_cap = shared_cap  # shared-prefix snapshots kept on disk (lab-snap), least recently used go first
        self.fp = fingerprint(self.ident)
        self.entries = {}   # name -> DiskEntry (this fingerprint's)
        self.foreign = {}   # path -> [bytes, mtime] (other fingerprints' directories)
        self.lock = threading.Lock()
        self.q = queue.Queue()
        self.write_ms = 0.0
        self.read_ms = 0.0
        self.stats = {"persists": 0, "persist_ms": 0.0, "queued_bytes": 0, "rejected": 0, "indexed": 0,
                      "dropped_at_scan": 0}
        self._big_logged = set()
        self.base.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._lockf = open(self.base / ".lock", "a+")
        try:
            fcntl.flock(self._lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.private = False
            self.root = self.base / self.fp[:16]
        except OSError as ex:
            if ex.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                raise
            self._lockf.close()
            self._lockf = None
            self.private = True
            self.root = self.base / f"private-{os.getpid()}"
            shutil.rmtree(self.root, ignore_errors=True)
            log(f"prefix cache: {self.base} is used by another server: private SSD tier {self.root.name} "
                "(not kept across restarts)")
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        if not self.private:
            self._scan()
        self.th = threading.Thread(target=self._writer, daemon=True, name="cache-disk")
        self.th.start()
        with self.lock:
            self._enforce()

    # -- start: index what earlier processes left
    def _scan(self):
        t0 = time.perf_counter()
        idf = self.root / "ident.json"
        if not idf.exists():
            tmp = self.root / "ident.json.tmp"
            tmp.write_text(json.dumps(self.ident, indent=1, sort_keys=True))
            os.replace(tmp, idf)
        for d in self.base.iterdir():
            if d == self.root or not d.is_dir():
                continue
            nm = d.name
            if nm.startswith("private-"):  # a second server's private tier: removed once its process is gone
                try:
                    os.kill(int(nm.split("-", 1)[1]), 0)
                except (ValueError, ProcessLookupError):
                    shutil.rmtree(d, ignore_errors=True)
                except PermissionError:
                    pass
            elif len(nm) == 16 and all(ch in "0123456789abcdef" for ch in nm):
                self.foreign[d] = [_dir_bytes(d), _newest_mtime(d)]
            elif nm[:1] == "c" and nm[1:].isdigit():  # the old wiped-at-start tier's directories
                shutil.rmtree(d, ignore_errors=True)
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            try:
                e = self._load_manifest(d)
            except (OSError, ValueError, KeyError, TypeError, IndexError) as ex:
                self.log(f"prefix cache: dropped damaged SSD entry {d.name} ({ex})")
                self.stats["dropped_at_scan"] += 1
                shutil.rmtree(d, ignore_errors=True)
                continue
            self.entries[e.name] = e
        self.stats["indexed"] = len(self.entries)
        nf = sum(v[0] for v in self.foreign.values())
        self.log(f"prefix cache: SSD tier {self.root} ({self.budget / 2**30:.0f} GB budget): {len(self.entries)} "
                 f"conversations, {self.bytes() / 2**30:.2f} GB, indexed in {(time.perf_counter() - t0) * 1e3:.0f} ms"
                 + (f"; {len(self.foreign)} other builds / packs {nf / 2**30:.2f} GB" if self.foreign else ""))

    def _load_manifest(self, d):
        man = json.loads((d / "manifest.json").read_text())
        if man.get("magic") != MAGIC or man.get("format") != FORMAT:
            raise ValueError("unknown format")
        if man.get("fingerprint") != self.fp:
            raise ValueError("written by another build / pack")
        e = DiskEntry(d.name, d)
        tf, tsz, tcrc = man["toks"]
        files = {tf: (tsz, tcrc)}
        for pos, f, sz, crc, meta in man["snaps"]:
            e.snaps[int(pos)] = (f, [int(x) for x in meta])
            files[f] = (sz, crc)
        for a, b, cap, ha, hb, hcap, f, sz, crc in man["segs"]:
            e.segs.append([int(a), int(b), int(cap), int(ha), int(hb), int(hcap), f])
            files[f] = (sz, crc)
        for f, (sz, _) in files.items():
            st = (d / f).stat()
            if st.st_size != sz:
                raise ValueError(f"{f}: {st.st_size} bytes, manifest says {sz}")
        raw = (d / tf).read_bytes()
        if zlib.crc32(raw) != tcrc:
            raise ValueError(f"{tf}: checksum mismatch")
        e.toks = np.frombuffer(raw, np.int64).tolist()
        e.n = int(man["n"])
        if len(e.toks) != e.n or not e.snaps or max(e.snaps) != e.n or not e.segs or e.segs[-1][1] != e.n:
            raise ValueError("inconsistent record")
        e.crc = files
        e.sizes = {f: sz for f, (sz, _) in files.items()}
        e.tokf = tf
        e.bytes = sum(e.sizes.values())
        e.serial = int(man.get("serial", 0))
        e.shared = bool(man.get("shared", False))
        e.manifest = man
        e.last = (d / "manifest.json").stat().st_mtime
        for f in d.iterdir():  # files a crash left behind (tmp, written but never named by a manifest)
            if f.name != "manifest.json" and f.name not in files:
                try:
                    f.unlink()
                except OSError:
                    shutil.rmtree(f, ignore_errors=True)
        return e

    def bytes(self):
        return sum(e.bytes for e in self.entries.values())

    # -- engine thread: write a conversation through
    def persist(self, conv, kv_src=None):
        """make conv's mirror = conv up to its newest checkpoint m: KV rows [0, m), the checkpoints, the tokens.
        kv_src(a, m, ha) -> (buf, cap, hcap) | None: rows [a, m) and head rows [ha, ..) from the slot holding conv
        (resident), or None (parked: its host segments are written). Queues the writes; -> bytes queued."""
        if not conv.snaps or max(conv.snaps) <= 0:
            return 0
        if not self._room():
            return 0
        t0 = time.perf_counter()
        e = conv.disk
        if e is not None and (e.bad or e.dead or e.name not in self.entries):
            e = conv.disk = None
        m = max(conv.snaps)
        if e is None:
            conv.dvalid = 0
            name = f"e{time.time_ns():x}-{os.getpid()}-{conv.id}"
            e = DiskEntry(name, self.root / name)
            e.shared = conv.shared
            e.conv = conv
            conv.disk = e
            with self.lock:
                self.entries[name] = e
        elif conv.dvalid == m == e.n and set(e.snaps) == set(conv.snaps):
            e.last = time.time()
            return 0
        keep = min(conv.dvalid, e.n, lcp(e.toks, conv.toks[:m]))
        if kv_src is None and keep < m:  # parked: its host segments replace the record's rows
            keep = 0
        hkeep = min(e.head_n(), max(0, keep - 1))
        drops, writes = [], []
        segs = []
        for a, b, cap, ha, hb, hcap, f in e.segs:
            if a < keep:
                segs.append([a, min(b, keep), cap, min(ha, hkeep), min(hb, hkeep), hcap, f])
            else:
                drops.append(f)
        snaps = {}
        for p, (f, meta) in e.snaps.items():
            if p <= keep and p in conv.snaps:
                snaps[p] = (f, meta)
            else:
                drops.append(f)
        if kv_src is not None:
            r = kv_src(keep, m, hkeep)
            if r is not None:
                buf, cap, hcap = r
                e.serial += 1
                f = f"seg{e.serial}.bin"
                segs.append([keep, m, cap, hkeep, hkeep + hcap, hcap, f])
                writes.append((f, buf))
        elif keep < m:
            for a, b, buf, cap, ha, hb, hcap in conv.segs:
                if a < m:
                    e.serial += 1
                    f = f"seg{e.serial}.bin"
                    segs.append([a, min(b, m), cap, ha, hb, hcap, f])
                    writes.append((f, buf))
        if not segs or segs[-1][1] != m:
            self.log(f"prefix cache: c{conv.id}: KV rows end at {segs[-1][1] if segs else 0} < {m}; not on SSD")
            with self.lock:
                self._evict(e)
            return 0
        for p, s in conv.snaps.items():
            if p not in snaps:
                e.serial += 1
                f = f"snap{e.serial}.bin"
                snaps[p] = (f, [int(x) for x in np.asarray(s.meta).reshape(-1)])
                writes.append((f, s.buf))
        e.serial += 1
        tf = f"toks{e.serial}.bin"
        toks = list(conv.toks[:m])
        writes.append((tf, np.asarray(toks, np.int64)))  # int64: image tokens are 63-bit prefix keys (lab-vision)
        if e.tokf is not None:
            drops.append(e.tokf)
        for f in drops:
            e.sizes.pop(f, None)
        for f, b in writes:
            e.sizes[f] = int(b.nbytes)
        e.toks, e.n, e.snaps, e.segs, e.tokf = toks, m, snaps, segs, tf
        e.bytes = sum(e.sizes.values())
        e.last = time.time()
        conv.dvalid = m
        rec = {"n": m, "toks": tf, "snaps": [[p, f, meta] for p, (f, meta) in sorted(snaps.items())],
               "segs": [list(s) for s in segs], "serial": e.serial, "shared": e.shared}
        qb = sum(int(b.nbytes) for _, b in writes)
        with self.lock:
            e.pending += 1
            e.idle.clear()
            self.q.put(("write", e, writes, rec, drops))
            self._enforce(protect=e)
        self.stats["persists"] += 1
        self.stats["queued_bytes"] += qb
        self.stats["persist_ms"] += (time.perf_counter() - t0) * 1e3
        return qb

    FREE_RESERVE = 8 << 30  # never fill the volume: no new writes while less than this is free

    def _room(self):
        try:
            free = shutil.disk_usage(self.root).free
        except OSError:
            return True
        low = free < self.FREE_RESERVE
        if low != getattr(self, "_low", False):
            self._low = low
            self.log(f"prefix cache: {free / 2**30:.1f} GB free on the SSD tier's volume: "
                     + ("writes paused" if low else "writes resumed"))
        return not low

    def _evict(self, e):
        """(holding self.lock) drop an entry: its queued writes are skipped, its directory goes"""
        e.dead = True
        self.entries.pop(e.name, None)
        if e.conv is not None and e.conv.disk is e:
            e.conv.disk = None
            e.conv.dvalid = 0
        e.conv = None
        self.q.put(("rm", e.dir))

    def _enforce(self, protect=None):
        """(holding self.lock) entries least recently used first until the budget holds; other fingerprints first"""
        while self.foreign and self.bytes() + sum(v[0] for v in self.foreign.values()) > self.budget:
            d = min(self.foreign, key=lambda k: self.foreign[k][1])
            nb = self.foreign.pop(d)[0]
            self.q.put(("rm", d))
            self.log(f"prefix cache: SSD removed {d.name} (another build / pack, {nb / 2**30:.2f} GB)")
        sh = sorted((e for e in self.entries.values() if e.shared), key=lambda x: (x is protect, x.last))
        while len(sh) > self.shared_cap:  # lab-snap: at most shared_cap shared-prefix snapshots on disk
            e = sh.pop(0)
            self.log(f"prefix cache: SSD evicted shared prefix {e.name} ({e.n} tokens)")
            self._evict(e)
        while self.entries and self.bytes() > self.budget:
            cand = [e for e in self.entries.values() if e is not protect]
            if not cand:
                e = protect
                if e.name not in self._big_logged:
                    self._big_logged.add(e.name)
                    self.log(f"prefix cache: a conversation of {e.n} tokens ({e.bytes / 2**30:.1f} GB) is larger than "
                             f"the SSD budget ({self.budget / 2**30:.1f} GB): not kept")
            else:
                e = min(cand, key=lambda x: x.last)
                self.log(f"prefix cache: SSD evicted {e.name} ({e.n} tokens, {e.bytes / 2**30:.2f} GB)")
            self._evict(e)

    def evicted_from_ram(self, conv):
        """a parked conversation leaves RAM: complete its mirror (its buffers stay referenced until written).
        -> True if it is on the SSD tier now"""
        self.persist(conv, None)
        e = conv.disk
        if e is None:
            return False
        with self.lock:
            e.conv = None
            if e.pending:
                e.ram = conv
        conv.disk = None
        return True

    def unlink(self, conv):
        """conv leaves RAM (dropped): its mirror stays as an SSD-only entry"""
        e = conv.disk
        if e is not None and e.conv is conv:
            with self.lock:
                e.conv = None
        conv.disk = None

    # -- writer thread
    def _writer(self):
        while True:
            job = self.q.get()
            try:
                if job is None:
                    return
                if job[0] == "rm":
                    shutil.rmtree(job[1], ignore_errors=True)
                else:
                    self._write(*job[1:])
            except Exception as ex:  # noqa: BLE001 (the writer must survive anything)
                self.log(f"prefix cache: SSD writer error {ex!r}")
            finally:
                self.q.task_done()

    def _write(self, e, writes, rec, drops):
        t0 = time.perf_counter()
        try:
            if e.dead or e.bad:
                return
            e.dir.mkdir(mode=0o700, exist_ok=True)
            for f, buf in writes:
                mv = memoryview(np.ascontiguousarray(buf).reshape(-1).view(np.uint8))
                tmp = e.dir / (f + ".tmp")
                with open(tmp, "wb") as fh:
                    fh.write(mv)
                e.crc[f] = (mv.nbytes, zlib.crc32(mv))
                os.replace(tmp, e.dir / f)
            if e.dead:
                return
            c = e.crc
            man = {"magic": MAGIC, "format": FORMAT, "fingerprint": self.fp, "n": rec["n"], "serial": rec["serial"],
                   "toks": [rec["toks"], *c[rec["toks"]]],
                   "snaps": [[p, f, *c[f], meta] for p, f, meta in rec["snaps"]],
                   "segs": [[*s[:7], *c[s[6]]] for s in rec["segs"]], "written": time.time()}
            if rec.get("shared"):
                man["shared"] = True
            tmp = e.dir / "manifest.json.tmp"
            tmp.write_text(json.dumps(man))
            os.replace(tmp, e.dir / "manifest.json")  # the commit point: a crash before it leaves the old record
            e.manifest = man
            live = {man["toks"][0]} | {s[1] for s in man["snaps"]} | {s[6] for s in man["segs"]}
            for f in drops:
                if f not in live:
                    try:
                        (e.dir / f).unlink()
                    except FileNotFoundError:
                        pass
                    c.pop(f, None)
        except (OSError, KeyError) as ex:
            self.log(f"prefix cache: SSD write of {e.name} failed ({ex!r}); entry dropped")
            e.bad = True
            with self.lock:
                self.entries.pop(e.name, None)
            shutil.rmtree(e.dir, ignore_errors=True)
        finally:
            self.write_ms += (time.perf_counter() - t0) * 1e3
            with self.lock:
                e.pending -= 1
                if e.pending == 0:
                    e.ram = None
                    e.idle.set()

    # -- engine thread: a hit
    def take(self, e, meta_dtype):
        """-> Conv (parked, in RAM, mirrored by e) or None if the entry is damaged (it is then dropped)"""
        t0 = time.perf_counter()
        with self.lock:
            c = e.ram
            e.ram = None
        if c is not None:  # evicted from RAM, writes not landed yet: its buffers are still here
            c.disk, c.dvalid = e, e.n
            c.shared = e.shared
            e.conv = c
            e.last = time.time()
            c.touch()
            return c
        e.idle.wait()
        if e.bad or e.dead or e.manifest is None:
            return None
        man = e.manifest
        try:
            c = Conv()
            c.toks = list(e.toks)
            for pos, f, sz, crc, meta in man["snaps"]:
                c.snaps[int(pos)] = Snap(int(pos), self._read(e, f, sz, crc), np.asarray(meta, meta_dtype))
            for a, b, cap, ha, hb, hcap, f, sz, crc in man["segs"]:
                c.segs.append([a, b, self._read(e, f, sz, crc), cap, ha, hb, hcap])
            if sorted(c.snaps) != e.snap_pos or len(c.segs) != len(e.segs) or int(man["n"]) != e.n:
                raise ValueError("record and manifest disagree")
        except (OSError, ValueError) as ex:
            self.log(f"prefix cache: SSD entry {e.name} rejected ({ex}); prefilling cold")
            self.stats["rejected"] += 1
            with self.lock:
                self._evict(e)
            return None
        c.disk, c.dvalid = e, e.n
        c.shared = e.shared
        with self.lock:
            e.conv = c
        e.last = time.time()
        try:
            os.utime(e.dir / "manifest.json")  # least recently used across restarts = the manifest's mtime
        except OSError:
            pass
        c.touch()
        self.read_ms += (time.perf_counter() - t0) * 1e3
        return c

    @staticmethod
    def _read(e, f, sz, crc):
        buf = np.fromfile(e.dir / f, np.uint8)
        if buf.nbytes != sz:
            raise ValueError(f"{f}: {buf.nbytes} bytes, manifest says {sz}")
        if zlib.crc32(buf) != crc:
            raise ValueError(f"{f}: checksum mismatch")
        return buf

    def flush(self, timeout=None):
        """wait until every queued job has run (True) or the timeout passed (False)"""
        t_end = None if timeout is None else time.monotonic() + timeout
        while self.q.unfinished_tasks:
            if t_end is not None and time.monotonic() > t_end:
                return False
            time.sleep(0.02)
        return True

    def close(self, timeout=60):
        """finish the queued writes (up to timeout s), stop the writer, release the directory"""
        ok = self.flush(timeout)
        self.q.put(None)
        self.th.join(timeout=5)
        if self.private:
            shutil.rmtree(self.root, ignore_errors=True)
        if self._lockf is not None:
            self._lockf.close()
            self._lockf = None
        return ok


def _dir_bytes(d):
    n = 0
    for f in Path(d).rglob("*"):
        try:
            if f.is_file():
                n += f.stat().st_size
        except OSError:
            pass
    return n


def _newest_mtime(d):
    try:
        return max([f.stat().st_mtime for f in Path(d).rglob("manifest.json")] + [Path(d).stat().st_mtime])
    except OSError:
        return 0.0


class PrefixCache:
    """all conversations (resident and parked) + the RAM budget (+ the SSD tier)"""

    def __init__(self, ram_budget=8 << 30, max_snaps=4, disk_dir=None, disk_budget=0, log=print, ident=None,
                 max_shared=8, max_shared_disk=32):
        self.ram_budget = ram_budget
        self.max_snaps = max(2, max_snaps)
        self.max_shared = max_shared  # lab-snap: shared-prefix snapshots kept in RAM (more go to the SSD tier / away)
        self.convs = {}  # id -> Conv
        self.log = log
        self.disk = DiskTier(disk_dir, disk_budget, ident, log, max_shared_disk) \
            if disk_dir and disk_budget > 0 else None
        self.stats = {"parks": 0, "park_ms": 0.0, "loads": 0, "load_ms": 0.0, "evicted": 0, "disk_hits": 0,
                      "shared_made": 0, "shared_hits": 0}

    def new(self, slot=None):
        c = Conv()
        c.slot = slot
        self.convs[c.id] = c
        return c

    def add(self, c):
        self.convs[c.id] = c
        return c

    def drop(self, c):
        self.convs.pop(c.id, None)
        if self.disk is not None:
            self.disk.unlink(c)

    def ram_bytes(self):
        seen = {}
        for c in self.convs.values():
            for b in c.buffers():
                seen[id(b)] = b.nbytes
        return int(sum(seen.values()))

    def candidates(self, prompt, n, live_slots):
        """-> list of (p, conv | DiskEntry, kind) with kind 'live' | 'snap' | 'disk'; live_slots: {slot: live}"""
        out = []
        for c in self.convs.values():
            if not c.toks:
                continue
            m = lcp(c.toks, prompt)
            if c.slot is not None and live_slots.get(c.slot) and m == len(c.toks) and 0 < m <= n - 1:
                out.append((m, c, "live"))
            p = c.best_snap(min(m, n - 1))
            if p > 0:
                out.append((p, c, "snap"))
        if self.disk:
            with self.disk.lock:  # entries mirroring a conversation in RAM: that conversation is the candidate
                ents = [e for e in self.disk.entries.values() if e.conv is None and not e.bad]
            for e in ents:
                m = lcp(e.toks, prompt)
                ps = [p for p in e.snaps if p <= min(m, n - 1)]
                if ps:
                    out.append((max(ps), e, "disk"))
        return out

    def longest_match(self, prompt, n):
        """lab-snap: -> (L, owner): the longest common prefix (capped at n) of the prompt with any token sequence the
        cache knows (conversations resident or parked, shared-prefix snapshots, SSD entries), whatever checkpoints it
        has, and the Conv / DiskEntry holding it (None when L = 0)"""
        best, own = 0, None
        prompt = prompt[:n]
        for c in self.convs.values():
            if c.toks and c.toks[0] == prompt[0]:
                m = lcp(c.toks[:n], prompt)
                if m > best:
                    best, own = m, c
        if self.disk:
            with self.disk.lock:
                ents = [e for e in self.disk.entries.values() if e.conv is None and not e.bad]
            for e in ents:
                if e.toks and e.toks[0] == prompt[0]:
                    m = lcp(e.toks[:n], prompt)
                    if m > best:
                        best, own = m, e
        return min(best, n), own

    def shared(self):
        return [c for c in self.convs.values() if c.shared]

    def find_shared(self, toks):
        """a shared-prefix snapshot of exactly these tokens (RAM or SSD), or None"""
        x = len(toks)
        for c in self.convs.values():
            if c.shared and len(c.toks) == x and c.toks == toks:
                return c
        if self.disk:
            with self.disk.lock:
                for e in self.disk.entries.values():
                    if e.shared and not e.bad and e.n == x and e.toks == toks:
                        return e
        return None

    def persist(self, conv, kv_src=None):
        """write conv through to the SSD tier (no-op without one); -> bytes queued"""
        return self.disk.persist(conv, kv_src) if self.disk is not None else 0

    def _out(self, v):
        """a parked conversation leaves RAM: to the SSD tier if there is one, else dropped"""
        self.stats["evicted"] += 1
        if self.disk and v.snaps and self.disk.evicted_from_ram(v):
            self.convs.pop(v.id, None)
            self.log(f"prefix cache: c{v.id} ({len(v.toks)} tokens{', shared prefix' if v.shared else ''}) -> SSD only")
        else:
            self.drop(v)
            self.log(f"prefix cache: evicted c{v.id} ({len(v.toks)} tokens{', shared prefix' if v.shared else ''})")

    def enforce(self, protect=()):
        """evict least recently used parked conversations until the RAM budget holds (to disk if enabled); at most
        max_shared shared-prefix snapshots stay in RAM (least recently used go first)"""
        sh = sorted((c for c in self.convs.values() if c.shared and c.id not in protect), key=lambda x: x.last)
        while len(sh) > self.max_shared:
            self._out(sh.pop(0))
        while self.ram_bytes() > self.ram_budget:
            parked = [c for c in self.convs.values() if c.slot is None and c.id not in protect]
            if not parked:
                # only resident conversations left: drop their host copies, then thin their checkpoints
                for c in sorted(self.convs.values(), key=lambda x: x.last):
                    if c.segs and c.id not in protect:
                        c.segs = []
                        break
                else:
                    for c in sorted(self.convs.values(), key=lambda x: x.last):
                        if len(c.snaps) > 2 and c.id not in protect:
                            c.thin(2)
                            break
                    else:
                        self.log(f"prefix cache: {self.ram_bytes() / 2**30:.1f} GB > budget with nothing evictable")
                        return
                continue
            self._out(min(parked, key=lambda x: x.last))

    def close(self, timeout=60):
        """the SSD tier's queued writes land (up to timeout s); -> False if some did not"""
        if self.disk is not None:
            return self.disk.close(timeout)
        return True

    def summary(self):
        res = [c for c in self.convs.values() if c.slot is not None]
        par = [c for c in self.convs.values() if c.slot is None and not c.shared]
        nsh = len(self.convs) - len(res) - len(par)
        d = f", SSD {len(self.disk.entries)} conv {self.disk.bytes() / 2**30:.2f} GB" if self.disk else ""
        return (f"{len(res)} resident + {len(par)} parked conversations + {nsh} shared prefixes, RAM "
                f"{self.ram_bytes() / 2**30:.2f} / {self.ram_budget / 2**30:.0f} GB{d}")
