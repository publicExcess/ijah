#!/usr/bin/env python3
"""Ijah comparison charts -> assets/charts/*.svg (static SVG, light + dark via prefers-color-scheme, <title> tooltips).

Numbers are the measured ones from the lab notes (M5 Max 128 GB, Qwen3.8-Flash-Next): the speed numbers below, the
per-window KLD results in data/charts/ (the engine's tests/quality runs and oMLX's, scored the same way). Edit and re-run:
python3 tools/make_charts.py
Error bars: KLD = 95% bootstrap CI of the mean over the set's windows (per-window results files below); decode = 95%
bootstrap CI of the mean over 6 prompts x 3 seeds; prefill = median with min-max over 5 cold runs. Speed numbers are the
18 Oct same-client runs (the head-to-head harness, not in this repository: one streaming client, same prompts and seeds,
each engine on its own pack and best settings). Ijah vs oMLX (Jundot oQ4e) and Sushi (its own Sushi-4bpw pack). The
DFlash2 drafter marker is a 19 Oct lab build, not part of this release (see the hollow-marker footnotes).
"""
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "charts"
DATA = ROOT / "data" / "charts"   # per-window KLD results (paths inside scrubbed)

# engine -> (light colour, dark colour); categorical order fixed (validated: dataviz validate_palette.js, light PASS,
# contrast WARN -> every mark is direct-labelled)
ENG = {
    "Ijah":   ("#eb6834", "#d95926"),
    "oMLX":   ("#1baf7a", "#199e70"),
    "Sushi":  ("#3d7dd8", "#639bec"),
}
ORDER = list(ENG)
# Ijah prefill: filled from the final same-client run after the 18 Oct prefill work (median, (min, max))
PF_IJAH_10K, PF_IJAH_10K_RNG = 2120, (2083, 2284)   # 18 Oct master, 2 sessions x 5 cold runs (median, min-max)
PF_IJAH_32K, PF_IJAH_32K_RNG = 2052, (2043, 2095)
# weight formats (from the packs' configs)
BITS = {
    "Ijah":  "5-bit EXL3 experts · 6/8-bit EXL3 trunk",
    "oMLX":  "Jundot oQ4e: mixed 4/5/6/8-bit affine (4.72 bpw)",
    "Sushi": "Sushi-4bpw: EXL3 4-bit experts · 8-bit trunk",
}
SHORT = {"Ijah": "5b · 6/8b", "oMLX": "oQ4e 4.72b", "Sushi": "4b · 8b"}

# (value or None, tooltip note, (lo, hi) or None)
SPEED = {
    "Decode with MTP (tok/s)": {
        "Ijah":  (98.9, "T6 pack, MTP depth 2 (--seed-mtp), SD 10.2, range 83-119; with 4-bit experts 105.2; "
                        "with the DFlash2 drafter 120.4 (19 Oct lab build, not in this release)", (94.4, 103.6)),
        "oMLX":  (105.2, "Jundot oQ4e, tuned settings (MTP depth 2, INT8 prefill); SD 9.2, range 91-120", (100.9, 109.2)),
        "Sushi": (85.8, "Sushi-4bpw, its own best settings (MTP on); SD 12.1, range 67-108; paired Ijah/Sushi "
                        "1.162 (CI 1.122-1.203), 18/18 wins", (80.4, 91.1)),
    },
    "Prefill, 10K-token prompt (tok/s)": {
        "Ijah":  (PF_IJAH_10K, "master after the 18 Oct prefill work, cold, 5 runs", PF_IJAH_10K_RNG),
        "oMLX":  (2412, "Jundot oQ4e, INT8-activation prefill, cold, 5 runs (first run 1,419: warm-up)", (1419, 2632)),
        "Sushi": (2014, "Sushi-4bpw, cold, 5 runs", (1949, 2028)),
    },
    "Prefill, 32K-token prompt (tok/s)": {
        "Ijah":  (PF_IJAH_32K, "master after the 18 Oct prefill work, cold, 5 runs", PF_IJAH_32K_RNG),
        "oMLX":  (2812, "Jundot oQ4e, INT8-activation prefill, cold, 5 runs", (2778, 2838)),
        "Sushi": (2026, "Sushi-4bpw, cold, 5 runs", (2021, 2050)),
    },
}

# KLD vs the bf16 teacher (lower is closer); sets A (wikitext), B (reasoning traces), C (32K long context).
# per-window results: (file, set) per engine and set
KLD_SRC = {
    "Ijah":  {"A": (DATA / "kld-t6fuse-T6-f32.json", "A"),
              "B": (DATA / "kld-master-1716-T6-B.json", "B"),
              "C": (DATA / "kld-t6fuse-T6-f32.json", "C")},
    "oMLX":  {s: (DATA / "kld-L48-oq4e.json", s) for s in "ABC"},
    "Sushi": {s: (DATA / "kld-L48-sushi.json", s) for s in "ABC"},
}
KLD_NOTE = {
    "Ijah":  "T6 pack: EXL3 K5 experts + K6/K8 trunk + bf16 mixer",
    "oMLX":  "Jundot oQ4e, scored through oMLX's own loader (omlx_kld.py)",
    "Sushi": "Sushi-4bpw (4-bit experts, 8-bit trunk), scored through Sushi's own loader",
}
SETS = (("A", "A · wikitext"), ("B", "B · reasoning"), ("C", "C · 32K context"))


def kld_stats():
    """engine -> set -> (mean, lo, hi, sd, n): 95% bootstrap CI of the mean over windows"""
    rng = random.Random(0)
    out = {}
    for e, sets in KLD_SRC.items():
        out[e] = {}
        for s, (f, key) in sets.items():
            v = json.load(open(f))["sets"][key]
            w = v["per_window"]["kld"]
            n = len(w)
            m = sum(w) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in w) / (n - 1)) if n > 1 else 0.0
            bs = sorted(sum(rng.choice(w) for _ in range(n)) / n for _ in range(4000))
            out[e][s] = (v.get("kld_mean", m), bs[100], bs[3899], sd, n)
    return out


FONT = "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif"


def style():
    light = "".join(f".c-{e}{{fill:{c[0]}}}.s-{e}{{stroke:{c[0]}}}" for e, c in ENG.items())
    dark = "".join(f".c-{e}{{fill:{c[1]}}}.s-{e}{{stroke:{c[1]}}}" for e, c in ENG.items())
    return f"""<style>
  .bg{{fill:#fcfcfb}} .t1{{fill:#1f1d1b}} .t2{{fill:#5d5a55}} .t3{{fill:#8a867f}} .grid{{stroke:#e7e4de}}
  .axis{{stroke:#bdb8b0}} .na{{fill:#f1eee8;stroke:#d9d4cb}} .gap{{stroke:#fcfcfb}} .err{{stroke:#3a3632}}
  text{{font-family:{FONT}}} {light}
  @media (prefers-color-scheme: dark) {{
    .bg{{fill:#1a1a19}} .t1{{fill:#f0efec}} .t2{{fill:#c3c2b7}} .t3{{fill:#8f8d86}} .grid{{stroke:#2c2b29}}
    .axis{{stroke:#4a4844}} .na{{fill:#242322;stroke:#3a3936}} .gap{{stroke:#1a1a19}} .err{{stroke:#e6e3dc}} {dark}
  }}
</style>"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v):
    return f"{v:,.0f}" if v >= 200 else f"{v:.1f}"


def hwhisker(x_lo, x_hi, y, cap=6):
    return (f'<line class="err" x1="{x_lo:.1f}" y1="{y:.1f}" x2="{x_hi:.1f}" y2="{y:.1f}" stroke-width="1.5"/>'
            f'<line class="err" x1="{x_lo:.1f}" y1="{y - cap:.1f}" x2="{x_lo:.1f}" y2="{y + cap:.1f}" stroke-width="1.5"/>'
            f'<line class="err" x1="{x_hi:.1f}" y1="{y - cap:.1f}" x2="{x_hi:.1f}" y2="{y + cap:.1f}" stroke-width="1.5"/>')


def vwhisker(x, y_lo, y_hi, cap=7):
    return (f'<line class="err" x1="{x:.1f}" y1="{y_lo:.1f}" x2="{x:.1f}" y2="{y_hi:.1f}" stroke-width="1.5"/>'
            f'<line class="err" x1="{x - cap:.1f}" y1="{y_lo:.1f}" x2="{x + cap:.1f}" y2="{y_lo:.1f}" stroke-width="1.5"/>'
            f'<line class="err" x1="{x - cap:.1f}" y1="{y_hi:.1f}" x2="{x + cap:.1f}" y2="{y_hi:.1f}" stroke-width="1.5"/>')


def legend(y):
    """engines in a 2 x 2 grid: swatch, name, weight format"""
    parts = []
    for i, e in enumerate(ORDER):
        x, yy = 40 + (i % 2) * 470, y + (i // 2) * 18
        parts.append(f'<rect class="c-{e}" x="{x}" y="{yy - 10}" width="12" height="12" rx="3"/>'
                     f'<text class="t1" x="{x + 18}" y="{yy}" font-size="12.5" font-weight="600">{e}</text>'
                     f'<text class="t3" x="{x + 70}" y="{yy}" font-size="11.5">{esc(BITS[e])}</text>')
    return "".join(parts)


def speed_chart():
    W, panel_h, top = 1040, 120, 126
    H = top + panel_h + 178
    pw, gx = 300, 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="Ijah, oMLX and Sushi: decode and prefill speed">', style(),
           f'<rect class="bg" width="{W}" height="{H}" rx="16"/>',
           f'<text class="t1" x="40" y="48" font-size="24" font-weight="700">Speed · Qwen3.8-Flash-Next on an M5 Max</text>',
           f'<text class="t2" x="40" y="72" font-size="14">Single stream, tokens per second, higher is better. '
           f'Hover a bar for what was measured.</text>', legend(98)]
    for i, (title, rows) in enumerate(SPEED.items()):
        col, row = i, 0
        x0 = 40 + col * (pw + gx)
        y0 = top + row * panel_h
        vmax = max((r[2][1] if r[2] else r[0]) for r in rows.values() if r[0]) * 1.12
        lw = 96
        bw = pw - lw - 40
        out.append(f'<text class="t1" x="{x0}" y="{y0 + 18}" font-size="15" font-weight="600">{esc(title)}</text>')
        for k in range(5):
            gxp = x0 + lw + bw * k / 4
            out.append(f'<line class="grid" x1="{gxp:.1f}" y1="{y0 + 30}" x2="{gxp:.1f}" y2="{y0 + 30 + len(ORDER) * 36}" '
                       f'stroke-width="1"/>')
        for j, e in enumerate(ORDER):
            v, note, rng = rows[e]
            by = y0 + 34 + j * 36
            bold = ' font-weight="700"' if e == "Ijah" else ""
            out.append(f'<text class="t2" x="{x0 + lw - 10}" y="{by + 14}" font-size="13" text-anchor="end"{bold}>{e}</text>'
                       f'<text class="t3" x="{x0 + lw - 10}" y="{by + 27}" font-size="9.5" text-anchor="end">{SHORT[e]}</text>')
            if v is None:
                out.append(f'<rect class="na" x="{x0 + lw}" y="{by + 4}" width="{bw * 0.42:.1f}" height="22" rx="4" '
                           f'stroke-dasharray="3 3"><title>{e}: {esc(note)}</title></rect>')
                out.append(f'<text class="t3" x="{x0 + lw + 10}" y="{by + 20}" font-size="12">{esc(note)}</text>')
                continue
            w = bw * v / vmax
            out.append(f'<path class="c-{e}" d="M{x0 + lw},{by + 4} h{w - 4:.1f} a4,4 0 0 1 4,4 v14 a4,4 0 0 1 -4,4 '
                       f'h{-(w - 4):.1f} z"><title>{e}: {fmt(v)} tok/s — {esc(note)}</title></path>')
            lx = x0 + lw + w + 8
            if rng:
                lo, hi = (x0 + lw + bw * r / vmax for r in rng)
                out.append(hwhisker(lo, hi, by + 15))
                lx = max(lx, hi + 8)
            out.append(f'<text class="t1" x="{lx:.1f}" y="{by + 20}" font-size="13" '
                       f'font-weight="{700 if e == "Ijah" else 500}">{fmt(v)}</text>')
    for k, line in enumerate((
            "Decode: mean of 6 prompts x 3 seeds (1,024 tokens, T=1, medium thinking), whiskers = 95% CI.",
            "Paired on the same prompt and seed: Ijah is 1.162x Sushi (CI 1.122-1.203, 18/18 wins) and 0.94x oMLX "
            "(CI 0.92-0.96, 2/18).",
            "Prefill: cold prompts with no cache hit, median of 5 runs, whiskers = min-max.",
            "One streaming client for all three, each on its own weights and best settings (oMLX: MTP depth 2, "
            "INT8-activation prefill).",
            "Engine + weights, not engine alone: Sushi ran its 4-bit-expert Sushi-4bpw pack, oMLX the 4.72 bpw oQ4e, "
            "Ijah its 5-bit-expert T6 pack.",
            "18 Oct 2026, M5 Max 128 GB. Ijah with the DFlash2 drafter measured 120.4 tok/s on a 19 Oct lab build "
            "(not in this release).")):
        out.append(f'<text class="t3" x="40" y="{H - 124 + 17 * k}" font-size="11.5">{esc(line)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def kld_chart(st):
    W, H = 1000, 488
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="Quality: KL divergence from the bf16 model, Ijah, oMLX and Sushi">', style(),
           f'<rect class="bg" width="{W}" height="{H}" rx="16"/>',
           f'<text class="t1" x="40" y="48" font-size="24" font-weight="700">Quality · distance from the full-precision model</text>',
           f'<text class="t2" x="40" y="72" font-size="14">KL divergence from the bf16 teacher on three test sets, '
           f'lower is closer.</text>', legend(98)]
    px0, pw, gap = 78, 274, 30
    vmax = 0.18
    y_base, h = 408, 240
    for si, (key, sname) in enumerate(SETS):
        x0 = px0 + si * (pw + gap)
        out.append(f'<text class="t1" x="{x0}" y="152" font-size="15" font-weight="600">{esc(sname)}</text>')
        for k in range(5):
            yy = y_base - h * k / 4
            out.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke-width="1"/>')
            if si == 0:
                out.append(f'<text class="t3" x="{x0 - 8}" y="{yy + 4:.1f}" font-size="10.5" text-anchor="end">'
                           f'{vmax * k / 4:.3f}</text>')
        bw = 64
        for j, e in enumerate(KLD_SRC):
            m, lo, hi, sd, n = st[e][key]
            bh = h * m / vmax
            bx = x0 + (pw - (len(KLD_SRC) * bw + (len(KLD_SRC) - 1) * 24)) / 2 + j * (bw + 24)
            cx = bx + bw / 2
            out.append(f'<path class="c-{e}" d="M{bx},{y_base} v{-(bh - 4):.1f} a4,4 0 0 1 4,-4 h{bw - 8} a4,4 0 0 1 4,4 '
                       f'v{bh - 4:.1f} z"><title>{e}, set {sname}: KLD {m:.4f}, 95% CI [{lo:.4f}, {hi:.4f}], '
                       f'SD over windows {sd:.4f} (n = {n}) — {esc(KLD_NOTE[e])}</title></path>')
            ylo, yhi = y_base - h * lo / vmax, y_base - h * hi / vmax
            out.append(vwhisker(cx, ylo, yhi))
            out.append(f'<text class="t1" x="{cx}" y="{min(yhi, y_base - bh) - 8:.1f}" font-size="13" '
                       f'text-anchor="middle" font-weight="{700 if e == "Ijah" else 500}">{m:.3f}</text>')
            out.append(f'<text class="t2" x="{cx}" y="{y_base + 18}" font-size="12.5" text-anchor="middle"'
                       f'{" font-weight=\"700\"" if e == "Ijah" else ""}>{e}</text>'
                       f'<text class="t3" x="{cx}" y="{y_base + 31}" font-size="9.5" text-anchor="middle">{SHORT[e]}</text>')
        out.append(f'<line class="axis" x1="{x0}" y1="{y_base}" x2="{x0 + pw}" y2="{y_base}" stroke-width="1"/>')
    ns = sorted({st[e][k][4] for e in st for k, _ in SETS})
    out.append(f'<text class="t3" x="40" y="{H - 30}" font-size="11.5">Whiskers: 95% bootstrap confidence interval of '
               f'the mean over each set\'s windows (n = {", ".join(map(str, ns))}); hover a bar for the SD.</text>')
    out.append(f'<text class="t3" x="40" y="{H - 13}" font-size="11.5">Ijah is 1.7-2x closer than '
               f'Sushi\'s 4bpw pack, and 4-5x closer than oMLX\'s Jundot oQ4e, on every set.</text>')
    out.append("</svg>")
    return "\n".join(out)


def tradeoff_chart(st):
    """speed vs quality: x = decode with MTP, y = KLD set B (log), bottom-right is best"""
    W, H = 800, 520
    x0, x1, y0, y1 = 96, 740, 410, 130
    xmin, xmax = 82, 126
    lmin, lmax = math.log10(0.006), math.log10(0.08)
    X = lambda v: x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)
    Y = lambda v: y0 - (math.log10(v) - lmin) / (lmax - lmin) * (y0 - y1)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="Speed versus quality">', style(), f'<rect class="bg" width="{W}" height="{H}" rx="16"/>',
           f'<text class="t1" x="40" y="48" font-size="24" font-weight="700">Faster and closer</text>',
           f'<text class="t2" x="40" y="72" font-size="14">Decode speed with MTP vs distance from the bf16 model '
           f'(set B, log scale). Best is bottom-right.</text>']
    for v in (85, 90, 95, 100, 105, 110, 115, 120):
        out.append(f'<line class="grid" x1="{X(v):.1f}" y1="{y1}" x2="{X(v):.1f}" y2="{y0}" stroke-width="1"/>')
        out.append(f'<text class="t3" x="{X(v):.1f}" y="{y0 + 20}" font-size="11" text-anchor="middle">{v}</text>')
    for v in (0.01, 0.02, 0.05):
        out.append(f'<line class="grid" x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke-width="1"/>')
        out.append(f'<text class="t3" x="{x0 - 8}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end">{v}</text>')
    out.append(f'<text class="t2" x="{(x0 + x1) / 2}" y="{y0 + 44}" font-size="12.5" text-anchor="middle">decode with '
               f'MTP, tokens per second →</text>')
    out.append(f'<text class="t2" x="28" y="{(y0 + y1) / 2}" font-size="12.5" text-anchor="middle" '
               f'transform="rotate(-90 28 {(y0 + y1) / 2})">← closer to bf16 (KLD, set B)</text>')
    dec = SPEED["Decode with MTP (tok/s)"]
    for e in ("Sushi", "oMLX", "Ijah"):
        sv, _, rng = dec[e]
        m, lo, hi, sd, n = st[e]["B"]
        cx, cy = X(sv), Y(m)
        if rng:
            out.append(hwhisker(X(rng[0]), X(rng[1]), cy, cap=5))
        out.append(vwhisker(cx, Y(lo), Y(hi), cap=5))
        r = 12 if e == "Ijah" else 9
        out.append(f'<circle class="c-{e} gap" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" stroke-width="2">'
                   f'<title>{e}: {sv} tok/s (95% CI {rng[0]}-{rng[1]}), KLD {m:.4f} [{lo:.4f}, {hi:.4f}] — '
                   f'{esc(BITS[e])}</title></circle>')
        # labels above-right of the point with a surface halo, so the speed whisker never runs through the text
        halo = 'stroke="#fcfcfb" stroke-width="5" stroke-linejoin="round" paint-order="stroke"'
        right = cx > x0 + 0.62 * (x1 - x0)   # near the right edge: label to the left of the point
        lx, anc = (cx - r - 6, ' text-anchor="end"') if right else (cx + r + 6, "")
        out.append(f'<text class="t1" x="{lx:.1f}" y="{cy - r - 16:.1f}" font-size="14" {halo}{anc} '
                   f'font-weight="{700 if e == "Ijah" else 500}">{e}</text>'
                   f'<text class="t3" x="{lx:.1f}" y="{cy - r - 3:.1f}" font-size="10.5" {halo}{anc}>{esc(BITS[e])}</text>')
    # Ijah with 4-bit experts (Sushi-4bpw's expert shards + Ijah's T6 trunk): hollow marker, same colour
    k4 = json.load(open(DATA / "kld-k4s-t6.json"))["sets"]["B"]["per_window"]["kld"]
    rng4 = random.Random(1)
    m4 = sum(k4) / len(k4)
    b4 = sorted(sum(rng4.choice(k4) for _ in range(len(k4))) / len(k4) for _ in range(4000))
    cx, cy = X(105.2), Y(m4)
    out.append(hwhisker(X(101.0), X(109.3), cy, cap=5))
    out.append(vwhisker(cx, Y(b4[100]), Y(b4[3899]), cap=5))
    out.append(f'<circle class="s-Ijah" cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="none" stroke-width="3">'
               f'<title>Ijah with 4-bit experts: 105.2 tok/s (95% CI 101.0-109.3), KLD {m4:.4f} '
               f'[{b4[100]:.4f}, {b4[3899]:.4f}] (Sushi-4bpw expert shards + Ijah\'s T6 trunk)</title></circle>')
    out.append(f'<text class="t1" x="{cx - 12:.1f}" y="{cy + 22:.1f}" font-size="13" text-anchor="end" {halo}>'
               f'Ijah, 4-bit experts (experiment)</text>')
    # Ijah + the DFlash2 block drafter (same T6 pack, so the same quality): hollow marker, same colour
    cx, cy = X(120.4), Y(st["Ijah"]["B"][0])
    out.append(hwhisker(X(116.9), X(123.8), cy, cap=5))
    out.append(f'<circle class="s-Ijah" cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="none" stroke-width="3" '
               f'stroke-dasharray="3 3"><title>Ijah with the DFlash2 drafter: 120.4 tok/s (95% CI 116.9-123.8), same T6 '
               f'pack so the same KLD {st["Ijah"]["B"][0]:.4f}; 19 Oct lab build, not in this release</title></circle>')
    out.append(f'<text class="t1" x="{cx - 12:.1f}" y="{cy + 40:.1f}" font-size="13" text-anchor="end" {halo}>'
               f'Ijah + DFlash2 drafter (lab build)</text>')
    out.append(f'<text class="t3" x="40" y="{H - 52}" font-size="11.5">Whiskers: decode 95% CI over 6 prompts x 3 seeds '
               f'(same client); KLD 95% bootstrap CI over windows.</text>')
    out.append(f'<text class="t3" x="40" y="{H - 35}" font-size="11.5">Hollow markers are lab work, not this release: '
               f'4-bit experts (not a published pack) and the DFlash2 drafter (not in this tree).</text>')
    out.append(f'<text class="t3" x="40" y="{H - 18}" font-size="11.5">The DFlash2 point is a 19 Oct build, whose MTP-only '
               f'baseline was 112.4 tok/s: read it against that, not against the 98.9 bar.</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    st = kld_stats()
    for e in st:
        print(e, {k: tuple(round(x, 4) for x in v[:4]) + (v[4],) for k, v in st[e].items()})
    for name, svg in (("speed", speed_chart()), ("quality", kld_chart(st)), ("tradeoff", tradeoff_chart(st))):
        (OUT / f"ijah-{name}.svg").write_text(svg)
        print("wrote", OUT / f"ijah-{name}.svg")
