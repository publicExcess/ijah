"""Ijah pixel-art logo: a 32x32 rottweiler pup in a badge -> SVG (crisp rects) + PNG via the browser."""
import math, sys
from pathlib import Path
N = 32
PAL = {"c": "#F4E7D3", "o": "#2A211C", "k": "#221D1A", "K": "#141110", "s": "#3A322D", "t": "#C9803F",
       "T": "#D18A48", "e": "#3B2A20", "p": "#0C0A09", "w": "#FFFFFF", "r": "#E07A8A", "b": "#2E6A8E",
       "g": "#E8B33E", "G": "#B8862A"}
g = [["." for _ in range(N)] for _ in range(N)]
def put(x, y, ch):
    if 0 <= x < N and 0 <= y < N: g[y][x] = ch
def mirror(x, y, ch): put(x, y, ch); put(N - 1 - x, y, ch)
def ell(cx, cy, rx, ry, ch, only=None):
    for y in range(N):
        for x in range(N):
            if ((x + .5 - cx) / rx) ** 2 + ((y + .5 - cy) / ry) ** 2 <= 1 and (only is None or g[y][x] in only): put(x, y, ch)
# badge: cream disc with a 1-px dark ring
for y in range(N):
    for x in range(N):
        d = math.hypot(x + .5 - 16, y + .5 - 16)
        if d <= 15.6: g[y][x] = "o" if d > 14.6 else "c"
# floppy ears (behind the head), drawn as tapering columns
for y, (x0, x1) in zip(range(10, 21), [(7, 9), (6, 9), (5, 9), (5, 9), (4, 9), (4, 9), (4, 8), (5, 8), (5, 8), (6, 8), (6, 7)]):
    for x in range(x0, x1 + 1): mirror(x, y, "K")
# head
ell(16, 15.2, 8.6, 9.6, "k")
for x in range(13, 19): put(x, 6, "s")                       # forehead sheen
for x in (11, 12): mirror(x, 10, "t")                          # brows
# eyes: brown ring, pupil, highlight top-left in each eye
for (ex, ey) in ((10, 12), (19, 12)):
    for dy in range(3):
        for dx in range(3): put(ex + dx, ey + dy, "e")
    for dy in (1, 2):
        for dx in (1, 2): put(ex + dx, ey + dy, "p")
    put(ex + 1, ey + 1, "w")
# cheeks + muzzle
ell(9.6, 18.6, 2.6, 2.1, "t", only="kK")
ell(22.4, 18.6, 2.6, 2.1, "t", only="kK")
ell(16, 19.4, 5.2, 4.3, "T")
# nose, mouth, tongue
for x in range(14, 18): put(x, 16, "p")
for x in range(14, 18): put(x, 17, "p")
put(14, 17, "T"); put(17, 17, "T"); put(15, 15, "T"); put(16, 15, "T")
for x in (13, 15, 16, 18): put(x, 18, "p")                   # philtrum + smile corners
put(14, 19, "p"); put(17, 19, "p")                             # the "w" of the smile
for (x, y) in ((15, 19), (16, 19), (15, 20), (16, 20)): put(x, y, "r")   # tongue
# collar + tag
for x in range(9, 23): put(x, 25, "b")
for x in range(10, 22): put(x, 26, "b")
for (x, y) in ((15, 26), (16, 26), (15, 27), (16, 27)): put(x, y, "g")
put(14, 27, "G"); put(17, 27, "G"); put(15, 28, "G"); put(16, 28, "G")
rows = ["".join(r) for r in g]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "assets"
print("\n".join(rows))
rects = []
for y, r in enumerate(rows):                                   # merge horizontal runs into one rect each
    x = 0
    while x < N:
        ch = r[x]; x1 = x
        while x1 < N and r[x1] == ch: x1 += 1
        if ch != ".": rects.append(f'<rect x="{x}" y="{y}" width="{x1 - x}" height="1" fill="{PAL[ch]}"/>')
        x = x1
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {N} {N}" width="512" height="512" shape-rendering="crispEdges" '
       f'role="img" aria-label="Ijah: a pixel-art rottweiler pup">\n  <title>Ijah</title>\n  ' + "\n  ".join(rects) + "\n</svg>\n")
(out / "ijah-logo.svg").write_text(svg)
