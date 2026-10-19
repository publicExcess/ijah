"""lab-vision end-to-end requests against a running server (tests/vision/e2e_job.sh starts one):
the chart prompt, the diagram -> Mermaid prompt (file path and data: URL), and a follow-up turn that must reuse the
image prefix. Prints the answers and the usage.

  $PY -I -B tests/vision/e2e_client.py --url http://127.0.0.1:8148 --key-file KEY [--out FILE.json]
"""
import argparse
import base64
import http.client
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
CHART = Path(os.environ.get("IJAH_TEST_CHART", ROOT / "tests/vision/assets/flow.png"))  # any bar-chart PNG (IJAH_TEST_CHART)
FLOW = ROOT / "tests/vision/assets/flow.png"


def post(url, key, body, timeout=3600):
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body),
              {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    r = c.getresponse()
    data = json.loads(r.read())
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8148")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--effort", default="low", help="reasoning effort of the real prompts (none = thinking off)")
    a = ap.parse_args()
    key = Path(a.key_file).read_text().strip()
    flow_url = "data:image/png;base64," + base64.b64encode(FLOW.read_bytes()).decode()
    res = {}

    def run(name, msgs, **kw):
        body = {"messages": msgs, "reasoning_effort": a.effort, **kw}
        t0 = time.time()
        d = post(a.url, key, body)
        m = d["choices"][0]["message"]
        res[name] = {"content": m.get("content"), "reasoning_chars": len(m.get("reasoning_content") or ""),
                     "usage": d["usage"], "s": time.time() - t0}
        print(f"=== {name}: {d['usage']} in {time.time() - t0:.1f} s\n{m.get('content')}\n", flush=True)
        return m

    q1 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": str(CHART)}},
                                       {"type": "text", "text": "List every bar with its exact value."}]}]
    m1 = run("chart", q1)
    q3 = q1 + [{"role": "assistant", "content": m1.get("content") or ""},
               {"role": "user", "content": "Which point has the lowest KLD, and what is its size in GiB?"}]
    run("chart_followup", q3)
    q2 = [{"role": "user", "content": [{"type": "text", "text": "Convert this diagram to a Mermaid flowchart. "
                                                                 "Reply with the Mermaid code only."},
                                       {"type": "image_url", "image_url": {"url": flow_url}}]}]
    run("mermaid", q2)
    q4 = q1 + [{"role": "assistant", "content": m1.get("content") or ""},
               {"role": "user", "content": "Which two points are closest in KLD?"}]
    run("chart_again", q4)  # after another image conversation: the chart conversation comes back from the cache
    for name in ("chart_followup", "chart_again"):
        cached = res[name]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"{name} reused {cached} cached tokens (the chart prompt was {res['chart']['usage']['prompt_tokens']})")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
