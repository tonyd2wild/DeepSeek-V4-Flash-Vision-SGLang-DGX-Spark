#!/usr/bin/env python3
"""Smoke test an OpenAI-compatible endpoint: one text prompt (streamed, reports TTFT + decode tok/s) and one image prompt.
Usage: tools/smoke.py <base_url> [served_model] [image_path_or_url]"""
import sys, json, time, base64, urllib.request, mimetypes, os

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else None
IMG = sys.argv[3] if len(sys.argv) > 3 else "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"

def models():
    return json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=20))["data"]

def chat(messages, stream=True, max_tokens=300):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "stream": stream}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            d = json.load(r); return d["choices"][0]["message"]["content"], None, None, time.time() - t0
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: ch = json.loads(payload)
            except Exception: continue
            delta = (ch.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                if first is None: first = time.time()
                n += 1; text.append(piece)
    t1 = time.time()
    return "".join(text), (first - t0) if first else None, (n - 1) / (t1 - first) if first and n > 1 else None, t1 - t0

if MODEL is None:
    MODEL = models()[0]["id"]
print("model:", MODEL)
out, ttft, tps, wall = chat([{"role": "user", "content": "In three sentences, explain what a DGX Spark is and who it is for."}])
print("TEXT  ttft %.2fs | decode ~%.1f chunks/s | wall %.1fs\n  %s" % (ttft or -1, tps or -1, wall, out.strip()[:300].replace("\n", " / ")))
if IMG.startswith("http"):
    image_url = IMG
else:
    mt = mimetypes.guess_type(IMG)[0] or "image/jpeg"
    image_url = "data:%s;base64,%s" % (mt, base64.b64encode(open(IMG, "rb").read()).decode())
out, ttft, tps, wall = chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": "Describe this image in two sentences."}]}])
print("IMAGE ttft %.2fs | decode ~%.1f chunks/s | wall %.1fs\n  %s" % (ttft or -1, tps or -1, wall, out.strip()[:300].replace("\n", " / ")))
