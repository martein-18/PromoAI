import os, json, time, urllib.request
key = os.environ["LLM_API_KEY"]
url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
model = "gemini-3.6 -flash"
body = {"model": model, "messages": [{"role": "user", "content": "Say hi in 3 words"}]}
req = urllib.request.Request(url, data=json.dumps(body).encode(),
      headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
t = time.time()
try:
    print(urllib.request.urlopen(req, timeout=30).read().decode()[:400])
except Exception as e:
    print("FAILED:", repr(e), getattr(e, "read", lambda: b"")()[:400])
print("seconds:", round(time.time() - t, 1))