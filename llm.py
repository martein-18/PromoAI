"""All model access lives here. Swap models / providers in config.yaml -> llm: only.

Providers:
  openai_compat : any OpenAI-compatible /chat/completions endpoint (Gemini, OpenRouter, Groq, OpenAI, Ollama...).
  mock          : offline deterministic stub.
Features:
  * retry with backoff on 429/5xx/timeouts (bad key / bad model name fails immediately, no pointless retries)
  * fallbacks: ordered list of other endpoints, used automatically when the primary stays down
  * primary_cooldown_s: after the primary fails, skip it for that long so later calls don't each wait minutes
  * min_interval_s: throttle between calls
  * resume: every answered call is cached from <out>/raw/calls.jsonl, so re-running the same --out folder never repeats work
Every call is logged untouched, including which model actually answered.
"""
import hashlib, json, os, time, urllib.request, urllib.error
from pathlib import Path


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    """Per-minute limit hit while a backup model is available: hand over now, try this model again after `wait` s."""
    def __init__(self, msg, wait):
        super().__init__(msg)
        self.wait = wait


class QuotaExhausted(LLMError):
    """Provider quota used up for the day: skip this endpoint for a long time instead of retrying."""


RETRYABLE = (408, 429, 500, 502, 503, 504)


def _is_daily_quota(body):
    b = body.lower()
    return "perday" in b or "per day" in b or "daily" in b


def _retry_after(err, body):
    """Seconds to wait, from a Retry-After header or Gemini's RetryInfo.retryDelay ("36s")."""
    import re
    h = err.headers.get("Retry-After") if err.headers else None
    if h and h.strip().isdigit():
        return int(h)
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
    return float(m.group(1)) if m else None


class Endpoint:
    """One OpenAI-compatible provider+model."""
    def __init__(self, cfg, defaults):
        self.cfg = {**{k: v for k, v in defaults.items() if k in ("temperature", "timeout_s", "retries", "retry_base_s", "max_tokens")}, **cfg}
        self.model = self.cfg["model"]
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.key_env = self.cfg.get("api_key_env", "LLM_API_KEY")

    def call(self, msgs, label="", fail_fast=False):
        """fail_fast: another model is waiting behind this one, so hand over as soon as possible instead of waiting:
        daily quota -> at once (QuotaExhausted); per-minute limit -> at once (RateLimited); timeout or connection
        error -> after 1 try; busy server (5xx) -> after 2 tries. The last model in a chain keeps retrying."""
        c = self.cfg
        key = os.environ.get(self.key_env, "").strip().strip("\"'").strip()   # setx often leaves quotes/spaces
        local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        if not key and not local:
            raise LLMError(f"{self.model}: set env var {self.key_env}")
        body = {"model": self.model, "messages": msgs, "temperature": c.get("temperature", 0.2)}
        if c.get("max_tokens"):
            body["max_tokens"] = c["max_tokens"]
        if c.get("extra_body"):
            body.update(c["extra_body"])
        last = None
        for attempt in range(c.get("retries", 6)):
            req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key or 'none'}"})
            try:
                with urllib.request.urlopen(req, timeout=c.get("timeout_s", 120)) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                raw = e.read().decode(errors="replace")
                last = f"{self.model}: HTTP {e.code} {raw[:300]}"
                if e.code not in RETRYABLE:
                    raise LLMError(last)            # bad key / model name: retrying won't help
                hinted = _retry_after(e, raw)
                if e.code == 429 and (_is_daily_quota(raw) or (hinted or 0) > 90):
                    raise QuotaExhausted(f"{self.model}: quota exhausted (daily limit)")   # won't reset for hours
                if e.code == 429 and fail_fast:
                    raise RateLimited(f"{self.model}: rate limited (per-minute)", (hinted or 30) + 1)
                if fail_fast and e.code != 429 and attempt >= 1:
                    raise LLMError(last + " (busy; moving to the next model)")
            except Exception as e:
                last = f"{self.model}: {e!r}"
                hinted = None
                if fail_fast:
                    raise LLMError(last + " (unreachable; moving to the next model)")
            wait = min(60, c.get("retry_base_s", 5) * 2 ** attempt)
            if hinted is not None:
                wait = hinted + 1                   # server told us exactly how long to wait
            print(f"  attempt {attempt + 1}/{c.get('retries', 6)} failed ({last[:160]}); retrying in {wait}s")
            time.sleep(wait)
        raise LLMError(last)


class BaseClient:
    def __init__(self, cfg, raw_dir):
        self.cfg = cfg
        self.model = cfg.get("model", "mock")       # primary model; also the cache identity
        self.last_model = self.model
        self.models_used = set()
        self.raw = Path(raw_dir)
        self.raw.mkdir(parents=True, exist_ok=True)
        self.min_interval = cfg.get("min_interval_s", 0)
        self.new_calls, self.model_seconds = 0, 0.0   # cost of THIS run (cache hits and reuses are free)
        self._last = 0.0
        self._cache = {}
        self._cache = self._read_log(self.raw / "calls.jsonl")
        # Read-only answers from OTHER runs (llm.cache_from: list of calls.jsonl paths). Only used for the exact same
        # model + prompt. A reused answer is copied into this run's own log, so this run's audit stays complete.
        self._ext = {}
        for src in cfg.get("cache_from") or []:
            if Path(src).resolve() != (self.raw / "calls.jsonl").resolve():
                self._ext.update(self._read_log(Path(src)))

    def _read_log(self, log):
        cache = {}
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    if "forget" in r:                  # a reply that was unusable: never serve it again
                        cache.pop(r["forget"], None)
                        continue
                    cache[self._key(r.get("cache_model", r["model"]), r.get("system"), r["prompt"])] = (r["response"], r["model"])
                except Exception:
                    pass
        return cache

    def forget(self, prompt, system=None):
        """Drop a cached reply that turned out unusable (broken JSON, wrong shape), so a retry or a later run asks
        the model again instead of getting the same broken answer from the cache. Logged, so it survives restarts."""
        k = self._key(self.model, system, prompt)
        self._cache.pop(k, None)
        self._ext.pop(k, None)
        with open(self.raw / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "forget": k}) + "\n")

    @staticmethod
    def _key(model, system, prompt):
        return hashlib.sha256(f"{model}|{system}|{prompt}".encode()).hexdigest()

    def _log(self, task, prompt, system, response, secs, reused=False):
        rec = {"ts": time.time(), "task": task, "model": self.last_model, "cache_model": self.model, "system": system,
               "prompt": prompt, "response": response, "seconds": round(secs, 2)}
        if reused:
            rec["reused"] = True                       # copied from an earlier run's log, not a new model call
        with open(self.raw / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def call(self, prompt, system=None, task=""):
        k = self._key(self.model, system, prompt)
        if k in self._cache:
            out, used = self._cache[k]
            self.models_used.add(used)
            self.last_model = used                     # so callers can record which model this answer came from
            return out
        if k in self._ext:                             # answered in an earlier run: reuse it, no new model call
            out, used = self._ext[k]
            self.last_model = used
            self.models_used.add(used)
            self._cache[k] = (out, used)
            self._log(task, prompt, system, out, 0.0, reused=True)
            return out
        budget = self.cfg.get("call_budget")           # shared {"left": n} set by the web server for public demos
        if budget is not None:
            if budget["left"] <= 0:
                raise LLMError("the daily cap of new AI calls set in config.yaml (web: daily_call_budget) is used up")
            budget["left"] -= 1
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        t = time.time()
        out = self._call(prompt, system, task)
        self._last = time.time()
        self.new_calls += 1
        self.model_seconds += self._last - t
        self.models_used.add(self.last_model)
        self._cache[k] = (out, self.last_model)
        self._log(task, prompt, system, out, self._last - t)
        return out

    def _call(self, prompt, system, task):
        raise NotImplementedError


_DOWN = {}   # (base_url, model) -> time until which it is skipped; shared by every client in this process,
             # so once a model's quota is gone, later runs on the same server skip it at once too


class ChainClient(BaseClient):
    """Primary endpoint + ordered fallbacks."""
    def __init__(self, cfg, raw_dir):
        super().__init__(cfg, raw_dir)
        self.endpoints = [Endpoint(cfg, cfg)] + [Endpoint(f, cfg) for f in (cfg.get("fallbacks") or [])]
        self.cooldown = cfg.get("primary_cooldown_s", 300)
        self.quota_cooldown = cfg.get("quota_cooldown_s", 6 * 3600)

    def _call(self, prompt, system, task):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        errors = []
        for i, ep in enumerate(self.endpoints):
            last_resort = i == len(self.endpoints) - 1
            if time.time() < _DOWN.get((ep.base_url, ep.model), 0) and not last_resort:
                continue                               # recently failed: skip straight to the next one
            try:
                out = ep.call(msgs, task, fail_fast=not last_resort)
                self.last_model = ep.model
                if i > 0:
                    print(f"  [fallback] answered by {ep.model}")
                return out
            except LLMError as e:
                errors.append(str(e))
                pause = (self.quota_cooldown if isinstance(e, QuotaExhausted) else
                         e.wait if isinstance(e, RateLimited) else self.cooldown)
                _DOWN[(ep.base_url, ep.model)] = time.time() + pause
                if i < len(self.endpoints) - 1:
                    print(f"  [fallback] {ep.model} unavailable, switching to {self.endpoints[i + 1].model}")
        raise LLMError(" || ".join(errors))


class MockClient(BaseClient):
    """Deterministic fake so the pipeline can be tested offline. Always labels itself MOCK in logs."""
    def __init__(self, cfg, raw_dir):
        super().__init__(cfg, raw_dir)
        self.model = self.last_model = "MOCK"

    def _call(self, prompt, system, task):
        if task == "rank_explain":
            return json.dumps({"overall": "mock", "items": [{"rank": i, "reason": "mock"} for i in range(1, 11)]})
        if task.startswith("candidates"):
            import re
            rows = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in
                    re.finditer(r"^\[(\d+)\] (\d+\.\d+)-(\d+\.\d+)", prompt, re.M)]
            if not rows:
                return json.dumps({"candidates": []})
            a, t0 = rows[0][0], rows[0][1]
            b = next((i for i, s, e in rows if e - t0 >= 35), rows[-1][0])
            return json.dumps({"candidates": [{"segments": [[a, b]], "hook_event": a, "cliffhanger_event": b,
                                               "hook": "mock hook", "cliffhanger": "mock cliffhanger", "why": "mock"}]})
        if task == "score":
            return json.dumps({"hook": {"score": 3, "reason": "mock"}, "cliffhanger": {"score": 3, "reason": "mock"},
                               "standalone": {"score": 3, "reason": "mock"}, "emotional": {"score": 3, "reason": "mock"},
                               "hook_type": "other"})
        if task == "cold_viewer":
            return json.dumps({"characters": "mock", "conflict": "mock", "next": "mock", "can_answer": True})
        if task == "assemble":                         # join the start of one clip with the end of another episode's
            import re
            blocks = re.findall(r"=== CLIP \S+ \| Episode (\d+):.*?===\n(.*?)(?=\n\n=== CLIP|\Z)", prompt, re.S)
            first = blocks[0] if blocks else None
            other = next((b for b in blocks[1:] if first and b[0] != first[0]), None)
            if not other:
                return json.dumps({"promos": []})
            a = [int(i) for _, i in re.findall(r"\[(\d+)\.(\d+)\]", first[1])]
            b = [int(i) for _, i in re.findall(r"\[(\d+)\.(\d+)\]", other[1])]
            s1, s2 = [int(first[0]), a[0], a[min(4, len(a) - 1)]], [int(other[0]), b[max(0, len(b) - 4)], b[-1]]
            return json.dumps({"promos": [{"segments": [s1, s2], "hook_event": s1[1], "cliffhanger_event": s2[2],
                                           "hook": "mock join hook", "cliffhanger": "mock join cliffhanger", "why": "mock"}]})
        if task == "bible":
            return json.dumps({"premise": "mock", "characters": [], "aliases": {}})
        return "{}"


def make_client(cfg, raw_dir, force_mock=False):
    lc = cfg["llm"]
    if force_mock or lc.get("provider") == "mock":
        return MockClient(lc, raw_dir)
    if lc.get("provider") == "openai_compat":
        return ChainClient(lc, raw_dir)
    raise LLMError(f"unknown provider {lc.get('provider')!r}")


def make_scoring_client(cfg, raw_dir, main_client, force_mock=False):
    """Client used for scoring (and the ranking explanation). With llm.scoring set, one fixed model scores every promo
    so scores are comparable. llm.scoring.fallbacks are used ONLY when that model is unavailable (e.g. daily quota):
    the run then still finishes, every promo records which model scored it (scored_by), such promos are flagged for
    review, and the audit warns that scores came from more than one model. Without llm.scoring it is the main client."""
    sc = cfg["llm"].get("scoring")
    if force_mock or not sc or cfg["llm"].get("provider") == "mock":
        return main_client
    lc = {**cfg["llm"], **sc, "fallbacks": sc.get("fallbacks") or []}
    if not lc.get("model"):
        raise LLMError("llm.scoring needs a model")
    return ChainClient(lc, raw_dir)


def extract_json(text):
    """Parse the JSON object/array a model reply starts with (handles ``` fences and preamble text).

    Strict on purpose: the FIRST { or [ must begin valid JSON. If that value is broken, the reply is a failure.
    (Scanning on to later brackets used to return a fragment from inside a broken reply, e.g. a segment list
    [[1, 30]] or one {"score": 3} sub-object, which was then silently treated as a real answer.)"""
    t = text or ""
    i = next((k for k, ch in enumerate(t) if ch in "{["), None)
    if i is None:
        raise ValueError("no JSON found in model reply")
    try:
        return json.JSONDecoder().raw_decode(t[i:])[0]
    except json.JSONDecodeError as e:
        raise ValueError(f"model reply is not valid JSON: {e}") from None