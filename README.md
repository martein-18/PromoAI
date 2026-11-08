# PromoAI

Picks 20-60s promo clips from episode screenplays. The model proposes clips by line ID; code enforces every rule
(timestamps, length, fast open, no cut lines) and ranks the survivors.

**The answer to the brief is in [`demo/PROMOS.md`](demo/PROMOS.md)**: which promos to post first and why (a day-by-day
plan), every promo with its episodes, in/out timestamps of each part, hook, cliffhanger and reason, and a measured
"where does it pick badly" section. Every run writes the same file to its output folder.

Pipeline: read files (code) -> learn the characters (AI) -> pick clips per episode (AI) -> check the rules (code) ->
score each clip blind (AI) -> join parts of checked clips across episodes (AI + code) -> rank, compare the top
head to head (code + AI) -> posting plan (code) -> explain the order (AI) -> audit everything (code).

## Quick start (run it on your own computer)

Easiest: open the deployed link you were sent; nothing to install. To run it locally instead you need
**Python 3.10 or newer** (python.org; on Windows tick "Add python.exe to PATH") and a **Gemini API key**
(free at https://aistudio.google.com/apikey). Use **Command Prompt** in VS Code (Terminal > New Terminal, then pick
"Command Prompt" from the arrow next to +).

1. Unzip the folder and open it in VS Code (File > Open Folder).
2. Once:

       python -m venv venv
       venv\Scripts\activate
       pip install -r requirements.txt

3. Every time, in the same terminal:

       venv\Scripts\activate
       set LLM_API_KEY=your-gemini-key
       python server.py

   (`set OPENROUTER_API_KEY=...` too if you have one; it is only for backup models. `setx` instead of `set` saves
   the key for new terminals. PowerShell: `venv\Scripts\Activate.ps1` and `$env:LLM_API_KEY="..."`.
   macOS/Linux: `source venv/bin/activate` and `export LLM_API_KEY=...`.)
4. The browser opens http://127.0.0.1:8000. Drop episode `.txt` files into the tray and press **Find promos**. Keys are only read from the terminal; the page
   never asks for one. If the key is missing, the page says so under **Find promos**.

Results are saved in `outputs/web/`. If the quota runs out, the run still writes what it finished, marked partial;
press **Find promos** again later to complete it without repeating any call. Ctrl+C in the terminal stops the server.

## Deploy (one public link, your API key, no setup for viewers)

The key is never in the code: it lives in the host's secret settings, and the browser never sees it.

1. Optional: copy a finished run into `demo/` so readers of the repo can open its `PROMOS.md`:
   `python make_demo.py web/<run id>` (or `run_A`).
2. Put the project on GitHub (private is fine). `data/`, `outputs/` and `venv/` are git-ignored; `demo/` is included.
3. On Render (render.com): New > Web Service > connect the repo, then set
   * Build command: `pip install -r requirements.txt`
   * Start command: `python server.py`  (it reads the host's `PORT` and listens publicly by itself)
   * Environment: `LLM_API_KEY` = your Gemini key, `PYTHON_VERSION` = `3.11.9`
     (optional: `OPENROUTER_API_KEY` for the backup models)
4. Send the `https://...onrender.com` link.

Protection for your key (`web:` in config.yaml): new AI calls are capped per day (`daily_call_budget`), runs of
episodes that were already processed reuse saved answers for free, only one run happens at a time, and only the page
itself can start runs.
Free hosting notes: the service sleeps when idle (first visit takes about a minute), and its disk is wiped on restart,
so live-run results and today's call count reset then; `demo/` is part of the code and always stays.

## Run it

    pip install -r requirements.txt
    python tests/test_fixtures.py              # every rule and fix, on synthetic episodes: no key, no dataset needed
    python tests/test_parser.py                # parser + validator (needs episodes in data/input, else skipped)
    python tests/test_repair.py                # duration-repair guardrails (needs episodes in data/input, else skipped)
    python tests/test_json.py                  # broken model replies count as failures, not as answers
    python tests/test_scoring.py               # repeat scoring, head-to-head comparison
    python tests/test_real_channel.py          # full pipeline against a misbehaving local stand-in model
    python tests/test_web.py                   # web page <-> server.py contract, against a stand-in model
    python main.py parse                       # no key needed
    python main.py all --mock                  # offline test of the whole pipeline (fake scores)
    python main.py all --stitch --out outputs/run_A
    streamlit run app.py
    python server.py                           # web interface at http://127.0.0.1:8000 (start it where your keys are set)
    python hand_check.py run_A                 # writes outputs/run_A/hand_check.md to judge the run by hand
    python measure_drift.py run_A gemini-3.6-flash 12   # how consistent is a scoring model? (24 calls)

### API keys (Windows)

    set LLM_API_KEY=your-gemini-key            :: this window only
    setx LLM_API_KEY your-gemini-key           :: saved for NEW windows (no quotes)
    setx OPENROUTER_API_KEY sk-or-v1-...       :: optional, for the OpenRouter fallbacks

macOS/Linux: `export LLM_API_KEY=...`. Keys are read from the environment only; nothing is stored in the repo.

## Free-tier setup (config.yaml -> llm)

* **Clip generation** uses `llm.model` and falls back through `llm.fallbacks` in order: extra Gemini models on the
  same key (each has its own free daily quota), then free OpenRouter models (one shared daily limit per account).
  A 429 that says the *daily* quota is used up skips that model at once (for `quota_cooldown_s`, default 6h)
  instead of retrying; a per-minute 429 waits exactly the delay the server asks for.
* **Scoring** uses `llm.scoring`: one fixed model with **no fallbacks**, so every promo is judged on the same scale.
  If it runs out, the run stops with "stopped: no model could answer". Re-run the same command later: every answered
  call is cached in `<out>/raw/calls.jsonl` and never repeated.
* Free OpenRouter `:free` models change often; check https://openrouter.ai/models?q=free if one returns 404.

## Scoring and ranking

* `prompts/score_prompt.txt` describes every score 1-5 for each criterion, so the model uses the whole scale.
  Scoring is **blind**: the scorer sees only the clip, never the show summary or the picker's claimed hook and
  cliffhanger, so "makes sense on its own" is judged as a stranger would see it.
* `llm.scoring.temperature: 0` and `samples: 3`: each clip is scored twice; only if the two passes disagree is it
  scored a third time. Each criterion is the mean of the passes, and every pass is kept in `scores.<criterion>.passes`.
  The audit line "Score stability" says how often the passes agreed.
* `ranking.compare`: the top `top_n` clips (after scoring, overlap removal and diversity) are then ordered by the
  scoring model comparing them directly, in two passes (list shown forward, then reversed) so list position cancels
  out. `score_rank` in the CSV keeps each clip's place by score alone. The audit line "Head-to-head ranking" shows how
  well the two passes agreed; a low or negative number means the comparison followed list position, not quality.
* Weights: hook .30, cliffhanger .35, standalone .25, emotional .10 (emotional barely separated clips in run_A).
  Ties are broken by cliffhanger, then hook, then standalone (never by episode order). A clip is dropped as a
  duplicate only if more than half of it repeats a better clip (`ranking.max_overlap_share`).
* Promos that need a human look are flagged (`needs_review` in the CSV, "check by hand" on the page): scoring passes
  disagreed, length auto-adjusted, hook/cliffhanger position unverified, or moved 3+ places by the head-to-head.

## Rules checked by code (validator.py, re-checked independently in verify.py)

* 20-60s total; parts in order and not overlapping; at most 4 parts.
* Something happens within 3s, and **the model's own hook line starts within 3s** of the clip start.
* **The clip ends at most 4s after the model's cliffhanger line** (`cliff_tail_max_s`): a reaction beat, never the answer.
* Clips end on the last real beat (dialogue, action or sound), so a visual cliffhanger ("the door bursts open") stays in.
* Quoted hook/cliffhanger lines must exist inside the clip itself.
* Every episode file needs a unique "EPISODE NUMBER"; a missing or repeated number stops the run and names the files.

## Cross-episode promos

Two sources: consecutive episodes whose notes say they pick up directly (`--stitch`), and a join stage
(`pipeline.assemble`) that asks the AI to combine parts of clips that already passed every rule, e.g. a hook from
episode 4 with a cliffhanger from episode 11. Code then checks that every part lies inside a checked clip, that at
least two episodes are used, and the usual length, hook and cliffhanger rules. A join is kept only if it scores
better than the clips it reuses (overlap removal keeps the better version).

## Extra hook check

The blind scorer also names the single line that would most stop a scroll. If that line is more than 3s into the
clip, the promo is flagged "strongest line is Ns in": the picker's chosen hook may not be the real one.

## If the quota runs out

The run still writes everything it finished (PROMOS.md, CSV, dashboard), marked PARTIAL at the top and in the audit.
Running the same command again completes it without repeating any call.

## Posting plan (slate.py -> posting_plan.md, and on the page)

A day-by-day plan built by code from the ranking: early episodes first so the big twists are not spoiled on day one,
at most 2 promos per hook type, one promo per episode, and a platform per clip (up to 30s: TikTok / Reels; longer:
YouTube Shorts). Settings: `slate` in config.yaml. Where a promo's day differs from its quality order, the plan says
why (for example "held to day 5 so it does not spoil episode 14"), and it lists top promos it left out and why.

## Limitations

* Clips are first picked one episode at a time; cross-episode promos come only from the stitching and join stages.
* Script text only: a slap or a look written as one short action line can be underrated; sarcasm can be read literally.
* Ranking weights are proposals, not set by the client. Free-tier models are weaker than the best paid ones.

## Pipeline settings (config.yaml -> pipeline)

* `run_cold_viewer: false` - the cold-viewer check (can someone who never saw the show follow the clip?) costs one
  call per promo, about a third of all calls, and does not affect ranking while `ranking.cold_viewer_fail_penalty`
  is 0. It is off to save free quota; set it to `true` (and give the penalty a weight) to use it.
* `duration_repair` - near-miss length fix, see below. `enabled: false` turns it off.

## Duration repair (repair.py)

The strict validator is unchanged. After it rejects a clip **only** for length, and the clip misses 20-60s by at
most `max_miss_s` (6s), code may move **one** edge by **one** line: drop a line after the cliffhanger or a setup line
before the hook (too long), or add one earlier setup line if the hook still starts within 3s (too short). Extending
past the cliffhanger is the last resort and is flagged. The hook and cliffhanger lines are never removed, the result
must pass the full strict check again, and repairs are used only when no retry produced a strictly valid clip.
Every repair is labelled: `auto_adjusted` in the CSV, a red tag in the dashboard, an audit line, and the original
rejection kept in `raw_results.json`.

## Files

screenplay_parser.py (parse) | validator.py + timeline.py (rules, ID->time) | repair.py (near-miss length fix) |
ai_analyzer.py + llm.py (model calls, fallbacks, cache) | promo_scorer.py (weights, dedupe, diversity) |
stitch.py (cross-episode promos) | slate.py (posting plan) | verify.py (audit) | dashboard_html.py (dashboard.html) | main.py (pipeline) |
app.py (Streamlit UI) | prompts/*.txt (prompts: edit these).

Outputs per run: raw_results.json (all candidates and every rejection, unedited), ranked_promos.csv/json, report.md, posting_plan.md,
audit.json, dashboard.html, raw/calls.jsonl (every model call and which model answered it).

Ranking weights are proposals (hook .30, cliffhanger .35, standalone .25, emotional .10), not from Kuku.
data/ and outputs/ are gitignored.
