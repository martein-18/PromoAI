"""Streamlit dashboard.  streamlit run app.py"""
import os, tempfile
from pathlib import Path
import streamlit as st
from config import load_config
from main import run_pipeline, load_episodes
from promo_scorer import to_dataframe
from dashboard_html import load_run, build_dashboard

st.set_page_config(page_title="PromoAI", layout="wide")
cfg = load_config("config.yaml")
OUT_ROOT = Path(cfg["paths"]["outputs_dir"])

with st.sidebar:
    st.title("PromoAI")
    st.header("Model")
    mock = st.checkbox("Mock model (fake, offline test only)", value=False)
    cfg["llm"]["base_url"] = st.text_input("Base URL", cfg["llm"]["base_url"])
    cfg["llm"]["model"] = st.text_input("Model", cfg["llm"]["model"])
    key = st.text_input("API key (kept in memory only)", type="password")
    if key:
        os.environ[cfg["llm"]["api_key_env"]] = key
    st.caption("Fallbacks: " + (", ".join(f["model"] for f in cfg["llm"].get("fallbacks", [])) or "none"))
    st.header("Pipeline")
    cfg["pipeline"]["candidates_per_episode"] = st.slider("Candidates per episode", 1, 6, cfg["pipeline"]["candidates_per_episode"])
    stitch = st.checkbox("Try cross-episode stitching", value=False)
    st.header("Ranking weights (proposed)")
    w = cfg["ranking"]["weights"]
    for k in list(w):
        w[k] = st.slider(k, 0.0, 1.0, float(w[k]), 0.05)
    cfg["ranking"]["cold_viewer_fail_penalty"] = st.slider("Cold-viewer fail penalty", 0.0, 2.0, 0.0, 0.25)
    cap = st.slider("Max per hook type in top 10 (0 = off)", 0, 10, 0)
    cfg["ranking"]["diversity"] = {"max_per_hook_type": cap, "top_k": 10}

st.title("PromoAI: finding promos in microdrama screenplays")
run_tab, view_tab = st.tabs(["1. Run on episodes", "2. View a finished run"])

with run_tab:
    up = st.file_uploader("Upload episode .txt files (or leave empty to use the data folder)", type="txt", accept_multiple_files=True)
    data_dir = cfg["paths"]["data_dir"]
    if up:
        data_dir = tempfile.mkdtemp()
        for f in up:
            Path(data_dir, f.name).write_bytes(f.getvalue())
    eps = load_episodes(data_dir)
    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes found", len(eps))
    c2.metric("Events parsed", sum(len(e.events) for e in eps))
    c3.metric("Parse warnings", sum(1 for e in eps if e.warnings))
    bad = [e for e in eps if e.warnings]
    if bad:
        with st.expander("Episodes with parse warnings"):
            for e in bad:
                st.write(e.ep, e.path, e.warnings)
    if eps:
        with st.expander("Peek at how an episode is parsed"):
            pick = st.selectbox("Episode", [e.ep for e in eps])
            ep = next(e for e in eps if e.ep == pick)
            st.write(f"**Logline:** {ep.logline}")
            st.dataframe([{"id": e.id, "start": e.start, "end": e.end, "type": e.type, "speaker": e.speaker, "text": e.text}
                          for e in ep.events], hide_index=True)
    chosen = st.multiselect("Episodes to analyse (empty = all)", [e.ep for e in eps])
    out_name = st.text_input("Output folder name", "run_1")
    if st.button("Run", type="primary", disabled=not eps):
        bar, msg = st.progress(0.0), st.empty()
        n = len(chosen or eps)
        state = {"i": 0}

        def progress(m):
            msg.write(m)
            if m.startswith("episode"):
                state["i"] += 1
                bar.progress(min(state["i"] / n, 1.0))
        try:
            run_pipeline(cfg, data_dir, OUT_ROOT / out_name, chosen or None, mock, stitch, progress)
            bar.progress(1.0)
            msg.write("done: open tab 2 and pick this run")
            st.session_state["last_run"] = out_name
        except Exception as ex:
            st.error(f"Run failed: {ex}. Re-running the same output folder resumes from saved answers.")

with view_tab:
    runs = sorted([p.name for p in OUT_ROOT.glob("*") if (p / "ranked_promos.json").exists()]) if OUT_ROOT.exists() else []
    if not runs:
        st.info("No finished runs yet. Use tab 1.")
    else:
        default = runs.index(st.session_state["last_run"]) if st.session_state.get("last_run") in runs else 0
        name = st.selectbox("Run", runs, index=default)
        R = load_run(OUT_ROOT / name)
        m, promos, aud = R["meta"], R["promos"], R["audit"]
        if aud.get("status") == "MOCK RUN":
            st.error("MOCK RUN: placeholder output from a fake model. Not real AI results.")
        else:
            st.success("REAL RUN: answers produced by " + ", ".join(m.get("models_used") or [m.get("model", "?")]))
        k = st.columns(6)
        for col, (lab, v) in zip(k, [("Episodes", len(m.get("episodes", []))), ("Events", R["events"]), ("LLM calls", len(R["calls"])),
                                      ("Valid candidates", len(R["candidates"])), ("Rejected", len(R["rejected"])), ("Ranked", len(promos))]):
            col.metric(lab, v)
        t_over, t_rank, t_rej, t_audit, t_calls = st.tabs(["Overview", "Ranked promos", "Rejected picks", "Audit: is it real?", "Raw model calls"])
        with t_over:
            st.markdown("**Pipeline:** parse (code) → show bible (LLM) → find candidates (LLM, by line ID) → validate (code) → "
                        "score + cold viewer (LLM) → rank + dedupe (code) → explain ranking (LLM)")
            if m.get("ranking_summary"):
                st.subheader("Post first")
                st.write(m["ranking_summary"])
                st.caption("Model rationale for the ordering, not a human decision.")
            if promos:
                st.subheader("Top 3")
                for p in promos[:3]:
                    st.markdown(f"**#{p['rank']}** · " + " + ".join(f"Ep {s['ep']} {s['start_ts']}–{s['end_ts']}" for s in p["segments"])
                                + f" · {p['total_duration']}s · score {p.get('final_score')}")
                    st.write(f"Hook: {p.get('hook')}  \nCliffhanger: {p.get('cliffhanger')}")
        with t_rank:
            if not promos:
                st.info("No valid promos in this run.")
            else:
                st.dataframe(to_dataframe(promos), hide_index=True)
                for p in promos:
                    segs = " + ".join(f"Ep {s['ep']} {s['start_ts']}–{s['end_ts']}" for s in p["segments"])
                    with st.expander(f"#{p['rank']}  {segs}  ({p['total_duration']}s, score {p.get('final_score')})"):
                        st.write(f"**Hook:** {p.get('hook')}")
                        st.write(f"**Cliffhanger:** {p.get('cliffhanger')}")
                        st.write(f"**Why it works:** {p.get('why')}")
                        if p.get("post_reason"):
                            st.write(f"**Why this rank:** {p['post_reason']}")
                        sc = p.get("scores") or {}
                        cols = st.columns(4)
                        for col, key_ in zip(cols, ["hook", "cliffhanger", "standalone", "emotional"]):
                            v = sc.get(key_, {}).get("score") if isinstance(sc.get(key_), dict) else None
                            col.metric(key_.title(), f"{v}/5" if v is not None else "n/a")
                            if isinstance(v, (int, float)):
                                col.progress(min(max(v / 5, 0), 1))
                        for note in (aud.get("per_promo_notes") or {}).get(p["id"], []) + p.get("warnings", []):
                            st.warning(note)
                        st.json({"cold_viewer": p.get("cold_viewer"), "score_reasons": sc})
                        st.code(p.get("clip_text", ""))
                st.download_button("Download CSV", to_dataframe(promos).to_csv(index=False), "ranked_promos.csv", "text/csv")
        with t_rej:
            st.caption("Picks the model made that our rules threw out, and why.")
            st.dataframe([{"source": x.get("source"), "reasons": " · ".join(x.get("issues", []))} for x in R["rejected"]], hide_index=True)
        with t_audit:
            icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}
            for c in aud.get("checks", []):
                st.write(f"{icon.get(c['status'], '')} **{c['name']}**: {c['detail']}")
            st.caption("Re-derived from files on disk after the run. Nothing here calls a model.")
        with t_calls:
            st.caption("Every prompt sent and every reply received, unedited.")
            for c in R["calls"][-30:]:
                with st.expander(f"{c['task']} · {c['model']} · {c['seconds']}s"):
                    st.text_area("Prompt", c["prompt"], height=160, key=f"p{c['ts']}")
                    st.text_area("Reply", c["response"], height=160, key=f"r{c['ts']}")
        html_path = build_dashboard(OUT_ROOT / name)
        st.download_button("Download presentation page (dashboard.html)", Path(html_path).read_bytes(), "dashboard.html", "text/html")