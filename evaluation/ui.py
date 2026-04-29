"""Gradio dashboard for RAG evaluation.

Design principles:
1. Layout is STATIC. Components never appear/disappear or resize between states —
   only their values change. Empty states still occupy their final dimensions.
2. Progress is rendered INSIDE the layout (custom HTML), never as a Gradio overlay.
3. No exception ever escapes a handler. Errors are rendered into the status row.
"""
from __future__ import annotations

import threading
import time
import traceback

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from evaluation.runner import TECHNIQUES, get_technique, run_evaluation
from evaluation.test_cases import ALL_CASES


_RESULTS_CACHE: dict[str, dict] = {}


def _filter_cases(difficulty: str) -> list[dict]:
    if difficulty == "all":
        return ALL_CASES
    if difficulty == "hard+medium":
        return [tc for tc in ALL_CASES if tc["difficulty"] in ("hard", "medium")]
    return [tc for tc in ALL_CASES if tc["difficulty"] == difficulty]

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 2rem 1.5rem 3rem !important;
}

#hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(139,92,246,0.06) 60%, rgba(236,72,153,0.04) 100%);
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 1.6rem 1.85rem;
    margin-bottom: 1.5rem;
}
#hero h1 {
    font-size: 1.55rem; font-weight: 700; margin: 0 0 0.4rem;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #4f46e5, #8b5cf6 60%, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; display: inline-block;
}
#hero p {
    color: var(--body-text-color-subdued); font-size: 0.93rem;
    margin: 0; max-width: 740px; line-height: 1.55;
}

.section-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--body-text-color-subdued);
    margin: 1.4rem 0 0.65rem;
}

.status-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.85rem 1.1rem;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 10px; min-height: 56px;
    margin: 0.5rem 0 0.4rem;
    transition: border-color 0.2s ease;
}
.status-row.running { border-color: rgba(99,102,241,0.45); }
.status-row.done    { border-color: rgba(16,185,129,0.4); }
.status-row.error   { border-color: rgba(220,38,38,0.5); background: rgba(220,38,38,0.04); }
.status-icon { font-size: 1.05rem; }
.status-text { flex: 1; font-size: 0.88rem; font-weight: 500; color: var(--body-text-color-subdued); overflow: hidden; text-overflow: ellipsis; }
.status-text strong { color: var(--body-text-color); font-weight: 600; }
.status-row.error .status-text { color: #b91c1c; }
.status-row.error .status-text strong { color: #991b1b; }
.status-progress { flex: 2; height: 6px; background: var(--border-color-primary); border-radius: 3px; overflow: hidden; }
.status-progress-bar {
    height: 100%; width: 0;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 3px; transition: width 0.35s ease;
}
.status-progress-bar.done { background: linear-gradient(90deg, #10b981 0%, #059669 100%); }
.status-percent {
    font-size: 0.82rem; color: var(--body-text-color-subdued);
    font-variant-numeric: tabular-nums; min-width: 56px; text-align: right;
}

.card-grid { display: grid; gap: 0.8rem; }
.card-grid-4 { grid-template-columns: repeat(4, 1fr); }
.card-grid-3 { grid-template-columns: repeat(3, 1fr); }

.metric-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px; padding: 1rem 1.15rem;
    min-height: 108px; display: flex; flex-direction: column;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.metric-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px -10px rgba(99,102,241,0.3);
    border-color: rgba(99,102,241,0.45);
}
.metric-label {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--body-text-color-subdued);
}
.metric-value {
    font-size: 1.7rem; font-weight: 700; margin-top: 0.35rem;
    letter-spacing: -0.025em; color: var(--body-text-color);
    font-variant-numeric: tabular-nums; line-height: 1.1;
}
.metric-value.empty { color: var(--body-text-color-subdued); opacity: 0.45; }
.metric-delta {
    margin-top: auto; padding-top: 0.4rem;
    font-size: 0.8rem; font-weight: 600;
    font-variant-numeric: tabular-nums; min-height: 1.1em;
}
.metric-delta.up   { color: #16a34a; }
.metric-delta.down { color: #dc2626; }
.metric-delta.flat { color: var(--body-text-color-subdued); font-weight: 500; }

footer { display: none !important; }
"""


HEADER_HTML = """
<div id="hero">
    <h1>📊 RAG Evaluation Dashboard</h1>
    <p>Compare 5 retrieval techniques across 100 test cases. Run <strong>baseline</strong> first, then any improvement to see how the metrics shift — green ▲ means it got better, red ▼ means worse.</p>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML builders
# ─────────────────────────────────────────────────────────────────────────────
def _esc(s: str) -> str:
    """Escape so an error message containing < or > can't break the HTML."""
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _status_html(state: str, done: int = 0, total: int = 100,
                 elapsed: float | None = None, error_msg: str = "") -> str:
    pct = (done / total * 100) if total else 0
    bar_class = "status-progress-bar"

    if state == "idle":
        icon, text, bar_w, right = "💤", "<strong>Ready</strong> — pick a technique and click Run evaluation", 0, ""
    elif state == "running":
        icon, text, bar_w, right = "⚙️", f"<strong>Running</strong> — question {done} of {total}", pct, f"{pct:.0f}%"
    elif state == "done":
        et = f" in {elapsed:.1f}s" if elapsed else ""
        icon, text, bar_w, right = "✅", f"<strong>Done</strong>{et} — {total} questions evaluated", 100, "100%"
        bar_class += " done"
    elif state == "error":
        icon = "❌"
        text = f"<strong>Error:</strong> {_esc(error_msg)[:300]}"
        bar_w, right = 0, ""
    else:
        icon, text, bar_w, right = "—", "", 0, ""

    return f"""
    <div class="status-row {state}">
        <div class="status-icon">{icon}</div>
        <div class="status-text">{text}</div>
        <div class="status-progress"><div class="{bar_class}" style="width:{bar_w}%"></div></div>
        <div class="status-percent">{right}</div>
    </div>
    """


def _delta_html(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return '<div class="metric-delta flat">&nbsp;</div>'
    d = value - baseline
    if abs(d) < 1e-4:
        return '<div class="metric-delta flat">— vs baseline</div>'
    arrow, cls = ("▲", "up") if d > 0 else ("▼", "down")
    return f'<div class="metric-delta {cls}">{arrow} {d:+.3f} vs baseline</div>'


def _card_html(label: str, value: float | None, baseline: float | None, fmt: str = "{:.3f}") -> str:
    if value is None:
        val_str, val_class = "—", "metric-value empty"
    else:
        val_str, val_class = fmt.format(value), "metric-value"
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{val_class}">{val_str}</div>
        {_delta_html(value, baseline)}
    </div>
    """


def _retrieval_cards_html(res: dict | None, baseline: dict | None) -> str:
    def b(*path):
        cur = baseline
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                return None
        return cur
    return f"""<div class="card-grid card-grid-4">
        {_card_html("MRR", res["retrieval"]["MRR"] if res else None, b("retrieval", "MRR"))}
        {_card_html("nDCG@5", res["retrieval"]["nDCG@5"] if res else None, b("retrieval", "nDCG@5"))}
        {_card_html("Hit@5", res["retrieval"]["Hit@5"] if res else None, b("retrieval", "Hit@5"), fmt="{:.0%}")}
        {_card_html("Overall", res["overall_score"] if res else None, b("overall_score"), fmt="{:.0%}")}
    </div>"""


def _answer_cards_html(res: dict | None, baseline: dict | None) -> str:
    def b(*path):
        cur = baseline
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                return None
        return cur
    return f"""<div class="card-grid card-grid-3">
        {_card_html("Accuracy",     res["answer"]["accuracy"]     if res else None, b("answer", "accuracy"),     fmt="{:.2f} / 5")}
        {_card_html("Completeness", res["answer"]["completeness"] if res else None, b("answer", "completeness"), fmt="{:.2f} / 5")}
        {_card_html("Relevance",    res["answer"]["relevance"]    if res else None, b("answer", "relevance"),    fmt="{:.2f} / 5")}
    </div>"""


def _comparison_chart(difficulty: str = "all") -> go.Figure:
    fig = go.Figure()
    suffix = f"::{difficulty}"
    relevant = {k.split("::")[0]: v for k, v in _RESULTS_CACHE.items() if k.endswith(suffix)}
    try:
        if relevant:
            techniques = list(relevant.keys())
            spec = [
                ("MRR",     lambda r: r["retrieval"]["MRR"]),
                ("nDCG@5",  lambda r: r["retrieval"]["nDCG@5"]),
                ("Hit@5",   lambda r: r["retrieval"]["Hit@5"]),
                ("Acc/5",   lambda r: r["answer"]["accuracy"] / 5),
                ("Comp/5",  lambda r: r["answer"]["completeness"] / 5),
                ("Overall", lambda r: r["overall_score"]),
            ]
            palette = ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981"]
            for i, tech in enumerate(techniques):
                r = relevant[tech]
                fig.add_trace(go.Bar(
                    name=tech,
                    x=[s[0] for s in spec],
                    y=[s[1](r) for s in spec],
                    marker_color=palette[i % len(palette)],
                ))
    except Exception:
        pass
    fig.update_layout(
        barmode="group",
        title=dict(text=f"Technique comparison · difficulty = {difficulty} · higher is better",
                   font=dict(size=14)),
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(showgrid=False),
        height=420,
        margin=dict(l=20, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if not relevant:
        fig.add_annotation(
            text=f"No runs yet for difficulty = {difficulty} — pick a technique and click Run",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="rgba(120,120,120,0.65)"),
        )
    return fig


_TABLE_COLS = [
    "question", "difficulty", "RR", "nDCG@5", "hit@5",
    "accuracy", "completeness", "relevance", "keyword_coverage",
]


def _table_for(res: dict | None) -> pd.DataFrame:
    if not res:
        return pd.DataFrame(columns=_TABLE_COLS)
    df = pd.DataFrame(res["per_question"])
    return df[[c for c in _TABLE_COLS if c in df.columns]]


def _safe_description(tech: str) -> str:
    """Get the description for a technique without ever throwing."""
    try:
        return f"**{tech}** — {get_technique(tech).DESCRIPTION}"
    except Exception as e:
        return f"**{tech}** — _failed to load module: {_esc(repr(e))}_"


def _safe_render(tech: str, difficulty: str = "all", *, status_state: str = "idle",
                 done: int = 0, total: int = 100,
                 elapsed: float | None = None, error_msg: str = "") -> tuple:
    """Build all UI updates. Never raises — errors become a status-row error."""
    try:
        cache_key = f"{tech}::{difficulty}"
        baseline_key = f"baseline::{difficulty}"
        res = _RESULTS_CACHE.get(cache_key)
        baseline = _RESULTS_CACHE.get(baseline_key) if tech != "baseline" else None
        return (
            _safe_description(tech),
            _status_html(status_state, done, total, elapsed, error_msg),
            _retrieval_cards_html(res, baseline),
            _answer_cards_html(res, baseline),
            _comparison_chart(difficulty),
            _table_for(res),
        )
    except Exception as e:
        return (
            f"**{tech}** — _render error_",
            _status_html("error", error_msg=repr(e)),
            _retrieval_cards_html(None, None),
            _answer_cards_html(None, None),
            go.Figure(),
            pd.DataFrame(columns=_TABLE_COLS),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Build app
# ─────────────────────────────────────────────────────────────────────────────
def build_demo(vectorstore):
    theme = gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_md,
    )

    with gr.Blocks(title="RAG Evaluation Dashboard") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=True):
            technique_dd = gr.Dropdown(
                choices=TECHNIQUES, value="baseline",
                label="Technique", scale=2,
            )
            difficulty_dd = gr.Dropdown(
                choices=["all", "easy", "medium", "hard", "hard+medium"],
                value="all",
                label="Difficulty filter",
                info="Restrict the eval to a subset to expose where techniques actually help",
                scale=2,
            )
            run_btn = gr.Button("▶ Run evaluation", variant="primary", scale=1, min_width=160)
            view_btn = gr.Button("👁 View cached", scale=1, min_width=140)

        description = gr.Markdown(value=_safe_description("baseline"))
        status = gr.HTML(value=_status_html("idle"))

        gr.HTML('<div class="section-label">Retrieval metrics</div>')
        retrieval_cards = gr.HTML(value=_retrieval_cards_html(None, None))

        gr.HTML('<div class="section-label">Answer quality (LLM-as-judge, 1–5)</div>')
        answer_cards = gr.HTML(value=_answer_cards_html(None, None))

        with gr.Tabs():
            with gr.Tab("📈 Comparison chart"):
                chart = gr.Plot(value=_comparison_chart())
            with gr.Tab("🔍 Per-question results"):
                table = gr.Dataframe(value=_table_for(None), interactive=False, wrap=True)

        outputs = [description, status, retrieval_cards, answer_cards, chart, table]

        # ─── Run handler ───────────────────────────────────────────────────
        def on_run(tech: str, difficulty: str):
            cases = _filter_cases(difficulty)
            total = len(cases)
            try:
                yield _safe_render(tech, difficulty, status_state="running", done=0, total=total)
            except Exception as e:
                yield _safe_render(tech, difficulty, status_state="error", error_msg=repr(e))
                return

            shared = {"done": 0, "total": total, "result": None, "error": None}

            def cb(d, t):
                shared["done"] = d
                shared["total"] = t

            def worker():
                try:
                    shared["result"] = run_evaluation(
                        vectorstore=vectorstore,
                        technique_name=tech,
                        test_cases=cases,
                        progress=False,
                        progress_callback=cb,
                    )
                except Exception as e:
                    shared["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"

            t = threading.Thread(target=worker, daemon=True)
            start = time.time()
            t.start()

            while t.is_alive():
                time.sleep(0.4)
                yield _safe_render(tech, difficulty, status_state="running",
                                   done=shared["done"], total=shared["total"])
            t.join()

            elapsed = time.time() - start
            if shared["error"]:
                yield _safe_render(tech, difficulty, status_state="error", error_msg=shared["error"])
                return

            _RESULTS_CACHE[f"{tech}::{difficulty}"] = shared["result"]
            yield _safe_render(tech, difficulty, status_state="done",
                               done=shared["total"], total=shared["total"], elapsed=elapsed)

        def on_view(tech: str, difficulty: str):
            return _safe_render(tech, difficulty, status_state="idle")

        def on_change(tech: str):
            return _safe_description(tech)

        run_btn.click(on_run, [technique_dd, difficulty_dd], outputs)
        view_btn.click(on_view, [technique_dd, difficulty_dd], outputs)
        technique_dd.change(on_change, [technique_dd], [description])

    return demo, CUSTOM_CSS, theme
