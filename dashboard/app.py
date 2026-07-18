"""Streamlit dashboard for the Failure-Aware Adaptive Reasoning Framework.

Run with:  streamlit run dashboard/app.py
"""

import random
import sys
import time
from pathlib import Path

import streamlit as st

# --- path setup ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.failures.taxonomy import FailureType

_METHOD_COLORS = {
    "Adaptive (Thompson)": "#16a34a",
    "Adaptive (UCB1)": "#2563eb",
    "Epsilon-greedy": "#dc2626",
    "Naive Retry": "#9ca3af",
    "Fixed Rule": "#f59e0b",
    "Random": "#a855f7",
    "One-Shot LLM": "#ef4444",
}

# ======================================================================
# Page config + custom CSS
# ======================================================================

st.set_page_config(
    page_title="Failure-Aware Agent Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0f172a; }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    [data-testid="stSidebar"] label { color: #cbd5e1 !important; }
    [data-testid="stMetric"] { background: #1e293b; border-radius: 8px; padding: 12px; }
    .badge-success { background:#16a34a; color:#fff; padding:2px 8px; border-radius:4px; font-weight:600; }
    .badge-fail    { background:#dc2626; color:#fff; padding:2px 8px; border-radius:4px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# Header
# ======================================================================

st.markdown("# Failure-Aware Adaptive Reasoning Framework -- Live Dashboard")
st.caption(
    "An LLM agent that detects, classifies, and learns to recover from execution failures "
    "using multi-armed bandit strategy selection."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Classifier Accuracy", "95%", help="TF-IDF + LinearSVC failure-type classifier")
c2.metric("Best Recovery Rate", "65.5%", help="Thompson Sampling late-stage success rate")
c3.metric("Significance", "p < 0.05", help="vs all baselines (paired t-test)")
c4.metric("Bandit Algorithms", "3", help="UCB1, Epsilon-greedy, Thompson Sampling")

st.divider()

# ======================================================================
# Lazy-loaded singletons
# ======================================================================

@st.cache_resource
def _load_classifier():
    """Load or train the FailureClassifier; returns (clf, metrics_dict)."""
    from src.failures.classifier import FailureClassifier
    from src.failures.injector import FailureInjector

    clf = FailureClassifier()
    joblib_path = _DATA_DIR / "classifier_model.joblib"

    if joblib_path.exists():
        clf.load(joblib_path)
        inj = FailureInjector(injection_rate=1.0, seed=777)
        for i in range(100):
            inj.inject({
                "step_index": i,
                "tool_name": ["web_search", "calculator", "db_query", "code_executor"][i % 4],
                "tool_args": {}, "goal": f"eval {i}", "tool_output": f"out {i}",
            })
        metrics = clf.evaluate(inj.get_injection_log()[:20])
    else:
        inj = FailureInjector(injection_rate=1.0, seed=42)
        for i in range(300):
            inj.inject({
                "step_index": i,
                "tool_name": ["web_search", "calculator", "db_query", "code_executor"][i % 4],
                "tool_args": {"q": str(i)}, "goal": f"train {i}", "tool_output": f"out {i}",
            })
        events = inj.get_injection_log()
        split = int(len(events) * 0.8)
        clf.train(events[:split])
        metrics = clf.evaluate(events[split:])
        clf.save(joblib_path)

    return clf, metrics


@st.cache_resource
def _run_selector_simulation(n_episodes: int = 500, algorithm: str = "thompson_sampling", seed: int = 42):
    """Run a selector simulation and return the selector."""
    from src.recovery.selector import AdaptiveSelector
    from src.failures.taxonomy import FailureEvent
    from src.recovery.strategies import execute_strategy

    sel = AdaptiveSelector(algorithm=algorithm, seed=seed)
    rng = random.Random(seed)
    for i in range(n_episodes):
        ft = rng.choice(list(FailureType))
        strat = sel.select_strategy(ft)
        ev = FailureEvent(failure_type=ft, raw_error="sim")
        outcome = execute_strategy(strat, ev, {"tool_name": "x", "step_index": i, "goal": "sim"}, rng=rng)
        sel.update(ft, strat, success=outcome["success"], episode_number=i)
    return sel


# ======================================================================
# Sidebar
# ======================================================================

with st.sidebar:
    st.markdown("# Controls")
    tab_choice = st.radio(
        "Navigate to",
        ["Live Simulation", "Algorithm Race", "Strategy Learning Heatmap",
         "Failure Classifier", "Live Agent Trace"],
        index=0,
    )
    st.divider()
    st.markdown("**Failure-Aware Agent** v0.2")
    st.caption("Adaptive recovery framework for LLM agents")


# ======================================================================
# Tab 1: Live Simulation
# ======================================================================

if tab_choice == "Live Simulation":
    st.markdown("# Live Simulation")
    st.caption("Compare Thompson Sampling, UCB1, and baselines over N episodes x M trials")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        num_episodes = st.slider("Episodes per trial", 50, 1000, 500, step=50)
    with c2:
        num_trials = st.slider("Trials (independent runs)", 10, 30, 30)
        st.caption("30 trials recommended for stable, statistically significant results.")
    with c3:
        st.markdown("")
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        from src.evaluation.harness import run_comparison, plot_comparison, plot_regret

        with st.spinner(f"Running {num_episodes} episodes x {num_trials} trials ..."):
            summary_df, raw_df = run_comparison(num_episodes=num_episodes, num_trials=num_trials)
            plot_comparison(raw_df, num_episodes, _DATA_DIR)
            plot_regret(raw_df, num_episodes, _DATA_DIR)

        st.session_state["sim_summary"] = summary_df
        st.session_state["sim_raw"] = raw_df
        st.session_state["sim_episodes"] = num_episodes

    if "sim_summary" in st.session_state:
        summary_df = st.session_state["sim_summary"]
        raw_df = st.session_state["sim_raw"]
        num_episodes = st.session_state["sim_episodes"]

        # --- Summary table ---
        st.subheader("Summary Results")
        display_cols = [
            "mean_late_stage_success_rate", "std_late_stage_success_rate",
            "mean_final_cumulative_regret", "std_final_cumulative_regret",
            "mean_total_successes",
        ]
        display_df = summary_df[display_cols].copy()
        display_df.columns = ["Late Rate", "Late Std", "Regret", "Regret Std", "Successes"]
        display_df["Late Rate"] = display_df["Late Rate"].map("{:.2%}".format)
        display_df["Late Std"] = display_df["Late Std"].map("{:.2%}".format)
        display_df["Regret"] = display_df["Regret"].map("{:.1f}".format)
        display_df["Regret Std"] = display_df["Regret Std"].map("{:.1f}".format)
        display_df["Successes"] = display_df["Successes"].map("{:.0f}".format)
        st.dataframe(display_df, use_container_width=True)

        # --- Significance test ---
        sig_df = summary_df.attrs.get("late_stage_significance")
        if sig_df is not None:
            st.subheader("Significance Test (Late-Stage vs Thompson Sampling)")
            styled = sig_df.style.applymap(
                lambda v: "background-color: #16a34a33" if "YES" in str(v) else "",
                subset=["significant"],
            ).format("{:.4f}".format, subset=["mean_difference", "t_statistic", "p_value"])
            st.dataframe(styled, use_container_width=True)

        # --- Success rate plot ---
        st.subheader("Cumulative Success Rate")
        try:
            st.image(str(_DATA_DIR / "method_comparison.png"), use_container_width=True)
        except FileNotFoundError:
            st.warning("Plot not found. Run the simulation first.")

        # --- Regret plot ---
        st.subheader("Cumulative Regret")
        st.caption(
            "Regret measures success lost by not always choosing the optimal strategy "
            "-- lower is better, and a flattening curve means convergence."
        )
        try:
            st.image(str(_DATA_DIR / "regret_curve.png"), use_container_width=True)
        except FileNotFoundError:
            st.warning("Regret plot not found. Run the simulation first.")

        # --- Interactive plotly chart ---
        st.subheader("Interactive View")
        try:
            import plotly.express as px
            methods = raw_df["method"].unique()
            agg_rows = []
            for m in methods:
                sub = raw_df[raw_df["method"] == m]
                pivot = sub.pivot_table(index="episode", columns="trial", values="cumulative_success_rate")
                mean_c = pivot.mean(axis=1)
                for ep, val in mean_c.items():
                    agg_rows.append({"episode": ep, "method": m, "success_rate": float(val)})
            agg_df = pd.DataFrame(agg_rows)
            fig = px.line(agg_df, x="episode", y="success_rate", color="method",
                          title="Mean Cumulative Success Rate by Method",
                          labels={"success_rate": "Success Rate", "episode": "Episode"},
                          color_discrete_map=_METHOD_COLORS,
                          category_orders={"method": list(_METHOD_COLORS)})
            fig.update_layout(legend_title_text="Method", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly for interactive charts: `pip install plotly`")


# ======================================================================
# Tab 2: Algorithm Race (live animated)
# ======================================================================

if tab_choice == "Algorithm Race":
    st.markdown("# Algorithm Race")
    st.caption(
        "Watch UCB1, Epsilon-greedy, and Thompson Sampling compete in real time. "
        "All three face the same sequence of failures."
    )

    race_eps = st.slider("Episodes to race", 50, 500, 200, step=25, key="race_eps")
    batch_size = st.slider("Batch size (episodes per frame)", 1, 20, 5, key="race_batch")

    if st.button("Start Race", type="primary", key="race_btn"):
        from src.recovery.selector import AdaptiveSelector
        from src.failures.taxonomy import FailureEvent
        from src.recovery.strategies import execute_strategy

        ALGO_CONFIGS = [
            ("Thompson Sampling", "thompson_sampling", "#16a34a"),
            ("UCB1", "ucb1", "#2563eb"),
            ("Epsilon-greedy", "epsilon_greedy", "#dc2626"),
        ]

        # Shared failure-type sequence
        race_seed = 42
        type_rng = random.Random(race_seed)
        fts = [type_rng.choice(list(FailureType)) for _ in range(race_eps)]

        selectors = {}
        curves = {}
        for label, algo, _ in ALGO_CONFIGS:
            selectors[label] = AdaptiveSelector(algorithm=algo, seed=race_seed)
            curves[label] = []

        # Live chart placeholder
        chart_placeholder = st.empty()
        status_text = st.empty()
        progress_bar = st.progress(0)

        rng = random.Random(race_seed + 7)
        running_avgs = {label: 0.0 for label, _, _ in ALGO_CONFIGS}

        for batch_start in range(0, race_eps, batch_size):
            batch_end = min(batch_start + batch_size, race_eps)
            for ep in range(batch_start, batch_end):
                ft = fts[ep]
                for label, _, _ in ALGO_CONFIGS:
                    sel = selectors[label]
                    strat = sel.select_strategy(ft)
                    ev = FailureEvent(failure_type=ft, raw_error="race")
                    ctx = {"tool_name": "web_search", "step_index": ep + 1, "goal": "race"}
                    outcome = execute_strategy(strat, ev, ctx, rng=rng)
                    sel.update(ft, strat, success=outcome["success"], episode_number=ep)

                    # Update running average
                    prev_count = running_avgs[label] * ep if ep > 0 else 0.0
                    new_count = prev_count + (1.0 if outcome["success"] else 0.0)
                    running_avgs[label] = new_count / (ep + 1)
                    curves[label].append(running_avgs[label])

            # Build chart DataFrame
            chart_data = pd.DataFrame({
                ep + 1: {label: curves[label][ep] for label, _, _ in ALGO_CONFIGS}
                for ep in range(len(curves[ALGO_CONFIGS[0][0]]))
            }).T
            chart_data.index.name = "Episode"

            with chart_placeholder.container():
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for label, _, color in ALGO_CONFIGS:
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(curves[label]) + 1)),
                            y=curves[label],
                            name=label, line=dict(color=color, width=3),
                        ))
                    fig.update_layout(
                        yaxis_title="Cumulative Success Rate",
                        xaxis_title="Episode",
                        yaxis_range=[0, 1],
                        height=400,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.line_chart(chart_data)

            # Status line
            best_label = max(running_avgs, key=running_avgs.get)
            best_rate = running_avgs[best_label]
            status_text.markdown(
                f"**Episode {batch_end}/{race_eps}** | "
                f"Leader: **{best_label}** at **{best_rate:.1%}** | "
                f"Thompson: {running_avgs['Thompson Sampling']:.1%} | "
                f"UCB1: {running_avgs['UCB1']:.1%} | "
                f"Eps-greedy: {running_avgs['Epsilon-greedy']:.1%}"
            )
            progress_bar.progress(batch_end / race_eps)
            time.sleep(0.15)

        status_text.success("Race complete!")
        st.session_state["race_results"] = {label: curves[label][-1] for label, _, _ in ALGO_CONFIGS}

    if "race_results" in st.session_state:
        st.subheader("Final Results")
        res = st.session_state["race_results"]
        cols = st.columns(3)
        for i, (label, rate) in enumerate(sorted(res.items(), key=lambda x: -x[1])):
            cols[i].metric(label, f"{rate:.1%}", help=f"Cumulative success rate after {race_eps} episodes")


# ======================================================================
# Tab 3: Strategy Learning Heatmap
# ======================================================================

if tab_choice == "Strategy Learning Heatmap":
    st.markdown("# Strategy Learning Heatmap")
    st.caption(
        "Learned success rates per (FailureType, RecoveryStrategy) pair. "
        "Default view uses Thompson Sampling (best-performing algorithm)."
    )

    algo_choice = st.selectbox(
        "Algorithm to visualise",
        ["thompson_sampling", "ucb1", "epsilon_greedy"],
        format_func=lambda x: {"thompson_sampling": "Thompson Sampling (best)", "ucb1": "UCB1", "epsilon_greedy": "Epsilon-greedy"}[x],
        key="heatmap_algo",
    )

    with st.spinner(f"Training {algo_choice} selector (500 episodes) ..."):
        sel = _run_selector_simulation(500, algorithm=algo_choice)

    stats = sel.get_stats_table()
    best = sel.get_best_strategies()

    # --- Heatmap ---
    st.subheader("Learned Success-Rate Matrix")
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Heatmap(
            z=stats.values,
            x=[c.replace("_", " ").title() for c in stats.columns],
            y=[r.replace("_", " ").title() for r in stats.index],
            text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in stats.values],
            texttemplate="%{text}",
            colorscale="YlOrRd",
            zmin=0, zmax=1,
            colorbar=dict(title="Success Rate"),
        ))
        fig.update_layout(
            xaxis_title="Recovery Strategy",
            yaxis_title="Failure Type",
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_h, ax_h = plt.subplots(figsize=(8, 5))
        im = ax_h.imshow(stats.values, cmap="YlOrRd", vmin=0, vmax=1)
        ax_h.set_xticks(range(len(stats.columns)))
        ax_h.set_yticks(range(len(stats.index)))
        ax_h.set_xticklabels(stats.columns, rotation=45, ha="right", fontsize=8)
        ax_h.set_yticklabels(stats.index, fontsize=8)
        for i in range(len(stats.index)):
            for j in range(len(stats.columns)):
                v = stats.values[i, j]
                if not np.isnan(v):
                    ax_h.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                              color="white" if v > 0.5 else "black")
        fig_h.colorbar(im, ax=ax_h, label="Success Rate")
        fig_h.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

    # --- Best strategy table ---
    st.subheader("Best Strategy per Failure Type")
    from src.recovery.strategies import SUCCESS_MATRIX, RecoveryStrategy

    true_optimal = {
        ft: max(rates, key=rates.get)
        for ft, rates in SUCCESS_MATRIX.items()
    }

    compare_rows = []
    for ft in FailureType:
        learned = best[ft]
        optimal = true_optimal[ft]
        match = learned == optimal
        learned_rate = stats.loc[ft.value, learned.value] if not np.isnan(stats.loc[ft.value, learned.value]) else 0.0
        optimal_rate = SUCCESS_MATRIX[ft][optimal]
        compare_rows.append({
            "Failure Type": ft.value.replace("_", " ").title(),
            "Learned Best": learned.value.replace("_", " ").title(),
            "Learned Rate": f"{learned_rate:.2%}",
            "True Optimal": optimal.value.replace("_", " ").title(),
            "True Rate": f"{optimal_rate:.2%}",
            "Match": "Yes" if match else "No",
        })
    compare_df = pd.DataFrame(compare_rows)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)


# ======================================================================
# Tab 4: Failure Classifier
# ======================================================================

if tab_choice == "Failure Classifier":
    st.markdown("# Failure Classifier")
    st.caption("TF-IDF + LinearSVC classifier that predicts FailureType from raw error text")

    clf, metrics = _load_classifier()

    st.subheader("Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Precision", f"{metrics['precision']:.2%}")
    m3.metric("Recall", f"{metrics['recall']:.2%}")
    m4.metric("F1 Score", f"{metrics['f1']:.2%}")

    st.subheader("Confusion Matrix")
    cm_path = _DATA_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        st.info("Confusion matrix not yet generated. Run `python tests/test_classifier.py` first.")

    st.subheader("Classify a Custom Error")
    st.caption("Type any error message below and the classifier will predict its FailureType.")

    example_errors = {
        "Tool timeout": "Request to web_search timed out after 30s. No data was received.",
        "Tool error": '{"error": "HTTP 503 Service Unavailable", "tool": "api_caller"}',
        "Hallucinated output": "The capital of Australia is Sydney. Based on the analysis.",
        "Malformed output": '{"result": "ok", "data": [1, 2,}',
        "Context loss": "Referencing output from step 42: the result was X. (Note: step 42 never occurred.)",
        "Wrong tool": "Attempted to use calculator for this subtask, but the task demands web_search.",
        "Ambiguous instruction": "Interpreted 'format' as 'PDF' based on ambiguous input value 'the usual'.",
    }
    chosen_example = st.selectbox("Or pick an example error:", ["(custom)"] + list(example_errors.keys()))
    default_text = example_errors.get(chosen_example, "") if chosen_example != "(custom)" else ""

    raw_input = st.text_area("Error text to classify:", value=default_text, height=100)
    classify_btn = st.button("Classify", type="primary")

    if classify_btn and raw_input.strip():
        predicted = clf.predict(raw_input, {"tool_name": "unknown"})
        proba = clf.predict_proba(raw_input, {"tool_name": "unknown"})

        st.success(f"**Predicted FailureType:** `{predicted.value}`")

        proba_df = pd.DataFrame([
            {"Failure Type": ft.value.replace("_", " ").title(), "Confidence": conf}
            for ft, conf in sorted(proba.items(), key=lambda x: -x[1])
        ])
        try:
            import plotly.express as px
            fig_p = px.bar(proba_df, x="Confidence", y="Failure Type", orientation="h",
                           color="Confidence", color_continuous_scale="Blues",
                           range_x=[0, 1], title="Predicted Probabilities")
            fig_p.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)
        except ImportError:
            st.bar_chart(proba_df.set_index("Failure Type"))


# ======================================================================
# Tab 5: Live Agent Trace
# ======================================================================

if tab_choice == "Live Agent Trace":
    st.markdown("# Live Agent Trace")
    st.caption("Step-by-step failure -> classify -> recover -> outcome for a single episode")

    if "trace_log" not in st.session_state:
        st.session_state["trace_log"] = []
    if "trace_sel" not in st.session_state:
        st.session_state["trace_sel"] = _run_selector_simulation(500, algorithm="thompson_sampling")
    if "trace_ep" not in st.session_state:
        st.session_state["trace_ep"] = 0

    trace_sel = st.session_state["trace_sel"]
    clf, _ = _load_classifier()

    if st.button("Simulate One Episode", type="primary"):
        rng = random.Random()
        ft = rng.choice(list(FailureType))

        from src.failures.injector import FailureInjector
        injector = FailureInjector(injection_rate=1.0, seed=st.session_state["trace_ep"])
        event = injector.inject({
            "step_index": st.session_state["trace_ep"],
            "tool_name": rng.choice(["web_search", "calculator", "db_query", "code_executor"]),
            "tool_args": {"q": "trace"}, "goal": "Live demo episode",
            "tool_output": "Tool returned a response.",
        })
        if event is None:
            event = injector.get_injection_log()[-1]

        predicted = clf.predict(event.raw_error, event.step_context)
        proba = clf.predict_proba(event.raw_error, event.step_context)
        conf = proba[predicted]

        strategy = trace_sel.select_strategy(predicted)

        from src.recovery.strategies import execute_strategy
        outcome = execute_strategy(strategy, event, event.step_context, rng=rng)
        trace_sel.update(predicted, strategy, success=outcome["success"], episode_number=st.session_state["trace_ep"])

        entry = {
            "episode": st.session_state["trace_ep"] + 1,
            "failure_type": event.failure_type.value,
            "raw_error": event.raw_error[:120],
            "classifier_pred": predicted.value,
            "confidence": conf,
            "strategy": strategy.value,
            "success": outcome["success"],
            "action": outcome["action"],
        }
        st.session_state["trace_log"].insert(0, entry)
        st.session_state["trace_log"] = st.session_state["trace_log"][:10]
        st.session_state["trace_ep"] += 1
        st.rerun()

    if st.session_state["trace_log"]:
        st.subheader(f"Last {len(st.session_state['trace_log'])} Episodes")
        for entry in st.session_state["trace_log"]:
            badge = '<span class="badge-success">SUCCESS</span>' if entry["success"] else '<span class="badge-fail">FAILURE</span>'
            with st.expander(
                f"Episode {entry['episode']} | "
                f"{entry['failure_type'].replace('_',' ').title()} | "
                f"{entry['strategy'].replace('_',' ').title()} | "
                f"{'SUCCESS' if entry['success'] else 'FAILURE'}",
                expanded=(entry["episode"] == st.session_state["trace_ep"]),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Failure detected:** `{entry['failure_type']}`")
                    st.code(entry["raw_error"], language=None)
                with c2:
                    st.markdown(f"**Classifier predicts:** `{entry['classifier_pred']}` ({entry['confidence']:.0%} confidence)")
                    st.markdown(f"**Recovery strategy:** `{entry['strategy']}`")
                    st.markdown(f"**Action:** {entry['action']}")
                st.markdown(f"**Outcome:** {badge}", unsafe_allow_html=True)
    else:
        st.info("Click **Simulate One Episode** to start the trace.")
