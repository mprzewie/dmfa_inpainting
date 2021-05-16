import streamlit as st
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import base64
from collections import defaultdict

st.set_page_config(
    page_title="DMFA metrics",
    page_icon=None,
    layout="wide",
)


def get_table_download_link(df, type="csv", filename="df"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    dumped = df.to_csv() if type == "csv" else df.to_latex()
    b64 = base64.b64encode(
        dumped.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a download="{filename}.{type}" href="data:file/csv;base64,{b64}">Download DF as {type}</a>'
    return href


def df_container(df, filename="df"):
    with st.beta_container():
        st.write(df)
        st.markdown(
            get_table_download_link(df, "csv", filename=filename),
            unsafe_allow_html=True,
        )
        st.markdown(
            get_table_download_link(df, "latex", filename=filename),
            unsafe_allow_html=True,
        )


HIST_FILE = "history.json"
st.title("DMFA inpainting metrics")

results_root = Path(".")

c1, c2, c3 = st.beta_columns(3)

with c1:

    # st.sidebar.write("Settings")
    task = st.selectbox("Model task", options=["classification", "generation"])

with c2:
    dataset = st.selectbox(
        "Dataset", options=[p.stem for p in (results_root / task).iterdir()]
    )


ds_root = results_root / task / dataset

experiments = [p.relative_to(ds_root).parent for p in ds_root.rglob(f"**/{HIST_FILE}")]

experiments_choices = sorted(
    {p.parent for p in experiments},
    key=lambda p: (ds_root / p).stat().st_mtime,
    reverse=True,
)
with c3:
    experiment = st.selectbox("Experiment", options=experiments_choices)

results_root = ds_root / experiment

experiment_id = f"{task}-{dataset}-{experiment}".replace("/", "-")
exp_results = dict()

for p in results_root.glob(f"*/{HIST_FILE}"):
    with p.open("r") as f:
        exp_results[p.parent.name] = json.load(f)

exp_results.keys()

rows = []

for exp_name, hist in exp_results.items():
    for h in hist:
        e = h["epoch"]

        for m_name, m_per_fold in h["metrics"].items():
            for fold, value in m_per_fold.items():
                row = {
                    "exp_name": exp_name,
                    "epoch": e,
                    "fold": fold,
                    "metric": m_name,
                    "value": value,
                }
                rows.append(row)

df = pd.DataFrame(rows)

table, charts = st.beta_columns([1, 1])


with table:
    # df_by_epoch = df.melt()
    """## Metrics"""
    fold_opts = list(df.fold.unique())
    fold = st.selectbox("Fold", options=fold_opts, index=fold_opts.index("val"))
    index_cols = ["exp_name", "epoch"]
    df_by_epoch = (
        df[df.fold == fold]
        .pivot(index=index_cols, columns=["metric"], values=["value"])
        .reset_index()
    )
    df_by_epoch.columns = [c[0] if c[1] == "" else c[1] for c in df_by_epoch.columns]
    # df_piv.columns = df_piv.columns.droplevel(0)

    with st.beta_expander("Final epoch metrics", expanded=True):
        fe_df = pd.concat(
            [
                df_by_epoch[df_by_epoch.exp_name == exp_name][
                    df_by_epoch.epoch == df[df.exp_name == exp_name].epoch.max()
                ]
                for exp_name in df.exp_name.unique()
            ]
        ).set_index("exp_name")
        df_container(fe_df, filename=f"final-epoch-{experiment_id}")
    # st.markdown(get_table_download_link(fe_df), unsafe_allow_html=True)

    with st.beta_expander("Best epoch metrics", expanded=True):
        metric = st.selectbox("Metric", options=df.metric.unique())

        l_or_h = st.selectbox(
            "Lower or higher?",
            options=["lower is better", "higher is better"],
            index=0
            if fe_df[metric].mean() < df_by_epoch[df_by_epoch.epoch == 0][metric].mean()
            else 1,  # if metric is decreasing we assume lower is better
        )
        better_fn = min if l_or_h == "lower is better" else max

        be_df = pd.concat(
            [
                df_by_epoch[df_by_epoch.exp_name == exp_name][
                    df_by_epoch[metric]
                    == better_fn(
                        df[df.fold == fold][df.exp_name == exp_name][
                            df.metric == metric
                        ].value
                    )
                ]
                for exp_name in df.exp_name.unique()
            ]
        )
        be_df = be_df[
            index_cols + [metric] + [m for m in df.metric.unique() if m != metric]
        ].set_index("exp_name")
        df_container(be_df, filename=f"best-epoch-{experiment_id}")

    with st.beta_expander("Ranking across epochs", expanded=True):
        epoch_to_start_from = st.slider(
            label="Epoch to start from",
            min_value=int(df_by_epoch.epoch.min()),
            max_value=int(df_by_epoch.epoch.max()),
            step=1,
            value=(int(df_by_epoch.epoch.min()) + int(df_by_epoch.epoch.max())) // 2,
        )

        exp_to_metric_to_score = defaultdict(lambda: defaultdict(int))
        for e in df_by_epoch.epoch.unique():
            if e < epoch_to_start_from:
                continue

            for m in df.metric.unique():
                e_max = df_by_epoch.epoch.max()
                lower_better = (
                    fe_df[m].mean() < df_by_epoch[df_by_epoch.epoch == 0][m].mean()
                )

                e_df = df_by_epoch[df_by_epoch.epoch == e].sort_values(
                    m, ascending=not lower_better
                )

                for score, (i, r) in enumerate(e_df.iterrows()):
                    exp_to_metric_to_score[r.exp_name][m] += score

        ranking_df = pd.DataFrame(
            [
                {"exp_name": exp_name, **r}
                for (exp_name, r) in exp_to_metric_to_score.items()
            ]
        ).set_index("exp_name")
        df_container(
            ranking_df,
            filename=f"ranking-from-epoch-{epoch_to_start_from}-{experiment_id}",
        )

    # todo maybe weighted?

    # df_by_epoch
    # df_mlt

with charts:
    """## Charts"""
    for m in sorted(df.metric.unique()):
        with st.beta_expander(m):
            st.write(
                alt.Chart(df[df.metric == m])
                .mark_line(point=True)
                .encode(
                    x="epoch",
                    y="value",
                    color="exp_name",
                    column=alt.Column("fold", sort=["val", "train"]),
                    tooltip="exp_name",
                )
                .interactive()
                .properties(title=m)
            )
