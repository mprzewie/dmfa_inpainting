import streamlit as st
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import base64


def get_table_download_link(df, type="csv"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    dumped = df.to_csv(index=False) if type == "csv" else df.to_latex(index=False)
    b64 = base64.b64encode(
        dumped.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download DF as {type}</a>'
    return href


HIST_FILE = "history.json"
st.title("DMFA inpainting metrics")

results_root = Path(".")


st.sidebar.write("Settings")


task = st.sidebar.selectbox("Model task", options=["classification", "generation"])

dataset = st.sidebar.selectbox(
    "Dataset", options=[p.stem for p in (results_root / task).iterdir()]
)

ds_root = results_root / task / dataset

experiments = [p.relative_to(ds_root).parent for p in ds_root.rglob(f"**/{HIST_FILE}")]

experiments_choices = sorted(
    {p.parent for p in experiments},
    key=lambda p: (ds_root / p).stat().st_mtime,
    reverse=True,
)
experiment = st.sidebar.selectbox("Experiment", options=experiments_choices)

results_root = ds_root / experiment

f"""Task: **{task}**"""
f"""Dataset: **{dataset}**"""
f"""Experiment: **{experiment}**"""

"""Trained models:"""

trained_models = [p.parent.stem for p in results_root.glob(f"*/{HIST_FILE}")]
trained_models

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

table, charts = st.beta_columns([5, 2])


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

    """### Final epoch metrics"""

    fe_df = pd.concat(
        [
            df_by_epoch[df_by_epoch.exp_name == exp_name][
                df_by_epoch.epoch == df[df.exp_name == exp_name].epoch.max()
            ]
            for exp_name in df.exp_name.unique()
        ]
    )
    fe_df
    # st.markdown(get_table_download_link(fe_df), unsafe_allow_html=True)

    """### Best epoch metrics"""
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
    ]
    be_df

    """### """

    # df_by_epoch
    # df_mlt

with charts:
    """## Charts"""
    for m in sorted(df.metric.unique()):
        st.write(
            alt.Chart(df[df.metric == m])
            .mark_line(point=True)
            .encode(
                x="epoch",
                y="value",
                color="exp_name",
                column="fold",
                tooltip="exp_name",
            )
            .interactive()
            .properties(title=m)
        )
