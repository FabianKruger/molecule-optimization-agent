import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")

with app.setup:
    import json
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import polars as pl
    import altair as alt


@app.cell
def _():
    mo.md("""
    # Setup
    """)
    return


@app.cell
def _():
    run_dir = Path("./data/runs").absolute()
    figure_dir = Path("./analysis/paper/figures/").absolute()
    return figure_dir, run_dir


@app.cell
def _():
    covid_traces = [
        (
            "boltz2_binding_optimization_conversation_20260115_190233.json",
            {"intervention": "with_explanation", "repeat": 1},
        ),
        (
            "boltz2_binding_optimization_conversation_20260116_191958.json",
            {"intervention": "no_explanation", "repeat": 1},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_120713.json",
            {"intervention": "no_protein", "repeat": 1},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_135913.json",
            {"intervention": "with_explanation", "repeat": 2},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_162408.json",
            {"intervention": "no_explanation", "repeat": 2},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_162644.json",
            {"intervention": "no_protein", "repeat": 2},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_191426.json",
            {"intervention": "with_explanation", "repeat": 3},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_192713.json",
            {"intervention": "no_explanation", "repeat": 3},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_192927.json",
            {"intervention": "no_protein", "repeat": 3},
        ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_125652.json",
        #     {"intervention": "with_explanation", "repeat": 4},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_130658.json",
        #     {"intervention": "no_explanation", "repeat": 4},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_134016.json",
        #     {"intervention": "no_protein", "repeat": 4},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_215532.json",
        #     {"intervention": "with_explanation", "repeat": 5},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_215909.json",
        #     {"intervention": "no_explanation", "repeat": 5},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_220929.json",
        #     {"intervention": "no_protein", "repeat": 5},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_232902.json",
        #     {"intervention": "with_explanation", "repeat": 6},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260123_031022.json",
        #     {"intervention": "no_explanation", "repeat": 6},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260123_030136.json",
        #     {"intervention": "no_protein", "repeat": 6},
        # ),
        (
            "boltz2_binding_optimization_conversation_20260123_194052.json",
            {"intervention": "with_explanation", "repeat": 7},
        ),
        (
            "boltz2_binding_optimization_conversation_20260123_194506.json",
            {"intervention": "no_explanation", "repeat": 7},
        ),
        (
            "boltz2_binding_optimization_conversation_20260123_193903.json",
            {"intervention": "no_protein", "repeat": 7},
        ),
        (
            "boltz2_binding_optimization_conversation_20260124_125014.json",
            {"intervention": "with_explanation", "repeat": 8},
        ),
        (
            "boltz2_binding_optimization_conversation_20260124_125338.json",
            {"intervention": "no_explanation", "repeat": 8},
        ),
        (
            "boltz2_binding_optimization_conversation_20260124_125441.json",
            {"intervention": "no_protein", "repeat": 8},
        ),
    ]

    mgyp_traces = [
        (
            "boltz2_binding_optimization_conversation_20260120_185114.json",
            {"intervention": "with_explanation", "repeat": 1},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_164334.json",
            {"intervention": "no_explanation", "repeat": 1},
        ),
        (
            "boltz2_binding_optimization_conversation_20260121_125157.json",
            {"intervention": "no_protein", "repeat": 1},
        ),
        (
            "boltz2_binding_optimization_conversation_20260121_125339.json",
            {"intervention": "with_explanation", "repeat": 2},
        ),
        (
            "boltz2_binding_optimization_conversation_20260121_130409.json",
            {"intervention": "no_explanation", "repeat": 2},
        ),
        (
            "boltz2_binding_optimization_conversation_20260121_130058.json",
            {"intervention": "no_protein", "repeat": 2},
        ),
        (
            "boltz2_binding_optimization_conversation_20260121_152419.json",
            {"intervention": "with_explanation", "repeat": 3},
        ),
        (
            "boltz2_binding_optimization_conversation_20260121_153547.json",
            {"intervention": "no_explanation", "repeat": 3},
        ),
        (
            "boltz2_binding_optimization_conversation_20260121_152832.json",
            {"intervention": "no_protein", "repeat": 3},
        ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_133341.json",
        #     {"intervention": "with_explanation", "repeat": 4},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_133657.json",
        #     {"intervention": "no_explanation", "repeat": 4},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_133801.json",
        #     {"intervention": "no_protein", "repeat": 4},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_213111.json",
        #     {"intervention": "with_explanation", "repeat": 5},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_213532.json",
        #     {"intervention": "no_explanation", "repeat": 5},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_214448.json",
        #     {"intervention": "no_protein", "repeat": 5},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260123_030400.json",
        #     {"intervention": "with_explanation", "repeat": 6},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260123_030815.json",
        #     {"intervention": "no_explanation", "repeat": 6},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260123_025506.json",
        #     {"intervention": "no_protein", "repeat": 6},
        # ),
        (
            "boltz2_binding_optimization_conversation_20260123_125515.json",
            {"intervention": "with_explanation", "repeat": 7},
        ),
        (
            "boltz2_binding_optimization_conversation_20260123_131656.json",
            {"intervention": "no_explanation", "repeat": 7},
        ),
        (
            "boltz2_binding_optimization_conversation_20260123_132736.json",
            {"intervention": "no_protein", "repeat": 7},
        ),
        (
            "boltz2_binding_optimization_conversation_20260123_164046.json",
            {"intervention": "with_explanation", "repeat": 8},
        ),
        (
            "boltz2_binding_optimization_conversation_20260123_163227.json",
            {"intervention": "no_explanation", "repeat": 8},
        ),
        (
            "boltz2_binding_optimization_conversation_20260123_163719.json",
            {"intervention": "no_protein", "repeat": 8},
        ),
    ]

    covid_interventions = [
        (
            "boltz2_binding_optimization_conversation_20260115_190233.json",
            {"intervention": "with_explanation", "repeat": 1},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_135913.json",
            {"intervention": "with_explanation", "repeat": 2},
        ),
        (
            "boltz2_binding_optimization_conversation_20260120_191426.json",
            {"intervention": "with_explanation", "repeat": 3},
        ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_125652.json",
        #     {"intervention": "with_explanation", "repeat": 4},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_215532.json",
        #     {"intervention": "with_explanation", "repeat": 5},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260122_232902.json",
        #     {"intervention": "with_explanation", "repeat": 6},
        # ),
        (
            "boltz2_binding_optimization_conversation_20260123_194052.json",
            {"intervention": "with_explanation", "repeat": 7},
        ),
        (
            "boltz2_binding_optimization_conversation_20260124_125014.json",
            {"intervention": "with_explanation", "repeat": 8},
        ),
        (
            "boltz2_binding_optimization_conversation_20260119_145428.json",
            {"intervention": "no_explanation, inverted", "repeat": 1},
        ),
        (
            "boltz2_binding_optimization_conversation_20260122_014613.json",
            {"intervention": "no_explanation, inverted", "repeat": 2},
        ),
        (
            "boltz2_binding_optimization_conversation_20260122_015405.json",
            {"intervention": "no_explanation, inverted", "repeat": 3},
        ),
        # (
        #     "boltz2_binding_optimization_conversation_20260123_032522.json",
        #     {"intervention": "no_explanation, inverted", "repeat": 4},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260123_033102.json",
        #     {"intervention": "no_explanation, inverted", "repeat": 5},
        # ),
        (
            "boltz2_binding_optimization_conversation_20260125_124117.json",
            {"intervention": "no_explanation, inverted", "repeat": 6},
        ),
        (
            "boltz2_binding_optimization_conversation_20260125_192645.json",
            {"intervention": "no_explanation, inverted", "repeat": 7},
        ),
        (
            "boltz2_binding_optimization_conversation_20260119_145603.json",
            {"intervention": "no_explanation, wrong_protein", "repeat": 1},
        ),
        (
            "boltz2_binding_optimization_conversation_20260122_012156.json",
            {"intervention": "no_explanation, wrong_protein", "repeat": 2},
        ),
        (
            "boltz2_binding_optimization_conversation_20260122_014329.json",
            {"intervention": "no_explanation, wrong_protein", "repeat": 3},
        ),
        # (
        #     "boltz2_binding_optimization_conversation_20260123_030309.json",
        #     {"intervention": "no_explanation, wrong_protein", "repeat": 4},
        # ),
        # (
        #     "boltz2_binding_optimization_conversation_20260123_033842.json",
        #     {"intervention": "no_explanation, wrong_protein", "repeat": 5},
        # ),
        (
            "boltz2_binding_optimization_conversation_20260125_124115.json",
            {"intervention": "no_explanation, wrong_protein", "repeat": 6},
        ),
        (
            "boltz2_binding_optimization_conversation_20260125_191555.json",
            {"intervention": "no_explanation, wrong_protein", "repeat": 7},
        ),
    ]
    return covid_interventions, covid_traces, mgyp_traces


@app.cell
def _():
    x_title = "Oracle Calls"
    y_title = "Best Boltz-2 binding affintiy probability up to call"
    legend_title = "Model"

    intervention_map = {
        "with_explanation": "Full explanation",
        "no_explanation": "No explanation",
        "no_protein": "No description",
        "no_explanation, inverted": "No explanation, probability inverted",
        "no_explanation, wrong_protein": "No explanation, wrong protein",
    }

    color_scale = [
        "#2E86AB",  # full
        # "#00a69d",  # partial
        "#ffcc00",  # no expl
        "#f58220",  # no descr
        "#ffcc00",  # no expl, score inverted
        "#f58220",  # no expl, wrong protein
    ]


    regular_color_map = {
        "Full explanation": "#2E86AB",
        "No explanation": "#ffcc00",
        "No description": "#f58220",
    }

    intervention_color_map = {
        "Full explanation": "#2E86AB",
        "No explanation, probability inverted": "#ffcc00",
        "No explanation, wrong protein": "#f58220",
    }
    return (
        intervention_color_map,
        intervention_map,
        legend_title,
        regular_color_map,
        x_title,
        y_title,
    )


@app.cell
def _():
    covid_fig_fn = "boltz2_covid.pdf"
    mgyp_fig_fn = "boltz2_mgyp.pdf"
    intervention_fig_fn = "boltz2_interventions.pdf"
    return covid_fig_fn, intervention_fig_fn, mgyp_fig_fn


@app.cell
def _():
    mo.md("""
    # Functions
    """)
    return


@app.function
def extract_iteration_score_table(
    conversation: dict[str, Any],
    max_it: int = 49,
) -> pl.DataFrame:
    trace: dict[str, Any] = conversation["trace"]
    df = pl.from_records(trace)

    if len(df) == 0:
        df = pl.DataFrame(
            {
                "iteration": range(max_it),
                "score": [float("nan")] * max_it,
            }
        )

    cols = [pl.col("iteration")]

    if "original_score" in df.columns:
        cols.append(pl.col("original_score").alias("score"))
    else:
        cols.append(pl.col("score"))

    df = df.select(*cols)

    max_it = max(df["iteration"].max(), max_it)
    missing_its = set(range(1, max_it + 1)).difference(df["iteration"])

    if len(missing_its) != 0:
        missing_df = pl.DataFrame(
            {
                "iteration": list(missing_its),
                "score": [float("nan")] * len(missing_its),
            }
        )
        df = pl.concat([df, missing_df])

    return df


@app.function
def load_traces(
    *,
    conversation_column_list: list[tuple[str, dict[str, Any]]],
    run_dir: Path,
) -> pl.DataFrame:
    df_list = []
    for fn, extra_col_dict in conversation_column_list:
        with open(run_dir / fn) as f:
            conversation = json.load(f)

        df = extract_iteration_score_table(conversation).with_columns(
            **{k: pl.lit(v) for k, v in extra_col_dict.items()}
        )
        df_list.append(df)

    return pl.concat(df_list)


@app.function
def prepare_plotting_df(
    *,
    df: pl.DataFrame,
    intervention_map: dict[str, str],
) -> pl.DataFrame:
    return (
        df.filter(pl.col("score").is_not_nan())
        .group_by("iteration", "intervention")
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.col("score").count().alias("n"),
        )
        .with_columns(
            (pl.col("mean") + pl.col("std")).alias("mean+std"),
            (pl.col("mean") - pl.col("std")).alias("mean-std"),
        )
        .sort("intervention", "iteration")
        .with_columns(
            pl.col("intervention").replace(intervention_map),
        )
    )


@app.function
def max_score_so_far(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("score").fill_nan(0.0)  # Lowest value, so we are below max
        )
        .sort(["intervention", "repeat", "iteration"])
        .with_columns(pl.col("score").cum_max().over("intervention", "repeat"))
    )


@app.function
def forward_fill_nan(df: pl.DataFrame) -> pl.DataFrame:
    # return df.sort(["intervention", "repeat", "iteration"]).with_columns(
    #     pl.col("score")
    #     .fill_nan(value=0.5)  # None
    #     .forward_fill()  # or .fill_null(strategy="forward")
    #     .over("intervention", "repeat")  # do it separately per intervention
    # )
    return df.filter(pl.col("score").is_finite())


@app.cell
def _(intervention_map, legend_title, x_title, y_title):
    def single_chart(
        *,
        df: pl.DataFrame,
        color_map: dict[str, str],
        y_title: str = y_title,
        xy_coords: tuple[int, int] | None = None,
    ) -> alt.Chart:
        dejavu = "DejaVu Sans"

        color_enc = alt.Color(
            "intervention:N",
            sort=list(intervention_map.values()),
            scale=alt.Scale(
                domain=list(color_map.keys()),
                range=list(color_map.values()),
            ),
        )

        legend_kwargs = {"orient": "bottom-right"}
        if xy_coords:
            legend_kwargs = {
                "orient": "none",
                "legendX": xy_coords[0],
                "legendY": xy_coords[1],
            }

        legend_config = alt.Legend(
            title=legend_title,
            fillColor="white",
            strokeColor="black",
            padding=4,
            titleFontWeight="normal",  # non-bold
            titleAnchor="middle",  # center the title
            **legend_kwargs,
        )

        _mean_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="iteration",
                y="mean",
                # color=alt.Color(
                #     "intervention",
                #     sort=intervention_map.values(),
                #     scale=alt.Scale(
                #         domain=color_map.keys(),
                #         range=color_map.values(),
                #     ),
                #     legend=alt.Legend(
                #         title=legend_title,
                #         orient="bottom-right",
                #         fillColor="white",
                #         strokeColor="black",
                #         padding=4,
                #         # cornerRadius=3,
                #     ),
                # ),
                color=color_enc.copy().legend(legend_config),
            )
        )

        _std_chart = (
            alt.Chart(df)
            .mark_errorband(
                opacity=0.1,
            )
            .encode(
                x=alt.X(
                    "iteration",
                    title=x_title,
                    axis=alt.Axis(
                        titleFontWeight="normal",
                        values=list(range(0, 60, 10)),
                    ),
                ),
                y=alt.Y(
                    "mean-std",
                    title=y_title,
                    axis=alt.Axis(titleFontWeight="normal"),
                ),
                y2="mean+std",
                # color=alt.Color(
                #     "intervention",
                #     sort=list(intervention_map.values()),
                #     scale=alt.Scale(
                #         domain=color_map.keys(),
                #         range=color_map.values(),
                #     ),
                #     legend=None,
                #     # legend=alt.Legend(
                #     #     title=legend_title,
                #     #     orient="bottom-right",
                #     #     fillColor="white",
                #     #     strokeColor="black",
                #     #     padding=4,
                #     #     # cornerRadius=3,
                #     # ),
                # ),
                color=color_enc.copy().legend(None),
            )
        )

        _chart = _std_chart + _mean_chart
        _chart = (
            _chart.resolve_scale(color="independent")
            .configure(font=dejavu)
            .configure_axis(
                labelFont=dejavu,
                titleFont=dejavu,
            )
            .configure_legend(
                labelFont=dejavu,
                titleFont=dejavu,
            )
            .configure_title(font=dejavu)
        )

        _chart = _chart.properties(
            # title="Mean binding affinity with std bands",
            width=500,
            height=310,
        )
        return _chart
    return (single_chart,)


@app.function
def count_missing_scores(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select("intervention", "repeat", pl.col("score").is_nan())
        .group_by("intervention", "repeat")
        .sum()
        .sort("intervention", "repeat")
    )


@app.cell
def _():
    mo.md("""
    # Load
    """)
    return


@app.cell
def _(covid_traces, run_dir):
    raw_covid_df = load_traces(
        conversation_column_list=covid_traces,
        run_dir=run_dir,
    ).sort("intervention", "iteration", "repeat")


    count_missing_scores(raw_covid_df)
    return (raw_covid_df,)


@app.cell
def _(raw_covid_df):
    maxed_raw_covid_df = max_score_so_far(raw_covid_df)
    return (maxed_raw_covid_df,)


@app.cell
def _(intervention_map, maxed_raw_covid_df):
    maxed_covid_df = prepare_plotting_df(
        df=maxed_raw_covid_df,
        intervention_map=intervention_map,
    )
    return (maxed_covid_df,)


@app.cell
def _(intervention_map, raw_covid_df):
    covid_df = prepare_plotting_df(
        df=raw_covid_df,
        intervention_map=intervention_map,
    )
    return (covid_df,)


@app.cell
def _(intervention_map, mgyp_traces, run_dir):
    _df = load_traces(
        conversation_column_list=mgyp_traces,
        run_dir=run_dir,
    )
    raw_mgyp_df = _df
    _df = max_score_so_far(_df)
    mgyp_df = prepare_plotting_df(
        df=_df,
        intervention_map=intervention_map,
    )

    count_missing_scores(raw_mgyp_df)
    return (mgyp_df,)


@app.cell
def _(covid_interventions, intervention_map, run_dir):
    _df = load_traces(
        conversation_column_list=covid_interventions,
        run_dir=run_dir,
    )
    raw_intervention_df = _df
    tmp_df = forward_fill_nan(_df)
    non_max_intervention_df = prepare_plotting_df(
        df=tmp_df,
        intervention_map=intervention_map,
    )

    _df = max_score_so_far(_df)
    intervention_df = prepare_plotting_df(
        df=_df,
        intervention_map=intervention_map,
    )

    count_missing_scores(raw_intervention_df)
    return intervention_df, non_max_intervention_df, tmp_df


@app.cell
def _(tmp_df):
    tmp_df.sort(["intervention", "iteration", "repeat"])
    return


@app.cell
def _():
    mo.md("""
    # Plotting
    """)
    return


@app.cell
def _(covid_df):
    _mean_chart = (
        alt.Chart(covid_df)
        .mark_line()
        .encode(
            x="iteration",
            y="mean",
            color="intervention",
        )
    )

    _std_chart = (
        alt.Chart(covid_df)
        .mark_errorband()
        .encode(
            x="iteration",
            y="mean-std",
            y2="mean+std",
            color="intervention",
        )
    )

    _chart = _std_chart + _mean_chart

    _chart = _chart.properties(
        title="Mean binding affinity with std bands",
    )

    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(
    covid_fig_fn,
    figure_dir,
    maxed_covid_df,
    regular_color_map,
    single_chart,
):
    _chart = single_chart(
        df=maxed_covid_df,
        color_map=regular_color_map,
    )

    _chart = _chart.encode(y=alt.Y(scale=alt.Scale(domain=[0, 1.1])))

    _chart.save(figure_dir / covid_fig_fn)
    _chart.save((figure_dir / covid_fig_fn).with_suffix(".png"), ppi=300)

    covid_chart = _chart

    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(figure_dir, mgyp_df, mgyp_fig_fn, regular_color_map, single_chart):
    _chart = single_chart(
        df=mgyp_df,
        color_map=regular_color_map,
    )

    _chart = _chart.encode(y=alt.Y(scale=alt.Scale(domain=[0, 1.1])))

    mgyp_chart = _chart

    _chart.save(figure_dir / mgyp_fig_fn, method="selenium")
    _chart.save((figure_dir / mgyp_fig_fn).with_suffix(".png"), ppi=300, method="selenium")
    _chart.save((figure_dir / mgyp_fig_fn).with_suffix(".svg"))

    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(
    figure_dir,
    intervention_color_map,
    intervention_df,
    intervention_fig_fn,
    single_chart,
):
    _chart = single_chart(
        df=intervention_df,
        color_map=intervention_color_map,
    )

    _chart = _chart.encode(y=alt.Y(scale=alt.Scale(domain=[0, 1.1])))
    _chart.save(figure_dir / intervention_fig_fn)
    _chart.save((figure_dir / intervention_fig_fn).with_suffix(".png"), ppi=300)

    intervention_chart = _chart

    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(
    figure_dir,
    intervention_color_map,
    intervention_fig_fn,
    non_max_intervention_df,
    single_chart,
):
    _chart = single_chart(
        df=non_max_intervention_df,
        color_map=intervention_color_map,
        y_title="Boltz-2 binding affinity probability",
        xy_coords=(302, 185),
    )

    _chart = _chart.encode(
        y=alt.Y(
            scale=alt.Scale(domain=[0, 1.1]),
        ),
    )

    _path = figure_dir / intervention_fig_fn
    _path = _path.parent / f"{_path.stem}_non_maxed{_path.suffix}"
    _chart.save(_path)
    _chart.save(_path.with_suffix(".png"), ppi=300)

    mo.ui.altair_chart(_chart)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
