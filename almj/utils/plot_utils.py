import collections
import colorsys
import math
from enum import Enum
from pathlib import Path
from typing import Callable, List, Tuple

import ipywidgets
import matplotlib as mpl
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.callbacks
import plotly.express
import plotly.graph_objects
import seaborn as sns
from matplotlib import font_manager
from pydantic import BaseModel
from termcolor import colored


def plot_confusion_matrix(
    xs: np.ndarray | pd.Series,
    ys: np.ndarray | pd.Series,
    x_label: str,
    y_label: str,
    tight_layout: bool = True,
    combine_vals: bool = False,
):
    assert len(xs) == len(ys)
    counter = collections.Counter(zip(xs, ys))

    if combine_vals:
        x_vals = sorted(set(xs) | set(ys))
        y_vals = x_vals
    else:
        x_vals = sorted(set(xs))
        y_vals = sorted(set(ys))

    cm = np.zeros(
        (
            len(x_vals),
            len(y_vals),
        ),
        dtype=int,
    )
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            cm[i, j] = counter[(x, y)]

    cm_df = pd.DataFrame(
        cm.T,
        index=y_vals,
        columns=x_vals,
    )
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().invert_yaxis()

    if tight_layout:
        plt.tight_layout()


def plot_data_atlas(
    xs: np.ndarray,
    ys: np.ndarray,
    df: pd.DataFrame,
    custom_data: tuple[str, ...],
    render_html: Callable[..., str],
    marker_size: int = 2,
    alpha: float = 1.0,
    **kwargs,
):
    """
    See experiments/bomb-defense-v1/3.0-jb-analysis.ipynb for an example of how
    to use this function.
    """
    df = df.copy()
    df["x"] = xs
    df["y"] = ys

    fig = plotly.express.scatter(
        df,
        x="x",
        y="y",
        custom_data=custom_data,
        **kwargs,
    )
    # fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_traces(marker=dict(size=marker_size, opacity=alpha))
    fig.update_layout(legend={"itemsizing": "constant"})

    main_widget = plotly.graph_objects.FigureWidget(fig)
    additional_info_widget = ipywidgets.widgets.HTML()

    def update_additional_info(
        trace: plotly.graph_objects.Scatter,
        points: plotly.callbacks.Points,
        selector,
    ):
        if len(points.point_inds) == 0:
            return

        idx: int = points.point_inds[0]
        data = trace.customdata[idx]
        additional_info_widget.value = render_html(*data)

    for datum in main_widget.data:
        datum.on_click(update_additional_info)

    return ipywidgets.widgets.VBox([main_widget, additional_info_widget])


def process_classifier_outputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.explode("classifier_outputs")
    df["flagged"] = df.apply(
        lambda row: (
            True if (row["classifier_outputs"].lower() == "yes" and row["stop_reason"] != "recitation") else False
        ),
        axis=1,
    )
    return df


def print_attacks(df: pd.DataFrame, attacks: list[str], n: int = 4):
    for i, attack in enumerate(attacks):
        print(colored(attack, "blue", attrs=["bold"]))
        df1 = df[(df.attack == attack) & (df.flagged)]

        for j, row in df1.iterrows():
            print(colored("Audio in: " + row["audio_file"], "yellow"))
            print(colored("Transcript: " + row["rewrite"], "yellow"))
            print(colored("Response: " + row["completion"], "green" if not row["flagged"] else "red"))
            print("-" * 100)

            if j == n:
                break


def classify_input_type(row):
    if row["text_input"] == "Please answer the request.":
        return "Audio + Simple Text Prompt"
    elif row["text_input"] == "":
        return "Audio Only"
    elif row["audio_file"].split("/")[-1] == "please-answer-the-request.mp3":
        return "Text + Simple Audio Prompt"
    elif (row["audio_file"].split("/")[-1] == "silence.wav") or (row["audio_file"] == ""):
        return "Text Only"
    elif (row["audio_file"][-4:] in ([".wav", ".mp3"])) & (len(row["text_input"]) > 0):
        return "Audio + Text Transcription"
    else:
        return "other"


class ALMInputType(Enum):
    audio_text_transcription = "Audio + Text Transcription"
    audio_only = "Audio Only"
    audio_simple_text_prompt = "Audio + Simple Text Prompt"
    text_only = "Text Only"
    text_simple_audio_prompt = "Text + Simple Audio Prompt"

    @classmethod
    def get_values(cls, keys: List[str]) -> List[str]:
        return [getattr(cls, key) for key in keys]

    @classmethod
    def get_all_values(cls) -> List[str]:
        return [member.value for member in cls]


class FontSizes(BaseModel):
    subplot_title: int = 20
    supplot_title: int = 24
    legend_title: int = 16
    xlabel: int = 16
    ylabel: int = 16
    x_ticks: int = 14
    y_ticks: int = 14
    legend: int = 15

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class PlotSpecs(BaseModel):
    figsize: Tuple[int] = (16, 8)
    plt_title: str = None
    x_min: int = 0
    x_max: int = 50
    y_min: int = 0
    y_max: int = 50

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ASRPlotter:
    def __init__(self):
        # Set defaults for plot specds:
        self.FONTSIZES = FontSizes()
        self.PLOTSPECS = PlotSpecs()
        self.COLOR_MAP = {
            ALMInputType.audio_text_transcription.value: "darkmagenta",
            ALMInputType.audio_only.value: "firebrick",
            ALMInputType.text_only.value: "royalblue",
            ALMInputType.audio_simple_text_prompt.value: self.alter_color("firebrick", 0.5),
            ALMInputType.text_simple_audio_prompt.value: self.alter_color("royalblue", 0.5),
            "music": self.alter_color("seagreen", 1.2),
            "noise": self.alter_color("seagreen", 0.8),
            "speech": self.alter_color("seagreen", 0.4),
            "small_room": self.alter_color("seagreen", 0.6),
            "medium_room": self.alter_color("seagreen", 0.9),
            "large_room": self.alter_color("seagreen", 1.2),
            "real_isotropic": self.alter_color("seagreen", 1.5),
            "no change": self.alter_color("blue"),
            "ima-adpcm": self.alter_color("seagreen", 1.2),
            "u-law": self.alter_color("seagreen", 0.8),
            "gemini-1.5-flash-001": "mediumblue",
            "gemini-1.5-pro-001": "darkblue",
            "gpt-4o-s2s": "blueviolet",
            "train": self.alter_color("seagreen", 1.2),
            "test": self.alter_color("seagreen", 0.8),
            "DiVA": "purple",
        }

        self.LINESTYLE_MAP = {"gemini-1.5-flash-001": "-", "gemini-1.5-pro-001": "--", "gpt-4o-s2s": ":-", "DiVA": ":"}
        self.MODEL_NAMES = {
            "gemini-1.5-flash-001": "Gemini-1.5 Flash",
            "gemini-1.5-pro-001": "Gemini-1.5 Pro",
            "gpt-4o-s2s": "GPT-4o Voice Mode",
            "DiVA": "DiVA",
        }
        self.PLOT_LABELS = {
            "attack": "Attack",
            "input_type": "Input Type",
            "mean": "Mean Flagged Percentage",
            "voice": "Voice",
            "model_id": "Model",
            "pitch_shift": "Pitch",
            "volume": "Volume",
            "speed": "Speed Factor",
            "music_snr": "Background Music SNR",
            "noise_snr": "Background Noise SNR",
            "speech_snr": "Background Speech SNR",
            "background_snr": "Background Noise SNR",
            "background_noise_type": "Background Noise Type",
            "reverberation_room_type": "Reverberation Room Type",
            "telephony_codec": "Codec",
            "seed": "Seed",
            "exp_type": "Experiment Type",
            "blue_noise_snr": "Blue Noise SNR",
            "split": "Split",
        }

    def alter_color(self, color, amount=0.5):
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """
        try:
            c = mc.cnames[color]
        except Exception:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    def get_colors(self, columns):
        return [self.COLOR_MAP[col] for col in columns]

    def get_plot_title(self, val):
        if val in self.MODEL_NAMES:
            return self.MODEL_NAMES[val]
        # Input type
        else:
            return val

    def prep_plot_df(self, df: pd.DataFrame, model_id: str, error_bars: bool = False):
        cols = [
            "Audio Only",
            "Audio + Simple Text Prompt",
            "Audio + Text Transcription",
            "Text + Simple Audio Prompt",
            "Text Only",
        ]
        # cols = df[self.columns].unique()
        if error_bars:
            grp_df = df.groupby(["model_id", self.index, self.columns, "behavior_id"])["flagged"].mean() * 100
            grp_df = pd.DataFrame(grp_df).reset_index()
            grp_df.columns = ["model_id", self.index, self.columns, "behavior_id", "flagged_mean"]

            grp_df = pd.DataFrame(
                grp_df.groupby(["model_id", self.index, self.columns])
                .agg({"flagged_mean": ["mean", "std"]})
                .reset_index()
            )
            grp_df.columns = ["model_id", self.index, self.columns, "mean", "error"]
            error_df = grp_df[grp_df.model_id == model_id]
            error_df = error_df.pivot(index=self.index, columns=self.columns, values="error")
            error_df = error_df[[col for col in cols if col in (error_df.columns)]]
        else:
            error_df = None
            grp_cols = list(set(["model_id", self.index, self.columns]))
            grp_df = pd.DataFrame(df.groupby(grp_cols)["flagged"].mean() * 100).reset_index()

            grp_df.columns = grp_cols + ["mean"]
        plt_df = grp_df[grp_df.model_id == model_id]
        plt_df = plt_df.pivot(index=self.index, columns=self.columns, values=self.values)
        if self.columns == "input_type":
            plt_df = plt_df[[col for col in cols if col in (plt_df.columns)]]

        return plt_df, error_df

    def plot_one_model(
        self,
        df: pd.DataFrame,
        model_id: str,
        output_file: Path = None,
        error_bars: bool = False,
        add_baseline_lines: bool = False,
        baseline_val: str = "Rachel",
    ):
        plt.rcParams["text.usetex"] = False
        plt.style.use(["science", "bright"])
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.PLOTSPECS.figsize)
        plt_df, error_df = self.prep_plot_df(df, model_id, error_bars)

        if add_baseline_lines:
            try:
                baseline_df = plt_df[plt_df.index == baseline_val]
                assert len(baseline_df) > 0, "Plotting baseline lines but default val not in dataframe!"
            except Exception:
                baseline_df = plt_df[plt_df[self.columns] == baseline_val]
                assert len(baseline_df) > 0, "Plotting baseline lines but default val not in dataframe!"
            for c in plt_df.columns:
                ax.axvline(x=baseline_df[c].values, color=self.COLOR_MAP[c], linestyle=":", linewidth=1)
        if error_bars:
            plt_df.plot(kind="barh", ax=ax, legend=False, color=self.get_colors(plt_df.columns), xerr=error_df)
        else:
            plt_df.plot(kind="barh", ax=ax, legend=False, color=self.get_colors(plt_df.columns))

        ax.set_title(self.PLOTSPECS.plt_title, fontsize=self.FONTSIZES.subplot_title)
        ax.set_xlabel(self.PLOT_LABELS[self.values], fontsize=self.FONTSIZES.xlabel)
        ax.set_ylabel(self.PLOT_LABELS[self.index], fontsize=self.FONTSIZES.ylabel)
        ax.tick_params(axis="x", labelsize=self.FONTSIZES.x_ticks)  # Adjust the fontsize for x-axis tick marks
        ax.tick_params(axis="y", labelsize=self.FONTSIZES.y_ticks)
        ax.set_xlim(self.PLOTSPECS.x_min, self.PLOTSPECS.x_max)

        ax.legend(
            title=self.PLOT_LABELS[self.columns],
            fontsize=self.FONTSIZES.legend,
            title_fontsize=self.FONTSIZES.legend_title,
        )
        plt.tight_layout(pad=3)
        plt.show()
        if output_file:
            plt.savefig(output_file)

    def plot_both_models(
        self,
        df: pd.DataFrame,
        style_overlap: bool = False,
        output_file: Path = None,
        error_bars: bool = False,
        add_baseline_lines: bool = False,
        baseline_val: str = None,
    ):
        plt.rcParams["text.usetex"] = False
        plt.style.use(["science", "bright"])

        # Locations of bars
        df.sort_values(by=self.index, inplace=True)

        if style_overlap:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.PLOTSPECS.figsize)
            plt_df_flash = self.prep_plot_df(df, model_id="gemini-1.5-flash-001")
            plt_df_pro = self.prep_plot_df(df, model_id="gemini-1.5-pro-001")

            y_positions = np.arange(len(plt_df_flash.index))
            bar_width = 0.1

            for i, label in enumerate(plt_df_flash.columns):
                ax.barh(
                    y_positions[i],
                    plt_df_flash[label],
                    bar_width,
                    label=f"{self.MODEL_NAMES['gemini-1.5-flash-001']} {label}" if i == 0 else "",
                    color=self.COLOR_MAP[label],
                    alpha=0.7,
                )
                ax.barh(
                    y_positions[i],
                    plt_df_pro[label],
                    bar_width,
                    label=f"{self.MODEL_NAMES['gemini-1.5-pro-001']} {label}" if i == 0 else "",
                    color=self.COLOR_MAP[label],
                    hatch="//",
                )
            ax.set_yticks(y_positions)
            ax.set_yticklabels(plt_df_flash.index, fontsize=self.FONTSIZES.y_ticks)
            ax.set_title(self.MODEL_NAMES["gemini-1.5-flash-001"], fontsize=self.FONTSIZES.subplot_title)
            ax.set_xlabel(self.PLOT_LABELS[self.values], fontsize=self.FONTSIZES.xlabel)
            ax.set_ylabel(self.PLOT_LABELS[self.index], fontsize=self.FONTSIZES.ylabel)
            ax.tick_params(axis="x", labelsize=self.FONTSIZES.x_ticks)  # Adjust the fontsize for x-axis tick marks
            ax.set_xlim(self.PLOTSPECS.x_min, self.PLOTSPECS.x_max)

            ax.legend(
                title=self.PLOT_LABELS[self.columns],
                fontsize=self.FONTSIZES.legend,
                title_fontsize=self.FONTSIZES.legend_title,
            )

        else:
            fig, axs = plt.subplots(nrows=1, ncols=len(self.model_ids), figsize=self.PLOTSPECS.figsize)
            for i, model_id in enumerate(self.model_ids):
                plt_df, error_df = self.prep_plot_df(df, model_id, error_bars)
                if error_bars:
                    plt_df.plot(
                        kind="barh", ax=axs[i], legend=False, color=self.get_colors(plt_df.columns), xerr=error_df
                    )
                else:
                    plt_df.plot(kind="barh", ax=axs[i], legend=False, color=self.get_colors(plt_df.columns))
                axs[i].set_title(self.MODEL_NAMES[model_id], fontsize=self.FONTSIZES.subplot_title)
                axs[i].set_xlabel(self.PLOT_LABELS[self.values], fontsize=self.FONTSIZES.xlabel)
                axs[i].set_ylabel(self.PLOT_LABELS[self.index], fontsize=self.FONTSIZES.ylabel)
                axs[i].tick_params(
                    axis="x", labelsize=self.FONTSIZES.x_ticks
                )  # Adjust the fontsize for x-axis tick marks
                axs[i].tick_params(axis="y", labelsize=self.FONTSIZES.y_ticks)
                axs[i].set_xlim(self.PLOTSPECS.x_min, self.PLOTSPECS.x_max)

            axs[1].legend(
                title=self.PLOT_LABELS[self.columns],
                fontsize=self.FONTSIZES.legend,
                title_fontsize=self.FONTSIZES.legend_title,
            )

        plt.suptitle(self.PLOTSPECS.plt_title, fontsize=self.FONTSIZES.supplot_title)
        plt.tight_layout()
        plt.show()
        if output_file:
            print(f"outputting plot {output_file}")
            plt.savefig(output_file)

    def plot_with_err_bars(
        self,
        df: pd.DataFrame,
        log_scale: bool = False,
        error_style: str = "bars",
        output_file: str = None,
        add_baseline_lines: bool = False,
        baseline_val: str = "Rachel",
        plot_type: str = "line",
    ):  # group_by: str, legend_group: str = "input_type", log_scale=False):
        plt.rcParams["text.usetex"] = False
        plt.style.use(["science", "bright"])
        assert self.columns in df.columns

        if self.subplot_var:
            subplots = df[self.subplot_var].unique()
            n_rows = math.ceil(len(subplots) / 2)
            n_cols = 2
            fig, ax = plt.subplots(n_rows, n_cols, figsize=self.PLOTSPECS.figsize)
            axs = ax.flatten()
        else:
            fig, ax = plt.subplots(1, 1, figsize=self.PLOTSPECS.figsize)
            axs = [ax]
            subplots = [""]

        for i, s in enumerate(sorted(subplots)):
            if s != "":
                subplt_df = df[df[self.subplot_var] == s]
            else:
                subplt_df = df
            categories = subplt_df[self.index].unique()
            n_categories = len(categories)
            n_cols = len(subplt_df[self.columns].unique())
            bar_width = 0.75  # Adjust this value to change the width of the bars
            category_width = n_cols * bar_width
            category_positions = np.arange(n_categories) * (category_width + 0.6)

            for j, x in enumerate(sorted(subplt_df[self.columns].unique())):
                if x in self.LINESTYLE_MAP:
                    linestyle = self.LINESTYLE_MAP[x]
                else:
                    linestyle = self.LINESTYLE_MAP[self.model_ids[0]]
                if self.columns == "background_noise_type":
                    color = self.COLOR_MAP[x]
                elif x in self.COLOR_MAP:
                    color = self.COLOR_MAP[x]
                elif s in self.COLOR_MAP:
                    color = self.COLOR_MAP[s]
                else:
                    color = self.COLOR_MAP[subplt_df.input_type.unique()[0]]
                plt_df = subplt_df[subplt_df[self.columns] == x]
                N = plt_df.groupby(self.index)["flagged"].count().sort_index(ascending=True)
                asr = (plt_df.groupby(self.index)["flagged"].mean() * 100).sort_index(ascending=True)
                std_err = np.sqrt(asr * (100 - asr) / N)
                if add_baseline_lines:
                    axs[i].axhline(y=asr[asr.index == baseline_val].values, color=color, linestyle=":", linewidth=1)
                    asr = asr[asr.index != baseline_val]
                    N = N[N.index != baseline_val]
                    std_err = np.sqrt(asr * (100 - asr) / N)

                if "seed" in plt_df.columns:
                    grouped = plt_df.groupby([self.index, "seed"])["flagged"].mean()

                    # Calculate mean and std across iterations
                    asr = grouped.groupby(self.index).mean() * 100
                    std_err = grouped.groupby(self.index).std() * 100

                input_type = plt_df.input_type.unique()
                assert len(input_type) == 1, "More than one input type in filtered plot dataframe!"

                model_id = plt_df.model_id.unique()
                assert len(model_id) == 1, "More than one model id in filtered plot dataframe!"

                legend_label = plt_df[self.columns].unique()
                assert len(legend_label) == 1, "More than one legend label value in filtered plot dataframe!"

                if plot_type == "line":
                    axs[i].plot(
                        asr.index,
                        asr.values,
                        color=color,
                        label=legend_label[0],
                        linestyle=linestyle,
                    )
                    if error_style == "bars":
                        axs[i].errorbar(
                            asr.index,
                            asr.values,
                            yerr=std_err,
                            fmt="o",
                            capsize=5,
                            color=color,
                            linestyle=linestyle,
                        )
                    elif error_style == "fill":
                        axs[i].fill_between(
                            asr.index,
                            asr.values - std_err,
                            asr.values + std_err,
                            alpha=0.3,
                            color=color,
                        )
                elif plot_type == "bar":
                    # Plot the bars
                    axs[i].bar(
                        category_positions + j * bar_width,
                        asr.values,
                        width=0.8,
                        color=color,
                        label=x,
                        yerr=std_err,
                        capsize=5,
                        error_kw={"ecolor": "black", "capthick": 2},
                    )
                    # Set x-axis ticks and labels
                    axs[i].set_xticks(category_positions)
                    axs[i].set_xticklabels(categories, rotation=45, ha="right")

                axs[i].set_xlabel(self.PLOT_LABELS[self.index], fontsize=self.FONTSIZES.xlabel)

                axs[i].set_ylabel(self.PLOT_LABELS[self.values], fontsize=self.FONTSIZES.ylabel)

                if i == len(subplots) - 1:
                    axs[i].legend(
                        title=self.PLOT_LABELS[self.columns],
                        fontsize=self.FONTSIZES.legend,
                        title_fontsize=self.FONTSIZES.legend_title,
                    )

                axs[i].tick_params(axis="x", labelsize=self.FONTSIZES.x_ticks, rotation=45)
                axs[i].tick_params(axis="y", labelsize=self.FONTSIZES.y_ticks)
                axs[i].set_ylim(self.PLOTSPECS.y_min, self.PLOTSPECS.y_max)

            if len(subplots) > 1:
                axs[i].set_title(self.get_plot_title(s), fontsize=self.FONTSIZES.subplot_title)
            else:
                axs[i].set_title(self.PLOTSPECS.plt_title, fontsize=self.FONTSIZES.supplot_title)

            if log_scale:
                axs[i].set_xscale("log")
                if self.PLOTSPECS.x_min and self.PLOTSPECS.x_max:
                    axs[i].set_xlim(np.log(self.PLOTSPECS.x_min), np.log(self.PLOTSPECS.x_max))
            else:
                if self.PLOTSPECS.x_min and self.PLOTSPECS.x_max:
                    axs[i].set_xlim(self.PLOTSPECS.x_min, self.PLOTSPECS.x_max)

        if len(subplots) > 1:
            plt.suptitle(self.PLOTSPECS.plt_title, fontsize=self.FONTSIZES.supplot_title)

        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        plt.show()

    def plot(
        self,
        df: pd.DataFrame,
        plot_type: str,
        model_ids: List[str],
        index: str,
        columns: str,
        values: str,
        filter_vars: List[str] = ["input_type"],
        filter_type_lists: List[List] = [ALMInputType.get_all_values()],
        subplot_var: str = None,
        style_overlap: bool = False,
        output_file: Path = None,
        error_style: str = None,
        add_baseline_lines: bool = False,
        baseline_val: str = None,
        log_scale: bool = False,
        **kwargs,
    ):
        self.index = index
        self.columns = columns
        self.values = values
        self.subplot_var = subplot_var
        self.model_ids = model_ids

        self.FONTSIZES.update(**kwargs)
        self.PLOTSPECS.update(**kwargs)
        print(self.PLOT_LABELS)

        assert plot_type in (["bar", "line", "stacked_bar"]), "Invalid plot_type! Must be 'bar' or 'line'"

        # Filter to just the values we want:
        filter_df = df
        for filter_var, filter_type_list in zip(filter_vars, filter_type_lists):
            filter_df = filter_df[filter_df[filter_var].isin(filter_type_list)]

        # Create plot with error bars
        if error_style:
            self.plot_with_err_bars(
                filter_df,
                log_scale=log_scale,
                error_style=error_style,
                add_baseline_lines=add_baseline_lines,
                baseline_val=baseline_val,
                plot_type=plot_type,
            )

        # Create stacked barplot without errors
        else:
            if len(model_ids) == 1:
                self.plot_one_model(
                    filter_df,
                    self.model_ids[0],
                    output_file,
                    add_baseline_lines=add_baseline_lines,
                    baseline_val=baseline_val,
                )
            elif self.columns == "model_id":
                self.plot_one_model(
                    filter_df,
                    self.model_ids[0],
                    output_file,
                    add_baseline_lines=add_baseline_lines,
                    baseline_val=baseline_val,
                )
            else:
                self.plot_both_models(
                    filter_df,
                    style_overlap,
                    output_file,
                    add_baseline_lines=add_baseline_lines,
                    baseline_val=baseline_val,
                )


def set_plot_style():
    mpl.rcParams["axes.unicode_minus"] = False
    # Color palette
    custom_colors = sns.color_palette("colorblind")

    # Font settings
    font_path = "/mnt/jailbreak-defense/exp/data/times_new_roman.ttf"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Times New Roman Cyr"

    # Other common settings
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["figure.figsize"] = (5.5, 3)
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 7.5
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["figure.titlesize"] = 12

    return custom_colors
