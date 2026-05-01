# cost_latency_context_tradeoff.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SUMMARY_FILE = Path("ai_pipeline_summary.json")
OUTPUT_FILE = Path("cost_latency_context_tradeoff.png")


def load_summary(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_tokens(value):
    value = value or 0

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def fmt_money(value):
    value = value or 0
    return f"${value:.6f}"


def fmt_num(value, digits=3):
    value = value or 0
    return f"{value:.{digits}f}"


def build_legend_label(row):
    lines = [
        f"{row['index']}. {row['label']}",
        f"   Quality: {fmt_num(row['quality'], 4)}",
        f"   Latency: {fmt_num(row['latency'], 2)} s",
        f"   Cost/run: {fmt_money(row['cost'])}",
        f"   Input: {fmt_tokens(row['input_tokens'])}",
        f"   Output: {fmt_tokens(row['output_tokens'])}",
    ]

    if row["reasoning_tokens"] > 0:
        lines.append(f"   Reasoning: {fmt_tokens(row['reasoning_tokens'])}")

    lines.append(f"   Context: {fmt_tokens(row['context_tokens'])}")
    lines.append(f"   Total: {fmt_tokens(row['total_tokens'])}")

    return "\n".join(lines)


def draw_break_marks(ax_left, ax_right):
    d = 0.012

    kwargs_left = dict(transform=ax_left.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs_left)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_left)

    kwargs_right = dict(transform=ax_right.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_right.plot((-d, +d), (-d, +d), **kwargs_right)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs_right)


def main():
    summary = load_summary(SUMMARY_FILE)
    pipelines = summary.get("pipelines", {})

    if not pipelines:
        raise ValueError("No pipeline data found in ai_pipeline_summary.json")

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    rows = []

    for index, (key, row) in enumerate(pipelines.items(), start=1):
        rows.append({
            "index": index,
            "key": key,
            "label": row.get("label", key),
            "latency": row.get("average_latency_seconds") or 0,
            "quality": row.get("average_relevance_score") or 0,
            "cost": row.get("average_cost_usd", row.get("average_estimated_cost_usd", 0)) or 0,
            "context_tokens": row.get("average_context_tokens") or 0,
            "input_tokens": row.get("average_input_tokens") or 0,
            "output_tokens": row.get("average_output_tokens") or 0,
            "reasoning_tokens": row.get("average_reasoning_tokens") or 0,
            "total_tokens": row.get("average_total_tokens") or 0,
            "color": colors[(index - 1) % len(colors)],
        })

    # Requested axis setup:
    # left: 0 to 8
    # right: 20 to 22
    left_xmin, left_xmax = 0, 8
    right_xmin, right_xmax = 20, 22

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [2.6, 1.0]}
    )

    bubble_size = 1000  # constant size

    for row in rows:
        x = row["latency"]
        y = row["quality"]

        if x <= left_xmax:
            ax_left.scatter(
                x, y,
                s=bubble_size,
                alpha=0.45,
                color=row["color"],
                edgecolors="black",
                linewidths=1.0,
                zorder=3
            )
            ax_left.annotate(
                str(row["index"]),
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                zorder=4
            )
        elif x >= right_xmin:
            ax_right.scatter(
                x, y,
                s=bubble_size,
                alpha=0.45,
                color=row["color"],
                edgecolors="black",
                linewidths=1.0,
                zorder=3
            )
            ax_right.annotate(
                str(row["index"]),
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                zorder=4
            )
        # Anything between 8 and 20 is intentionally omitted from the plot
        # because that region is truncated.

    ax_left.set_xlim(left_xmin, left_xmax)
    ax_right.set_xlim(right_xmin, right_xmax)

    for ax in (ax_left, ax_right):
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(labelright=False)
    ax_right.tick_params(right=False)

    draw_break_marks(ax_left, ax_right)

    fig.suptitle("Cost–Latency–Context Trade-off", fontsize=16, y=0.96)
    ax_left.set_ylabel("Answer Quality / Relevance Score")
    fig.supxlabel("Average Response Latency Seconds")

    legend_handles = []
    legend_labels = []

    for row in rows:
        handle = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=row["color"],
            markeredgecolor="black",
            markersize=10,
            linewidth=0,
        )
        legend_handles.append(handle)
        legend_labels.append(build_legend_label(row))

    ax_right.legend(
        legend_handles,
        legend_labels,
        title="Pipeline / Model Metrics\nBubble Size = Constant",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
        title_fontsize=11,
        labelspacing=1.4,
        handletextpad=0.8,
    )

    plt.subplots_adjust(right=0.64, wspace=0.08)
    plt.savefig(OUTPUT_FILE, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved graph to: {OUTPUT_FILE.resolve()}")
    print("Used broken x-axis with 8–20 seconds omitted and max x-axis capped at 22.")


if __name__ == "__main__":
    main()