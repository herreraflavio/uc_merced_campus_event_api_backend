import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Dummy experimental data
# ------------------------------------------------------------
# One row = one tested deployment configuration.
# You can replace this with real experiment results later.

configs = [
    # category, pipeline, model, kb_size, token_budget, accuracy, cost_cents, latency_ms

    # Events
    {"category": "Events", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 100, "token_budget": 1000, "accuracy": 3.6, "cost_cents": 0.7, "latency_ms": 1100},
    {"category": "Events", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 250, "token_budget": 2000, "accuracy": 3.8, "cost_cents": 1.3, "latency_ms": 1700},
    {"category": "Events", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 500, "token_budget": 4000, "accuracy": 4.0, "cost_cents": 2.7, "latency_ms": 2900},

    {"category": "Events", "pipeline": "Enhanced", "model": "GPT-4o", "kb_size": 100, "token_budget": 1000, "accuracy": 4.1, "cost_cents": 0.6, "latency_ms": 950},
    {"category": "Events", "pipeline": "Enhanced", "model": "GPT-4o", "kb_size": 250, "token_budget": 2000, "accuracy": 4.3, "cost_cents": 1.0, "latency_ms": 1350},
    {"category": "Events", "pipeline": "Enhanced", "model": "GPT-5", "kb_size": 500, "token_budget": 4000, "accuracy": 4.5, "cost_cents": 3.5, "latency_ms": 3100},

    # Dining
    {"category": "Dining", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 100, "token_budget": 1000, "accuracy": 3.3, "cost_cents": 0.8, "latency_ms": 1200},
    {"category": "Dining", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 250, "token_budget": 2000, "accuracy": 3.5, "cost_cents": 1.4, "latency_ms": 1900},
    {"category": "Dining", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 500, "token_budget": 4000, "accuracy": 3.7, "cost_cents": 2.9, "latency_ms": 3300},

    {"category": "Dining", "pipeline": "Enhanced", "model": "GPT-4o", "kb_size": 100, "token_budget": 1000, "accuracy": 4.0, "cost_cents": 0.7, "latency_ms": 1050},
    {"category": "Dining", "pipeline": "Enhanced", "model": "GPT-4o", "kb_size": 250, "token_budget": 2000, "accuracy": 4.2, "cost_cents": 1.1, "latency_ms": 1450},
    {"category": "Dining", "pipeline": "Enhanced", "model": "GPT-5", "kb_size": 500, "token_budget": 4000, "accuracy": 4.4, "cost_cents": 3.6, "latency_ms": 3400},

    # Buildings / Services
    {"category": "Buildings/Services", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 100, "token_budget": 1000, "accuracy": 3.8, "cost_cents": 0.7, "latency_ms": 1000},
    {"category": "Buildings/Services", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 250, "token_budget": 2000, "accuracy": 4.0, "cost_cents": 1.2, "latency_ms": 1600},
    {"category": "Buildings/Services", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 500, "token_budget": 4000, "accuracy": 4.1, "cost_cents": 2.6, "latency_ms": 2800},

    {"category": "Buildings/Services", "pipeline": "Enhanced", "model": "GPT-4o", "kb_size": 100, "token_budget": 1000, "accuracy": 4.3, "cost_cents": 0.6, "latency_ms": 900},
    {"category": "Buildings/Services", "pipeline": "Enhanced", "model": "GPT-4o", "kb_size": 250, "token_budget": 2000, "accuracy": 4.5, "cost_cents": 1.0, "latency_ms": 1300},
    {"category": "Buildings/Services", "pipeline": "Enhanced", "model": "GPT-5", "kb_size": 500, "token_budget": 4000, "accuracy": 4.6, "cost_cents": 3.4, "latency_ms": 3000},

    # User Reports
    {"category": "User Reports", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 100, "token_budget": 1000, "accuracy": 3.1, "cost_cents": 0.8, "latency_ms": 1250},
    {"category": "User Reports", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 250, "token_budget": 2000, "accuracy": 3.3, "cost_cents": 1.5, "latency_ms": 2100},
    {"category": "User Reports", "pipeline": "Baseline", "model": "GPT-4o", "kb_size": 500, "token_budget": 4000, "accuracy": 3.5, "cost_cents": 3.0, "latency_ms": 3600},

    {"category": "User Reports", "pipeline": "Enhanced", "model": "GPT-4o", "kb_size": 100, "token_budget": 1000, "accuracy": 3.9, "cost_cents": 0.7, "latency_ms": 1100},
    {"category": "User Reports", "pipeline": "Enhanced", "model": "GPT-4o", "kb_size": 250, "token_budget": 2000, "accuracy": 4.1, "cost_cents": 1.2, "latency_ms": 1550},
    {"category": "User Reports", "pipeline": "Enhanced", "model": "GPT-5", "kb_size": 500, "token_budget": 4000, "accuracy": 4.3, "cost_cents": 3.8, "latency_ms": 3600},
]


# ------------------------------------------------------------
# Visual settings
# ------------------------------------------------------------

categories = ["Events", "Dining", "Buildings/Services", "User Reports"]

category_markers = {
    "Events": "o",
    "Dining": "s",
    "Buildings/Services": "^",
    "User Reports": "D",
}

pipeline_styles = {
    "Baseline": {
        "alpha": 0.45,
        "edgecolors": "black",
        "linewidths": 1,
    },
    "Enhanced": {
        "alpha": 0.85,
        "edgecolors": "black",
        "linewidths": 1.5,
    },
}


def get_bubble_size(kb_size):
    """
    Bubble size represents knowledge base size.
    Larger KB size = larger bubble.
    """
    return kb_size * 0.7


def label_points(ax, data):
    """
    Label each point with token budget.
    """
    for row in data:
        ax.annotate(
            f"{row['token_budget'] // 1000}k",
            (row["x"], row["y"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8
        )


# ------------------------------------------------------------
# Graph 1: Accuracy vs Cost
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))
ax = plt.gca()

for category in categories:
    for pipeline in ["Baseline", "Enhanced"]:
        subset = [
            row for row in configs
            if row["category"] == category and row["pipeline"] == pipeline
        ]

        x = [row["cost_cents"] for row in subset]
        y = [row["accuracy"] for row in subset]
        sizes = [get_bubble_size(row["kb_size"]) for row in subset]

        plt.scatter(
            x,
            y,
            s=sizes,
            marker=category_markers[category],
            label=f"{category} - {pipeline}",
            **pipeline_styles[pipeline]
        )

        label_data = [
            {"x": row["cost_cents"], "y": row["accuracy"], "token_budget": row["token_budget"]}
            for row in subset
        ]
        label_points(ax, label_data)

plt.title("Accuracy vs Cost by Pipeline, Category, Token Budget, and KB Size")
plt.xlabel("Cost per Query (cents)")
plt.ylabel("Average Answer Relevance Score")
plt.ylim(1, 5)
plt.grid(alpha=0.3)

plt.text(
    0.02,
    0.02,
    "Bubble size = knowledge base size | Labels = token budget",
    transform=ax.transAxes,
    fontsize=9
)

plt.legend(fontsize=7, loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig("graph1_accuracy_vs_cost.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Graph 2: Accuracy vs Latency
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))
ax = plt.gca()

for category in categories:
    for pipeline in ["Baseline", "Enhanced"]:
        subset = [
            row for row in configs
            if row["category"] == category and row["pipeline"] == pipeline
        ]

        x = [row["latency_ms"] for row in subset]
        y = [row["accuracy"] for row in subset]
        sizes = [get_bubble_size(row["kb_size"]) for row in subset]

        plt.scatter(
            x,
            y,
            s=sizes,
            marker=category_markers[category],
            label=f"{category} - {pipeline}",
            **pipeline_styles[pipeline]
        )

        label_data = [
            {"x": row["latency_ms"], "y": row["accuracy"], "token_budget": row["token_budget"]}
            for row in subset
        ]
        label_points(ax, label_data)

plt.title("Accuracy vs Latency by Pipeline, Category, Token Budget, and KB Size")
plt.xlabel("Latency per Query (ms)")
plt.ylabel("Average Answer Relevance Score")
plt.ylim(1, 5)
plt.grid(alpha=0.3)

plt.text(
    0.02,
    0.02,
    "Bubble size = knowledge base size | Labels = token budget",
    transform=ax.transAxes,
    fontsize=9
)

plt.legend(fontsize=7, loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig("graph2_accuracy_vs_latency.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Graph 3: Accuracy vs Knowledge Base Size
# ------------------------------------------------------------
# This graph shows scalability directly.
# Each subplot represents one category.
# Lines compare Baseline vs Enhanced.

kb_sizes = [100, 250, 500]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.flatten()

for i, category in enumerate(categories):
    ax = axes[i]

    for pipeline in ["Baseline", "Enhanced"]:
        subset = [
            row for row in configs
            if row["category"] == category and row["pipeline"] == pipeline
        ]

        subset = sorted(subset, key=lambda row: row["kb_size"])

        x = [row["kb_size"] for row in subset]
        y = [row["accuracy"] for row in subset]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=pipeline
        )

        for row in subset:
            ax.annotate(
                f"{row['token_budget'] // 1000}k",
                (row["kb_size"], row["accuracy"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

    ax.set_title(category)
    ax.set_xlabel("Knowledge Base Size")
    ax.set_xticks(kb_sizes)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Average Answer Relevance Score")
axes[2].set_ylabel("Average Answer Relevance Score")

for ax in axes:
    ax.set_ylim(1, 5)
    ax.legend()

fig.suptitle("Accuracy as Knowledge Base Size Increases", fontsize=15)

plt.tight_layout()
plt.savefig("graph3_accuracy_vs_kb_size.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Optional: Print best configurations
# ------------------------------------------------------------

print("\nBest configurations by accuracy-per-cent:")
for row in sorted(configs, key=lambda r: r["accuracy"] / r["cost_cents"], reverse=True)[:8]:
    print(
        f"{row['pipeline']} | {row['category']} | {row['model']} | "
        f"KB={row['kb_size']} | Tokens={row['token_budget']} | "
        f"Accuracy={row['accuracy']} | Cost={row['cost_cents']}¢ | "
        f"Latency={row['latency_ms']}ms | "
        f"Accuracy/cent={row['accuracy'] / row['cost_cents']:.2f}"
    )


print("\nBest configurations by accuracy-per-second:")
for row in sorted(configs, key=lambda r: r["accuracy"] / (r["latency_ms"] / 1000), reverse=True)[:8]:
    print(
        f"{row['pipeline']} | {row['category']} | {row['model']} | "
        f"KB={row['kb_size']} | Tokens={row['token_budget']} | "
        f"Accuracy={row['accuracy']} | Cost={row['cost_cents']}¢ | "
        f"Latency={row['latency_ms']}ms | "
        f"Accuracy/sec={row['accuracy'] / (row['latency_ms'] / 1000):.2f}"
    )