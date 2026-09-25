"""
Publication Visualization Module
Generates side-by-side comparison charts benchmarking replication results against Chen et al. (EMNLP 2023).
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .config import OUTPUTS_DIR, FIGURES_DIR, MODEL_SMALL, MODEL_LARGE

sns.set_theme(style="whitegrid", font_scale=1.1)

def _clean_model_label(name):
    if "20b" in name.lower():
        return "Small (GPT-OSS 20B)"
    elif "120b" in name.lower():
        return "Large (GPT-OSS 120B)"
    return name.split("/")[-1]

def generate_all_plots(metrics):
    t4 = metrics.get("table4_simulation_precision", {})
    t5 = metrics.get("table5_simulation_generality", {})

    # -------------------------------------------------------------
    # Plot 1: Table 4 Simulation Precision with Paper Benchmarks
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Categories: Original Paper (GPT-3.5 vs GPT-4) and Replication (20B vs 120B)
    categories = ["Original Paper\n(GPT-3.5)", "Replication\n(GPT-OSS 20B)", 
                  "Original Paper\n(GPT-4)", "Replication\n(GPT-OSS 120B)"]

    # Original paper values from Table 4
    cot_paper_gpt3 = 77.3
    posthoc_paper_gpt3 = 76.8
    cot_paper_gpt4 = 81.1
    posthoc_paper_gpt4 = 83.9

    # Replicated values from metrics.json
    cot_rep_20b = t4.get(f"{MODEL_SMALL}_cot", {}).get("macro_precision", 0.0)
    posthoc_rep_20b = t4.get(f"{MODEL_SMALL}_posthoc", {}).get("macro_precision", 0.0)
    cot_rep_120b = t4.get(f"{MODEL_LARGE}_cot", {}).get("macro_precision", 0.0)
    posthoc_rep_120b = t4.get(f"{MODEL_LARGE}_posthoc", {}).get("macro_precision", 0.0)

    cot_vals = [cot_paper_gpt3, cot_rep_20b, cot_paper_gpt4, cot_rep_120b]
    posthoc_vals = [posthoc_paper_gpt3, posthoc_rep_20b, posthoc_paper_gpt4, posthoc_rep_120b]

    x = range(len(categories))
    width = 0.35

    b1 = ax.bar([i - width/2 for i in x], cot_vals, width, label="Chain-of-Thought (CoT)", color="#386cb0")
    b2 = ax.bar([i + width/2 for i in x], posthoc_vals, width, label="Post-Hoc", color="#7fc97f")

    ax.set_ylabel("Simulation Precision (%)", fontweight="bold", fontsize=12)
    ax.set_title("Table 4 Comparison: Original Paper (OpenAI) vs. Replication (GPT-OSS)", pad=15, fontweight="bold", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc="upper right")

    for bar in b1 + b2:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add dividing line between small and large models
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.6)
    ax.text(0.5, 103, "Smaller Scale Models", ha="center", fontweight="bold", color="#555555")
    ax.text(2.5, 103, "Larger Scale Models", ha="center", fontweight="bold", color="#555555")

    fig.tight_layout()
    p1 = FIGURES_DIR / "table4_simulation_precision.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 2: Table 5 Simulation Generality Across Metrics
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ["BLEU Diversity", "Cosine Diversity", "Jaccard Diversity"]
    keys = list(t5.keys())

    x = range(len(metrics_names))
    width = 0.18
    palette = ["#386cb0", "#7fc97f", "#beaed4", "#fdc086"]

    for idx, k in enumerate(keys):
        g = t5[k]
        vals = [g.get("bleu_generality", 0), g.get("cosine_generality", 0), g.get("jaccard_generality", 0)]
        label = k.replace(MODEL_SMALL, "20B").replace(MODEL_LARGE, "120B").replace("_", " ").upper()
        ax.bar([i + (idx - len(keys)/2 + 0.5)*width for i in x], vals, width, label=label, color=palette[idx % len(palette)])

    ax.set_ylabel("Diversity Score (1 - similarity)", fontweight="bold", fontsize=12)
    ax.set_title("Table 5 Replication: Simulation Generality Across Metrics", pad=15, fontweight="bold", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_names, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    fig.tight_layout()
    p2 = FIGURES_DIR / "table5_simulation_generality.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)

    print(f"[FIGURES GENERATED] Saved updated figures with paper comparison to {FIGURES_DIR}")
