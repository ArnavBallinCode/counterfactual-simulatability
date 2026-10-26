"""
Phase 2 Visualization Module
Generates Before vs. After comparison figures benchmarking the proposed CDBP extension
against the Phase 1 baseline (Chen et al. replication) across precision and simulatable yield.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .config import OUTPUTS_DIR, FIGURES_DIR, MODEL_SMALL, MODEL_LARGE

sns.set_theme(style="whitegrid", font_scale=1.1)

def plot_phase2_comparisons(comparison_metrics=None):
    """
    Generate grouped comparison bar charts for:
    1. Simulation Precision: Phase 1 Baseline vs Phase 2 CDBP
    2. Simulatable Yield (% non-bot): Phase 1 Baseline vs Phase 2 CDBP
    """
    if comparison_metrics is None:
        metrics_file = OUTPUTS_DIR / "metrics_phase2.json"
        if not metrics_file.exists():
            print(f"[WARNING] {metrics_file} not found. Cannot plot Phase 2 figures.")
            return
        with open(metrics_file, "r", encoding="utf-8") as f:
            comparison_metrics = json.load(f)

    conditions = comparison_metrics.get("conditions", {})
    if not conditions:
        print("[WARNING] No condition data found in Phase 2 metrics.")
        return

    # Labels and data containers
    labels = [
        "Small 20B\nCoT",
        "Small 20B\nPost-Hoc",
        "Large 120B\nCoT",
        "Large 120B\nPost-Hoc"
    ]

    key_mapping = [
        f"{MODEL_SMALL}_contrastive_cot",
        f"{MODEL_SMALL}_contrastive_posthoc",
        f"{MODEL_LARGE}_contrastive_cot",
        f"{MODEL_LARGE}_contrastive_posthoc"
    ]

    p1_precisions = []
    p2_precisions = []
    p1_yields = []
    p2_yields = []

    for k in key_mapping:
        cond = conditions.get(k, {})
        prec = cond.get("precision", {})
        yld = cond.get("simulatable_yield", {})

        p1_p = prec.get("phase1_macro", 0.0)
        p2_p = prec.get("phase2_macro", 0.0)
        p1_y = yld.get("phase1_yield_pct", 0.0)
        p2_y = yld.get("phase2_yield_pct", 0.0)

        p1_precisions.append(p1_p if isinstance(p1_p, (int, float)) else 0.0)
        p2_precisions.append(p2_p if isinstance(p2_p, (int, float)) else 0.0)
        p1_yields.append(p1_y if isinstance(p1_y, (int, float)) else 0.0)
        p2_yields.append(p2_y if isinstance(p2_y, (int, float)) else 0.0)

    x = range(len(labels))
    width = 0.35

    # ------------------------------------------------------------------
    # Figure 1: Simulation Precision (Before vs After)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar([i - width/2 for i in x], p1_precisions, width, label="Phase 1 Baseline (Chen et al.)", color="#4a7bb0")
    b2 = ax.bar([i + width/2 for i in x], p2_precisions, width, label="Phase 2 Extension (CDBP Ours)", color="#2ca02c")

    ax.set_ylabel("Macro Simulation Precision (%)", fontweight="bold", fontsize=12)
    ax.set_title("Phase 2 Extension: Simulation Precision (Baseline vs. Contrastive Decision Boundaries)", pad=15, fontweight="bold", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc="upper right")

    # Add human ceiling dashed line
    ax.axhline(y=91.5, color="crimson", linestyle=":", linewidth=1.5, label="Human Explanation Ceiling (91.5%)")
    ax.text(3.3, 92.5, "Human Ceiling (91.5%)", color="crimson", fontweight="bold", fontsize=9, ha="right")

    for bar in b1 + b2:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Divider between Small and Large
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(0.5, 104, "GPT-OSS 20B (Small)", ha="center", fontweight="bold", color="#555555")
    ax.text(2.5, 104, "GPT-OSS 120B (Large)", ha="center", fontweight="bold", color="#555555")

    fig.tight_layout()
    f1 = FIGURES_DIR / "phase2_before_vs_after_precision.png"
    fig.savefig(f1, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Simulatable Yield (% non-bot rescued)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    b1_y = ax.bar([i - width/2 for i in x], p1_yields, width, label="Phase 1 Baseline Yield (Severe Filter Collapse)", color="#e7298a")
    b2_y = ax.bar([i + width/2 for i in x], p2_yields, width, label="Phase 2 CDBP Yield (Rescued Counterfactuals)", color="#1b9e77")

    ax.set_ylabel("Simulatable Yield % (|C*| / |C|)", fontweight="bold", fontsize=12)
    ax.set_title("Phase 2 Extension: Filter Attrition Rescue (Percentage of Non-Bot Counterfactuals)", pad=15, fontweight="bold", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, loc="upper left")

    for bar in b1_y + b2_y:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(0.5, 108, "GPT-OSS 20B", ha="center", fontweight="bold", color="#555555")
    ax.text(2.5, 108, "GPT-OSS 120B", ha="center", fontweight="bold", color="#555555")

    fig.tight_layout()
    f2 = FIGURES_DIR / "phase2_before_vs_after_yield.png"
    fig.savefig(f2, dpi=300)
    plt.close(fig)

    print(f"[PHASE 2 FIGURES GENERATED] Saved to {FIGURES_DIR}:")
    print(f" - {f1.name}")
    print(f" - {f2.name}")

