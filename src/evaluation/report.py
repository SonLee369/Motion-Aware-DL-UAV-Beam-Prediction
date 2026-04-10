"""
PDF Report Generator for Beam Prediction Experiments
Uses fpdf2 to generate 2-3 page PDF reports with tables, plots, and analysis.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ExperimentReport(FPDF):
    """Custom FPDF class with header/footer for experiment reports"""

    def __init__(self, title: str = "Experiment Report", experiment_id: str = "E0"):
        super().__init__()
        self.report_title = title
        self.experiment_id = experiment_id
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"{self.experiment_id} | {self.report_title}", align="L")
        self.cell(0, 6, datetime.now().strftime("%Y-%m-%d %H:%M"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ------------------------------------------------------------------ #
    #  Reusable building blocks                                           #
    # ------------------------------------------------------------------ #
    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 60, 120)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def key_value(self, key: str, value: str, bold_value: bool = False):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        self.cell(55, 6, key, new_x="END")
        style = "B" if bold_value else ""
        self.set_font("Helvetica", style, 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def add_table(self, headers: List[str], rows: List[List[str]],
                  col_widths: Optional[List[float]] = None,
                  highlight_last_row: bool = False,
                  highlight_row: Optional[int] = None):
        """Render a table. highlight_row (0-based) marks the best result in gold."""
        n = len(headers)
        if col_widths is None:
            usable_w = self.w - 2 * self.l_margin
            col_widths = [usable_w / n] * n

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        fill = False
        for r_idx, row in enumerate(rows):
            is_best = (r_idx == highlight_row)
            is_last = (r_idx == len(rows) - 1) and highlight_last_row
            if is_best:
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(255, 243, 180)   # gold
                self.set_text_color(30, 30, 30)
            elif is_last:
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(230, 240, 255)
                self.set_text_color(30, 30, 30)
            else:
                self.set_font("Helvetica", "", 9)
                self.set_fill_color(245, 245, 245) if fill else self.set_fill_color(255, 255, 255)
                self.set_text_color(40, 40, 40)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6.5, val, border=1, align="C",
                          fill=(is_best or is_last or fill))
            self.ln()
            if not is_best:   # don't flip stripe on highlighted rows
                fill = not fill

    def verdict_box(self, passed: bool, text: str):
        """Draw a colored verdict box."""
        if passed:
            self.set_fill_color(220, 245, 220)
            self.set_draw_color(60, 160, 60)
            icon = "PASS"
        else:
            self.set_fill_color(255, 230, 230)
            self.set_draw_color(200, 60, 60)
            icon = "FAIL"
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 30, 30)
        y0 = self.get_y()
        self.rect(10, y0, 190, 12, style="DF")
        self.set_xy(12, y0 + 2)
        self.cell(20, 8, icon)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 8, text)
        self.set_y(y0 + 15)

    def add_image_safe(self, path: str, w: float = 180):
        """Add image if file exists, otherwise insert placeholder text."""
        if Path(path).exists():
            self.image(path, w=w)
            self.ln(3)
        else:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(160, 160, 160)
            self.cell(0, 6, f"[Image not found: {path}]", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)


# ====================================================================== #
#  E1 Baseline Report                                                     #
# ====================================================================== #

def generate_e1_report(
    results: dict,
    trainer_history: dict,
    save_dir: str = "results/E1_baseline",
    speed_results: dict = None,
    dataset_stats: dict = None,
    config: dict = None,
    plot_dir: str = None,
):
    """
    Generate a 2-3 page PDF report for E1 Baseline Reproduction.

    Args:
        results: evaluate_model() output
        trainer_history: trainer.get_training_history() output
        save_dir: Where to write the PDF
        speed_results: Per-speed-category breakdown
        dataset_stats: dict with total/train/val/test counts
        config: Training config dict
        plot_dir: Directory containing PNG plots (defaults to save_dir)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if plot_dir is None:
        plot_dir = str(save_path)

    pdf = ExperimentReport(
        title="Baseline Reproduction",
        experiment_id="E1"
    )
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Page 1: Config + Main Results ────────────────────────────────── #
    pdf.section_title("1. Experiment Overview")
    pdf.body_text(
        "Reproduce the baseline beam prediction model from the reference paper "
        "using position-only GPS features (5D). Architecture: CNN encoder + "
        "GRU encoder-decoder + FC classifier, trained on DeepSense 6G Scenario 23."
    )

    # Config table
    if config:
        pdf.sub_title("Training Configuration")
        cfg_items = [
            ("Seed", str(config.get("random_seed", "N/A"))),
            ("Batch Size", str(config.get("batch_size", "N/A"))),
            ("Epochs", str(config.get("num_epochs", "N/A"))),
            ("Learning Rate", str(config.get("learning_rate", "5e-4"))),
            ("LR Schedule", str(config.get("lr_schedule", "[12, 18]"))),
            ("Device", str(config.get("device", "N/A"))),
        ]
        for k, v in cfg_items:
            pdf.key_value(k, v)
        pdf.ln(3)

    # Dataset stats
    if dataset_stats:
        pdf.sub_title("Dataset")
        total = dataset_stats.get("total", 0)
        for split in ("train", "val", "test"):
            n = dataset_stats.get(split, 0)
            pct = f"  ({n/total*100:.1f}%)" if total else ""
            pdf.key_value(split.capitalize(), f"{n:,}{pct}")
        pdf.ln(3)

    # ── Main accuracy table ──────────────────────────────────────────── #
    pdf.section_title("2. Results - Accuracy by Prediction Step")
    headers = ["Step", "Top-1 (%)", "Top-3 (%)", "Top-5 (%)"]
    col_w = [30, 40, 40, 40]

    has_power = "mean_power_loss_db" in results.get("overall", {})
    if has_power:
        headers.append("PL (dB)")
        col_w = [25, 35, 35, 35, 35]

    rows = []
    for v in range(4):
        m = results[f"v{v}"]
        row = [
            f"v={v}",
            f"{m['top1_acc']*100:.2f}",
            f"{m['top3_acc']*100:.2f}",
            f"{m['top5_acc']*100:.2f}",
        ]
        if has_power:
            row.append(f"{m.get('mean_power_loss_db', 0):.3f}")
        rows.append(row)

    overall = results["overall"]
    ov_row = [
        "Overall",
        f"{overall['top1_acc']*100:.2f}",
        f"{overall['top3_acc']*100:.2f}",
        f"{overall['top5_acc']*100:.2f}",
    ]
    if has_power:
        ov_row.append(f"{overall.get('mean_power_loss_db', 0):.3f}")
    rows.append(ov_row)

    pdf.add_table(headers, rows, col_widths=col_w, highlight_last_row=True)
    pdf.ln(4)

    # ── Speed category ───────────────────────────────────────────────── #
    if speed_results:
        pdf.section_title("3. Accuracy by Speed Category")
        speed_map = {
            "slow": "Slow (<=10 mph)",
            "medium": "Medium (10-20 mph)",
            "fast": "Fast (>20 mph)"
        }
        s_headers = ["Category", "Top-1 (%)", "Top-3 (%)", "Samples"]
        s_col_w = [45, 35, 35, 35]
        s_rows = []
        s_top1_vals = []
        for key, label in speed_map.items():
            if key in speed_results:
                sp = speed_results[key]
                sp_ov = sp.get("overall", sp)
                n_samples = sp.get("num_samples", "?")
                top1 = sp_ov['top1_acc'] * 100
                s_top1_vals.append(top1)
                s_rows.append([
                    label,
                    f"{top1:.2f}",
                    f"{sp_ov['top3_acc']*100:.2f}",
                    str(n_samples),
                ])
        if s_rows:
            best_spd_cat = int(np.argmax(s_top1_vals)) if s_top1_vals else None
            pdf.add_table(s_headers, s_rows, col_widths=s_col_w, highlight_row=best_spd_cat)
        pdf.ln(4)

    # ── Comparison with paper ────────────────────────────────────────── #
    PAPER_TOP1 = 70.5
    PAPER_PL = 0.59
    ACC_TOL = 2.0
    PL_TOL = 0.15

    our_top1 = overall["top1_acc"] * 100
    diff_acc = our_top1 - PAPER_TOP1
    acc_pass = abs(diff_acc) <= ACC_TOL

    section_num = 4 if speed_results else 3
    pdf.section_title(f"{section_num}. Comparison with Reference Paper")

    comp_headers = ["Metric", "Paper", "Ours", "Diff", "Status"]
    comp_w = [35, 30, 30, 30, 30]
    comp_rows = [
        ["Top-1 Acc", f"{PAPER_TOP1:.1f}%", f"{our_top1:.2f}%",
         f"{diff_acc:+.2f}%", "PASS" if acc_pass else "FAIL"],
    ]

    all_pass = acc_pass
    if has_power:
        our_pl = overall["mean_power_loss_db"]
        diff_pl = our_pl - PAPER_PL
        pl_pass = abs(diff_pl) <= PL_TOL
        all_pass = all_pass and pl_pass
        comp_rows.append([
            "Power Loss", f"{PAPER_PL:.2f} dB", f"{our_pl:.3f} dB",
            f"{diff_pl:+.3f} dB", "PASS" if pl_pass else "FAIL",
        ])

    pdf.add_table(comp_headers, comp_rows, col_widths=comp_w)
    pdf.ln(4)

    pdf.verdict_box(
        all_pass,
        "Baseline successfully reproduced - all metrics within tolerance."
        if all_pass else
        "Some metrics outside tolerance - see details above."
    )
    pdf.ln(4)

    # ── Page 2: Plots ────────────────────────────────────────────────── #
    section_num += 1
    pdf.add_page()
    pdf.section_title(f"{section_num}. Training Curves")
    pdf.add_image_safe(os.path.join(plot_dir, "E1_training_curves.png"), w=175)

    section_num += 1
    pdf.section_title(f"{section_num}. Accuracy by Prediction Step")
    pdf.add_image_safe(os.path.join(plot_dir, "E1_accuracy_by_step.png"), w=175)

    # ── Training history summary ─────────────────────────────────────── #
    if trainer_history:
        section_num += 1
        pdf.section_title(f"{section_num}. Training Summary")
        best_epoch = trainer_history.get("best_epoch", "N/A")
        best_val = trainer_history.get("best_val_acc", 0)
        final_train_loss = (
            trainer_history["train_losses"][-1]
            if trainer_history.get("train_losses") else "N/A"
        )
        final_val_loss = (
            trainer_history["val_losses"][-1]
            if trainer_history.get("val_losses") else "N/A"
        )
        pdf.key_value("Best Epoch", str(best_epoch))
        pdf.key_value("Best Val Accuracy", f"{best_val:.4f}")
        pdf.key_value(
            "Final Train Loss",
            f"{final_train_loss:.4f}" if isinstance(final_train_loss, float) else str(final_train_loss)
        )
        pdf.key_value(
            "Final Val Loss",
            f"{final_val_loss:.4f}" if isinstance(final_val_loss, float) else str(final_val_loss)
        )

    # ── Save ─────────────────────────────────────────────────────────── #
    pdf_path = save_path / "E1_report.pdf"
    pdf.output(str(pdf_path))
    logger.info(f"PDF report saved to {pdf_path}")
    return str(pdf_path)


# ====================================================================== #
#  E2 Feature Ablation Report                                             #
# ====================================================================== #

def generate_e2_report(
    all_results: Dict,
    comparison_df,  # pandas DataFrame
    save_dir: str = "results/E2_feature_ablation",
    plot_dir: str = None,
):
    """
    Generate PDF report for E2 Feature Ablation Study.

    Args:
        all_results: Dict mapping feature_set_name -> results dict
        comparison_df: pandas DataFrame from compare_results()
        save_dir: Where to write the PDF
        plot_dir: Directory containing PNG plots
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if plot_dir is None:
        plot_dir = str(save_path)

    pdf = ExperimentReport(
        title="Feature Ablation Study",
        experiment_id="E2"
    )
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Overview ─────────────────────────────────────────────────────── #
    pdf.section_title("1. Experiment Overview")
    pdf.body_text(
        "Incremental ablation study comparing GPS-derived motion features "
        "(E2.0-E2.9), direct sensor-measured velocity (E2.10-E2.11), and "
        "angular rate / directed tangential vector features (E2.12-E2.14) for beam "
        "prediction. All experiments use the same BaselineBeamPredictor architecture "
        "(260K params) - only the input feature set varies. "
        "Motion features are computed on complete sequences before train/val/test "
        "splitting to avoid Savitzky-Golay boundary artifacts."
    )

    # Dynamic feature set overview from actual results
    pdf.sub_title("Feature Sets Evaluated")
    fs_headers = ["Set", "Dim", "Description"]
    fs_w = [38, 14, 138]
    fs_rows = []
    for _, row in comparison_df.iterrows():
        fs_rows.append([
            str(row["Feature Set"]),
            str(int(row["Num Features"])),
            # Extract description from all_results if available
            str(all_results.get(
                next((k for k, v in all_results.items()
                      if v.get("feature_set") == row["Feature Set"]), ""),
                {}
            ).get("feature_set", str(row["Feature Set"]))),
        ])
    pdf.add_table(fs_headers, fs_rows, col_widths=fs_w)
    pdf.ln(4)

    def format_mean_std(mean_val: float, std_val: float) -> str:
        return f"{mean_val:.2f} +/- {std_val:.2f}"

    # ── Main comparison table ────────────────────────────────────────── #
    pdf.section_title("2. Results - Overall Comparison")
    best_overall_idx = int(comparison_df["Overall_top1"].idxmax())
    headers = ["Feature Set", "Dim", "Top-1 (%)", "Top-3 (%)", "Improv."]
    col_w = [48, 14, 40, 40, 25]
    rows = []
    for _, row in comparison_df.iterrows():
        rows.append([
            str(row["Feature Set"]),
            str(int(row["Num Features"])),
            format_mean_std(row['Overall_top1'], row.get('Overall_top1_std', 0.0)),
            format_mean_std(row['Overall_top3'], row.get('Overall_top3_std', 0.0)),
            f"{row['Improvement (%)']:+.2f}%",
        ])
    pdf.add_table(headers, rows, col_widths=col_w, highlight_row=best_overall_idx)
    pdf.ln(4)

    # ── Per-step breakdown ───────────────────────────────────────────── #
    pdf.section_title("3. Top-1 Accuracy by Prediction Step")
    baseline_row = comparison_df.iloc[0]
    baseline_step = {v: baseline_row[f'v{v}_top1'] for v in range(4)}

    step_headers = ["Feature Set", "v=0", "v=1", "v=2", "v=3", "Improv."]
    step_w = [44, 28, 28, 28, 28, 22]
    step_rows = []
    avg_step_improvs = []
    for _, row in comparison_df.iterrows():
        avg_imp = np.mean([row[f'v{v}_top1'] - baseline_step[v] for v in range(4)])
        avg_step_improvs.append(avg_imp)
        step_rows.append([
            str(row["Feature Set"]),
            format_mean_std(row['v0_top1'], row.get('v0_top1_std', 0.0)),
            format_mean_std(row['v1_top1'], row.get('v1_top1_std', 0.0)),
            format_mean_std(row['v2_top1'], row.get('v2_top1_std', 0.0)),
            format_mean_std(row['v3_top1'], row.get('v3_top1_std', 0.0)),
            f"{avg_imp:+.2f}%",
        ])
    best_step_idx = int(np.argmax(avg_step_improvs))
    pdf.add_table(step_headers, step_rows, col_widths=step_w, highlight_row=best_step_idx)
    pdf.ln(4)

    # ── Speed sensitivity analysis ──────────────────────────────────── #
    speed_order = ["slow", "medium", "fast"]
    speed_display = ["Slow", "Medium", "Fast"]
    feature_labels = []
    speed_top1 = {k: [] for k in speed_order}
    speed_top1_std = {k: [] for k in speed_order}
    speed_samples = {k: [] for k in speed_order}
    has_speed_data = False

    for feature_set_key, results in all_results.items():
        label = results.get("feature_set", feature_set_key)
        feature_labels.append(label)
        speed_metrics = results.get("speed_metrics") or {}
        for speed_key in speed_order:
            if speed_key in speed_metrics:
                has_speed_data = True
                overall = speed_metrics[speed_key].get("overall", speed_metrics[speed_key])
                speed_top1[speed_key].append(overall.get("top1_acc", 0.0) * 100)
                speed_top1_std[speed_key].append(overall.get("top1_acc_std", 0.0) * 100)
                speed_samples[speed_key].append(speed_metrics[speed_key].get("num_samples", 0))
            else:
                speed_top1[speed_key].append(0.0)
                speed_top1_std[speed_key].append(0.0)
                speed_samples[speed_key].append(0)

    section_num = 4
    if has_speed_data:
        pdf.section_title("4. Speed Sensitivity Analysis")

        # Grouped bar chart
        speed_plot_path = os.path.join(plot_dir, "E2_speed_by_category.png")
        x = np.arange(len(speed_order))
        n_features = max(len(feature_labels), 1)
        width = 0.8 / n_features

        fig, ax = plt.subplots(figsize=(8.5, 5))
        for idx, label in enumerate(feature_labels):
            offsets = x + (idx - (n_features - 1) / 2) * width
            values = [speed_top1[s][idx] for s in speed_order]
            ax.bar(offsets, values, width, label=label, alpha=0.85)

        ax.set_xlabel("Speed Category")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title("Speed Sensitivity by Feature Set", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(speed_display)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(speed_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pdf.sub_title("Top-1 Accuracy by Speed Category")
        pdf.add_image_safe(speed_plot_path, w=170)

        # Per-speed improvement table
        pdf.sub_title("Top-1 Accuracy by Speed Category vs Baseline")
        spd_headers = ["Feature Set", "Slow (%)", "Medium (%)", "Fast (%)", "Fast Improv."]
        spd_w = [44, 36, 36, 36, 26]
        baseline_speed_top1 = {s: speed_top1[s][0] if speed_top1[s] else 0.0 for s in speed_order}
        spd_rows = []
        fast_improvs = []
        for idx, label in enumerate(feature_labels):
            fast_imp = speed_top1['fast'][idx] - baseline_speed_top1['fast']
            fast_improvs.append(fast_imp)
            spd_rows.append([
                label,
                format_mean_std(speed_top1['slow'][idx],   speed_top1_std['slow'][idx]),
                format_mean_std(speed_top1['medium'][idx], speed_top1_std['medium'][idx]),
                format_mean_std(speed_top1['fast'][idx],   speed_top1_std['fast'][idx]),
                f"{fast_imp:+.2f}%",
            ])
        best_fast_idx = int(np.argmax(speed_top1['fast']))
        pdf.add_table(spd_headers, spd_rows, col_widths=spd_w, highlight_row=best_fast_idx)
        pdf.ln(4)
        section_num = 5

    # ── Key findings ─────────────────────────────────────────────────── #
    pdf.section_title(f"{section_num}. Key Findings")

    if len(comparison_df) >= 1:
        baseline_acc = comparison_df.iloc[0]["Overall_top1"]
        baseline_name = comparison_df.iloc[0]["Feature Set"]
        best_idx  = comparison_df["Overall_top1"].idxmax()
        best_name = comparison_df.loc[best_idx, "Feature Set"]
        best_acc  = comparison_df.loc[best_idx, "Overall_top1"]
        best_improv = best_acc - baseline_acc

        # Find best GPS-derived velocity set (E2.1-E2.5) and best GPS full (E2.9)
        gps_vel_sets = comparison_df[comparison_df["Feature Set"].str.contains("GPS vel", na=False)]
        best_gps_vel = gps_vel_sets.loc[gps_vel_sets["Overall_top1"].idxmax()] if len(gps_vel_sets) > 0 else None

        sensor_sets = comparison_df[comparison_df["Feature Set"].str.contains("sensor", na=False)]
        best_sensor = sensor_sets.loc[sensor_sets["Overall_top1"].idxmax()] if len(sensor_sets) > 0 else None

        findings = [
            f"1. Baseline (E2.0, position only): {baseline_acc:.2f}% Top-1.",
            f"2. Best overall: {best_name} at {best_acc:.2f}% ({best_improv:+.2f}% over baseline).",
        ]

        if best_gps_vel is not None:
            findings.append(
                f"3. Best GPS-derived velocity set: {best_gps_vel['Feature Set']} at "
                f"{best_gps_vel['Overall_top1']:.2f}% ({best_gps_vel['Overall_top1'] - baseline_acc:+.2f}%)."
            )
        if best_sensor is not None:
            findings.append(
                f"4. Best sensor velocity set: {best_sensor['Feature Set']} at "
                f"{best_sensor['Overall_top1']:.2f}% ({best_sensor['Overall_top1'] - baseline_acc:+.2f}%). "
                f"Sensor features use direct IMU measurements vs GPS-derived estimation."
            )

        # Acceleration finding
        acc_sets = comparison_df[comparison_df["Feature Set"].str.contains("acc", na=False)]
        if len(acc_sets) > 0:
            best_acc_set = acc_sets.loc[acc_sets["Overall_top1"].idxmax()]
            findings.append(
                f"5. GPS-derived acceleration - best set {best_acc_set['Feature Set']} at "
                f"{best_acc_set['Overall_top1']:.2f}% ({best_acc_set['Overall_top1'] - baseline_acc:+.2f}%). "
                f"GPS acceleration noise (double differentiation) may limit contribution."
            )

        # Angular rate / directed tangential finding
        angular_sets = comparison_df[
            comparison_df["Feature Set"].str.contains("angular|dir v_tan|omega", case=False, na=False)
        ]
        if len(angular_sets) > 0:
            best_ang = angular_sets.loc[angular_sets["Overall_top1"].idxmax()]
            findings.append(
                f"{len(findings)+1}. Angular/tangential features - best set {best_ang['Feature Set']} at "
                f"{best_ang['Overall_top1']:.2f}% ({best_ang['Overall_top1'] - baseline_acc:+.2f}%). "
                f"These capture UE-BS direction rotation rate and directed tangential velocity."
            )

        n = len(findings)
        for i, f in enumerate(findings):
            pdf.body_text(f)

    # ── Page 2: Plots ────────────────────────────────────────────────── #
    pdf.add_page()
    pdf.section_title(f"{section_num + 1}. Visualizations")

    pdf.sub_title("Overall Accuracy Comparison")
    pdf.add_image_safe(os.path.join(plot_dir, "E2_overall_comparison.png"), w=170)

    pdf.sub_title("Accuracy by Prediction Step")
    pdf.add_image_safe(os.path.join(plot_dir, "E2_by_step.png"), w=170)

    pdf.sub_title("Accuracy Heatmap")
    pdf.add_image_safe(os.path.join(plot_dir, "E2_heatmap.png"), w=170)

    # ── Save ─────────────────────────────────────────────────────────── #
    pdf_path = save_path / "E2_report.pdf"
    try:
        pdf.output(str(pdf_path))
        logger.info(f"PDF report saved to {pdf_path}")
        return str(pdf_path)
    except PermissionError:
        fallback_name = f"E2_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        fallback_path = save_path / fallback_name
        pdf.output(str(fallback_path))
        logger.warning(
            f"Permission denied writing {pdf_path}. Saved to {fallback_path} instead."
        )
        return str(fallback_path)


# ====================================================================== #
#  E3 Speed-Aware Training Report                                         #
# ====================================================================== #

def generate_e3_report(
    all_results: Dict,
    comparison_df,                          # pandas DataFrame from compare_strategies()
    save_dir: str = "results/E3_speed_aware",
    plot_dir: str = None,
    train_df=None,                          # optional, for speed distribution vis
):
    """
    Generate a comprehensive PDF report for E3 Speed-Aware Training.

    Sections:
        1. Experiment Overview
        2. Overall Comparison
        3. Per-Speed Breakdown
        4. Statistical Testing (paired t-test)
        5. Training Dynamics (learning curves)
        6. Speed Distribution in Training
        7. Ablation of Combined Strategy (E3.5)
        8. Key Findings
        9. Visualizations (external plots)

    Args:
        all_results:    dict  key -> aggregated results (must contain 'overall', 'by_speed', etc.)
        comparison_df:  pandas DataFrame from compare_strategies()
        save_dir:       output directory for PDF and auxiliary PNGs
        plot_dir:       directory containing existing PNG plots (defaults to save_dir)
        train_df:       training DataFrame with 'speed_category' column (for distribution plot)
    """
    import pandas as pd
    from scipy import stats as sp_stats

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if plot_dir is None:
        plot_dir = str(save_path)

    pdf = ExperimentReport(title="Speed-Aware Training", experiment_id="E3")
    pdf.alias_nb_pages()
    pdf.add_page()

    def _fmt(mean_val, std_val=0.0):
        return f"{mean_val:.2f} +/- {std_val:.2f}" if std_val else f"{mean_val:.2f}"

    # ────────────────────────────────────────────────────────────────── #
    # 1. Overview                                                        #
    # ────────────────────────────────────────────────────────────────── #
    sec = 1
    pdf.section_title(f"{sec}. Experiment Overview")
    pdf.body_text(
        "E3 evaluates six speed-aware training strategies to improve beam prediction "
        "performance, especially for high-speed UAVs.  All experiments share the same "
        "position + velocity feature set (8-dim) and differ only in training strategy."
    )

    pdf.sub_title("Strategy Summary")
    strat_headers = ["ID", "Strategy", "Description"]
    strat_w = [18, 45, 127]
    strat_rows = [
        ["E3.0", "Baseline",            "Standard training, no speed-aware techniques"],
        ["E3.1", "Stratified Sampling",  "Balanced sampling across speed categories"],
        ["E3.2", "Speed-Conditioned",    "Speed embedding concatenated to features"],
        ["E3.3", "Weighted Loss",        "Higher loss weight for fast-speed samples"],
        ["E3.4", "Curriculum Learning",  "Train slow -> medium -> all speeds"],
        ["E3.5", "Combined",             "All strategies together"],
    ]
    pdf.add_table(strat_headers, strat_rows, col_widths=strat_w)
    pdf.ln(4)

    # ────────────────────────────────────────────────────────────────── #
    # 2. Overall Comparison                                              #
    # ────────────────────────────────────────────────────────────────── #
    sec += 1
    pdf.section_title(f"{sec}. Overall Accuracy Comparison")

    ov_headers = ["Strategy", "Top-1 (%)", "Top-3 (%)", "Improv."]
    ov_w = [55, 40, 40, 28]
    ov_rows = []
    for _, row in comparison_df.iterrows():
        ov_rows.append([
            str(row['Strategy']),
            _fmt(row['Overall_Top1'], row.get('Overall_Top1_std', 0.0)),
            _fmt(row['Overall_Top3'], row.get('Overall_Top3_std', 0.0)),
            f"{row['Overall_Improvement']:+.2f}",
        ])
    best_ov_idx = int(comparison_df['Overall_Top1'].idxmax())
    pdf.add_table(ov_headers, ov_rows, col_widths=ov_w, highlight_row=best_ov_idx)
    pdf.ln(4)

    # ────────────────────────────────────────────────────────────────── #
    # 3. Per-Speed Breakdown                                             #
    # ────────────────────────────────────────────────────────────────── #
    speed_keys = ['slow', 'medium', 'fast']
    speed_display = {'slow': 'Slow (<10 mph)', 'medium': 'Medium (10-20 mph)', 'fast': 'Fast (>20 mph)'}

    has_speed = any(
        col in comparison_df.columns
        for col in ['Slow_Top1', 'Medium_Top1', 'Fast_Top1']
    )

    if has_speed:
        sec += 1
        pdf.section_title(f"{sec}. Performance by Speed Category")

        for spd in speed_keys:
            col = f'{spd.capitalize()}_Top1'
            col_std = f'{spd.capitalize()}_Top1_std'
            imp_col = f'{spd.capitalize()}_Improvement'
            if col not in comparison_df.columns:
                continue

            pdf.sub_title(speed_display[spd])
            sp_headers = ["Strategy", "Top-1 (%)", "Improvement"]
            sp_w = [65, 50, 40]
            sp_rows = []
            for _, row in comparison_df.iterrows():
                sp_rows.append([
                    str(row['Strategy']),
                    _fmt(row[col], row.get(col_std, 0.0)),
                    f"{row.get(imp_col, 0.0):+.2f}" if imp_col in row.index else "--",
                ])
            best_spd_idx = int(comparison_df[col].idxmax())
            pdf.add_table(sp_headers, sp_rows, col_widths=sp_w, highlight_row=best_spd_idx)
            pdf.ln(3)

        # Fast-speed improvement highlight
        if 'Fast_Improvement' in comparison_df.columns:
            best_fast_idx = comparison_df['Fast_Top1'].idxmax()
            best_fast = comparison_df.loc[best_fast_idx]
            pdf.body_text(
                f"Largest fast-speed gain: {best_fast['Strategy']} at "
                f"{best_fast['Fast_Top1']:.2f}% Top-1 "
                f"({best_fast['Fast_Improvement']:+.2f}% vs baseline)."
            )
            pdf.ln(2)

    # ────────────────────────────────────────────────────────────────── #
    # 4. Statistical Testing (paired t-test vs baseline)                 #
    # ────────────────────────────────────────────────────────────────── #
    sec += 1
    pdf.add_page()
    pdf.section_title(f"{sec}. Statistical Significance (vs Baseline)")
    pdf.body_text(
        "Paired two-sided t-test on overall Top-1 accuracy across seeds. "
        "Null hypothesis: strategy performance = baseline performance."
    )

    baseline_key = list(all_results.keys())[0]
    baseline_res = all_results[baseline_key]

    # Reconstruct per-seed overall Top-1 from training_histories
    def _get_seed_top1(res_dict):
        """Return list of per-seed overall top-1 accuracy."""
        histories = res_dict.get('training_histories', [])
        if histories:
            return [h.get('best_val_acc', 0.0) for h in histories]
        # fallback: single value repeated
        return [res_dict['overall']['top1_acc']]

    baseline_accs = _get_seed_top1(baseline_res)

    ttest_headers = ["Strategy", "Mean Top-1", "t-stat", "p-value", "Sig."]
    ttest_w = [55, 30, 25, 25, 25]
    ttest_rows = []

    for key, res in all_results.items():
        strat_accs = _get_seed_top1(res)
        strategy_name = res.get('strategy', key)
        mean_acc = np.mean(strat_accs) * 100

        if key == baseline_key or len(strat_accs) < 2:
            ttest_rows.append([strategy_name, f"{mean_acc:.2f}", "--", "--", "ref"])
            continue

        # Ensure equal length for paired test
        n = min(len(baseline_accs), len(strat_accs))
        t_stat, p_val = sp_stats.ttest_rel(strat_accs[:n], baseline_accs[:n])
        sig = "Yes" if p_val < 0.05 else "No"
        ttest_rows.append([
            strategy_name,
            f"{mean_acc:.2f}",
            f"{t_stat:.3f}",
            f"{p_val:.4f}",
            sig,
        ])

    # Highlight row with lowest p-value (most significant), skip baseline row (ref)
    best_ttest_idx = None
    min_p = 1.0
    for i, r in enumerate(ttest_rows):
        try:
            p = float(r[3])
            if p < min_p:
                min_p = p; best_ttest_idx = i
        except (ValueError, IndexError):
            pass
    pdf.add_table(ttest_headers, ttest_rows, col_widths=ttest_w, highlight_row=best_ttest_idx)
    pdf.ln(4)

    # ────────────────────────────────────────────────────────────────── #
    # 5. Training Dynamics (learning curves)                             #
    # ────────────────────────────────────────────────────────────────── #
    sec += 1
    pdf.section_title(f"{sec}. Training Dynamics")

    # Generate learning-curve plot
    lc_path = os.path.join(plot_dir, "E3_learning_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, res in all_results.items():
        label = res.get('strategy', key)
        hist = res.get('training_history', {})
        tl = hist.get('train_losses', [])
        vl = hist.get('val_losses', [])
        va = hist.get('val_accuracies', [])
        epochs = list(range(1, len(tl) + 1))

        if tl:
            axes[0].plot(epochs, tl, label=label, linewidth=1.5)
        if va:
            axes[1].plot(list(range(1, len(va) + 1)),
                         [a * 100 for a in va], label=label, linewidth=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Curves", fontweight="bold")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy (%)")
    axes[1].set_title("Validation Accuracy Curves", fontweight="bold")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(lc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.add_image_safe(lc_path, w=180)
    pdf.ln(3)

    # ────────────────────────────────────────────────────────────────── #
    # 6. Speed Distribution in Training                                  #
    # ────────────────────────────────────────────────────────────────── #
    sec += 1
    pdf.section_title(f"{sec}. Speed Distribution in Training Data")

    if train_df is not None and 'speed_category' in train_df.columns:
        dist_path = os.path.join(plot_dir, "E3_speed_distribution.png")
        cat_counts = train_df['speed_category'].value_counts().sort_index()
        cat_labels = ['Slow (<10 mph)', 'Medium (10-20 mph)', 'Fast (>20 mph)']
        colors = ['#5DA5DA', '#FAA43A', '#F17CB0']

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Bar chart
        bars = axes[0].bar(range(len(cat_counts)), cat_counts.values, color=colors, alpha=0.85)
        axes[0].set_xticks(range(len(cat_counts)))
        axes[0].set_xticklabels(cat_labels[:len(cat_counts)])
        axes[0].set_ylabel("Number of Samples")
        axes[0].set_title("Speed Category Distribution", fontweight="bold")
        for bar, val in zip(bars, cat_counts.values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                         f"{val:,}", ha='center', fontsize=9)

        # Pie chart
        axes[1].pie(cat_counts.values, labels=cat_labels[:len(cat_counts)],
                    colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1].set_title("Proportions", fontweight="bold")

        fig.tight_layout()
        fig.savefig(dist_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pdf.add_image_safe(dist_path, w=175)
        pdf.body_text(
            "The natural distribution is imbalanced towards slow speeds. Speed-aware "
            "strategies (E3.1 stratified sampling, E3.3 weighted loss) aim to mitigate "
            "this imbalance."
        )
    else:
        pdf.body_text(
            "[Speed distribution plot skipped - train_df not provided or missing "
            "'speed_category' column.]"
        )
    pdf.ln(3)

    # ────────────────────────────────────────────────────────────────── #
    # 7. Ablation of Combined Strategy (E3.5)                            #
    # ────────────────────────────────────────────────────────────────── #
    sec += 1
    pdf.add_page()
    pdf.section_title(f"{sec}. Ablation of Combined Strategy (E3.5)")

    # Identify keys for components and combined
    component_map = {
        'E3_1': 'Stratified Sampling',
        'E3_2': 'Speed-Conditioned',
        'E3_3': 'Weighted Loss',
    }
    combined_key = 'E3_5'
    baseline_top1 = all_results.get(baseline_key, {}).get('overall', {}).get('top1_acc', 0) * 100
    combined_top1 = all_results.get(combined_key, {}).get('overall', {}).get('top1_acc', 0) * 100

    abl_headers = ["Component", "Individual Top-1 (%)", "Contribution vs Baseline"]
    abl_w = [55, 50, 55]
    abl_rows = []

    for comp_key, comp_name in component_map.items():
        if comp_key in all_results:
            comp_acc = all_results[comp_key]['overall']['top1_acc'] * 100
            comp_std = all_results[comp_key]['overall'].get('top1_acc_std', 0) * 100
            abl_rows.append([
                comp_name,
                _fmt(comp_acc, comp_std),
                f"{comp_acc - baseline_top1:+.2f}%",
            ])

    # Add combined row
    combined_std = all_results.get(combined_key, {}).get('overall', {}).get('top1_acc_std', 0) * 100
    abl_rows.append([
        "Combined (E3.5)",
        _fmt(combined_top1, combined_std),
        f"{combined_top1 - baseline_top1:+.2f}%",
    ])

    # Highlight the row with the highest individual/combined Top-1
    best_abl_idx = len(abl_rows) - 1  # Combined row as default
    try:
        accs = []
        for r in abl_rows:
            try:
                accs.append(float(r[1].split()[0]))
            except Exception:
                accs.append(0.0)
        best_abl_idx = int(np.argmax(accs))
    except Exception:
        pass
    pdf.add_table(abl_headers, abl_rows, col_widths=abl_w, highlight_row=best_abl_idx)
    pdf.ln(3)

    # Generate ablation waterfall chart
    waterfall_path = os.path.join(plot_dir, "E3_ablation_waterfall.png")
    comp_names = []
    comp_improvements = []
    for comp_key, comp_name in component_map.items():
        if comp_key in all_results:
            comp_names.append(comp_name)
            comp_improvements.append(
                all_results[comp_key]['overall']['top1_acc'] * 100 - baseline_top1
            )
    comp_names.append("Combined")
    comp_improvements.append(combined_top1 - baseline_top1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = ['#5DA5DA'] * (len(comp_names) - 1) + ['#60BD68']
    bars = ax.bar(comp_names, comp_improvements, color=colors_bar, alpha=0.85, edgecolor='grey')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Improvement over Baseline (%)")
    ax.set_title("Component Contribution to E3.5 Combined Strategy", fontweight="bold")
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, comp_improvements):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:+.2f}%", ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.add_image_safe(waterfall_path, w=170)
    pdf.ln(3)

    # Additivity check
    sum_individual = sum(
        all_results[k]['overall']['top1_acc'] * 100 - baseline_top1
        for k in component_map if k in all_results
    )
    additivity = combined_top1 - baseline_top1
    pdf.body_text(
        f"Sum of individual improvements: {sum_individual:+.2f}%. "
        f"Combined improvement: {additivity:+.2f}%. "
        f"{'Synergistic' if additivity > sum_individual else 'Sub-additive'} effect "
        f"(ratio: {additivity / sum_individual:.2f}x)."
        if abs(sum_individual) > 1e-6 else
        "Individual improvements are near zero; combined effect is marginal."
    )
    pdf.ln(3)

    # ────────────────────────────────────────────────────────────────── #
    # 8. Key Findings                                                    #
    # ────────────────────────────────────────────────────────────────── #
    sec += 1
    pdf.section_title(f"{sec}. Key Findings")

    # Auto-generate
    best_key = max(all_results, key=lambda k: all_results[k]['overall']['top1_acc'])
    best_res = all_results[best_key]
    best_strat = best_res.get('strategy', best_key)
    best_acc = best_res['overall']['top1_acc'] * 100

    findings = [
        f"1. Best overall strategy: {best_strat} at {best_acc:.2f}% Top-1 "
        f"({best_acc - baseline_top1:+.2f}% over baseline).",
    ]

    # Fast-speed improvement
    if has_speed and 'Fast_Top1' in comparison_df.columns:
        best_fast_idx = comparison_df['Fast_Top1'].idxmax()
        bf = comparison_df.loc[best_fast_idx]
        findings.append(
            f"2. Best fast-speed strategy: {bf['Strategy']} at "
            f"{bf['Fast_Top1']:.2f}% Top-1 ({bf.get('Fast_Improvement', 0):+.2f}% "
            f"over baseline fast category)."
        )

    # Combined vs best individual
    if combined_key in all_results and len(component_map) > 0:
        best_individual_key = max(component_map, key=lambda k:
                                  all_results.get(k, {}).get('overall', {}).get('top1_acc', 0))
        best_ind_acc = all_results[best_individual_key]['overall']['top1_acc'] * 100
        findings.append(
            f"3. Combined strategy (E3.5) achieves {combined_top1:.2f}% vs best "
            f"individual component at {best_ind_acc:.2f}% "
            f"({combined_top1 - best_ind_acc:+.2f}% difference)."
        )

    findings.append(
        f"4. Baseline (E3.0): {baseline_top1:.2f}% Top-1. "
        f"Speed-aware techniques provide up to {best_acc - baseline_top1:+.2f}% improvement."
    )

    for f in findings:
        pdf.body_text(f)

    # ────────────────────────────────────────────────────────────────── #
    # 9. Visualizations (from plot_comparison)                           #
    # ────────────────────────────────────────────────────────────────── #
    sec += 1
    pdf.add_page()
    pdf.section_title(f"{sec}. Visualizations")

    pdf.sub_title("Overall Accuracy Comparison")
    pdf.add_image_safe(os.path.join(plot_dir, "E3_overall_comparison.png"), w=170)

    pdf.sub_title("Accuracy by Speed Category")
    pdf.add_image_safe(os.path.join(plot_dir, "E3_by_speed.png"), w=170)

    pdf.sub_title("Improvement Heatmap")
    pdf.add_image_safe(os.path.join(plot_dir, "E3_improvement_heatmap.png"), w=170)

    # ── Save ─────────────────────────────────────────────────────────── #
    pdf_path = save_path / "E3_report.pdf"
    try:
        pdf.output(str(pdf_path))
        logger.info(f"PDF report saved to {pdf_path}")
        return str(pdf_path)
    except PermissionError:
        fallback_name = f"E3_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        fallback_path = save_path / fallback_name
        pdf.output(str(fallback_path))
        logger.warning(
            f"Permission denied writing {pdf_path}. Saved to {fallback_path} instead."
        )
        return str(fallback_path)

def generate_generic_report(
    experiment_id: str,
    title: str,
    description: str,
    results_table_headers: List[str],
    results_table_rows: List[List[str]],
    col_widths: Optional[List[float]] = None,
    findings: Optional[List[str]] = None,
    save_dir: str = "results",
    plot_files: Optional[List[str]] = None,
    extra_tables: Optional[List[dict]] = None,
):
    """
    Generate a generic experiment PDF report.

    Args:
        experiment_id: e.g. "E3"
        title: Report title
        description: Experiment description paragraph
        results_table_headers: Main table headers
        results_table_rows: Main table rows
        col_widths: Column widths for main table
        findings: Key findings as list of strings
        save_dir: Output directory
        plot_files: List of PNG file paths to embed
        extra_tables: List of dicts with keys: title, headers, rows, col_widths
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    pdf = ExperimentReport(title=title, experiment_id=experiment_id)
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.section_title("1. Experiment Overview")
    pdf.body_text(description)

    pdf.section_title("2. Results")
    pdf.add_table(results_table_headers, results_table_rows,
                  col_widths=col_widths, highlight_last_row=True)
    pdf.ln(4)

    if extra_tables:
        for i, tbl in enumerate(extra_tables, start=3):
            pdf.section_title(f"{i}. {tbl.get('title', 'Additional Results')}")
            pdf.add_table(tbl["headers"], tbl["rows"],
                          col_widths=tbl.get("col_widths"), highlight_last_row=False)
            pdf.ln(4)

    sec = (3 + len(extra_tables)) if extra_tables else 3
    if findings:
        pdf.section_title(f"{sec}. Key Findings")
        for f in findings:
            pdf.body_text(f)
        sec += 1

    if plot_files:
        pdf.add_page()
        pdf.section_title(f"{sec}. Visualizations")
        for pf in plot_files:
            pdf.add_image_safe(pf, w=170)
            pdf.ln(2)

    pdf_path = save_path / f"{experiment_id}_report.pdf"
    pdf.output(str(pdf_path))
    logger.info(f"PDF report saved to {pdf_path}")
    return str(pdf_path)


# ====================================================================== #
#  E4 Architecture Ablation Report                                        #
# ====================================================================== #

def generate_e4_report(
    grand_results: Dict,
    feat_comparison_dfs: Dict,   # feat_name -> pd.DataFrame from compare_architectures()
    architectures: Dict,          # OrderedDict: arch_name -> config dict
    save_dir: str = "results/E4_architecture",
    plot_dir: str = None,
) -> str:
    """
    Generate a comprehensive PDF report for E4 Architecture Ablation.

    Sections:
        1. Experiment Overview (arch table + design rationale)
        2. Grand Summary Table (arch x feature-set pivot)
        3. Per-Feature-Set Detailed Results (one table per feat set)
        4. Per-Prediction-Step Analysis
        5. Training Dynamics (learning curves)
        6. Key Findings
        7. Visualizations (grand heatmap + grand bar + per-feat bars)
    """
    import pandas as pd

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if plot_dir is None:
        plot_dir = str(save_path)

    feat_names = list(feat_comparison_dfs.keys())
    arch_names = list(architectures.keys())

    def _safe(s: str) -> str:
        """Replace non-Latin-1 characters so fpdf Helvetica font doesn't crash."""
        return s.encode('latin-1', errors='replace').decode('latin-1')

    def _fmt(mean_val, std_val=0.0):
        if mean_val is None or (isinstance(mean_val, float) and np.isnan(mean_val)):
            return "--"
        return f"{mean_val:.2f} +/- {std_val:.2f}"

    pdf = ExperimentReport(title="Architecture Ablation Study", experiment_id="E4")
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── 1. Experiment Overview ───────────────────────────────────────── #
    pdf.section_title("1. Experiment Overview")
    pdf.body_text(
        "E4 evaluates 8 architectural variants of the CNN-GRU beam predictor "
        "across 4 motion feature sets, using 5 random seeds each. "
        "Each variant changes exactly ONE component from the E4.0 baseline, "
        "enabling clean ablation of: multi-scale CNN, cross-attention, hidden "
        "dimension, GRU depth, bidirectional GRU, and Transformer encoder. "
        "Attention variants (E4.2, E4.3) are skipped for 6-D feature sets "
        "where motion_dim=1 degenerates cross-attention to a scalar gate."
    )

    # Arch summary table
    pdf.sub_title("Architecture Variants (E4.0 - E4.7)")
    arch_headers = ["ID", "Name", "Change vs Baseline", "Params"]
    arch_w = [20, 42, 100, 28]
    arch_rows = []
    for arch_name, cfg in architectures.items():
        arch_id = arch_name.split('_')[0]   # e.g. "E4.0"
        # Estimate params from grand_results
        params_str = "--"
        for feat in feat_names:
            key = f"{feat}_{arch_name}"
            if key in grand_results:
                p = grand_results[key].get('num_parameters', 0)
                params_str = f"{p/1e3:.1f}K" if p < 1e6 else f"{p/1e6:.2f}M"
                break
        arch_rows.append([arch_id, _safe(cfg['name']), _safe(cfg['description']), params_str])
    pdf.add_table(arch_headers, arch_rows, col_widths=arch_w)
    pdf.ln(4)

    # Feature sets table
    pdf.sub_title("Feature Sets Evaluated")
    fs_headers = ["Tag", "Dims", "Features"]
    fs_w = [22, 14, 154]
    fs_rows = [
        ["8D",       "8",  "pos (5D) + v_radial + v_tangential + v_mag"],
        ["6D_vmag",  "6",  "pos (5D) + v_mag"],
        ["6D_vtan",  "6",  "pos (5D) + v_tangential"],
        ["6D_sensor","6",  "pos (5D) + sensor speed scalar"],
    ]
    pdf.add_table(fs_headers, fs_rows, col_widths=fs_w)
    pdf.ln(4)

    # ── 2. Grand Summary ─────────────────────────────────────────────── #
    pdf.add_page()
    pdf.section_title("2. Grand Summary - Top-1 Accuracy (%) by Architecture x Feature Set")
    pdf.body_text("Values shown as mean +/- std across 5 seeds. '--' = skipped (attention on 6D). "
                  "Best cell per column highlighted in gold.")

    grand_headers = ["Architecture"] + feat_names
    # col widths: 50 + 35*4 = 190
    grand_w = [50] + [35] * len(feat_names)

    # Find best per column for highlighting
    best_per_feat = {}
    for feat in feat_names:
        best_val, best_row_idx = -1.0, -1
        for i, arch_name in enumerate(arch_names):
            key = f"{feat}_{arch_name}"
            if key in grand_results:
                v = grand_results[key]['overall']['top1_acc'] * 100
                if v > best_val:
                    best_val, best_row_idx = v, i
        best_per_feat[feat] = best_row_idx

    grand_rows = []
    highlight_cells = {}  # row_idx -> set of col indices to highlight
    for i, arch_name in enumerate(arch_names):
        row = [architectures[arch_name]['name']]
        for j, feat in enumerate(feat_names):
            key = f"{feat}_{arch_name}"
            if key in grand_results:
                r = grand_results[key]
                mean = r['overall']['top1_acc'] * 100
                std  = r['overall'].get('top1_acc_std', 0.0) * 100
                row.append(f"{mean:.2f} +/- {std:.2f}")
                if best_per_feat[feat] == i:
                    highlight_cells.setdefault(i, set()).add(j + 1)
            else:
                row.append("--")
        grand_rows.append(row)

    # Find the row with most gold cells (overall best) for row highlight
    overall_best_row = max(
        range(len(arch_names)),
        key=lambda i: sum(
            grand_results.get(f"{feat}_{arch_names[i]}", {}).get('overall', {}).get('top1_acc', 0)
            for feat in feat_names
        )
    )
    pdf.add_table(grand_headers, grand_rows, col_widths=grand_w, highlight_row=overall_best_row)
    pdf.ln(4)

    # ── 3. Per-Feature-Set Detailed Results ──────────────────────────── #
    pdf.add_page()
    pdf.section_title("3. Per-Feature-Set Detailed Results")

    tbl_headers = ["Architecture", "Params", "Top-1 (%)", "Top-3 (%)", "Delta Top-1"]
    tbl_w = [50, 20, 42, 42, 22]

    for feat_name, cdf in feat_comparison_dfs.items():
        pdf.sub_title(f"Feature Set: {feat_name}")
        tbl_rows = []
        best_idx = int(cdf['Overall_top1'].idxmax())
        for _, row in cdf.iterrows():
            p = row['Parameters']
            params_str = f"{p/1e3:.1f}K" if p < 1e6 else f"{p/1e6:.2f}M"
            tbl_rows.append([
                str(row['Architecture']),
                params_str,
                _fmt(row['Overall_top1'], row.get('Overall_top1_std', 0.0)),
                _fmt(row['Overall_top3'], row.get('Overall_top3_std', 0.0)),
                f"{row['Improvement (%)']:+.2f}%",
            ])
        pdf.add_table(tbl_headers, tbl_rows, col_widths=tbl_w, highlight_row=best_idx)
        pdf.ln(5)

    # ── 4. Per-Prediction-Step Analysis (reference: 8D or first feat) ── #
    ref_feat = '8D' if '8D' in feat_comparison_dfs else feat_names[0]
    ref_cdf  = feat_comparison_dfs[ref_feat]

    pdf.add_page()
    pdf.section_title(f"4. Accuracy by Prediction Step - {ref_feat} Feature Set")
    pdf.body_text(
        f"Top-1 accuracy at each of the 4 future beam prediction steps (v=0..3) "
        f"for the {ref_feat} feature set. A higher drop across steps indicates "
        f"the architecture struggles with longer horizons."
    )

    baseline_steps = {v: ref_cdf.iloc[0][f'v{v}_top1'] for v in range(4)}
    step_headers = ["Architecture", "v=0", "v=1", "v=2", "v=3", "Avg Delta"]
    step_w = [50, 30, 30, 30, 30, 20]
    step_rows = []
    avg_deltas = []
    for _, row in ref_cdf.iterrows():
        delta = np.mean([row[f'v{v}_top1'] - baseline_steps[v] for v in range(4)])
        avg_deltas.append(delta)
        step_rows.append([
            str(row['Architecture']),
            _fmt(row['v0_top1'], row.get('v0_top1_std', 0.0)),
            _fmt(row['v1_top1'], row.get('v1_top1_std', 0.0)),
            _fmt(row['v2_top1'], row.get('v2_top1_std', 0.0)),
            _fmt(row['v3_top1'], row.get('v3_top1_std', 0.0)),
            f"{delta:+.2f}%",
        ])
    best_step_idx = int(np.argmax(avg_deltas))
    pdf.add_table(step_headers, step_rows, col_widths=step_w, highlight_row=best_step_idx)
    pdf.ln(4)

    # ── 5. Training Dynamics ─────────────────────────────────────────── #
    pdf.section_title(f"5. Training Dynamics - {ref_feat} Feature Set")

    lc_path = os.path.join(plot_dir, "E4_learning_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for arch_name in arch_names:
        key = f"{ref_feat}_{arch_name}"
        if key not in grand_results:
            continue
        label = architectures[arch_name]['name']
        hist = grand_results[key].get('training_history', {})
        tl = hist.get('train_losses', [])
        va = hist.get('val_accuracies', [])
        if tl:
            axes[0].plot(range(1, len(tl) + 1), tl, label=label, linewidth=1.5)
        if va:
            axes[1].plot(range(1, len(va) + 1), [a * 100 for a in va],
                         label=label, linewidth=1.5)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Curves", fontweight="bold")
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Accuracy (%)")
    axes[1].set_title("Validation Accuracy Curves", fontweight="bold")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(lc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.add_image_safe(lc_path, w=180)
    pdf.ln(3)

    # ── 6. Key Findings ──────────────────────────────────────────────── #
    pdf.add_page()
    pdf.section_title("6. Key Findings")

    # Best overall combo
    best_val, best_combo_str = -1.0, ""
    for arch_name in arch_names:
        for feat in feat_names:
            key = f"{feat}_{arch_name}"
            if key in grand_results:
                v = grand_results[key]['overall']['top1_acc'] * 100
                if v > best_val:
                    best_val = v
                    best_combo_str = f"{_safe(architectures[arch_name]['name'])} x {feat}"

    # Baseline (E4.0 × ref_feat)
    baseline_key = f"{ref_feat}_E4.0_baseline"
    baseline_acc = grand_results.get(baseline_key, {}).get('overall', {}).get('top1_acc', 0) * 100

    findings = [
        f"1. Best overall combination: {best_combo_str} - {best_val:.2f}% Top-1 "
        f"({best_val - baseline_acc:+.2f}% vs E4.0 baseline on {ref_feat}).",
    ]

    # Best per feature set
    for feat, cdf in feat_comparison_dfs.items():
        best_row = cdf.loc[cdf['Overall_top1'].idxmax()]
        findings.append(
            f"   Best [{feat}]: {best_row['Architecture']} - "
            f"{best_row['Overall_top1']:.2f}% +/- {best_row.get('Overall_top1_std', 0.0):.2f}% "
            f"({best_row['Improvement (%)']:+.2f}% vs E4.0 baseline)."
        )

    # Attention note
    findings.append(
        "2. Attention variants (E4.2, E4.3) were skipped for 6D feature sets "
        "(motion_dim=1 reduces cross-attention to a trivial scalar gate)."
    )

    # Param efficiency note
    ref_key_base = f"{ref_feat}_E4.0_baseline"
    if ref_key_base in grand_results:
        base_params = grand_results[ref_key_base].get('num_parameters', 0)
        findings.append(
            f"3. E4.0 baseline has {base_params/1e3:.1f}K parameters. "
            f"Larger variants (E4.4 hidden=256, E4.7 Transformer) use more params - "
            f"check accuracy/params trade-off in grand bar chart."
        )

    for f in findings:
        pdf.body_text(f)
    pdf.ln(3)

    # ── 7. Visualizations ────────────────────────────────────────────── #
    pdf.section_title("7. Visualizations")

    pdf.sub_title("Grand Summary Heatmap (Architecture x Feature Set)")
    pdf.add_image_safe(os.path.join(plot_dir, "E4_grand_heatmap.png"), w=175)

    pdf.add_page()
    pdf.sub_title("Grand Grouped Bar Chart (Architecture x Feature Set)")
    pdf.add_image_safe(os.path.join(plot_dir, "E4_grand_bar.png"), w=175)

    for feat in feat_names:
        bar_path = os.path.join(plot_dir, f"E4_{feat}_bar_comparison.png")
        step_path = os.path.join(plot_dir, f"E4_{feat}_by_step.png")
        if Path(bar_path).exists() or Path(step_path).exists():
            pdf.add_page()
            pdf.sub_title(f"Feature Set: {feat}")
            pdf.add_image_safe(bar_path, w=170)
            pdf.add_image_safe(step_path, w=170)

    # ── Save ─────────────────────────────────────────────────────────── #
    pdf_path = save_path / "E4_report.pdf"
    try:
        pdf.output(str(pdf_path))
        logger.info(f"E4 PDF report saved to {pdf_path}")
        return str(pdf_path)
    except PermissionError:
        fallback = save_path / f"E4_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(str(fallback))
        logger.warning(f"Permission denied writing {pdf_path}. Saved to {fallback} instead.")
        return str(fallback)
