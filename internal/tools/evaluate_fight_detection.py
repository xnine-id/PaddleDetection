"""
Evaluasi Akurasi Model Fight Detection PaddleDetection
======================================================
Menghitung: Precision, Recall, F1-Score, mAP, Confusion Matrix

Input:
  - predictions.json  : output pipeline PaddleDetection (frame_id video asli)
  - ground_truth.json : export dari Label Studio (frame extract fps=5)

Cara pakai:
  python evaluate_fight_detection.py \
    --predictions  /path/to/predictions.json \
    --ground_truth /path/to/ground_truth.json \
    --output_dir   /path/to/eval_output/ \
    --video_fps    25 \
    --extract_fps  5
"""

import json
import os
import re
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
)


# ─────────────────────────────────────────────
# 1. LOAD GROUND TRUTH dari Label Studio JSON
# ─────────────────────────────────────────────
def load_ground_truth(gt_json_path: str) -> Dict[int, int]:
    """
    Parse Label Studio export JSON.
    Format: [{image, id, label, ...}, ...]
    Returns: {frame_number (int): label (int)}  0=no_fight, 1=fight
    """
    with open(gt_json_path) as f:
        data = json.load(f)

    gt = {}
    for item in data:
        image_path = item.get("image", "")

        # Ekstrak nomor frame: frame_00052.jpg → 52
        match = re.search(r"frame_(\d+)", image_path)
        if not match:
            continue
        frame_num = int(match.group(1))
        label = 1 if item.get("label", "").lower() == "fight" else 0
        gt[frame_num] = label

    fight_count = sum(gt.values())
    no_fight_count = len(gt) - fight_count
    print(f"[GT]   Loaded {len(gt)} labeled frames")
    print(f"       fight={fight_count}, no_fight={no_fight_count}")
    return gt


# ─────────────────────────────────────────────
# 2. LOAD PREDIKSI dari pipeline output JSON
# ─────────────────────────────────────────────
def load_predictions(pred_json_path: str) -> List[Any]:
    """
    Parse predictions.json dari pipeline PaddleDetection.
    Format: [{class, score, frame_ids: [...]}, ...]
    """
    with open(pred_json_path) as f:
        data = json.load(f)
    print(f"[PRED] Loaded {len(data)} prediction clips")
    return data


# ─────────────────────────────────────────────
# 3. COCOKKAN prediksi clip → frame GT
# ─────────────────────────────────────────────
def align_predictions_to_frames(
    predictions: List[Any], ground_truth: Dict[int, int], video_fps: int = 25, extract_fps: int = 5
) -> Tuple[List[int], List[int], List[float]]:
    """
    Konversi frame_ids (video asli) → frame extract,
    lalu cocokkan dengan ground truth per frame.

    video_fps=25, extract_fps=5 → interval=5
    frame_id_extract = frame_id_video / interval
    """
    interval = video_fps / extract_fps  # = 5.0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []
    matched_frames = set()

    for clip in predictions:
        clip_class = int(clip["class"])
        clip_score = float(clip["score"])
        frame_ids = clip.get("frame_ids", [])

        if not frame_ids:
            continue

        # Konversi frame_id video asli → frame extract
        extract_ids = [round((fid+1) / interval) for fid in frame_ids]
        min_extract = min(extract_ids)
        max_extract = max(extract_ids)

        # Assign label clip ke semua frame GT dalam range ini
        for frame_num in range(min_extract, max_extract + 1):
            if frame_num in ground_truth and frame_num not in matched_frames:
                y_true.append(ground_truth[frame_num])
                y_pred.append(clip_class)
                # Score = probabilitas "fight"
                y_scores.append(clip_score if clip_class == 1 else 1.0 - clip_score)
                matched_frames.add(frame_num)

    # Frame GT yang tidak tercakup prediksi → diasumsikan no_fight
    unmatched = set(ground_truth.keys()) - matched_frames
    if unmatched:
        print(
            f"[WARN] {len(unmatched)} frame GT tidak tercakup prediksi → diasumsikan no_fight"
        )
        for frame_num in sorted(unmatched):
            y_true.append(ground_truth[frame_num])
            y_pred.append(0)
            y_scores.append(0.0)

    print(
        f"[ALIGN] matched={len(matched_frames)}, unmatched={len(unmatched)}, total={len(y_true)}"
    )
    return y_true, y_pred, y_scores


# ─────────────────────────────────────────────
# 4. HITUNG & PRINT METRIK
# ─────────────────────────────────────────────
def compute_metrics(y_true: List[int], y_pred: List[int], y_scores: List[float]) -> Tuple[float, float, float, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mAP = average_precision_score(y_true, y_scores)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print("\n" + "=" * 48)
    print("       HASIL EVALUASI — FIGHT DETECTION")
    print("=" * 48)
    print(f"  Precision   : {precision:.4f}  ({precision*100:.1f}%)")
    print(f"  Recall      : {recall:.4f}  ({recall*100:.1f}%)")
    print(f"  F1-Score    : {f1:.4f}  ({f1*100:.1f}%)")
    print(f"  mAP         : {mAP:.4f}  ({mAP*100:.1f}%)")
    print("-" * 48)
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Total frames: {len(y_true)}")
    print(f"  Fight GT    : {sum(y_true)}")
    print(f"  No-fight GT : {len(y_true) - sum(y_true)}")
    print("=" * 48 + "\n")

    return precision, recall, f1, mAP


# ─────────────────────────────────────────────
# 5. PLOT CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true: List[int], y_pred: List[int], output_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Fight", "Fight"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(
        "Confusion Matrix — Fight Detection", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# 6. PLOT PRECISION-RECALL CURVE
# ─────────────────────────────────────────────
def plot_pr_curve(y_true: List[int], y_scores: List[float], mAP: float, output_dir: str) -> None:
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
    baseline = sum(y_true) / len(y_true)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        rec_curve,
        prec_curve,
        color="steelblue",
        linewidth=2,
        label=f"Model (mAP={mAP:.4f})",
    )
    ax.fill_between(rec_curve, prec_curve, alpha=0.1, color="steelblue")
    ax.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Baseline ({baseline:.2f})",
    )
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))

    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# 7. PLOT ROC CURVE
# ─────────────────────────────────────────────
def plot_roc_curve(y_true: List[int], y_scores: List[float], output_dir: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        fpr, tpr, color="darkorange", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})"
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — Fight Detection", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# 8. BUILD FRAME ARRAYS untuk timeline
# ─────────────────────────────────────────────
def build_frame_arrays(predictions: List[Any], ground_truth: Dict[int, int], video_fps: int = 25, extract_fps: int = 5) -> Tuple[Any, Any, int]:
    """
    Bangun array GT dan prediksi berindeks frame extract,
    digunakan untuk plot timeline dengan sumbu waktu.
    """
    interval = video_fps / extract_fps
    total_frames = max(ground_truth.keys()) + 1
    gt_array = np.zeros(total_frames, dtype=int)
    pred_array = np.zeros(total_frames, dtype=int)

    for frame_num, label in ground_truth.items():
        gt_array[frame_num] = label

    matched = set()
    for clip in predictions:
        clip_class = int(clip["class"])
        frame_ids = clip.get("frame_ids", [])
        if not frame_ids:
            continue
        extract_ids = [round((fid+1) / interval) for fid in frame_ids]

        for frame_num in range(min(extract_ids), max(extract_ids) + 1):
            if frame_num < total_frames and frame_num not in matched:
                pred_array[frame_num] = clip_class
                matched.add(frame_num)

    return gt_array, pred_array, total_frames


# ─────────────────────────────────────────────
# 9. HELPER: format detik → "mm:ss"
# ─────────────────────────────────────────────
def seconds_to_mmss(seconds: float) -> str:
    m = round(seconds) // 60
    s = round(seconds) % 60
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────
# 10. PLOT TIMELINE dengan durasi waktu
# ─────────────────────────────────────────────
def plot_timeline(gt_array: Any, pred_array: Any, total_frames: int, extract_fps: int, output_dir: str) -> None:
    seconds_per_frame = 1.0 / extract_fps
    total_seconds = total_frames * seconds_per_frame
    time_axis = np.arange(total_frames) * seconds_per_frame

    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    fig.patch.set_facecolor("#f9f9f9")

    # ── Ground Truth ──
    axes[0].fill_between(time_axis, gt_array, alpha=0.75, color="#3B82F6", step="post")
    axes[0].set_ylabel("Ground Truth", fontsize=11, fontweight="bold")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["No Fight", "Fight"], fontsize=9)
    axes[0].set_ylim(-0.15, 1.45)
    axes[0].set_facecolor("#f0f4ff")
    axes[0].grid(True, axis="x", alpha=0.3, linestyle="--")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Helper: ekstrak segmen fight dari array
    def get_segments(arr: Any, time_ax: Any) -> List[Tuple[float, float]]:
        segments = []
        in_seg, start = False, 0
        for i, val in enumerate(arr):
            t = time_ax[i]
            if val == 1 and not in_seg:
                in_seg = True
                start = t
            elif val == 0 and in_seg:
                in_seg = False
                segments.append((start, t))
        if in_seg:
            segments.append((start, time_ax[-1]))
        return segments

    # Annotate timestamp GT — posisi vertikal bergantian atas/bawah supaya tidak tabrakan
    gt_segments = get_segments(gt_array, time_axis)
    print("[GROUND TRUTH] gt_segments", gt_segments)
    for idx, (seg_start, seg_end) in enumerate(gt_segments):
        mid_t = (seg_start + seg_end) / 2
        y_pos = 1.22 if idx % 2 == 0 else 1.10  # bergantian tinggi/rendah
        label = f"{seconds_to_mmss(seg_start)}"
        axes[0].text(
            mid_t,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#1D4ED8",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="#3B82F6",
                alpha=0.7,
                linewidth=0.8,
            ),
        )

    # ── Prediction ──
    axes[1].fill_between(
        time_axis, pred_array, alpha=0.75, color="#EF4444", step="post"
    )
    axes[1].set_ylabel("Prediction", fontsize=11, fontweight="bold")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["No Fight", "Fight"], fontsize=9)
    axes[1].set_ylim(-0.15, 1.45)
    axes[1].set_facecolor("#fff0f0")
    axes[1].grid(True, axis="x", alpha=0.3, linestyle="--")
    axes[1].spines[["top", "right"]].set_visible(False)

    # Annotate timestamp Prediction — bergantian juga
    pred_segments = get_segments(pred_array, time_axis)
    print("[PREDICTION] pred_segments", pred_segments)
    for idx, (seg_start, seg_end) in enumerate(pred_segments):
        mid_t = (seg_start + seg_end) / 2
        y_pos = 1.22 if idx % 2 == 0 else 1.10
        label = f"{seconds_to_mmss(seg_start)}"
        axes[1].text(
            mid_t,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#991B1B",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="#EF4444",
                alpha=0.7,
                linewidth=0.8,
            ),
        )

    # ── X-axis: format mm:ss setiap 30 detik ──
    axes[1].set_xlabel("Waktu", fontsize=11, fontweight="bold")
    axes[1].set_xlim(0, total_seconds)
    tick_positions = np.arange(0, total_seconds + 30, 30)
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels([seconds_to_mmss(t) for t in tick_positions], fontsize=9)

    # Garis vertikal & label menit
    for m in range(0, int(total_seconds / 60) + 2):
        t = m * 60
        if t <= total_seconds:
            axes[0].axvline(x=t, color="gray", alpha=0.2, linewidth=0.8)
            axes[1].axvline(x=t, color="gray", alpha=0.2, linewidth=0.8)
            axes[1].text(
                t, -0.13, f"{m}m", ha="center", va="top", fontsize=8, color="gray"
            )

    # ── Judul & legend ──
    duration_str = seconds_to_mmss(total_seconds)
    fig.suptitle(
        f"Timeline: Ground Truth vs Prediction  |  Durasi: {duration_str}  |  "
        f"Extract: {extract_fps} fps  (1 frame = {seconds_per_frame:.1f}s)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3B82F6", alpha=0.75, label="Fight (Ground Truth)"),
        Patch(facecolor="#EF4444", alpha=0.75, label="Fight (Prediction)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 1.0),
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "timeline.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    print(f"  Durasi total: {duration_str} ({total_seconds:.1f} detik)")


# ─────────────────────────────────────────────
# 9. SIMPAN SUMMARY ke JSON
# ─────────────────────────────────────────────
def save_summary(precision: float, recall: float, f1: float, mAP: float, y_true: List[int], y_pred: List[int], output_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel().tolist()

    summary = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mAP": round(mAP, 4),
        "confusion_matrix": {
            "labels": ["no_fight", "fight"],
            "matrix": cm.tolist(),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        },
        "total_frames": len(y_true),
        "fight_frames": sum(y_true),
        "no_fight_frames": len(y_true) - sum(y_true),
    }
    path = os.path.join(output_dir, "evaluation_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluasi Fight Detection")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path ke predictions.json (pipeline output)",
    )
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="Path ke ground_truth.json (Label Studio export)",
    )
    parser.add_argument(
        "--output_dir", default="./eval_output", help="Folder output grafik & summary"
    )
    parser.add_argument(
        "--video_fps", type=int, default=25, help="FPS video asli (default: 25)"
    )
    parser.add_argument(
        "--extract_fps",
        type=int,
        default=5,
        help="FPS saat extract frames (default: 5)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nKonfigurasi:")
    print(f"  video_fps={args.video_fps}, extract_fps={args.extract_fps}")
    print(f"  interval konversi = {args.video_fps / args.extract_fps} frame\n")

    # Load data
    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)

    # Align frame_id video asli → frame extract → cocokkan GT
    y_true, y_pred, y_scores = align_predictions_to_frames(
        predictions,
        ground_truth,
        video_fps=args.video_fps,
        extract_fps=args.extract_fps,
    )

    if len(y_true) == 0:
        print("[ERROR] Tidak ada data yang cocok antara prediksi dan ground truth!")
        print(
            "  Cek apakah frame_ids di predictions.json sesuai dengan nama file di ground_truth.json"
        )
        return

    # Hitung metrik
    precision, recall, f1, mAP = compute_metrics(y_true, y_pred, y_scores)

    # Plot semua grafik
    plot_confusion_matrix(y_true, y_pred, args.output_dir)
    plot_pr_curve(y_true, y_scores, mAP, args.output_dir)
    plot_roc_curve(y_true, y_scores, args.output_dir)

    # Build frame arrays untuk timeline dengan durasi waktu
    gt_array, pred_array, total_frames = build_frame_arrays(
        predictions,
        ground_truth,
        video_fps=args.video_fps,
        extract_fps=args.extract_fps,
    )

    plot_timeline(gt_array, pred_array, total_frames, args.extract_fps, args.output_dir)

    # Simpan summary
    save_summary(precision, recall, f1, mAP, y_true, y_pred, args.output_dir)

    print(f"\nSelesai! Semua output tersimpan di: {args.output_dir}")


if __name__ == "__main__":
    main()
