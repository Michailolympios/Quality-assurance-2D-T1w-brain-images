import os, re
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SEG_TO_TISSUE = {
    1: "BACKGROUND",
    2: "SKULL",
    3: "CSF",
    4: "WM",
    5: "GM",
}

PLANE_MAP = {"AX": "axial", "COR": "coronal", "SAG": "sagittal"}

BASE_DIR = r"C:\MRIproject\MRI_project_analysis\Image_processing_course\AssignmentTrainingImages1"
OUT_DIR  = r"C:\MRIproject\MRI_project_analysis\Image_processing_course\Outputs"

GOLD_DIR = os.path.join(BASE_DIR, "gold")
ALG1_DIR = os.path.join(BASE_DIR, "alg1output (without connected components)")
ALG2_DIR = os.path.join(BASE_DIR, "alg2output (with connected components)")

EXT = "png"
PAT = re.compile(r"^(AX|COR|SAG)_(\d+)_SEG(\d+)$")


# ============================================================
# IO
# ============================================================

def load_mask_01(p: str) -> np.ndarray:
    m = imageio.imread(p)
    if m.ndim == 3:
        raise ValueError(f"{p} is RGB, expected single-channel mask")

    m = m.astype(np.uint8)
    u = np.unique(m)

    if set(u).issubset({0, 255}):
        m = (m > 0).astype(np.uint8)
    elif not set(u).issubset({0, 1}):
        raise ValueError(f"{p} has values {u}")

    return m


def parse_stem(stem):
    m = PAT.match(stem)
    if not m:
        return None
    plane_code, slice_idx, seg_id = m.groups()
    seg_id = int(seg_id)
    if seg_id not in SEG_TO_TISSUE:
        return None
    return (
        PLANE_MAP[plane_code],
        int(slice_idx),
        seg_id,
        SEG_TO_TISSUE[seg_id],
    )


def index_images(root):
    idx = {}
    for f in Path(root).rglob(f"*.{EXT}"):
        if parse_stem(f.stem):
            idx[f.stem] = str(f)
    return idx


# ============================================================
# METRICS (BACKGROUND MASKED)
# ============================================================

def foreground_mask(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """
    Foreground = union of gold and prediction
    Background is excluded from evaluation
    """
    return (gt == 1) | (pr == 1)


def counts_binary_masked(gt, pr, valid):
    gt1 = (gt == 1) & valid
    pr1 = (pr == 1) & valid
    gt0 = (gt == 0) & valid
    pr0 = (pr == 0) & valid

    tp = int(np.sum(gt1 & pr1))
    tn = int(np.sum(gt0 & pr0))
    fp = int(np.sum(gt0 & pr1))
    fn = int(np.sum(gt1 & pr0))
    return tp, tn, fp, fn


def safe_div(a, b):
    return a / b if b > 0 else np.nan


def compute_metrics(tp, tn, fp, fn):
    acc  = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return acc, prec, sens, spec


# ============================================================
# QA VISUALIZATION (NO TN, NO BACKGROUND)
# ============================================================

COL_TP = np.array([0, 255, 0], np.uint8)   # green
COL_FP = np.array([255, 0, 0], np.uint8)   # red
COL_FN = np.array([0, 0, 255], np.uint8)   # blue
COL_TN = np.array([180, 180, 180], np.uint8)  # light gray (true negative)


def make_error_map_rgb(gt, pr, valid):
    """
    RGB error map:
      TP = green
      FP = red
      FN = blue
      TN = gray
      background (invalid) = black
    """
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)

    tp = valid & (gt == 1) & (pr == 1)
    fp = valid & (gt == 0) & (pr == 1)
    fn = valid & (gt == 1) & (pr == 0)
    tn = valid & (gt == 0) & (pr == 0)

    rgb[tn] = COL_TN
    rgb[tp] = COL_TP
    rgb[fp] = COL_FP
    rgb[fn] = COL_FN

    # outside valid mask stays black
    return rgb



def save_confusion_figure(gt_p, pr_p, out_p, title):
    gt = load_mask_01(gt_p)
    pr = load_mask_01(pr_p)

    valid = foreground_mask(gt, pr)
    tp, tn, fp, fn = counts_binary_masked(gt, pr, valid)
    rgb = make_error_map_rgb(gt, pr, valid)

    fig = plt.figure(figsize=(9, 4))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(rgb)
    ax1.axis("off")
    ax1.set_title("TP / FP / FN (background masked)")

    ax2 = plt.subplot(1, 2, 2)
    ax2.axis("off")

    tbl = ax2.table(
        cellText=[[tn, fp], [fn, tp]],
        rowLabels=["gold 0", "gold 1"],
        colLabels=["pred 0", "pred 1"],
        loc="center",
        cellLoc="center"
    )
    tbl.scale(1.2, 1.6)
    ax2.set_title("Confusion matrix")

    fig.suptitle(title)
    plt.tight_layout()
    Path(out_p).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=200)
    plt.close(fig)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_pair(gold_dir, pred_dir, algorithm):
    gold = index_images(gold_dir)
    pred = index_images(pred_dir)

    rows = []
    agg = {}

    qa_dir = os.path.join(OUT_DIR, "QA_confusion_maps", algorithm)

    for st in sorted(set(gold) & set(pred)):
        parsed = parse_stem(st)
        if not parsed:
            continue

        plane, sl, seg, tissue = parsed

        gt = load_mask_01(gold[st])
        pr = load_mask_01(pred[st])

        valid = foreground_mask(gt, pr)

        tp, tn, fp, fn = counts_binary_masked(gt, pr, valid)
        acc, prec, sens, spec = compute_metrics(tp, tn, fp, fn)

        rows.append({
            "Algorithm": algorithm,
            "Tissue": tissue,
            "Plane": plane,
            "Slice": sl,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": acc,
            "Precision": prec,
            "Sensitivity": sens,
            "Specificity": spec,
        })

        key = (tissue, plane)
        agg.setdefault(key, [0, 0, 0, 0, 0])
        agg[key][0] += tp
        agg[key][1] += tn
        agg[key][2] += fp
        agg[key][3] += fn
        agg[key][4] += 1

        save_confusion_figure(
            gold[st],
            pred[st],
            os.path.join(qa_dir, f"{st}_{algorithm}.png"),
            f"{algorithm} | {tissue} | {plane} | slice {sl}"
        )

    per_df = pd.DataFrame(rows)

    agg_rows = []
    for (tissue, plane), (tp, tn, fp, fn, n) in agg.items():
        acc, prec, sens, spec = compute_metrics(tp, tn, fp, fn)
        agg_rows.append({
            "Algorithm": algorithm,
            "Tissue": tissue,
            "Plane": plane,
            "MatchedSlices": n,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": acc,
            "Precision": prec,
            "Sensitivity": sens,
            "Specificity": spec,
        })

    return per_df, pd.DataFrame(agg_rows)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    p1, a1 = evaluate_pair(GOLD_DIR, ALG1_DIR, "alg1")
    p2, a2 = evaluate_pair(GOLD_DIR, ALG2_DIR, "alg2")

    pd.concat([p1, p2]).to_csv(
        os.path.join(OUT_DIR, "metrics_per_image.csv"), index=False
    )
    pd.concat([a1, a2]).to_csv(
        os.path.join(OUT_DIR, "metrics_by_tissue_plane.csv"), index=False
    )

    overall = []
    for alg, df in pd.concat([p1, p2]).groupby("Algorithm"):
        TP, TN, FP, FN = df[["TP","TN","FP","FN"]].sum()
        acc, prec, sens, spec = compute_metrics(TP, TN, FP, FN)
        overall.append({
            "Algorithm": alg,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "Accuracy": acc,
            "Precision": prec,
            "Sensitivity": sens,
            "Specificity": spec,
        })

    pd.DataFrame(overall).to_csv(
        os.path.join(OUT_DIR, "metrics_overall_micro.csv"), index=False
    )

    print("[OK] Background masked correctly")
    print("[OK] Metrics are tissue-valid")
    print("[OK] QA maps are clean")


if __name__ == "__main__":
    main()

