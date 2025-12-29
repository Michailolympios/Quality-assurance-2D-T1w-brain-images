import os, glob, re, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# set tissues and dimension
SEG_TO_TISSUE = {2: "SKULL", 3: "CSF", 4: "WM", 5: "GM"}
PLANE_MAP = {"AX": "axial", "COR": "coronal", "SAG": "sagittal"}  # <-- FIXED missing }

# Where the images are situated and where we want to save them
BASE_DIR = r"C:\MRIproject\MRI_project_analysis\Image_processing_course\AssignmentTrainingImages"
OUT_DIR  = r"C:\MRIproject\MRI_project_analysis\Image_processing_course\Outputs"

# in which subfolder we have the images for each
GOLD_DIR = os.path.join(BASE_DIR, "gold")
ALG1_DIR = os.path.join(BASE_DIR, "alg1")
ALG2_DIR = os.path.join(BASE_DIR, "alg2")

EXT = "png"

PAT = re.compile(r"^(AX|COR|SAG)_(\d+)_SEG(\d+)$") # filename pattern

#make sure that is binary, we load the image 
def load_mask_01(p: str) -> np.ndarray:
    m = imageio.imread(p)
    if m.ndim == 3:
        raise ValueError(f"{p} is RGB (shape={m.shape}). Expected single-channel 0/1 mask.")
    m = m.astype(np.uint8)

    u = np.unique(m)
    if not np.all(np.isin(u, [0, 1])):
        raise ValueError(f"{p} has values {u[:20]} (showing up to 20). Expected only 0 and 1.")
    return m


def stem(p: str) -> str:
    return Path(p).stem


def parse_stem(st: str):
    """
    Parse file stem like 'AX_6_SEG3' -> (plane='axial', slice=6, seg=3, tissue='CSF')
    Returns None if pattern doesn't match or seg not in {2,3,4,5}.
    """
    m = PAT.match(st)
    if not m:
        return None
    plane_code = m.group(1)
    slice_idx  = int(m.group(2))
    seg_id     = int(m.group(3))
    if seg_id not in SEG_TO_TISSUE:
        return None
    plane = PLANE_MAP[plane_code]
    tissue = SEG_TO_TISSUE[seg_id]
    return plane, slice_idx, seg_id, tissue


def index_images(root_dir: str, ext: str):
    """
    Index all images under a root folder (recursively).
    key: stem (AX_6_SEG3)
    val: full path
    Only keeps files that match the expected naming scheme.
    """
    files = list(Path(root_dir).rglob(f"*.{ext}"))
    idx = {}
    for f in files:
        st = f.stem
        if parse_stem(st) is None:
            continue
        idx[st] = str(f)
    return idx


def counts_binary(gt: np.ndarray, pr: np.ndarray):
    gt1 = (gt == 1)  # where gold standard is true and false where it is not
    pr1 = (pr == 1)  # where algorithm is true and where the algorithm is not

    # we calculate tp fp fn and tn
    tp = int(np.sum(gt1 & pr1))
    fp = int(np.sum(~gt1 & pr1))
    fn = int(np.sum(gt1 & ~pr1))
    tn = int(gt.size - tp - fp - fn)
    return tp, tn, fp, fn


def safe_div(a, b):
    return (a / b) if b != 0 else np.nan


# We try to compute the accuracy and precision alogn with the sensitivity and specificity
def compute_metrics(tp, tn, fp, fn):
    acc  = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    sens = safe_div(tp, tp + fn)     
    spec = safe_div(tn, tn + fp)
    return acc, prec, sens, spec


# We create a contigency table with a QA image of TP TN FN FP with different colours 
COL_TP = np.array([0, 255, 0], dtype=np.uint8)       # green
COL_TN = np.array([180, 180, 180], dtype=np.uint8)   # gray
COL_FN = np.array([0, 0, 255], dtype=np.uint8)       # blue
COL_FP = np.array([255, 0, 0], dtype=np.uint8)       # red
COL_MASKED = np.array([0, 0, 0], dtype=np.uint8)     # black


def confusion_counts_binary(gt: np.ndarray, pr: np.ndarray, valid: np.ndarray | None = None):
    """
    gt/pr: 0/1
    valid: optional 0/1 mask; evaluate only where valid==1
    """
    if valid is None:
        valid = np.ones_like(gt, dtype=bool)
    else:
        valid = (valid == 1)

    gt1 = (gt == 1) & valid
    pr1 = (pr == 1) & valid
    gt0 = (gt == 0) & valid
    pr0 = (pr == 0) & valid

    tp = int(np.sum(gt1 & pr1))
    tn = int(np.sum(gt0 & pr0))
    fp = int(np.sum(gt0 & pr1))
    fn = int(np.sum(gt1 & pr0))
    return tp, tn, fp, fn


def make_error_map_rgb(gt: np.ndarray, pr: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """
    Returns an RGB uint8 image where each pixel is colored as TP/TN/FP/FN.
    Pixels outside 'valid' are black.
    """
    H, W = gt.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    if valid is None:
        valid_bool = np.ones((H, W), dtype=bool)
    else:
        valid_bool = (valid == 1)

    gt1 = (gt == 1)
    pr1 = (pr == 1)

    tp = valid_bool & gt1 & pr1
    tn = valid_bool & (~gt1) & (~pr1)
    fp = valid_bool & (~gt1) & pr1
    fn = valid_bool & gt1 & (~pr1)

    rgb[tn] = COL_TN
    rgb[tp] = COL_TP
    rgb[fn] = COL_FN
    rgb[fp] = COL_FP

    rgb[~valid_bool] = COL_MASKED
    return rgb


def save_confusion_figure(gt_path: str, pr_path: str, out_png: str,
                          title: str = "", valid_path: str | None = None):
    gt = load_mask_01(gt_path)
    pr = load_mask_01(pr_path)
    if gt.shape != pr.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pr.shape}")

    valid = load_mask_01(valid_path) if valid_path else None
    if valid is not None and valid.shape != gt.shape:
        raise ValueError(f"Valid mask shape mismatch: {valid.shape} vs {gt.shape}")

    tp, tn, fp, fn = confusion_counts_binary(gt, pr, valid)
    rgb = make_error_map_rgb(gt, pr, valid)

    # ---- plot ----
    fig = plt.figure(figsize=(9, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(rgb)
    ax1.set_title("TP/TN/FP/FN map")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.axis("off")

    # Confusion matrix table: rows=Reference (gold), cols=Segmentation 
    #            pred 0      pred 1
    # gold 0      TN        FP
    # gold 1      FN        TP
    cell_text = [[tn, fp],
                 [fn, tp]]
    row_labels = ["gold 0", "gold 1"]
    col_labels = ["pred 0", "pred 1"]

    tbl = ax2.table(cellText=cell_text,
                    rowLabels=row_labels,
                    colLabels=col_labels,
                    loc="center",
                    cellLoc="center")
    tbl.scale(1.2, 1.6)
    ax2.set_title("Confusion matrix")

    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# We compare the images with the reference and we save all of the information in a dataframe
def evaluate_pair(gold_root: str, pred_root: str, ext: str, algorithm_name: str,
                  make_qa: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      per_image_df: one row per image
      by_tissue_plane_df: aggregated totals per (tissue, plane)
    """
    gold_idx = index_images(gold_root, ext)
    pred_idx = index_images(pred_root, ext)

    common = sorted(set(gold_idx.keys()) & set(pred_idx.keys()))
    if len(common) == 0:
        raise RuntimeError(f"No matching files between gold and {algorithm_name} by filename stem.")

    per_rows = []
    agg = {}  # key=(tissue, plane) -> [TP,TN,FP,FN,n]

    qa_dir = os.path.join(OUT_DIR, "QA_confusion_maps", algorithm_name)

    for st in common:
        parsed = parse_stem(st)
        if parsed is None:
            continue
        plane, slice_idx, seg_id, tissue = parsed

        gt = load_mask_01(gold_idx[st])
        pr = load_mask_01(pred_idx[st])

        if gt.shape != pr.shape:
            raise ValueError(f"Shape mismatch at {st}: gold {gt.shape} vs {algorithm_name} {pr.shape}")

        tp, tn, fp, fn = counts_binary(gt, pr)
        acc, prec, sens, spec = compute_metrics(tp, tn, fp, fn)

        per_rows.append({
            "Algorithm": algorithm_name,
            "FileStem": st,
            "Plane": plane,
            "Slice": slice_idx,
            "SEG": seg_id,
            "Tissue": tissue,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": acc,
            "Precision": prec,
            "Sensitivity": sens,
            "Specificity": spec,
        })

        key = (tissue, plane)
        if key not in agg:
            agg[key] = [0, 0, 0, 0, 0]
        agg[key][0] += tp
        agg[key][1] += tn
        agg[key][2] += fp
        agg[key][3] += fn
        agg[key][4] += 1

   
        if make_qa:
            out_png = os.path.join(qa_dir, f"{st}_{algorithm_name}_QA.png")
            title = f"{algorithm_name} vs gold | {tissue} | {plane} | slice {slice_idx} | {st}"
            save_confusion_figure(gold_idx[st], pred_idx[st], out_png, title=title)

    per_df = pd.DataFrame(per_rows)

    agg_rows = []
    for (tissue, plane), (TP, TN, FP, FN, nimg) in agg.items():
        acc, prec, sens, spec = compute_metrics(TP, TN, FP, FN)
        agg_rows.append({
            "Algorithm": algorithm_name,
            "Tissue": tissue,
            "Plane": plane,
            "MatchedSlices": nimg,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "Accuracy": acc,
            "Precision": prec,
            "Sensitivity": sens,
            "Specificity": spec,
        })

    agg_df = pd.DataFrame(agg_rows)
    return per_df, agg_df


# Now we can create the main run script below
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # gold vs alg1
    per1, by1 = evaluate_pair(GOLD_DIR, ALG1_DIR, EXT, algorithm_name="alg1", make_qa=True)

    # gold vs alg2
    per2, by2 = evaluate_pair(GOLD_DIR, ALG2_DIR, EXT, algorithm_name="alg2", make_qa=True)

    # Save per image
    per = pd.concat([per1, per2], ignore_index=True)
    out_per = os.path.join(OUT_DIR, "metrics_per_image.csv")
    per.to_csv(out_per, index=False)
    print(f"[OK] wrote {out_per}")

    # Save tissue + plane
    df = pd.concat([by1, by2], ignore_index=True)
    out1 = os.path.join(OUT_DIR, "metrics_by_tissue_plane.csv")
    df.to_csv(out1, index=False)
    print(f"[OK] wrote {out1}")

    # Overall performance of algorithm 
    overall = []
    for alg, sub in per.groupby("Algorithm"):
        TP = int(sub["TP"].sum())
        TN = int(sub["TN"].sum())
        FP = int(sub["FP"].sum())
        FN = int(sub["FN"].sum())
        acc, prec, sens, spec = compute_metrics(TP, TN, FP, FN)
        overall.append({
            "Algorithm": alg,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "Accuracy": acc,
            "Precision": prec,
            "Sensitivity": sens,
            "Specificity": spec,
        })

    out2 = os.path.join(OUT_DIR, "metrics_overall_micro.csv")
    pd.DataFrame(overall).to_csv(out2, index=False)
    print(f"[OK] wrote {out2}")

    print(f"[OK] QA images saved under: {os.path.join(OUT_DIR, 'QA_confusion_maps')}")


if __name__ == "__main__":
    main()
