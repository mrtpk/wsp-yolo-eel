"""Export the study's numbers to one JSON file for the kaizen blog post.

Computed from result files: every evaluation in results/v2/runs.jsonl, recall by eel length
(results/diag_baseline.csv), robustness (results/robustness_baseline.json) and the mask IoU ceiling
(results/mask_ceiling.log). Copied from the logs named in "sources": numbers that exist only in console logs.

Usage: uv run python experiments/export_blog_numbers.py <out.json>
"""
import csv
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def fitness(b, m):
    return 0.1 * b["mAP50"] + 0.9 * b["mAP"] + 0.1 * m["mAP50"] + 0.9 * m["mAP"]


runs = {}
for line in open(R / "v2/runs.jsonl"):
    r = json.loads(line)
    b, m = r["box_m"], r["mask_m"]
    runs[r["cfg"]["tag"]] = {
        "box": {k: round(b[k], 6) for k in ("P", "R", "mAP50", "mAP")},
        "mask": {k: round(m[k], 6) for k in ("P", "R", "mAP50", "mAP")},
        "fitness": round(fitness(b, m), 6),
    }

bins = [(0, 60), (60, 100), (100, 150), (150, 250), (250, 10**6)]
rows = list(csv.DictReader(open(R / "diag_baseline.csv")))
recall = []
for lo, hi in bins:
    sel = [r for r in rows if lo <= float(r["length_px"]) < hi]
    found = sum(int(r["found"]) for r in sel)
    recall.append({"lo": lo, "hi": hi, "eels": len(sel), "found": found, "recall": round(found / len(sel), 4)})
miss = [r for r in rows if r["found"] == "0"]
loc_fail = [r for r in miss if float(r["best_iou"]) < 0.5]
imgsz = {}
for size in (640, 768):
    rr = list(csv.DictReader(open(R / f"diag_imgsz{size}.csv")))
    short = [r for r in rr if float(r["length_px"]) < 60]
    mid = [r for r in rr if 60 <= float(r["length_px"]) < 100]
    imgsz[size] = {"under60": round(sum(int(r["found"]) for r in short) / len(short), 4),
                   "60to100": round(sum(int(r["found"]) for r in mid) / len(mid), 4),
                   "all": round(sum(int(r["found"]) for r in rr) / len(rr), 4)}
diag = {
    "misses": len(miss), "score_failures": len(miss) - len(loc_fail), "loc_failures": len(loc_fail),
    "median_miss_score": round(float(sorted(float(r["best_ok_score"]) for r in miss if float(r["best_iou"]) >= 0.5)[(len(miss) - len(loc_fail)) // 2]), 3),
    "imgsz": imgsz,
    "eels": len(rows),
    "found": sum(int(r["found"]) for r in rows),
    "by_stride": {s: sum(1 for r in rows if r["found"] == "1" and r["det_stride"] == s) for s in ("8", "16", "32")},
}

rob = json.load(open(R / "robustness_baseline.json"))
robustness = {k: round(v["box_mAP50"], 4) for k, v in rob["results"].items()}
robustness["frames"] = rob["frames"]

ceiling = []
for line in open(R / "mask_ceiling.log"):
    v = line.split()
    if v and v[0].isdigit():
        ceiling.append({"imgsz": int(v[0]), "grid": v[1], "mean_iou": float(v[2]),
                        "ge75": float(v[5].rstrip("%")), "ge90": float(v[6].rstrip("%"))})

logged = {
    "sources": "weight_space_report.md sections 1-15; console logs in results/ and results/v2/queue_logs/",
    "params_original": 3.06e6, "params_v2": 3.18e6, "frames": 4986, "labels": 5013, "heldout_frames": 2493,
    "class_bias": {"dead": [-12.98, -12.03, -8.0], "eel": [-2.17, 1.18, -8.0]},
    "stride8_edits": [["bias +1.0", 2, 38], ["bias +2.0", 7, 77], ["LDA a=4 b=-1", 12, 156], ["LDA a=8 b=0", 38, 1103]],
    "stride8_targets": 67,
    "probe": [[21, 117, 129, 37], [30, 245, 234, 112], [51, 689, 429, 256]],
    "census": {"tp": 4613, "dup": 329, "dup_same": 166, "dup_adjacent": 165, "dup_cross": 163,
               "misloc": 73, "misloc_on_missed": 62, "background": 1, "dup_score_ratio": 0.67},
    "after_peak": {"dup": 1, "misloc": 58, "tp": 4611},
    "head_blend": {"t0.5_mAP50": 0.145, "t0.25_mAP50": 0.794, "cls_only_mAP50": 0.501, "cls_only_P": 0.028},
    "merge_max_err": 4e-5,
    "temporal": {"variants": 49, "best_gain": 0.0002, "missed": 401, "missed_with_neighbour": 51},
    "boundary": {"eels": 2319, "iou": 0.733, "absd": 1.76, "sd": 0.93, "area": 1.14, "bf1": 0.535, "bf2": 0.743,
                 "fn_outside_box": 0.23, "vertices": 45, "vertex_px": 7, "eel_width": 9},
    "refined": {"iou": 0.733, "absd": 1.57, "sd": 0.19, "area": 0.99, "bf1": 0.572, "bf2": 0.770,
                "params": 37921, "fit_eels": 2321, "epochs": 12, "ms_per_eel": 60},
    "wrapper29": {"bf1": [0.54, 0.60], "absd": [1.66, 1.46], "area": [1.14, 1.00], "iou": [0.730, 0.740]},
    "failed_mask_fixes": [["threshold -2..+2 logits", "0.632-0.734"], ["guided filter", "0.689-0.733"],
                          ["inference 640 / 768 px", "0.734 / 0.733"], ["box grown 1-8 px", "0.708-0.730"]],
}

out = {"runs": runs, "recall_by_length": recall, "diag": diag, "robustness": robustness,
       "mask_ceiling": ceiling, "logged": logged}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"wrote {sys.argv[1]}: {len(runs)} runs; recall bins {[r['recall'] for r in recall]}; diag {diag}")
