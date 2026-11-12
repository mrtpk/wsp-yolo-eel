"""Print results/v2/runs.jsonl as a table (optional tag filter), with fitness = sum over box+mask of 0.1*mAP50 + 0.9*mAP50-95."""
import json
import sys
flt = sys.argv[1] if len(sys.argv) > 1 else ""
sort = "--sort" in sys.argv
rows = [json.loads(l) for l in open("results/v2/runs.jsonl") if l.strip()]
rows = [r for r in rows if flt in r["cfg"]["tag"]]
fit = lambda r: 0.1 * r["box_m"]["mAP50"] + 0.9 * r["box_m"]["mAP"] + 0.1 * r["mask_m"]["mAP50"] + 0.9 * r["mask_m"]["mAP"]
if sort:
    rows.sort(key=fit, reverse=True)
print(f"{'tag':34} {'boxP':>6} {'boxR':>6} {'bmAP50':>7} {'bmAP':>7} {'mmAP50':>7} {'mmAP':>7} {'fitness':>8}")
for r in rows:
    b, m = r["box_m"], r["mask_m"]
    print(f"{r['cfg']['tag']:34} {b['P']:6.3f} {b['R']:6.3f} {b['mAP50']:7.4f} {b['mAP']:7.4f} {m['mAP50']:7.4f} {m['mAP']:7.4f} {fit(r):8.4f}")
