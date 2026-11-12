"""Evaluate a checkpoint on the eel dataset.
Usage: uv run python experiments/evaluate.py <weights.pt> <data.yaml> [key=value ...]   (extra val args, e.g. end2end=False iou=0.6)"""
import sys
import json
from ultralytics import YOLO

weights, data = sys.argv[1], sys.argv[2]


def parse(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"True": True, "False": False, "None": None}.get(v, v)


extra = {k: parse(v) for k, v in (a.split("=", 1) for a in sys.argv[3:])}

# Optional NMS on top of the end-to-end head (removes duplicate boxes the one-to-one head failed to suppress).
POSTNMS = extra.pop("postnms", None)
if POSTNMS is not None:
    import torchvision
    from ultralytics.models.yolo.detect.val import DetectionValidator

    _orig = DetectionValidator.postprocess

    def _postprocess_nms(self, preds):
        outs = _orig(self, preds)
        for o in outs:
            if len(o["bboxes"]) > 1:
                keep = torchvision.ops.nms(o["bboxes"].float(), o["conf"].float(), float(POSTNMS))
                for k in ("bboxes", "conf", "cls", "extra"):
                    o[k] = o[k][keep]
        return outs

    DetectionValidator.postprocess = _postprocess_nms
m = YOLO(weights)
r = m.val(data=data, **{"imgsz": 512, "batch": 32, "device": "cpu", "plots": False, "verbose": False, "workers": 4, **extra})
out = {
    "weights": weights,
    "extra": extra,
    "postnms": POSTNMS,
    "box": {"P": r.box.mp, "R": r.box.mr, "mAP50": r.box.map50, "mAP50-95": r.box.map},
    "mask": {"P": r.seg.mp, "R": r.seg.mr, "mAP50": r.seg.map50, "mAP50-95": r.seg.map},
    "ms_per_img": r.speed,
}
print(json.dumps(out, indent=2))
