"""Mask boundary metrics and the guided filter, shared by the boundary diagnosis and the refiner."""
import cv2
import numpy as np


def score(pm, gm):
    """IoU, boundary F at 1/2 px, mean |boundary distance|, signed distance (+ = outside GT), area ratio."""
    pm = pm.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    inter, uni = (pm & gm).sum(), (pm | gm).sum()
    p_edge = (pm - cv2.erode(pm, k)) > 0
    g_edge = (gm - cv2.erode(gm, k)) > 0
    dtg = cv2.distanceTransform((~g_edge).astype(np.uint8), cv2.DIST_L2, 3)
    dtp = cv2.distanceTransform((~p_edge).astype(np.uint8), cv2.DIST_L2, 3)
    out = dict(iou=inter / max(uni, 1), area=pm.sum() / max(gm.sum(), 1))
    for tol in (1, 2):
        pr = (dtg[p_edge] <= tol).mean() if p_edge.any() else 0.0
        rc = (dtp[g_edge] <= tol).mean() if g_edge.any() else 0.0
        out[f"bf{tol}"] = 2 * pr * rc / max(pr + rc, 1e-9)
    d = dtg[p_edge] * np.where(gm[p_edge] > 0, -1, 1)
    out["absd"] = float(np.abs(d).mean()) if len(d) else 0.0
    out["sd"] = float(d.mean()) if len(d) else 0.0
    return out


def guided_filter(I, p, r, eps):
    """He et al. guided filter: edge-preserving smoothing of p that follows the edges of guide I."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)
