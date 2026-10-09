"""Sim-side preprocessing for the online Pred-Tac bridge: turns the full-resolution render of each env into the two
small inputs the tactile predictor actually consumes, so the server no longer receives (or crops) 960x720 frames.

  small_rgb (N, 224, 224, 3) uint8 : the full frame resized for the DINOv2 branch (same cv2.resize/INTER_LINEAR as
                                     sim_cond_cache.dino_forward_batch -- the server only normalizes it)
  crops     (N, 2, 256, 256, 3) uint8 : one WiLoR hand crop per (env, hand slot 0=left/1=right), RGB, flip already
                                     applied for left hands -- the server only normalizes it (mean/std from WiLoR's cfg)

The crop is a faithful port of WiLoR's ViTDetDataset (wilor/datasets/vitdet_dataset.py, rescale_factor=2.0):
box -> center/size, optional anti-alias Gaussian blur when the box is large (sigma=(ds-1)/2, ds=bbox/256/2 > 1.1),
mirror for left hands, affine warp to 256x256. Two deliberate differences, both for speed and both tested against the
original in test_predtac_crop_equiv.py: the blur/warp run on a ROI around the box instead of a copy of the whole frame
(the original's per-hand full-frame copy + skimage gaussian was ~19 ms/hand of the server's ~68 ms/env), and the input
here is RGB, so the BGR->RGB reversal of the original is not needed.

Pure numpy + cv2 (importable from the isaacgym-pinned env and from the touchanything env alike).
"""
import cv2
import numpy as np

PATCH = 256
DINO_SIZE = 224
RESCALE = 2.0


def _trans(c_x, c_y, src_w, src_h, dst_w, dst_h):
    """gen_trans_from_patch_cv with scale=1, rot=0 (WiLoR wilor/datasets/utils.py)."""
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = (c_x, c_y)
    src[1] = (c_x, c_y + src_h * 0.5)
    src[2] = (c_x + src_w * 0.5, c_y)
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = (dst_w * 0.5, dst_h * 0.5)
    dst[1] = (dst_w * 0.5, dst_h * 0.5 + dst_h * 0.5)
    dst[2] = (dst_w * 0.5 + dst_w * 0.5, dst_h * 0.5)
    return cv2.getAffineTransform(src, dst)


def wilor_patch(frame_rgb, box, is_right, patch=PATCH, rescale=RESCALE):
    """frame_rgb (H, W, 3) uint8, box [x1, y1, x2, y2] in frame pixels -> (patch, patch, 3) uint8 RGB crop."""
    H, W = frame_rgb.shape[:2]
    box = np.asarray(box, dtype=np.float32)
    cx, cy = float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)
    bbox_size = float(rescale * max(box[2] - box[0], box[3] - box[1]))
    flip = not bool(is_right)
    ds = (bbox_size / patch) / 2.0
    sigma = (ds - 1.0) / 2.0 if ds > 1.1 else 0.0
    rad = int(4.0 * sigma + 0.5) if sigma > 0 else 0          # skimage's truncate=4 kernel radius
    half = bbox_size / 2.0
    margin = rad + 2
    x0 = max(int(np.floor(cx - half)) - margin, 0)
    x1 = min(int(np.ceil(cx + half)) + margin + 1, W)
    y0 = max(int(np.floor(cy - half)) - margin, 0)
    y1 = min(int(np.ceil(cy + half)) + margin + 1, H)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((patch, patch, 3), dtype=np.uint8)
    roi = frame_rgb[y0:y1, x0:x1]
    if rad > 0:
        k = 2 * rad + 1
        roi = cv2.GaussianBlur(roi.astype(np.float32), (k, k), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    lx, ly = cx - x0, cy - y0
    if flip:
        roi = roi[:, ::-1]
        lx = (x1 - x0) - 1 - lx          # mirror about the ROI (equivalent to mirroring the full frame, see module doc)
    trans = _trans(lx, ly, bbox_size, bbox_size, patch, patch)
    out = cv2.warpAffine(np.ascontiguousarray(roi), trans, (patch, patch), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(out, 0, 255).astype(np.uint8) if out.dtype != np.uint8 else out


def build_payload(frames_rgb, boxes, has_hand, pool=None):
    """frames_rgb (N,H,W,3) uint8 RGB; boxes (N,2,4) float32 [left,right], NaN where absent; has_hand (N,2) bool.
    Returns (small_rgb (N,224,224,3) uint8, crops (N,2,256,256,3) uint8). `pool`: optional concurrent.futures executor
    (cv2 releases the GIL, so a small thread pool speeds this up)."""
    N = frames_rgb.shape[0]
    small = np.empty((N, DINO_SIZE, DINO_SIZE, 3), dtype=np.uint8)
    crops = np.zeros((N, 2, PATCH, PATCH, 3), dtype=np.uint8)

    def one(i):
        small[i] = cv2.resize(frames_rgb[i], (DINO_SIZE, DINO_SIZE), interpolation=cv2.INTER_LINEAR)
        for h in (0, 1):
            if has_hand[i, h]:
                crops[i, h] = wilor_patch(frames_rgb[i], boxes[i, h], is_right=(h == 1))

    if pool is None:
        for i in range(N):
            one(i)
    else:
        list(pool.map(one, range(N)))
    return small, crops
