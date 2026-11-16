"""FlexGlue forward pass from a .jmtw weights file only: the executable specification for the C++ backends.

Uses torch as an array library (conv2d, matmul, sort); nothing from the upstream repository.

    w = load(path)                                   # dict of tensors
    f = describe(w, image)                           # image (3,H,W) float in [0,1] -> features (256,h,w)
    out = match(w, f0, xy0, size0, f1, xy1, size1)   # sparse matching of two keypoint sets

Conventions (upstream's): keypoints are pixels of the original image with (0,0) at the top-left corner of the
top-left pixel, sizes are (w, h). "Resized pixels" are pixels of the network input (8 per feature cell).
Every function takes an optional `stages` dict and records its intermediate tensors there (golden dumps).
"""
import math

import torch
import torch.nn.functional as F

import jmtfile

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEPTHS = (3, 3, 27, 3)
HEADS = 8
TOPK = 512  # the entmax threshold is computed from the TOPK largest logits


def load(path, device="cpu", dtype=torch.float32):
    meta, tensors = jmtfile.read(path)
    w = {k: torch.from_numpy(v.copy()).to(device=device, dtype=dtype) for k, v in tensors.items()}
    w["_meta"] = meta
    return w


def rec(stages, name, x):
    if stages is not None:
        stages[name] = x.detach().float().cpu()
    return x


# ---------------------------------------------------------------- basic operations

def linear(w, name, x):
    return x @ w[name + ".weight"].T + w[name + ".bias"] if name + ".bias" in w else x @ w[name + ".weight"].T


def layer_norm(w, name, x, eps=1e-5):
    """Over the last axis."""
    return F.layer_norm(x, x.shape[-1:], w[name + ".weight"], w[name + ".bias"], eps)


def hwc(x):
    return x.permute(1, 2, 0)


def chw(x):
    return x.permute(2, 0, 1)


def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))


def cell_centers(h, w, sx, sy, device):
    """(h*w, 2) centres of the cells of an h x w grid in [-1, 1], x then y, raster order."""
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack([(x + 0.5) / sx, (y + 0.5) / sy], -1).flatten(0, 1) * 2 - 1


def sample(features, pts):
    """Bilinear samples (N, C) of features (C, h, w) at pts (N, 2) in [-1, 1]; outside = nearest border."""
    out = F.grid_sample(features[None], pts[None, None], "bilinear", padding_mode="border", align_corners=False)
    return out[0, :, 0].T


# ---------------------------------------------------------------- describe (once per image)

def preprocess(image, resize=(512, 512)):
    """(3,H,W) in [0,1] -> normalized network input (3,h,w). resize: (h, w), both multiples of 32."""
    x = F.interpolate(image[None], resize, mode="bilinear", align_corners=False)[0]
    mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype)[:, None, None]
    std = torch.tensor(STD, device=x.device, dtype=x.dtype)[:, None, None]
    return (x - mean) / std


def backbone(w, x, stages=None):
    """ConvNeXt-B: four feature maps at strides 4, 8, 16, 32 with 128, 256, 512, 1024 channels."""
    feats = []
    for s, depth in enumerate(DEPTHS):
        p = f"backbone.downsample_layers.{s}"
        if s == 0:
            x = F.conv2d(x[None], w[p + ".0.weight"], w[p + ".0.bias"], stride=4)[0]
            x = chw(layer_norm(w, p + ".1", hwc(x), 1e-6))
        else:
            x = chw(layer_norm(w, p + ".0", hwc(x), 1e-6))
            x = F.conv2d(x[None], w[p + ".1.weight"], w[p + ".1.bias"], stride=2)[0]
        rec(stages, f"down{s}", x)
        for b in range(depth):
            p = f"backbone.stages.{s}.{b}"
            y = F.conv2d(x[None], w[p + ".dwconv.weight"], w[p + ".dwconv.bias"], padding=3, groups=x.shape[0])[0]
            y = layer_norm(w, p + ".norm", hwc(y), 1e-6)
            y = linear(w, p + ".pwconv2", gelu(linear(w, p + ".pwconv1", y)))  # layer scale is folded in
            x = x + chw(y)
        feats.append(rec(stages, f"stage{s}", x))
    return feats


def neck(w, feats, stages=None):
    """Merge the four stages at stride 8 -> (256, h/8, w/8)."""
    fine, *coarse = feats
    x = chw(layer_norm(w, "neck.stem.0", hwc(fine)))
    x = F.conv2d(x[None], w["neck.stem.1.weight"], w["neck.stem.1.bias"], padding=1)[0]
    x = gelu(layer_norm(w, "neck.stem.2", hwc(x)))
    x = hwc(F.pixel_unshuffle(chw(x)[None], 2)[0])
    x = linear(w, "neck.fine_proj", x)
    rec(stages, "neck_fine", chw(x))
    h, wd = x.shape[:2]
    for i, c in enumerate(coarse):
        c = chw(linear(w, f"neck.projs.{i}", layer_norm(w, f"neck.norms.{i}", hwc(c))))
        if c.shape[-2:] != (h, wd):
            c = F.interpolate(c[None], (h, wd), mode="bilinear", align_corners=False)[0]
        x = x + hwc(c)
    rec(stages, "neck_sum", chw(x))
    y = linear(w, "neck.mlp.3", gelu(linear(w, "neck.mlp.1", layer_norm(w, "neck.mlp.0", x))))
    x = layer_norm(w, "neck.out_norm", linear(w, "neck.down", x + y))
    return rec(stages, "features", chw(x))


def describe(w, image, resize=(512, 512), stages=None):
    x = rec(stages, "input", preprocess(image.to(w["desc0.weight"]), resize))
    return neck(w, backbone(w, x, stages), stages)


def global_tokens(w, features):
    """Layer-0 descriptors of every cell: (h*w, 64), L2-normalized."""
    return F.normalize(linear(w, "desc0", hwc(features).flatten(0, 1)), dim=-1)


# ---------------------------------------------------------------- transformer (once per pair)

def rotary_encoding(w, h, wd, device):
    """(2, h*w, 1, 64): cos and sin factors for every token, shared by all heads."""
    centers = cell_centers(h, wd, max(h, wd), max(h, wd), device).to(w["posenc.Wr.weight"])
    proj = centers @ w["posenc.Wr.weight"].T  # (P, 32)
    enc = torch.stack([proj.cos(), proj.sin()], 0).repeat_interleave(2, dim=-1)  # (2, P, 64)
    return enc[:, :, None]


def rotate(enc, t):
    """Rotate consecutive pairs of t (P, H, 64) by the token's angles."""
    t1, t2 = t.unflatten(-1, (-1, 2)).unbind(-1)
    half = torch.stack((-t2, t1), -1).flatten(-2)
    return t * enc[0] + half * enc[1]


def attention(q, k, v):
    """q (P, H, d), k and v (S, H, d) -> (P, H*d). Softmax over the S source tokens, per head."""
    logits = torch.einsum("phd,shd->hps", q, k) / math.sqrt(q.shape[-1])
    return torch.einsum("hps,shd->phd", logits.softmax(-1), v).flatten(-2)


def ffn(w, name, x, message):
    y = torch.cat([x, layer_norm(w, name[: -len("ffn")] + "message_norm", message)], -1)
    y = gelu(layer_norm(w, name + ".1", linear(w, name + ".0", y)))
    return x + linear(w, name + ".3", y)


def self_block(w, p, x, enc):
    q, k, v = linear(w, p + ".Wqkv", x).unflatten(-1, (3, HEADS, -1)).unbind(-3)
    q = rotate(enc, layer_norm(w, p + ".q_norm", q))
    k = rotate(enc, layer_norm(w, p + ".k_norm", k))
    return ffn(w, p + ".ffn", x, linear(w, p + ".out_proj", attention(q, k, v)))


def cross_block(w, p, x0, x1):
    qk0, qk1 = (layer_norm(w, p + ".qk_norm", linear(w, p + ".to_qk", x).unflatten(-1, (HEADS, -1))) for x in (x0, x1))
    v0, v1 = (linear(w, p + ".to_v", x).unflatten(-1, (HEADS, -1)) for x in (x0, x1))
    m0 = linear(w, p + ".to_out", attention(qk0, qk1, v1))
    m1 = linear(w, p + ".to_out", attention(qk1, qk0, v0))
    return ffn(w, p + ".ffn", x0, m0), ffn(w, p + ".ffn", x1, m1)


def covisibility(w, layer, x0, x1):
    """Mean over the tokens of both images of sigmoid(linear(token)). Layer 0 has no such head: 1."""
    if layer == 0:
        return 1.0
    return torch.sigmoid(linear(w, f"heads.{layer}.covisibility", torch.cat([x0, x1], 0))).mean().item()


def decode_pair(w, f0, f1, n_layers=9, min_covisibility=0.0, early_stop_decay=None, stages=None):
    """Tokens (P, 512) of both images after the transformer, the layer reached and the covisibility score.

    Early stop: after layer k (2 <= k < n_layers) stop when score_k < min_covisibility * (1 - decay)^(9 - k).
    """
    hw0, hw1 = f0.shape[-2:], f1.shape[-2:]
    x0 = linear(w, "input_proj", hwc(f0).flatten(0, 1))
    x1 = linear(w, "input_proj", hwc(f1).flatten(0, 1))
    enc0, enc1 = rotary_encoding(w, *hw0, f0.device), rotary_encoding(w, *hw1, f1.device)
    rec(stages, "tokens0_l0", x0), rec(stages, "tokens1_l0", x1)
    layer, score = n_layers, None
    for k in range(1, n_layers + 1):
        p = f"transformers.{k - 1}"
        x0, x1 = self_block(w, p + ".self_attn", x0, enc0), self_block(w, p + ".self_attn", x1, enc1)
        rec(stages, f"tokens0_l{k}_self", x0)
        x0, x1 = cross_block(w, p + ".cross_attn", x0, x1)
        rec(stages, f"tokens0_l{k}", x0), rec(stages, f"tokens1_l{k}", x1)
        if early_stop_decay is None or k in (1, n_layers):
            continue
        s = covisibility(w, k, x0, x1)
        if s < min_covisibility * (1 - early_stop_decay) ** (9 - k):
            layer, score = k, s
            break
    if score is None:
        score = covisibility(w, layer, x0, x1)
    return x0, x1, layer, score


# ---------------------------------------------------------------- head (per query point)

def entmax15_threshold(ys):
    """Threshold tau of 1.5-entmax p = (y - tau)_+^2 from scores sorted in descending order."""
    rho = torch.arange(1, ys.shape[-1] + 1, device=ys.device, dtype=ys.dtype)
    mean = ys.cumsum(-1) / rho
    mean_sq = ys.square().cumsum(-1) / rho
    tau = mean - ((1 - rho * (mean_sq - mean.square())) / rho).clamp_min(0).sqrt()
    support = (tau <= ys).sum(-1, keepdim=True)
    return tau.gather(-1, support - 1)


def window_readout(logits, grid, h, wd):
    """Expected position (N, 2) in the 3x3 window around the best cell, and the probability mass there."""
    top = logits.topk(min(TOPK, logits.shape[-1]), dim=-1)
    vmax, idx = top.values[:, :1], top.indices[:, :1]
    y = (logits - vmax) / 2
    tau = entmax15_threshold((top.values - vmax) / 2)
    r = torch.arange(-1, 2, device=logits.device)
    ny = (idx // wd)[..., None] + r[:, None]
    nx = (idx % wd)[..., None] + r[None, :]
    inside = ((ny >= 0) & (ny < h) & (nx >= 0) & (nx < wd)).flatten(-2)
    nidx = (ny.clamp(0, h - 1) * wd + nx.clamp(0, wd - 1)).flatten(-2)
    p = (y.gather(-1, nidx) - tau).clamp_min(0).square()
    position = (p[..., None] * grid[nidx]).sum(-2) / p.sum(-1, keepdim=True)
    return position, (p * inside).sum(-1)


def mlp3(w, name, x):
    return linear(w, name + ".3", F.relu(layer_norm(w, name + ".1", linear(w, name + ".0", x))))


def head(w, layer, pts, cf_i, cf_j, stages=None, tag=""):
    """Points (N, 2) of image i in [-1, 1] -> predicted points in image j in [-1, 1], and confidence (N).

    cf_i, cf_j: tokens of both images as (512, h, w) maps, after `layer` transformer layers.
    """
    p = f"heads.{layer}"
    h, wd = cf_j.shape[-2:]
    search = F.normalize(linear(w, p + ".to_search", hwc(cf_j).flatten(0, 1)), dim=-1)
    f_query = sample(cf_i, pts)
    query = F.normalize(linear(w, p + ".to_search", f_query), dim=-1)
    logits = query @ search.T * w[p + ".temperature"]
    grid = cell_centers(h, wd, wd, h, logits.device).to(logits)
    coarse, mass = window_readout(logits, grid, h, wd)
    rec(stages, f"coarse{tag}", coarse), rec(stages, f"mass{tag}", mass)
    f_match = sample(cf_j, coarse)
    x = torch.cat([mlp3(w, p + ".offset_head.query_encoder", f_query),
                   mlp3(w, p + ".offset_head.match_encoder", f_match)], -1)
    logit = mlp3(w, p + ".offset_head.head", x).unflatten(-1, (2, 18))[..., :17]  # 16 bins + 1; last is unused
    bins = torch.arange(17, device=x.device, dtype=x.dtype)
    delta = ((logit.softmax(-1) * bins).sum(-1) / 16 - 0.5) * 2  # in cells, within +-1
    refined = coarse + torch.stack([delta[:, 0] * 2 / wd, delta[:, 1] * 2 / h], -1)
    if layer == 0:
        return refined, mass
    matchability = torch.sigmoid(linear(w, p + ".matchability", f_query)[:, 0])
    return refined, matchability * mass


# ---------------------------------------------------------------- sparse matching

def match(w, f0, xy0, size0, f1, xy1, size1, n_layers=9, filter_threshold=0.1, max_distance_px=3.0,
          min_covisibility=0.0, early_stop_decay=None, stages=None):
    """Mutual nearest neighbours between keypoints and predicted points.

    f0, f1: features (256, h, w); xy0 (N, 2), xy1 (M, 2): keypoints in image pixels; size0, size1: (w, h).
    Returns matches0 (N; index into xy1 or -1), scores0 (N), p2d0_i1 (N, 2) and p2d1_i0 (M, 2) in image pixels,
    confidence0, confidence1, covisibility, stop_layer.
    """
    dev, dt = f0.device, f0.dtype
    xy = [xy0.to(dev, dt), xy1.to(dev, dt)]
    size = [torch.tensor(size0, device=dev, dtype=dt), torch.tensor(size1, device=dev, dtype=dt)]
    x0, x1, layer, score = decode_pair(w, f0, f1, n_layers, min_covisibility, early_stop_decay, stages)
    cf = [chw(x0.unflatten(0, f0.shape[-2:])), chw(x1.unflatten(0, f1.shape[-2:]))]
    resized = [torch.tensor([8.0 * f.shape[-1], 8.0 * f.shape[-2]], device=dev, dtype=dt) for f in (f0, f1)]
    pred, conf = [], []
    for i in (0, 1):
        pts = xy[i] / size[i] * 2 - 1
        p, c = head(w, layer, pts, cf[i], cf[1 - i], stages, str(i))
        pred.append((p + 1) / 2 * size[1 - i])
        conf.append(c)
    # everything below is in resized pixels
    k = [xy[i] * resized[i] / size[i] for i in (0, 1)]
    q = [pred[0] * resized[1] / size[1], pred[1] * resized[0] / size[0]]  # q[0] lives in image 1
    inside = lambda pts, s: ((pts > 0) & (pts < s)).all(-1)  # noqa: E731
    valid = [inside(k[0], resized[0]) & inside(q[0], resized[1]), inside(k[1], resized[1]) & inside(q[1], resized[0])]
    if score < min_covisibility:
        valid = [torch.zeros_like(v) for v in valid]
    # distance of keypoint a (image 0) and keypoint b (image 1): mean of both transfer errors
    d2 = torch.cdist(q[0], k[1]).square() + torch.cdist(k[0], q[1]).square()  # argmin as upstream's sum of squares
    m0, m1 = d2.argmin(1), d2.argmin(0)
    mutual = m1[m0] == torch.arange(len(m0), device=dev)
    dist = 0.5 * ((q[0] - k[1][m0]).norm(dim=-1) + (k[0] - q[1][m0]).norm(dim=-1))
    scores = (conf[0] * conf[1][m0]).sqrt()
    ok = mutual & (scores >= filter_threshold) & valid[0] & valid[1][m0]
    if max_distance_px:
        ok &= dist < max_distance_px
    return {
        "matches0": torch.where(ok, m0, -1).cpu(), "scores0": torch.where(ok, scores, 0).float().cpu(),
        "p2d0_i1": pred[0].float().cpu(), "p2d1_i0": pred[1].float().cpu(),
        "confidence0": conf[0].float().cpu(), "confidence1": conf[1].float().cpu(),
        "covisibility": float(score), "stop_layer": layer,
    }
