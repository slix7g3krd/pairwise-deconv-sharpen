"""图像去卷积锐化流水线。

针对题目列出的六类退化，本模块的对应修复为：

* :func:`calibrated_psf`：接收实拍标定图/标定 PSF，估计非高斯 PSF；
* :func:`mirror_pad`：使用镜像（symmetric mirror）延拓，避免补零暗环；
* :func:`overlap_window` / :func:`fft_block`：足够重叠 + 余弦窗融合，消除块缝；
* :func:`global_normalize`：所有分块共用同一组全局统计/增益；
* :func:`soft_clip`：对过冲做软拐点限幅，避免高光被硬截成死白；
* :func:`metrics` / :func:`validation`：与参考图做 MSE/RMSE/PSNR/SSIM 误差校验。

其余 ``audit``、``fallback``、``determinism``、``ringing_check``、
``noise_ratio``、``output`` 为流水线配套入口。标量输入也可直接调用，
便于契约测试和交互式检查。
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 常量与审计日志
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_SIZE = 64
DEFAULT_OVERLAP = 16
DEFAULT_REG = 1e-2
DEFAULT_SOFT_KNEE = 0.88
PSF_SIZE = 21
ERROR_RMSE_TOLERANCE = 0.03

_AUDIT_LOG: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# 基础转换工具
# ---------------------------------------------------------------------------

def _is_integer(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _as_float_image(x: Any) -> np.ndarray:
    """转换为 float64；uint8/uint16 按标准范围归一化。

    与流水线内部的安全转换不同，这里保留 NaN/Inf，方便 validation 发现问题。
    """
    if x is None:
        return np.zeros((1, 1), dtype=np.float64)
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)

    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float64) / 65535.0
    return arr.astype(np.float64, copy=False)


def _safe_float_image(x: Any) -> np.ndarray:
    arr = _as_float_image(x)
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)


def _gray2d(arr: np.ndarray) -> np.ndarray:
    """转为二维灰度图；0/1 维输入会被整理成可计算的二维形状。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[-1] >= 3:
            return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
        return a[..., 0]
    # 批图像：取第一张，保证指标函数不直接崩溃。
    return _gray2d(a[0])


def _spatial_shape(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim == 0:
        return 1, 1
    if arr.ndim == 1:
        return 1, int(arr.shape[0])
    return int(arr.shape[0]), int(arr.shape[1])


def _centroid(psf: np.ndarray) -> tuple[float, float]:
    p = np.asarray(psf, dtype=np.float64)
    if p.ndim != 2 or p.size == 0:
        return 0.0, 0.0
    total = float(p.sum())
    if not np.isfinite(total) or total <= 1e-15:
        return float((p.shape[0] - 1) / 2.0), float((p.shape[1] - 1) / 2.0)
    yy, xx = np.indices(p.shape)
    return float((yy * p).sum() / total), float((xx * p).sum() / total)


def _normalize_psf(psf: Any) -> np.ndarray:
    p = np.asarray(psf, dtype=np.float64)
    if p.ndim == 0:
        p = p.reshape(1, 1)
    if p.ndim != 2:
        g = _gray2d(p)
        p = np.asarray(g, dtype=np.float64)
    p = np.maximum(p, 0.0)
    total = float(p.sum())
    if not np.isfinite(total) or total <= 1e-15:
        p = _aberration_psf(min(PSF_SIZE, max(1, p.shape[0])))
        total = float(p.sum())
    return p / total


# ---------------------------------------------------------------------------
# 1. 实拍标定 PSF
# ---------------------------------------------------------------------------

def _aberration_psf(size: int = PSF_SIZE, seed: int = 0) -> np.ndarray:
    """无实拍标定时的非高斯像差核（仅作确定性回退/演示）。"""
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if size == 1:
        return np.ones((1, 1), dtype=np.float64)

    radius = size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    u = r / max(1.0, size / 4.0)

    # 像散、彗差和衍射环的简化模型；它不是固定高斯。
    astig = 1.0 + 0.18 * np.cos(2.0 * theta)
    coma = 1.0 + 0.10 * x / max(1.0, radius)
    airy = np.sinc(0.62 * u * astig) ** 2
    ring = 1.0 + 0.10 * np.exp(-((u - 1.75) / 0.22) ** 2)
    envelope = np.exp(-(u ** 2) / 9.0)
    psf = np.maximum(airy * ring * envelope * coma, 0.0)

    # 加入一个确定性的微小非对称扰动，避免误被当作理想对称高斯。
    rng = np.random.default_rng(int(seed))
    psf *= 1.0 + 0.015 * rng.standard_normal(psf.shape)
    psf = np.maximum(psf, 0.0)
    s = float(psf.sum())
    return psf / s if s > 1e-15 else np.ones((size, size)) / (size * size)


def _extract_centered_patch(spot: np.ndarray, cy: float, cx: float,
                            radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    padded = np.pad(spot, radius + 2, mode="symmetric")
    y = int(round(cy)) + radius + 2
    x = int(round(cx)) + radius + 2
    patch = padded[y - radius:y + radius + 1, x - radius:x + radius + 1]
    target = 2 * radius + 1
    if patch.shape != (target, target):
        out = np.zeros((target, target), dtype=np.float64)
        h = min(target, patch.shape[0])
        w = min(target, patch.shape[1])
        out[:h, :w] = patch[:h, :w]
        patch = out
    return patch


def _estimate_psf(calibration: np.ndarray, size: int = PSF_SIZE) -> np.ndarray:
    gray = _gray2d(_safe_float_image(calibration))
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    radius = size // 2

    border = np.concatenate([
        gray[0, :].ravel(), gray[-1, :].ravel(),
        gray[:, 0].ravel(), gray[:, -1].ravel()
    ]) if min(gray.shape) >= 2 else gray.ravel()
    background = float(np.nanmedian(border)) if border.size else 0.0
    spot = np.clip(gray - background, 0.0, None)

    # 若输入本身就是一个小 PSF 切片而不是标定场景，直接归一化使用。
    if float(spot.sum()) <= 1e-12:
        spot = np.clip(gray, 0.0, None)
    if float(spot.sum()) <= 1e-12:
        return _aberration_psf(size)

    cy, cx = _centroid(spot)
    patch = _extract_centered_patch(spot, cy, cx, radius)

    floor = float(np.percentile(patch, 15.0))
    patch = np.clip(patch - floor, 0.0, None)
    return _normalize_psf(patch)


def calibrated_psf(calibration: Any = None, size: int | None = None,
                   **kwargs: Any) -> dict[str, Any]:
    """生成或估计标定 PSF。

    用法：

    * ``calibrated_psf(spot_image)``：从实拍标定点/PSF 切片估计；
    * ``calibrated_psf(21)``：契约/标量调用，返回 21x21 回退像差核；
    * ``calibrated_psf(size=31)``：指定回退核尺寸。

    返回字典中的 ``psf`` 为能量归一化的二维 ``ndarray``。
    """
    if size is None and _is_integer(calibration):
        size = int(calibration)
        calibration = None
    if size is None:
        size = PSF_SIZE

    source = "synthetic_lens_aberration_fallback"
    warning: str | None = None
    measured: np.ndarray | None = None

    try:
        if isinstance(calibration, dict):
            for key in ("psf", "calibrated_psf", "spot", "image", "calibration"):
                if key in calibration:
                    calibration = calibration[key]
                    break
        if isinstance(calibration, (str, Path)):
            from PIL import Image
            calibration = np.asarray(Image.open(calibration).convert("L"))

        if calibration is not None and not _is_integer(calibration):
            arr = np.asarray(calibration)
            if arr.size > 1:
                measured = _estimate_psf(arr, size)
                source = "measured_calibration"
    except Exception as exc:  # 标定输入损坏时不能让整个锐化流程中断。
        warning = f"calibration_read_failed: {exc}"

    psf = measured if measured is not None else _aberration_psf(
        size, seed=int(kwargs.get("seed", 0))
    )
    psf = _normalize_psf(psf)
    cy, cx = _centroid(psf)
    radius = (max(psf.shape) - 1) / 2.0
    centroid_error = float(np.hypot(cy - (psf.shape[0] - 1) / 2.0,
                                    cx - (psf.shape[1] - 1) / 2.0))

    return {
        "psf": psf,
        "source": source,
        "measured": source == "measured_calibration",
        "normalized": True,
        "energy": float(psf.sum()),
        "shape": tuple(psf.shape),
        "centroid": (cy, cx),
        "centroid_error": centroid_error,
        "centered": centroid_error <= 0.75,
        "gaussian_assumption": False,
        "warning": warning,
        "support_radius": radius,
    }


# ---------------------------------------------------------------------------
# 2. 镜像边界延拓
# ---------------------------------------------------------------------------

def mirror_pad(image: Any, pad: Any = None, *, pad_y: int | None = None,
               pad_x: int | None = None, mode: str = "mirror") -> np.ndarray:
    """对图像做镜像延拓；HxWxC 图像只延拓空间轴，不延拓通道轴。"""
    arr = np.asarray(image)
    scalar_input = arr.ndim == 0
    if scalar_input:
        arr = np.atleast_1d(arr)

    if pad_y is None or pad_x is None:
        if pad is None:
            h, w = _spatial_shape(arr)
            default_pad = max(1, min(8, max(1, min(h, w)) // 4))
            pad_y = default_pad if pad_y is None else pad_y
            pad_x = default_pad if pad_x is None else pad_x
        elif _is_integer(pad):
            p = max(0, int(pad))
            if pad_y is None:
                pad_y = p
            if pad_x is None:
                pad_x = p
        elif isinstance(pad, (tuple, list)) and len(pad) == 2 and _is_integer(pad[0]):
            py, px = max(0, int(pad[0])), max(0, int(pad[1]))
            pad_y = py if pad_y is None else pad_y
            pad_x = px if pad_x is None else pad_x
        else:
            # 允许直接传入 NumPy 风格的完整 pad_width。
            np_mode = "symmetric" if mode in ("mirror", "symmetric") else mode
            return np.pad(arr, pad, mode=np_mode)

    py = max(0, int(pad_y or 0))
    px = max(0, int(pad_x or 0))
    np_mode = "symmetric" if mode in ("mirror", "symmetric") else mode

    if arr.ndim == 1:
        width = (px or py, px or py)
    elif arr.ndim == 2:
        width = ((py, py), (px, px))
    elif arr.ndim == 3:
        width = ((py, py), (px, px), (0, 0))
    else:
        width = ((py, py), (px, px)) + ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, width, mode=np_mode)


# ---------------------------------------------------------------------------
# 3. 重叠区余弦窗
# ---------------------------------------------------------------------------

def _ramp_1d(n: int, overlap: int) -> np.ndarray:
    n = max(1, int(n))
    w = np.ones(n, dtype=np.float64)
    max_o = 1 if n == 1 else n // 2
    o = max(0, min(int(overlap), max_o))
    if o == 0:
        return w
    for i in range(o):
        # sin^2 斜坡；同一段重叠中左右两块权重满足 L + R = 1。
        value = float(np.sin(np.pi * (i + 0.5) / (2.0 * o)) ** 2)
        w[i] = value
        w[n - 1 - i] = value
    return w


def overlap_window(size: Any, overlap: int | tuple[int, int] | None = None,
                   *, kind: str = "cosine") -> np.ndarray:
    """生成一维或二维重叠融合窗。

    ``size`` 为整数时返回一维窗；为 ``(H, W)``、图像数组时返回二维窗。
    """
    if isinstance(size, np.ndarray):
        h, w = _spatial_shape(size)
        if overlap is None:
            overlap = DEFAULT_OVERLAP
        if isinstance(overlap, (tuple, list)):
            oy, ox = int(overlap[0]), int(overlap[1])
        else:
            oy = ox = int(overlap)
        return _ramp_1d(h, oy)[:, None] * _ramp_1d(w, ox)[None, :]

    if isinstance(size, (tuple, list)):
        if len(size) == 1:
            return overlap_window(int(size[0]), overlap)
        h, w = int(size[0]), int(size[1])
        if overlap is None:
            ov = min(DEFAULT_OVERLAP, max(1, min(h, w) // 4))
            oy = ox = ov
        elif isinstance(overlap, (tuple, list)):
            oy, ox = int(overlap[0]), int(overlap[1])
        else:
            oy = ox = int(overlap)
        return _ramp_1d(h, oy)[:, None] * _ramp_1d(w, ox)[None, :]

    n = max(1, int(size))
    ov = DEFAULT_OVERLAP if overlap is None else int(overlap)
    if kind not in ("cosine", "hann", "sin2"):
        raise ValueError(f"unsupported window kind: {kind}")
    return _ramp_1d(n, ov)


def _starts(length: int, block: int, step: int) -> list[int]:
    block = min(max(1, int(block)), max(1, int(length)))
    step = max(1, int(step))
    starts = list(range(0, max(1, length - block + 1), step))
    last = max(0, length - block)
    if not starts:
        starts = [0]
    if starts[-1] < last:
        starts.append(last)
    return starts


def _resolve_overlap(overlap: int | None, psf: np.ndarray,
                     block_size: int) -> int:
    if overlap is not None:
        return max(0, int(overlap))
    radius = max(psf.shape) // 2
    return int(min(max(8, 2 * radius + 2), max(1, block_size // 2 - 1)))


def _psf_otf(psf: np.ndarray, shape: tuple[int, int], reg: float) -> np.ndarray:
    h, w = shape
    kernel = np.zeros((h, w), dtype=np.float64)
    p = psf
    if p.shape[0] > h or p.shape[1] > w:
        y0 = max(0, (p.shape[0] - h) // 2)
        x0 = max(0, (p.shape[1] - w) // 2)
        p = p[y0:y0 + min(h, p.shape[0]), x0:x0 + min(w, p.shape[1])]
    kernel[:p.shape[0], :p.shape[1]] = p
    kernel = np.roll(kernel, (-p.shape[0] // 2, -p.shape[1] // 2), axis=(0, 1))
    h_fft = np.fft.rfft2(kernel)
    return np.conj(h_fft) / (np.abs(h_fft) ** 2 + max(1e-12, float(reg)))


def _wiener_tile(tile: np.ndarray, otf: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft2(tile)
    restored = np.fft.irfft2(spectrum * otf, s=tile.shape)
    return np.real(restored)


def _global_gain(before: np.ndarray, after: np.ndarray) -> float:
    b = before[np.isfinite(before)]
    a = after[np.isfinite(after)]
    if b.size == 0 or a.size == 0:
        return 1.0
    mean_b = float(b.mean())
    mean_a = float(a.mean())
    if abs(mean_a) < 1e-10:
        energy_b = float(np.mean(b ** 2))
        energy_a = float(np.mean(a ** 2))
        return math.sqrt(max(0.0, energy_b) / max(1e-12, energy_a))
    gain = mean_b / mean_a
    if not np.isfinite(gain) or gain <= 0.0:
        return 1.0
    return float(np.clip(gain, 0.5, 2.0))


def _fft_block_gray(image: np.ndarray, psf: np.ndarray, block_size: int,
                    overlap: int | None, reg: float,
                    window_kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    g = _safe_float_image(image)
    if g.ndim == 0:
        g = g.reshape(1, 1)
    elif g.ndim == 1:
        g = g.reshape(1, -1)

    h, w = g.shape
    ov = _resolve_overlap(overlap, psf, int(block_size))
    padded = mirror_pad(g, ov, mode="symmetric")
    ph, pw = padded.shape

    bh = max(1, min(int(block_size), ph))
    bw = max(1, min(int(block_size), pw))
    step_y = max(1, bh - min(ov, bh // 2))
    step_x = max(1, bw - min(ov, bw // 2))
    ys = _starts(ph, bh, step_y)
    xs = _starts(pw, bw, step_x)

    accum = np.zeros((ph, pw), dtype=np.float64)
    weights = np.zeros((ph, pw), dtype=np.float64)
    otf_cache: dict[tuple[int, int], np.ndarray] = {}
    boxes: list[tuple[int, int, int, int]] = []

    for y in ys:
        for x in xs:
            tile = padded[y:y + bh, x:x + bw]
            th, tw = tile.shape
            if (th, tw) not in otf_cache:
                otf_cache[(th, tw)] = _psf_otf(psf, (th, tw), reg)
            restored = _wiener_tile(tile, otf_cache[(th, tw)])
            win = overlap_window((th, tw), min(ov, min(th, tw) // 2),
                                 kind=window_kind)
            accum[y:y + th, x:x + tw] += restored * win
            weights[y:y + th, x:x + tw] += win
            boxes.append((int(x), int(y), int(tw), int(th)))

    safe_weights = np.where(weights > 1e-10, weights, 1.0)
    merged_padded = accum / safe_weights
    merged_padded = np.where(weights > 1e-10, merged_padded, 0.0)
    merged = merged_padded[ov:ov + h, ov:ov + w]

    gain = _global_gain(g, merged)
    merged = merged * gain

    info = {
        "block_size": [bh, bw],
        "overlap": int(ov),
        "step": [step_y, step_x],
        "blocks": boxes,
        "block_count": len(boxes),
        "window": "cosine_sin2",
        "padding": "mirror",
        "global_gain": gain,
        "normalization": "global_mean",
    }
    return merged, info


def fft_block(image: Any = 1, psf: Any = None, *,
              block_size: int = DEFAULT_BLOCK_SIZE,
              overlap: int | None = None,
              reg: float = DEFAULT_REG,
              window_kind: str = "cosine",
              return_report: bool = True,
              **kwargs: Any) -> Any:
    """带镜像边界、重叠窗融合和全局增益的分块 FFT Wiener 反卷积。"""
    arr = _safe_float_image(image)
    psf_report: dict[str, Any] | None = None
    if psf is None:
        psf_report = calibrated_psf(min(PSF_SIZE, max(1, int(block_size) // 3)))
        kernel = psf_report["psf"]
    elif isinstance(psf, dict):
        psf_report = psf
        kernel = _normalize_psf(psf["psf"])
    else:
        kernel = _normalize_psf(psf)

    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        channels = []
        info: dict[str, Any] = {}
        for c in range(arr.shape[-1]):
            out_c, info = _fft_block_gray(
                arr[..., c], kernel, block_size, overlap, reg, window_kind
            )
            channels.append(out_c)
        out = np.stack(channels, axis=-1)
    else:
        out, info = _fft_block_gray(
            arr, kernel, block_size, overlap, reg, window_kind
        )

    if psf_report is not None:
        info["psf_source"] = psf_report.get("source")
    if not return_report:
        return out
    return {"image": out, **info}


# ---------------------------------------------------------------------------
# 4. 全局统一归一化
# ---------------------------------------------------------------------------

def global_normalize(blocks: Any = None, *, stats: dict[str, float] | None = None,
                     mode: str = "percentile",
                     percentile_low: float = 0.5,
                     percentile_high: float = 99.5,
                     target_mean: float | None = None,
                     **kwargs: Any) -> dict[str, Any]:
    """对所有分块使用同一组 min/max/scale，禁止逐块独立拉伸。"""
    if blocks is None:
        blocks = 1
    if isinstance(blocks, dict) and "blocks" in blocks:
        blocks = blocks["blocks"]
    if isinstance(blocks, np.ndarray):
        arrays = [blocks]
    elif isinstance(blocks, (list, tuple)):
        arrays = [np.asarray(b, dtype=np.float64) for b in blocks]
    else:
        arrays = [np.asarray(blocks, dtype=np.float64)]

    values = np.concatenate([np.asarray(a).ravel() for a in arrays if np.asarray(a).size])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.zeros(1, dtype=np.float64)

    if stats is not None:
        lo = float(stats.get("low", stats.get("min", 0.0)))
        hi = float(stats.get("high", stats.get("max", 1.0)))
        mean = float(stats.get("mean", float(finite.mean())))
    elif mode == "mean":
        mean = float(finite.mean())
        target = mean if target_mean is None else float(target_mean)
        gain = target / mean if abs(mean) > 1e-12 else 1.0
        normalized = [np.asarray(a, dtype=np.float64) * gain for a in arrays]
        return {
            "blocks": normalized,
            "coefficient": float(gain),
            "global_mean": mean,
            "scope": "global",
            "mode": mode,
            "per_block_coefficients": False,
        }
    else:
        lo = float(np.percentile(finite, percentile_low))
        hi = float(np.percentile(finite, percentile_high))
        mean = float(finite.mean())

    degenerate = hi - lo < 1e-12
    if degenerate:
        # 常量图没有可拉伸的动态范围；保持原值，避免所有块被统一改成 0。
        lo, hi = 0.0, 1.0
    scale = 1.0 / (hi - lo)
    normalized = [(np.asarray(a, dtype=np.float64) - lo) * scale for a in arrays]

    return {
        "blocks": normalized,
        "low": lo,
        "high": hi,
        "mean": mean,
        "scale": scale,
        "offset": -lo * scale,
        "coefficient": scale,
        "degenerate_range": degenerate,
        "scope": "global",
        "mode": mode,
        "per_block_coefficients": False,
    }


# ---------------------------------------------------------------------------
# 5. 软限幅
# ---------------------------------------------------------------------------

def soft_clip(image: Any = None, limit: float = 1.0,
              knee: float | None = None, *,
              ceiling: float | None = None,
              threshold: float | None = None,
              lower: float | None = None,
              lower_knee: float | None = None,
              **kwargs: Any) -> Any:
    """膝区以下直通，膝区以上指数滚降并平滑逼近 ``limit``。

    与 ``np.clip`` 的硬截断不同，有限过冲不会被压成同一条水平线，
    因此高光层次不会立刻变成死白。
    """
    if image is None:
        image = 1
    if ceiling is not None:
        limit = float(ceiling)
    limit = float(limit)
    if threshold is not None:
        knee = float(threshold)
    if knee is None:
        knee = DEFAULT_SOFT_KNEE * limit if limit > 0 else 0.0
    knee = float(knee)
    if limit > 0 and not (0.0 < knee < limit):
        knee = 0.8 * limit

    arr = np.asarray(image, dtype=np.float64)
    width = max(1e-12, limit - knee)
    z = np.clip(np.maximum(arr - knee, 0.0) / width, 0.0, 80.0)
    upper = knee + width * (1.0 - np.exp(-z))
    out = np.where(arr > knee, upper, arr)

    # 负值/下冲只在调用方显式要求下软下限；默认保持 0 和暗部直通，
    # 避免把纯黑抬成灰雾。最终输出阶段仍会做合法值域量化。
    if lower is not None:
        lower = float(lower)
        if lower_knee is None:
            lower_knee = lower + 0.2 * limit if limit > 0 else lower
        lk = float(lower_knee)
        low_width = max(1e-12, lk - lower)
        dz = np.clip(np.maximum(lk - arr, 0.0) / low_width, 0.0, 80.0)
        lower_curve = lk - low_width * (1.0 - np.exp(-dz))
        out = np.where(arr < lk, lower_curve, out)

    if np.ndim(image) == 0:
        return float(out.item())
    return out


# ---------------------------------------------------------------------------
# 误差/质量指标
# ---------------------------------------------------------------------------

def _align_pair(result: Any, reference: Any) -> tuple[np.ndarray, np.ndarray]:
    a = _as_float_image(result)
    b = _as_float_image(reference)
    a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
    b = np.nan_to_num(b, nan=0.0, posinf=1e6, neginf=-1e6)
    if a.shape == b.shape:
        return a, b
    try:
        return np.broadcast_arrays(a, b)
    except ValueError:
        shape = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
        if not shape:
            return a, b
        sl = tuple(slice(0, s) for s in shape)
        return a[sl], b[sl]


def _box_mean(a: np.ndarray, win: int = 7) -> np.ndarray:
    h, w = a.shape
    win = max(3, min(int(win), h, w))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return a
    r = win // 2
    padded = np.pad(a, r, mode="symmetric")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    total = (
        integral[win:, win:]
        - integral[:-win, win:]
        - integral[win:, :-win]
        + integral[:-win, :-win]
    )
    return total[:h, :w] / float(win * win)


def _ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    ga, gb = _gray2d(a), _gray2d(b)
    if ga.shape != gb.shape:
        ga, gb = _align_pair(ga, gb)
    h, w = ga.shape
    win = min(7, h, w)
    if win % 2 == 0:
        win -= 1
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    if win >= 3:
        ux = _box_mean(ga, win)
        uy = _box_mean(gb, win)
        uxx = _box_mean(ga * ga, win)
        uyy = _box_mean(gb * gb, win)
        uxy = _box_mean(ga * gb, win)
        vx = np.maximum(uxx - ux * ux, 0.0)
        vy = np.maximum(uyy - uy * uy, 0.0)
        cov = uxy - ux * uy
        ssim_map = (((2 * ux * uy + c1) * (2 * cov + c2))
                    / ((ux * ux + uy * uy + c1) * (vx + vy + c2)))
        return float(np.mean(ssim_map))

    ux, uy = float(ga.mean()), float(gb.mean())
    vx, vy = float(ga.var()), float(gb.var())
    cov = float(np.mean((ga - ux) * (gb - uy)))
    return float(((2 * ux * uy + c1) * (2 * cov + c2))
                 / ((ux * ux + uy * uy + c1) * (vx + vy + c2)))


def metrics(result: Any = 1, reference: Any = None, **kwargs: Any) -> dict[str, Any]:
    """计算输出统计；提供 reference 时增加 MSE/RMSE/MAE/PSNR/SSIM。"""
    arr = _as_float_image(result)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        finite = np.zeros(1)
    gray = _gray2d(arr)
    out: dict[str, Any] = {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "shape": tuple(arr.shape),
        "finite": bool(np.all(np.isfinite(arr))),
    }

    if reference is None:
        return out

    a, b = _align_pair(result, reference)
    diff = a - b
    mse = float(np.mean(diff ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0

    auto_range = max(1.0, float(np.nanmax(np.abs(b)))) if b.size else 1.0
    data_range = float(kwargs.get("data_range", 255.0 if auto_range > 1.5 else 1.0))
    if mse <= 1e-12:
        psnr = 99.0
    else:
        psnr = float(10.0 * math.log10((data_range ** 2) / mse))

    out.update({
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "psnr_db": psnr,
        "ssim": _ssim(a, b, data_range=max(1e-6, data_range)),
        "data_range": data_range,
        "within_rmse_tolerance": rmse <= float(
            kwargs.get("rmse_tolerance", ERROR_RMSE_TOLERANCE)
        ),
    })
    return out


# ---------------------------------------------------------------------------
# 6. 振铃与噪声
# ---------------------------------------------------------------------------

def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        n = m.copy()
        n[1:, :] |= m[:-1, :]
        n[:-1, :] |= m[1:, :]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n
    return m


def ringing_check(result: Any = 1, reference: Any = None, *,
                  threshold: float = 0.18,
                  edge_threshold: float = 0.08,
                  band: int = 2,
                  **kwargs: Any) -> dict[str, Any]:
    """检查强边缘附近的过冲/振铃强度。"""
    g = _gray2d(_safe_float_image(result))
    if min(g.shape) < 3 or g.size < 9:
        return {
            "score": 0.0, "ringing_index": 0.0, "overshoot": 0.0,
            "undershoot": 0.0, "edge_pixels": 0, "band_pixels": 0,
            "ok": True, "threshold": float(threshold),
        }

    if reference is not None:
        edge_source = _gray2d(_safe_float_image(reference))
        if edge_source.shape != g.shape:
            edge_source, g2 = _align_pair(edge_source, g)
            if g2.shape == edge_source.shape:
                g = g2
    else:
        edge_source = g

    gy, gx = np.gradient(edge_source)
    grad = np.hypot(gx, gy)
    strong = grad >= max(float(edge_threshold), float(np.quantile(grad, 0.90)))
    if not strong.any():
        strong = grad >= max(edge_threshold, float(grad.max()) * 0.35)

    wide = _dilate(strong, int(band))
    near = _dilate(strong, max(1, int(band) - 1))
    band_mask = wide & ~near
    if not band_mask.any():
        band_mask = wide

    baseline = _box_mean(g, win=max(5, 2 * int(band) + 3))
    residual = g - baseline
    band_values = residual[band_mask]
    band_rms = float(np.sqrt(np.mean(band_values ** 2))) if band_values.size else 0.0
    edge_contrast = float(grad[strong].mean()) if strong.any() else float(grad.mean())
    score = band_rms / max(edge_contrast, 0.05)

    flat = grad < max(0.02, float(np.quantile(grad, 0.35)))
    flat_rms = float(np.sqrt(np.mean(residual[flat] ** 2))) if flat.any() else 0.0
    ringing_index = band_rms / max(flat_rms, 1e-6)

    return {
        "score": float(score),
        "ringing_index": float(min(ringing_index, 999.0)),
        "band_rms": band_rms,
        "flat_rms": flat_rms,
        "overshoot": float(np.maximum(band_values, 0).max()) if band_values.size else 0.0,
        "undershoot": float(np.maximum(-band_values, 0).max()) if band_values.size else 0.0,
        "edge_pixels": int(strong.sum()),
        "band_pixels": int(band_mask.sum()),
        "threshold": float(threshold),
        "ok": bool(score <= threshold),
    }


def _noise_sigma(image: Any) -> float:
    g = _gray2d(_safe_float_image(image))
    h, w = g.shape
    if h < 3 or w < 3:
        return float(np.std(g))
    lap = (
        g[:-2, 1:-1] + g[2:, 1:-1]
        + g[1:-1, :-2] + g[1:-1, 2:]
        - 4.0 * g[1:-1, 1:-1]
    )
    count = max(1, lap.size)
    return float(abs(np.sqrt(np.pi / 2.0) / (6.0 * count) * np.sum(np.abs(lap))))


def noise_ratio(before: Any = 1, after: Any = None, **kwargs: Any) -> dict[str, Any]:
    """估计噪声标准差；给出 after 时返回噪声放大比和 SNR。"""
    sigma_before = _noise_sigma(before)
    mean_before = float(_gray2d(_safe_float_image(before)).mean())
    report: dict[str, Any] = {
        "noise_sigma": sigma_before,
        "snr": mean_before / max(sigma_before, 1e-8),
        "ratio": 1.0,
        "noise_gain": 1.0,
    }
    if after is None:
        return report

    sigma_after = _noise_sigma(after)
    mean_after = float(_gray2d(_safe_float_image(after)).mean())
    gain = sigma_after / max(sigma_before, 1e-8)
    report.update({
        "noise_sigma_before": sigma_before,
        "noise_sigma_after": sigma_after,
        "snr_before": mean_before / max(sigma_before, 1e-8),
        "snr_after": mean_after / max(sigma_after, 1e-8),
        "noise_gain": float(gain),
        "ratio": float(gain),
        "amplified": bool(gain > 1.25),
    })
    return report


# ---------------------------------------------------------------------------
# 校验、审计、回退、确定性
# ---------------------------------------------------------------------------

def _validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    block_size = int(cfg.get("block_size", DEFAULT_BLOCK_SIZE))
    overlap = int(cfg.get("overlap", DEFAULT_OVERLAP))
    if block_size <= 0:
        errors.append("block_size must be positive")
    if overlap < 0:
        errors.append("overlap must be non-negative")
    if block_size > 0 and overlap >= block_size:
        errors.append("overlap must be smaller than block_size")
    elif block_size > 0 and overlap * 2 > block_size:
        warnings.append("overlap larger than block_size/2 is unusual")

    psf = cfg.get("psf")
    if isinstance(psf, dict):
        psf = psf.get("psf")
    if psf is not None:
        p = np.asarray(psf, dtype=np.float64)
        if p.size == 0 or not np.all(np.isfinite(p)):
            errors.append("psf must be finite and non-empty")
        elif abs(float(p.sum()) - 1.0) > 1e-4:
            warnings.append("psf is not energy-normalized")
        radius = max(p.shape) // 2 if p.ndim == 2 else 0
        if overlap < radius:
            warnings.append("overlap is smaller than PSF radius; edge FFT artifacts may remain")

    padding = str(cfg.get("padding", "mirror"))
    if padding not in ("mirror", "symmetric", "reflect"):
        errors.append("padding must be mirror/symmetric/reflect, not zero padding")
    if cfg.get("normalization", "global") not in (
        "global", "global_mean", "global_minmax", "percentile"
    ):
        errors.append("normalization must be global")
    if cfg.get("clipping", "soft") == "hard":
        errors.append("hard clipping causes dead highlights; use soft clipping")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "kind": "config"}


def validation(value: Any = None, reference: Any = None, *,
               config: dict[str, Any] | None = None,
               rmse_tolerance: float = ERROR_RMSE_TOLERANCE,
               **kwargs: Any) -> dict[str, Any]:
    """校验配置或结果；结果可同时与参考图做误差校验。"""
    if config is not None:
        return _validate_config(config)
    if isinstance(value, dict) and any(
        k in value for k in ("block_size", "overlap", "padding", "normalization")
    ) and "image" not in value:
        return _validate_config(value)

    arr = _as_float_image(value)
    finite = bool(np.all(np.isfinite(arr))) if arr.size else True
    in_range = bool(np.all((arr >= -1e-8) & (arr <= 1.0 + 1e-8))) if arr.size else True
    errors: list[str] = []
    warnings: list[str] = []
    if not finite:
        errors.append("output contains NaN or Inf")
    if not in_range:
        warnings.append("output has values outside [0,1] before/without clipping")
    if arr.size == 0:
        errors.append("output is empty")

    report: dict[str, Any] = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "finite": finite,
        "in_range": in_range,
        "shape": tuple(arr.shape),
        "kind": "result",
    }
    if reference is not None:
        err = metrics(value, reference, rmse_tolerance=rmse_tolerance, **kwargs)
        report["metrics"] = err
        if not err.get("within_rmse_tolerance", False):
            report["ok"] = False
            errors.append(
                f"RMSE {err['rmse']:.6f} exceeds tolerance {rmse_tolerance:.6f}"
            )
    return report


def audit(event: Any = None, detail: Any = None, **kwargs: Any) -> dict[str, Any]:
    """记录一次关键流水线决策并返回该条审计记录。"""
    if isinstance(event, dict):
        record = dict(event)
        record.setdefault("event", "audit")
    else:
        record = {
            "event": "audit" if event is None else str(event),
            "detail": detail if isinstance(detail, dict) else {"value": detail},
        }
    record.setdefault("timestamp", time.time())
    record.setdefault("ok", True)
    record.update(kwargs)
    _AUDIT_LOG.append(record)
    return record


def get_audit_log() -> list[dict[str, Any]]:
    return list(_AUDIT_LOG)


def reset_audit_log() -> None:
    _AUDIT_LOG.clear()


def fallback(image: Any = 1, reason: Any = None, *,
              original: Any = None, error: BaseException | str | None = None,
              **kwargs: Any) -> dict[str, Any]:
    """反卷积失败时的安全路径：不做逆滤波，只做有限值整理和软限幅。"""
    source = original if original is not None else image
    reasons = []
    if reason is not None:
        reasons.append(str(reason))
    if error is not None:
        reasons.append(str(error))
    try:
        arr = _safe_float_image(source)
        out = soft_clip(arr)
        return {
            "ok": True,
            "used_fallback": True,
            "mode": "bypass_deconvolution_soft_clip",
            "image": out,
            "reasons": reasons,
        }
    except Exception as exc:  # pragma: no cover - 极端输入的最后防线
        return {
            "ok": False,
            "used_fallback": True,
            "mode": "black_frame",
            "image": np.zeros((1, 1), dtype=np.uint8),
            "reasons": reasons + [f"fallback_failed: {exc}"],
        }


def _stable_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
        except Exception:
            return False
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_stable_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_stable_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def determinism(fn: Any = None, *args: Any, seed: Any = 0,
                **kwargs: Any) -> bool:
    """同一输入运行两次，结果必须完全一致。

    支持 ``determinism(1)``（把整数当作自检随机种子）和
    ``determinism(func, arg1, arg2)``（校验指定函数两次调用一致）。
    """
    seed_value = int(seed) if _is_integer(seed) else 0
    if _is_integer(fn):
        seed_value = int(fn)
        fn = kwargs.pop("fn", None)
    elif fn is not None and not callable(fn):
        fn = kwargs.pop("fn", None)

    if callable(fn):
        first = fn(*args, **kwargs)
        second = fn(*args, **kwargs)
        return _stable_equal(first, second)

    rng = np.random.default_rng(seed_value)
    img = rng.random((24, 24), dtype=np.float64) * 0.18
    img[:, 12:] += 0.55
    yy, xx = np.indices(img.shape)
    img += 0.18 * (((yy - 8) ** 2 + (xx - 18) ** 2) <= 12).astype(float)
    img = np.clip(img, 0.0, 1.0)
    psf = _aberration_psf(9, seed=seed_value)

    first = deconvolve(
        img, psf=psf, block_size=16, overlap=6, reg=DEFAULT_REG,
        return_report=False, audit_log=False
    )
    second = deconvolve(
        img, psf=psf, block_size=16, overlap=6, reg=DEFAULT_REG,
        return_report=False, audit_log=False
    )
    return _stable_equal(first, second)


# ---------------------------------------------------------------------------
# 7. 完整去卷积入口与输出
# ---------------------------------------------------------------------------

def deconvolve(image: Any = 1, psf: Any = None, reference: Any = None, *,
               method: str = "wiener",
               block_size: int = DEFAULT_BLOCK_SIZE,
               overlap: int | None = None,
               reg: float = DEFAULT_REG,
               soft: bool = True,
               return_report: bool = True,
               audit_log: bool = True,
               **kwargs: Any) -> Any:
    """完整修复流水线：标定 PSF、镜像延拓、重叠 FFT、全局归一和软限幅。"""
    try:
        arr = _safe_float_image(image)
        if isinstance(psf, dict):
            psf_report = psf
            psf_source = psf.get("psf", psf.get("kernel", _aberration_psf(9)))
        else:
            psf_report = calibrated_psf(
                PSF_SIZE if min(_spatial_shape(arr)) >= PSF_SIZE else 9
            )
            psf_source = psf_report["psf"] if psf is None else psf
        kernel = _normalize_psf(psf_source)

        # 当前所有显式方法名都走稳定的正则化 FFT 路径；保留 method 字段
        # 是为了兼容后续接入 Richardson-Lucy 等迭代实现。
        block = fft_block(
            arr, kernel, block_size=block_size, overlap=overlap,
            reg=reg, return_report=True
        )
        raw = block.pop("image")
        block_info = block

        final = soft_clip(raw) if soft else raw
        err = metrics(final, reference) if reference is not None else None
        ring = ringing_check(final, reference)
        noise = noise_ratio(arr, final)
        check = validation(final, reference)

        report = {
            "ok": bool(check["ok"]),
            "image": final,
            "raw": raw,
            "method": f"{str(method).lower()}_regularized_fft_block",
            "psf": psf_report,
            "block_processing": block_info,
            "metrics": err,
            "ringing": ring,
            "noise": noise,
            "validation": check,
            "padding": "mirror",
            "window_fusion": "cosine",
            "normalization": "global",
            "clipping": "soft" if soft else "none",
        }
        if audit_log:
            audit("deconvolve", {
                "shape": _spatial_shape(arr),
                "block_count": block_info.get("block_count"),
                "rmse": None if err is None else err.get("rmse"),
                "ringing_ok": ring.get("ok"),
                "validation_ok": check.get("ok"),
            })
        if return_report:
            return report
        return final
    except Exception as exc:
        fb = fallback(image, reason="deconvolve_failed", error=exc)
        if audit_log:
            audit("deconvolve_fallback", {"error": str(exc)}, ok=False)
        if return_report:
            return {**fb, "ok": False, "deconvolution_ok": False,
                    "error": str(exc)}
        return fb["image"]


def output(image: Any = 1, path: str | Path | None = None, *,
           reference: Any = None, dtype: Any = np.uint8,
           meta: dict[str, Any] | None = None,
           **kwargs: Any) -> dict[str, Any]:
    """软限幅/量化输出；可选保存为 PNG/TIFF，并计算参考图误差。"""
    arr = _as_float_image(image)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = soft_clip(arr)
    arr = np.clip(arr, 0.0, 1.0)

    if dtype in ("uint8", np.uint8):
        out = np.round(arr * 255.0).astype(np.uint8)
        dtype_name = "uint8"
        max_value = 255
    elif dtype in ("uint16", np.uint16):
        out = np.round(arr * 65535.0).astype(np.uint16)
        dtype_name = "uint16"
        max_value = 65535
    else:
        out = arr.astype(dtype)
        dtype_name = str(np.dtype(dtype))
        max_value = 1

    result: dict[str, Any] = {
        "image": out,
        "dtype": dtype_name,
        "sha256": hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest(),
        "shape": tuple(out.shape),
        "soft_clipped": True,
    }
    if reference is not None:
        result["metrics"] = metrics(out.astype(np.float64) / max_value, reference)
    if meta is not None:
        result["meta"] = meta
    if path is not None:
        path = Path(path)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        img = out
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        Image.fromarray(img).save(path)
        result["path"] = str(path)
    return result
