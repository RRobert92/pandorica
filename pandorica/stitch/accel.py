#######################################################################
#  Serial Stitcher - An Automatic tool for tomograms stitching        #
#                                                                     #
#  https://github.com/RRobert92                                       #
#                                                                     #
#  Robert Kiewisz                                                     #
#  PolyForm Noncommercial License 1.0.0 - see LICENSE                 #
#######################################################################
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Robert Kiewisz

"""
GPU-accelerated volume warp (torch ``grid_sample``), with memory-capped Z-chunking.

The export's per-slice resampling is the heavy compute on a full-res stack. This
module runs it on the GPU (CUDA → MPS → CPU, auto) via ``grid_sample``, processing
**a few Z-slices at a time** so peak device memory is bounded (it never holds the
whole multi-GB stack). It reproduces the Z-varying blend of the CPU path:
the displacement at slice ``k`` interpolates the bottom field ``b_grid`` (low-Z)
and top field ``t_grid`` (high-Z); a uniform warp is just ``b_grid == t_grid``.

``align_corners=True`` + bilinear matches scipy ``map_coordinates(order=1,
mode='constant', cval=0)`` so GPU and CPU outputs agree (verified in tests).
"""

from typing import Optional, Tuple

import numpy as np

from pandorica.stitch.transform.solver import Pose, linear_part


def gpu_available() -> bool:
    """True if a non-CPU torch device (CUDA or MPS) is usable."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(torch.cuda.is_available() or torch.backends.mps.is_available())


def pick_device(prefer_gpu: bool = True) -> str:
    """Choose a torch device string: 'cuda' > 'mps' > 'cpu'."""
    if not prefer_gpu:
        return "cpu"
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _per_slice_bytes(out_hw: Tuple[int, int], in_hw: Tuple[int, int]) -> int:
    """
    Bytes per Z-slice that :func:`warp_volume_torch` allocates on the device.

    Per chunk it builds (all float32): the input slab ``vin`` (``4·Hin·Win``),
    the displacement ``disp`` (``8·Hc·Wc``), the source coords ``src`` (``8``),
    the sampling grid ``grid`` (``8``), the sampled output ``sampled`` (``4``)
    and the cast/clamp temp ``chunk_out`` (``4``). Sum per slice ≈
    ``32·Hc·Wc + 4·Hin·Win`` bytes. Persistent tensors (``out_t``, ``b_t``,
    ``t_t``: ~24·Hc·Wc) are amortised across chunks and a small constant
    relative to a multi-slice chunk, so they are not modelled here.
    """
    hc, wc = out_hw
    hin, win = in_hw
    return 32 * int(hc) * int(wc) + 4 * int(hin) * int(win)


def device_free_bytes(device: str) -> Optional[int]:
    """
    Free device memory in bytes, or ``None`` when the device has no query API.

    CUDA exposes :func:`torch.cuda.mem_get_info` (free, total). MPS has no
    free-memory API as of torch 2.12 — return ``None`` and let callers choose
    a conservative fallback.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    if device == "cuda" and torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            return int(free)
        except Exception:  # noqa: BLE001
            return None
    return None


def auto_gpu_chunk(
    device: str,
    out_hw: Tuple[int, int],
    in_hw: Tuple[int, int],
    safety_frac: float = 0.5,
    max_chunk: int = 64,
    min_chunk: int = 1,
    fallback: int = 4,
) -> int:
    """
    Pick a Z-slice chunk size that fits in available device memory.

    The export warp loop processes ``chunk`` slices at a time. Bigger chunks
    amortise H2D/D2H overhead but raise peak device memory. We size the chunk
    so peak per-chunk allocation stays under ``safety_frac`` of free device
    memory, clamped to ``[min_chunk, max_chunk]``. ``safety_frac`` defaults to
    0.5 so transient allocations elsewhere (other tensors, other processes)
    do not OOM the device.

    Returns ``fallback`` on:

    * CPU device (no GPU memory to budget against)
    * MPS (no free-memory API in torch — apply the fallback rather than guess)
    * any query failure

    :param device: ``'cuda'`` / ``'mps'`` / ``'cpu'``.
    :param out_hw: output canvas ``(Hc, Wc)``.
    :param in_hw: input section ``(Hin, Win)``.
    :param safety_frac: fraction of free device memory we allow ourselves.
    :param max_chunk: hard upper bound (returns diminish past ~64 on CUDA).
    :param min_chunk: floor — must be ≥ 1.
    :param fallback: used when no memory query is available for this device.
    """
    per_slice = _per_slice_bytes(out_hw, in_hw)
    if per_slice <= 0 or device == "cpu":
        return int(fallback)

    free_bytes = device_free_bytes(device)
    if free_bytes is None:
        return int(fallback)

    frac = max(0.05, min(0.95, float(safety_frac)))
    budget = int(free_bytes * frac)
    chunk = budget // per_slice
    chunk = max(int(min_chunk), min(int(max_chunk), int(chunk)))
    return chunk


def warp_volume_torch(
    volume,
    inv_pose: Pose,
    out_hw,
    out_pts: np.ndarray,
    b_grid: np.ndarray,
    t_grid: np.ndarray,
    device: Optional[str] = None,
    chunk: int = 4,
    dtype=np.uint8,
    vmax: float = 255.0,
    out: Optional[np.ndarray] = None,
):
    """
    Z-blend warp of a ``[Z, Y, X]`` volume on the GPU, ``chunk`` slices at a time.

    Mirrors :func:`.stitch._warp_volume_zblend`: slice ``k`` samples the input at
    ``inv_pose(out_pts − [α_k·b_grid + (1−α_k)·t_grid])`` with ``α`` = 1 at low-Z
    (k=0) → 0 at high-Z. Only ``chunk`` slices' grids/data live on the device at
    once, so memory ≈ ``chunk × Hc × Wc`` — set it down if a device runs tight.

    :param inv_pose: inverse of the section's absolute (pixel, canvas-offset) pose.
    :param out_pts / b_grid / t_grid: ``[M, 2]`` output ``(x, y)`` and bottom/top
        displacement fields (M = Hc·Wc).
    :param out: optional destination. An ``[Z, Hc, Wc]`` array written in place, OR
        a writable **binary file** (anything with ``.write``) — in which case each
        Z-chunk's bytes are streamed out in order (C-order, row-major) and nothing
        larger than one chunk is held in RAM (file writes go through the OS page
        cache, not process RSS). If ``None``, a host array is allocated and returned.
    :return: the warped array, or the file handle when streaming.
    """
    import torch
    import torch.nn.functional as F

    dev = torch.device(device or pick_device())
    z, h_in, w_in = volume.shape
    hc, wc = out_hw

    # apply_pose(inv_pose, p) = p @ L.T + t. Use the full 2x2 linear part so an
    # anisotropic / sheared inverse pose warps correctly; for an isotropic pose
    # L = Scale * R and this reduces to the old (Angle, Scale) construction.
    rt = torch.tensor(
        np.ascontiguousarray(linear_part(inv_pose).T), dtype=torch.float32, device=dev
    )  # L.T
    tt = torch.tensor([inv_pose["Tx"], inv_pose["Ty"]], dtype=torch.float32, device=dev)
    out_t = torch.as_tensor(np.asarray(out_pts, np.float32), device=dev)  # [M,2]
    b_t = torch.as_tensor(np.asarray(b_grid, np.float32), device=dev)
    t_t = torch.as_tensor(np.asarray(t_grid, np.float32), device=dev)

    stream = hasattr(out, "write")
    if out is None:
        out = np.empty((z, hc, wc), dtype=dtype)
    for z0 in range(0, z, chunk):
        z1 = min(z0 + chunk, z)
        ks = torch.arange(z0, z1, device=dev, dtype=torch.float32)
        alpha = 1.0 - ks / (z - 1) if z > 1 else torch.ones_like(ks)
        # disp[n, M, 2]; src = inv_pose(out - disp)
        disp = (
            alpha[:, None, None] * b_t[None] + (1.0 - alpha)[:, None, None] * t_t[None]
        )
        src = (out_t[None] - disp) @ rt + tt  # [n, M, 2] input (x, y)
        gx = 2.0 * src[..., 0] / max(w_in - 1, 1) - 1.0
        gy = 2.0 * src[..., 1] / max(h_in - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).reshape(z1 - z0, hc, wc, 2)
        vin = torch.as_tensor(
            np.asarray(volume[z0:z1], np.float32), device=dev
        ).unsqueeze(
            1
        )  # [n,1,Hin,Win]
        sampled = F.grid_sample(
            vin, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        chunk_out = sampled.squeeze(1).clamp(0.0, vmax)
        chunk_arr = chunk_out.to("cpu").numpy().astype(dtype)
        if stream:
            out.write(np.ascontiguousarray(chunk_arr).tobytes())
        else:
            out[z0:z1] = chunk_arr
        del disp, src, gx, gy, grid, vin, sampled, chunk_out, chunk_arr
    return out
