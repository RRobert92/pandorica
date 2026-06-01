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
Cross-section filament chaining.

The matcher pairs MT stubs across each interface ``(section_k, section_k+1)``;
that pairing already drives the alignment. This module reuses the *same*
pairings to merge the per-section spline IDs of a single microtubule into one
global filament ID — so the exported spatial graph contains one connected
spline per MT, instead of one disjoint spline per section it crosses.

Design:

* Union-find over ``(section_idx, mt_id)`` nodes.
* For each interface ``k`` where ``InterfaceQC.accepted`` is True, union every
  ``(k, ref_mt_id)`` with ``(k+1, mov_mt_id)``.
* Flagged interfaces are *not* unioned — the chain breaks there. This keeps
  the global ID change visible exactly where the pipeline already says "I'm
  not confident at this joint."
* Components with no chain (singletons) still get their own global ID.
* Output: a ``dict`` mapping ``(section_idx, local_mt_id)`` to a contiguous
  ``global_filament_id`` in ``0 … n_filaments - 1``.

The matcher's Hungarian assignment guarantees ≤1 partner per stub at each
interface, so each chain is linear by construction.
"""

from typing import Dict, Iterable, List, Set, Tuple

import numpy as np


_Node = Tuple[int, int]
_Pair = Tuple[int, int, float]


class _UnionFind:
    """Minimal union-find with path compression + union-by-rank."""

    def __init__(self) -> None:
        self.parent: Dict[_Node, _Node] = {}
        self.rank: Dict[_Node, int] = {}

    def add(self, x: _Node) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: _Node) -> _Node:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: _Node, b: _Node) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def chain_filaments(
    sections_mt_ids: List[Iterable[int]],
    interface_id_pairs: List[List[_Pair]],
    interface_accepted: List[bool],
) -> Tuple[Dict[_Node, int], int]:
    """
    Build per-MT global filament IDs from per-interface match pairs.

    :param sections_mt_ids: ``[n_sections]``, each entry the set/list of local
        MT IDs present in that section.
    :param interface_id_pairs: ``[n_sections - 1]``, each entry the matcher's
        surviving ``(ref_mt_id, mov_mt_id, cost)`` for that interface.
    :param interface_accepted: ``[n_sections - 1]`` booleans (one per
        interface) — chains are *not* extended across an interface where the
        QC certificate did not accept. The cost field is currently unused
        (the matcher already gated by cost; chain time only enforces QC).
    :return: ``(id_map, n_filaments)`` where ``id_map[(section_idx, mt_id)] =
        global_filament_id`` is contiguous ``0 .. n_filaments - 1``.
    """
    n_sections = len(sections_mt_ids)
    if n_sections == 0:
        return {}, 0
    n_interfaces = max(0, n_sections - 1)
    if len(interface_id_pairs) != n_interfaces:
        raise ValueError(
            f"interface_id_pairs length {len(interface_id_pairs)} "
            f"does not match n_sections-1 = {n_interfaces}"
        )
    if len(interface_accepted) != n_interfaces:
        raise ValueError(
            f"interface_accepted length {len(interface_accepted)} "
            f"does not match n_sections-1 = {n_interfaces}"
        )

    uf = _UnionFind()
    # Seed nodes for every MT — singletons get their own component for free.
    for k, ids in enumerate(sections_mt_ids):
        for mt_id in ids:
            uf.add((k, int(mt_id)))

    # Union across accepted interfaces.
    for k in range(n_interfaces):
        if not interface_accepted[k]:
            continue
        section_k_ids: Set[int] = {int(i) for i in sections_mt_ids[k]}
        section_kp1_ids: Set[int] = {int(i) for i in sections_mt_ids[k + 1]}
        for ref_id, mov_id, _cost in interface_id_pairs[k]:
            ref_node = (k, int(ref_id))
            mov_node = (k + 1, int(mov_id))
            # Defensive: skip pairs whose endpoints reference IDs that aren't
            # in the section graphs (shouldn't happen, but keeps the API
            # robust to mismatched inputs).
            if int(ref_id) not in section_k_ids or int(mov_id) not in section_kp1_ids:
                continue
            uf.add(ref_node)
            uf.add(mov_node)
            uf.union(ref_node, mov_node)

    # Assign contiguous global IDs by stable order: section first, then mt_id.
    # Determinism matters for reproducible exports and snapshot tests.
    root_to_gid: Dict[_Node, int] = {}
    id_map: Dict[_Node, int] = {}
    next_gid = 0
    for k in range(n_sections):
        for mt_id in sorted(int(i) for i in sections_mt_ids[k]):
            node = (k, mt_id)
            root = uf.find(node)
            if root not in root_to_gid:
                root_to_gid[root] = next_gid
                next_gid += 1
            id_map[node] = root_to_gid[root]

    return id_map, next_gid


def orient_chain_blocks(coords: np.ndarray, section_idx: np.ndarray) -> np.ndarray:
    """
    Concatenate per-section sub-blocks of each chain in a continuous direction.

    The merge step yields a ``[N, ≥4]`` ``coords`` array — columns
    ``[global_id, x, y, z, ...]`` — where rows are stable-sorted by global id,
    and within each global id rows are grouped per section in section order.
    Each (gid, section_idx) sub-block is in the original trace order recorded
    in the spatial graph (which is along arc length, *not* Z — a bendy MT
    visits non-monotonic Z values, so sorting by Z would zigzag the spline).

    This function preserves the within-section trace order but reverses
    individual sub-blocks where needed so that the join between consecutive
    sections is continuous. For chains spanning two or more sections it also
    chooses the first sub-block's orientation by which endpoint is closest to
    the second sub-block's nearest endpoint — without this the first section's
    orientation is arbitrary and the chain could read "backwards".

    :param coords: ``[N, ≥4]`` array, columns ``[gid, x, y, z, ...]``,
        stable-sorted by ``gid`` with per-section sub-blocks in section order.
    :param section_idx: ``[N]`` int array, the section each row came from
        (aligned with ``coords``).
    :return: a new ``[N, ≥4]`` array with each sub-block possibly reversed.
        ``gid`` values are unchanged.
    """
    if coords.size == 0:
        return coords.copy()
    n = coords.shape[0]
    gid = coords[:, 0].astype(np.int64)
    sec = np.asarray(section_idx).astype(np.int64)
    out_blocks: List[np.ndarray] = []

    i = 0
    while i < n:
        g = gid[i]
        # Collect (start, end) for every (gid, section) sub-block in this gid.
        sub_runs: List[Tuple[int, int]] = []
        while i < n and gid[i] == g:
            s = i
            s_id = sec[i]
            while i < n and gid[i] == g and sec[i] == s_id:
                i += 1
            sub_runs.append((s, i))

        if len(sub_runs) == 1:
            s0, e0 = sub_runs[0]
            out_blocks.append(coords[s0:e0])
            continue

        # Orient sub-0 against sub-1's nearest endpoint: the chain's direction
        # is otherwise ambiguous (a single MT in section 0 can be traced
        # high→low or low→high; we don't know which face joins to section 1
        # until we look).
        s0, e0 = sub_runs[0]
        s1, e1 = sub_runs[1]
        a_head = coords[s0, 1:4]
        a_tail = coords[e0 - 1, 1:4]
        b_head = coords[s1, 1:4]
        b_tail = coords[e1 - 1, 1:4]
        d_tail = min(
            float(np.linalg.norm(a_tail - b_head)),
            float(np.linalg.norm(a_tail - b_tail)),
        )
        d_head = min(
            float(np.linalg.norm(a_head - b_head)),
            float(np.linalg.norm(a_head - b_tail)),
        )
        block0 = coords[s0:e0][::-1] if d_head < d_tail else coords[s0:e0]
        out_blocks.append(block0)
        running_tail = block0[-1, 1:4]

        for s, e in sub_runs[1:]:
            head = coords[s, 1:4]
            tail = coords[e - 1, 1:4]
            if float(np.linalg.norm(tail - running_tail)) < float(
                np.linalg.norm(head - running_tail)
            ):
                block = coords[s:e][::-1]
            else:
                block = coords[s:e]
            out_blocks.append(block)
            running_tail = block[-1, 1:4]

    return np.concatenate(out_blocks, axis=0)


def split_chains_at_joints(
    coords: np.ndarray,
    section_idx: np.ndarray,
    *,
    max_angle_deg: float = 45.0,
) -> np.ndarray:
    """
    Break chains at joints where the OVERALL direction on each side disagrees.

    The criterion is *not* the local bend at the joint. A microtubule whose two
    halves are laterally shifted (e.g. from imperfect section alignment) has a
    sharp local bend at the cut while the bulk direction of each half stays
    the same — we want to KEEP that chain. Conversely, two unrelated MTs that
    happened to be close at the cut have DIFFERENT bulk directions — we want
    to BREAK that chain.

    For each joint we therefore compute a chord through the whole sub-block on
    each side (start → end of that section's contribution) **in XY** (the
    user's viewing projection) and compare the two with a *signed* angle so a
    fold-back reads as ~180°, not as ~0°. Sub-blocks shorter than 2 points
    yield no signal — the joint is skipped.

    The caller is expected to have run :func:`orient_chain_blocks` first so
    each sub-block reads in a consistent direction.

    :param coords: ``[N, ≥4]`` array, columns ``[gid, x, y, z, ...]``,
        grouped by gid, with per-section sub-blocks oriented consistently.
    :param section_idx: ``[N]`` int, the section each row came from.
    :param max_angle_deg: max acceptable angle between the two sub-blocks'
        overall direction vectors. Higher = more permissive.
    :return: a new ``coords`` array with split chains assigned new gids (the
        old gid is preserved on the first segment, subsequent segments get
        fresh ids starting from ``max(gid) + 1``).
    """
    if coords.size == 0:
        return coords.copy()
    n = coords.shape[0]
    gid = coords[:, 0].astype(np.int64).copy()
    sec = np.asarray(section_idx).astype(np.int64)
    next_id = int(gid.max()) + 1
    cos_max = float(np.cos(np.deg2rad(max_angle_deg)))

    i = 0
    while i < n:
        chain_start = i
        g0 = gid[i]
        while i < n and gid[i] == g0:
            i += 1
        chain_end = i
        # Sub-block boundaries inside this chain.
        boundaries = [chain_start]
        for j in range(chain_start + 1, chain_end):
            if sec[j] != sec[j - 1]:
                boundaries.append(j)
        boundaries.append(chain_end)
        if len(boundaries) < 3:
            continue  # single sub-block — no joint to test

        for k in range(1, len(boundaries) - 1):
            joint = boundaries[k]
            pre_start, pre_end = boundaries[k - 1], boundaries[k]
            post_start, post_end = boundaries[k], boundaries[k + 1]
            if pre_end - pre_start < 2 or post_end - post_start < 2:
                continue
            # XY only (the projection the user judges) and SIGNED, so a
            # 160° fold-back reads as 160°, not 20°. orient_chain_blocks
            # has already ensured both chords flow in the chain's natural
            # direction, so cos == 1 means "coherent" and the angular
            # gate fires above ``max_angle_deg``.
            v_pre = coords[pre_end - 1, 1:3] - coords[pre_start, 1:3]
            v_post = coords[post_end - 1, 1:3] - coords[post_start, 1:3]
            n_pre = float(np.linalg.norm(v_pre))
            n_post = float(np.linalg.norm(v_post))
            if n_pre < 1e-8 or n_post < 1e-8:
                continue
            cos_a = float(np.dot(v_pre, v_post)) / (n_pre * n_post)
            if cos_a < cos_max:
                # Overall directions disagree: split. Every row from this
                # joint to the end of the (original) chain gets a fresh id;
                # later joints inside this chain that also fail will overwrite
                # the tail with another fresh id, producing one new gid per
                # surviving split point.
                gid[joint:chain_end] = next_id
                next_id += 1

    out = coords.copy()
    out[:, 0] = gid.astype(coords.dtype)
    return out


def compute_chain_labels(
    coords: np.ndarray,
    section_idx: np.ndarray,
    pre_split_gid: np.ndarray | None = None,
    local_window: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Per-filament + per-point diagnostic labels for the chained spatial graph.

    Two kinds of bend angle are computed at every joint:

    * **Local** (the angle the eye sees in napari) — chord through the last
      ``local_window`` points of the pre-block versus the chord through the
      first ``local_window`` points of the post-block. This captures the
      *visual* sharpness right at the cut. The primary metric (saved as
      ``MaxJointAngleDeg`` and ``JointAngleDeg``).
    * **Overall** (the "is it the same MT in bulk?" question) — chord
      through the whole pre-block versus the whole post-block. A real MT
      cut and laterally shifted has a sharp *local* angle (visible) but a
      coherent *overall* direction (still the same fiber). Saved as
      ``MaxJointOverallDeg`` and ``JointOverallDeg``.

    Returns a dict with:

    Per-point arrays (length ``N``):
      * ``point_section_idx`` (int32) — which input section this point came from.
      * ``point_at_joint`` (int32, 0/1) — 1 if this point is the last sample
        before, or the first sample after, a section boundary inside its chain.
      * ``point_joint_angle_deg`` (float32) — local joint angle; 0 elsewhere.
      * ``point_joint_overall_deg`` (float32) — overall joint angle; 0 elsewhere.

    Per-edge arrays (length ``n_filaments``, in array-order of first
    appearance — i.e. the order the writer emits edges):
      * ``edge_chain_length`` (int32) — number of sections this filament
        spans. ``1`` = a single-section MT (no chaining).
      * ``edge_n_joints`` (int32) — number of joints = chain_length − 1.
      * ``edge_max_joint_angle_deg`` (float32) — worst LOCAL joint angle.
      * ``edge_max_joint_overall_deg`` (float32) — worst OVERALL joint angle.
      * ``edge_was_split`` (int32, 0/1) — 1 if this filament's gid is one of
        the fresh ids assigned by :func:`split_chains_at_joints`. Only
        populated when ``pre_split_gid`` is given.

    :param coords: ``[N, ≥4]`` array, columns ``[gid, x, y, z, ...]``,
        sorted by gid, with sub-blocks oriented and (optionally) split.
    :param section_idx: ``[N]`` int, the section each row came from.
    :param pre_split_gid: optional ``[N]`` array of the gid each row had
        *before* :func:`split_chains_at_joints`. Used to compute
        ``edge_was_split``. Pass ``None`` to skip.
    :param local_window: number of points on each side of a joint used for
        the local-bend computation. Larger = smoother (averages over more
        tracing noise), smaller = more local. Default 5.
    """
    if coords.size == 0:
        return {
            "point_section_idx": np.zeros(0, dtype=np.int32),
            "point_at_joint": np.zeros(0, dtype=np.int32),
            "point_joint_angle_deg": np.zeros(0, dtype=np.float32),
            "point_joint_overall_deg": np.zeros(0, dtype=np.float32),
            "edge_chain_length": np.zeros(0, dtype=np.int32),
            "edge_n_joints": np.zeros(0, dtype=np.int32),
            "edge_max_joint_angle_deg": np.zeros(0, dtype=np.float32),
            "edge_max_joint_overall_deg": np.zeros(0, dtype=np.float32),
            "edge_was_split": np.zeros(0, dtype=np.int32),
        }
    n = coords.shape[0]
    gid = coords[:, 0].astype(np.int64)
    sec = np.asarray(section_idx).astype(np.int64)
    point_section_idx = sec.astype(np.int32)
    point_at_joint = np.zeros(n, dtype=np.int32)
    point_joint_angle_deg = np.zeros(n, dtype=np.float32)
    point_joint_overall_deg = np.zeros(n, dtype=np.float32)
    chain_length: List[int] = []
    n_joints: List[int] = []
    max_local: List[float] = []
    max_overall: List[float] = []
    was_split: List[int] = []

    # All angle calculations operate in XY only — that is the projection
    # the eye sees in napari when looking down the Z axis (the usual
    # viewing convention for serial-section EM). Including Z adds a
    # constant ~45-90° "cross-cut" component to every joint that swamps
    # the actual XY bend the user is judging. We also keep the SIGN of
    # cos so a 160° fold-back reads as 160°, not 20°.

    def _xy_turn_deg(v_in: np.ndarray, v_out: np.ndarray) -> float:
        """Turning angle (0..180°) at a polyline vertex, in XY only.

        ``v_in`` is the incoming chord (points TOWARD the vertex),
        ``v_out`` is the outgoing chord (points AWAY from the vertex).
        When both are along the same direction the path continues
        straight → 0°. Anti-parallel → fold-back, 180°. Right angle → 90°.
        """
        a = np.asarray(v_in)[:2]
        b = np.asarray(v_out)[:2]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        cos_a = float(np.dot(a, b)) / (na * nb)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def _overall_angle(
        p_start: int, p_end: int, q_start: int, q_end: int
    ) -> float:
        """Angle between the chord of pre-block and the chord of post-block.

        Both chords flow in the chain's natural direction
        (``orient_chain_blocks`` enforces this), so cos == 1 means a
        coherent chain. Reported as the deviation from straight.
        """
        if p_end - p_start < 2 or q_end - q_start < 2:
            return 0.0
        v_p = coords[p_end - 1, 1:4] - coords[p_start, 1:4]
        v_q = coords[q_end - 1, 1:4] - coords[q_start, 1:4]
        return _xy_turn_deg(v_p, v_q)

    def _local_joint_angle(
        pre_start: int, pre_end: int, post_start: int, post_end: int, window: int
    ) -> float:
        """Worst visual bend (XY) at the joint segment ``A → B``.

        Polyline around the cut::

            ... ── A_prev ── A ── B ── B_next ── ...

        Two corners can be sharp: at A (incoming vs joint segment) and
        at B (joint segment vs outgoing). We use chords through up to
        ``window`` points on each side, so single-sample tracing noise
        does not dominate. Each angle reads 0° for a straight pass and
        180° for a complete fold-back. The local joint angle is the
        worst of the two corners. When A and B coincide in XY (no
        visible jump in the projection), the joint segment is undefined
        and we compare incoming to outgoing directly.
        """
        if pre_end - pre_start < 2 or post_end - post_start < 2:
            return 0.0
        A = coords[pre_end - 1, 1:4]
        B = coords[post_start, 1:4]
        pre_back_idx = max(pre_start, pre_end - int(window))
        post_fwd_idx = min(post_end - 1, post_start + int(window) - 1)
        t_in = A - coords[pre_back_idx, 1:4]
        t_out = coords[post_fwd_idx, 1:4] - B
        t_joint = B - A
        if float(np.linalg.norm(np.asarray(t_joint)[:2])) < 1e-8:
            return _xy_turn_deg(t_in, t_out)
        return max(
            _xy_turn_deg(t_in, t_joint),
            _xy_turn_deg(t_joint, t_out),
        )

    i = 0
    while i < n:
        chain_start = i
        g0 = gid[i]
        while i < n and gid[i] == g0:
            i += 1
        chain_end = i
        boundaries = [chain_start]
        for j in range(chain_start + 1, chain_end):
            if sec[j] != sec[j - 1]:
                boundaries.append(j)
        boundaries.append(chain_end)
        n_subs = len(boundaries) - 1
        chain_length.append(int(n_subs))
        n_joints.append(int(max(0, n_subs - 1)))
        worst_local = 0.0
        worst_overall = 0.0

        for k in range(1, len(boundaries) - 1):
            pre_start, pre_end = boundaries[k - 1], boundaries[k]
            post_start, post_end = boundaries[k], boundaries[k + 1]
            ang_local = _local_joint_angle(
                pre_start, pre_end, post_start, post_end, int(local_window)
            )
            ang_overall = _overall_angle(
                pre_start, pre_end, post_start, post_end
            )
            # Mark both sides of the joint; record both metrics on the rows.
            point_at_joint[pre_end - 1] = 1
            point_at_joint[post_start] = 1
            point_joint_angle_deg[pre_end - 1] = ang_local
            point_joint_angle_deg[post_start] = ang_local
            point_joint_overall_deg[pre_end - 1] = ang_overall
            point_joint_overall_deg[post_start] = ang_overall
            if ang_local > worst_local:
                worst_local = ang_local
            if ang_overall > worst_overall:
                worst_overall = ang_overall

        max_local.append(worst_local)
        max_overall.append(worst_overall)

        if pre_split_gid is not None:
            current_gid = int(gid[chain_start])
            originated_pre_split = bool(
                np.any(np.asarray(pre_split_gid).astype(np.int64) == current_gid)
            )
            was_split.append(0 if originated_pre_split else 1)
        else:
            was_split.append(0)

    return {
        "point_section_idx": point_section_idx,
        "point_at_joint": point_at_joint,
        "point_joint_angle_deg": point_joint_angle_deg,
        "point_joint_overall_deg": point_joint_overall_deg,
        "edge_chain_length": np.array(chain_length, dtype=np.int32),
        "edge_n_joints": np.array(n_joints, dtype=np.int32),
        "edge_max_joint_angle_deg": np.array(max_local, dtype=np.float32),
        "edge_max_joint_overall_deg": np.array(max_overall, dtype=np.float32),
        "edge_was_split": np.array(was_split, dtype=np.int32),
    }
