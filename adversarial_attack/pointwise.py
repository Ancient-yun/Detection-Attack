"""PointWise: Pixel-level L0 adversarial attack for object detection.

Decision-based black-box attack that iteratively replaces pixels
to reduce L0 perturbation while maintaining attack success.

Adapted from classification-based PointWise to work with detection
models via the MMDetModelAdapter interface.
"""

import torch
import numpy as np
import random
from typing import Tuple

from .metrics import compute_l0


class PointWiseAtt:
    """PointWise Adversarial Attack.

    Starting from an adversarial image, iteratively tries to replace
    perturbed pixels with original values while maintaining adversarial
    status, then refines with binary search.

    Args:
        model: Model adapter with predict_label(x) → int interface.
        flag: If True, targeted attack; if False, untargeted attack.
        log_interval: Print progress every N queries.
    """

    def __init__(self, model, flag: bool = True, log_interval: int = 50,
                 verbose: bool = True):
        self.model = model
        self.flag = flag
        self.log_interval = log_interval
        self.verbose = verbose

    def _check_adv_status(
        self,
        img: np.ndarray,
        olabel: int,
        tlabel: int,
    ) -> bool:
        """Check if an image is adversarial (decision-based).

        Args:
            img: Image as numpy array, shape matching original.
            olabel: Original label.
            tlabel: Target label.

        Returns:
            True if adversarial.
        """
        pred_label = self.model.predict_label(
            torch.from_numpy(img).float().cuda()
        )
        if self.flag:
            return pred_label == tlabel
        else:
            return pred_label != olabel

    def _resolve_npix(self, npix, N: int) -> int:
        """Convert npix from ratio to absolute count if needed.

        Args:
            npix: If float < 1.0, treat as ratio of N.
                  Otherwise treat as absolute count.
            N: Total number of elements (oimg.size).

        Returns:
            Absolute count (int, >= 1).
        """
        if isinstance(npix, float) and npix < 1.0:
            return max(1, int(N * npix))
        return max(1, int(npix))

    def _binary_search(
        self,
        x: np.ndarray,
        index,
        adv_value,
        non_adv_value,
        shape: tuple,
        olabel: int,
        tlabel: int,
        n_steps: int = 10,
    ) -> Tuple[np.ndarray, int]:
        """Binary search to find the boundary between adv and non-adv.

        Args:
            x: Flattened image array.
            index: Pixel index or indices to search.
            adv_value: Current adversarial value.
            non_adv_value: Original (non-adversarial) value.
            shape: Original image shape for reshaping.
            olabel: Original label.
            tlabel: Target label.
            n_steps: Number of binary search steps.

        Returns:
            Tuple of (best adversarial value, number of queries used).
        """
        n_queries = 0
        for _ in range(n_steps):
            mid_value = (adv_value + non_adv_value) / 2
            x[index] = mid_value
            n_queries += 1
            if self._check_adv_status(x.reshape(shape), olabel, tlabel):
                adv_value = mid_value
            else:
                non_adv_value = mid_value
        return adv_value, n_queries

    def pw_perturb(
        self,
        oimg: np.ndarray,
        timg: np.ndarray,
        olabel: int,
        tlabel: int,
        max_query: int = 1000,
        snapshot_interval: int = 200,
    ) -> Tuple[np.ndarray, int, np.ndarray, dict]:
        """Run single-pixel PointWise attack.

        Phase 1: Try replacing each perturbed pixel with original value.
        Phase 2: Refine remaining pixels with binary search.

        Args:
            oimg: Original image (numpy array).
            timg: Starting adversarial image (numpy array).
            olabel: Original label.
            tlabel: Target label.
            max_query: Maximum number of model queries.
            snapshot_interval: Queries between snapshots.

        Returns:
            Tuple of:
                - Adversarial image (flattened numpy array)
                - Number of queries used
                - L0 distance trace array
                - Snapshots dictionary
        """
        shape = oimg.shape
        N = oimg.size
        start_qry = 0
        end_qry = 0

        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        n_queries = 0
        D = np.zeros(max_query + 500, dtype=int)
        d = compute_l0(
            torch.from_numpy(oimg).cuda(),
            torch.from_numpy(x.reshape(shape)).cuda(),
        )
        snapshots = {}
        next_snap_query = snapshot_interval

        # === Phase 1: Greedy pixel replacement ===
        terminate = False
        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)

            for index in indices:
                old_value = x[index]
                new_value = original[index]
                if old_value == new_value:
                    continue

                x[index] = new_value
                n_queries += 1
                is_adv = self._check_adv_status(
                    x.reshape(shape), olabel, tlabel
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = n_queries
                    D[start_qry:end_qry] = d
                    d = compute_l0(
                        torch.from_numpy(oimg).cuda(),
                        torch.from_numpy(x.reshape(shape)).cuda(),
                    )
                    if self.verbose and n_queries % self.log_interval == 0:
                        print(
                            f"[PW Phase1] query={n_queries}/{max_query}, L0={d}"
                        )
                else:
                    x[index] = old_value

                if n_queries >= next_snap_query:
                    snapshots[next_snap_query] = torch.from_numpy(x.reshape(shape)).clone().float().cuda()
                    next_snap_query += snapshot_interval

                if n_queries >= max_query:
                    terminate = True
                    break
            else:
                # No pixel replacement was successful
                terminate = True

        # === Phase 2: Binary search refinement ===
        terminate = n_queries >= max_query

        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)
            improved = False

            for index in indices:
                old_value = x[index]
                original_value = original[index]
                if old_value == original_value:
                    continue

                x[index] = original_value
                n_queries += 1
                is_adv = self._check_adv_status(
                    x.reshape(shape), olabel, tlabel
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = n_queries
                    D[start_qry:end_qry] = d
                    d = compute_l0(
                        torch.from_numpy(oimg).cuda(),
                        torch.from_numpy(x.reshape(shape)).cuda(),
                    )
                    improved = True
                else:
                    # Binary search for optimal value
                    best_adv_value, nqry = self._binary_search(
                        x, index, old_value, original_value,
                        shape, olabel, tlabel,
                        n_steps=min(10, max_query - n_queries),
                    )
                    n_queries += nqry

                    if old_value != best_adv_value:
                        x[index] = best_adv_value
                        improved = True
                        start_qry = end_qry
                        end_qry = n_queries
                        D[start_qry:end_qry] = d
                        d = compute_l0(
                            torch.from_numpy(oimg).cuda(),
                            torch.from_numpy(x.reshape(shape)).cuda(),
                        )
                        if n_queries % self.log_interval == 0:
                            print(
                                f"[PW Phase2] query={n_queries}/{max_query}, L0={d}, "
                                f"index={index}"
                            )
                    else:
                        x[index] = old_value

                if n_queries >= next_snap_query:
                    snapshots[next_snap_query] = torch.from_numpy(x.reshape(shape)).clone().float().cuda()
                    next_snap_query += snapshot_interval

                if n_queries >= max_query:
                    terminate = True
                    break

            if not improved:
                terminate = True

        # Final L0 measurement
        d = compute_l0(
            torch.from_numpy(oimg).cuda(),
            torch.from_numpy(x.reshape(shape)).cuda(),
        )
        D[end_qry:n_queries] = d

        return x, n_queries, D[:n_queries], snapshots

    def pw_perturb_multiple(
        self,
        oimg: np.ndarray,
        timg: np.ndarray,
        olabel: int,
        tlabel: int,
        npix: float = 196,
        max_query: int = 1000,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Run multi-pixel PointWise attack (group-based).

        Same as pw_perturb but processes pixels in groups for efficiency.

        Args:
            oimg: Original image (numpy array).
            timg: Starting adversarial image (numpy array).
            olabel: Original label.
            tlabel: Target label.
            npix: Number of pixels per group. If float < 1.0, ratio of total.
            max_query: Maximum number of model queries.

        Returns:
            Tuple of (adversarial_image, n_queries, l0_trace).
        """
        shape = oimg.shape
        N = oimg.size
        npix = self._resolve_npix(npix, N)
        start_qry = 0
        end_qry = 0

        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        n_queries = 0
        D = np.zeros(max_query + 500, dtype=int)
        d = compute_l0(
            torch.from_numpy(oimg).cuda(),
            torch.from_numpy(x.reshape(shape)).cuda(),
        )
        n_groups = N // npix
        snapshots = {}
        next_snap_query = snapshot_interval

        if self.verbose:
            print(f"[PW-Multi] N={N}, npix={npix}, n_groups={n_groups}")

        # === Phase 1: Group replacement ===
        terminate = False
        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)

            for group_idx in range(n_groups):
                idx = indices[group_idx * npix:(group_idx + 1) * npix]
                old_value = x[idx].copy()
                new_value = original[idx]
                if np.abs(old_value - new_value).sum() == 0:
                    continue

                x[idx] = new_value
                n_queries += 1
                is_adv = self._check_adv_status(
                    x.reshape(shape), olabel, tlabel
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = n_queries
                    D[start_qry:end_qry] = d
                    d = compute_l0(
                        torch.from_numpy(oimg).cuda(),
                        torch.from_numpy(x.reshape(shape)).cuda(),
                    )
                    if self.verbose and n_queries % self.log_interval == 0:
                        print(
                            f"[PW-Multi Phase1] query={n_queries}/{max_query}, L0={d}"
                        )
                else:
                    x[idx] = old_value

                if n_queries >= max_query:
                    terminate = True
                    break
            else:
                terminate = True

        # === Phase 2: Group refinement with binary search ===
        terminate = n_queries >= max_query

        if self.verbose:
            print("[PW-Multi] Entering refinement stage")
        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)
            improved = False

            for group_idx in range(n_groups):
                idx = indices[group_idx * npix:(group_idx + 1) * npix]
                old_value = x[idx].copy()
                original_value = original[idx]
                if np.abs(old_value - original_value).sum() == 0:
                    continue

                x[idx] = original_value
                n_queries += 1
                is_adv = self._check_adv_status(
                    x.reshape(shape), olabel, tlabel
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = n_queries
                    D[start_qry:end_qry] = d
                    d = compute_l0(
                        torch.from_numpy(oimg).cuda(),
                        torch.from_numpy(x.reshape(shape)).cuda(),
                    )
                    improved = True
                else:
                    best_adv_value, nqry = self._binary_search(
                        x, idx, old_value, original_value,
                        shape, olabel, tlabel,
                        n_steps=min(10, max_query - n_queries),
                    )
                    n_queries += nqry
                    diff = old_value - best_adv_value
                    if diff.sum() != 0:
                        x[idx] = best_adv_value
                        improved = True
                        start_qry = end_qry
                        end_qry = n_queries
                        D[start_qry:end_qry] = d
                        d = compute_l0(
                            torch.from_numpy(oimg).cuda(),
                            torch.from_numpy(x.reshape(shape)).cuda(),
                        )
                        if n_queries % self.log_interval == 0:
                            print(
                                f"[PW-Multi Phase2] query={n_queries}/{max_query}, "
                                f"L0={d}"
                            )
                    else:
                        x[idx] = old_value

                if n_queries >= max_query:
                    terminate = True
                    break

            if not improved:
                terminate = True

        # Final L0
        d = compute_l0(
            torch.from_numpy(oimg).cuda(),
            torch.from_numpy(x.reshape(shape)).cuda(),
        )
        D[end_qry:n_queries] = d

        return x, n_queries, D[:n_queries], snapshots

    def pw_perturb_multiple_scheduling(
        self,
        oimg: np.ndarray,
        timg: np.ndarray,
        olabel: int,
        tlabel: int,
        npix: float = 196,
        max_query: int = 1000,
        snapshot_interval: int = 200,
    ) -> Tuple[np.ndarray, int, np.ndarray, dict]:
        """Run multi-pixel PointWise attack with coarse-to-fine scheduling.

        Same as pw_perturb_multiple but halves npix after each Phase 1 pass,
        progressively refining from large groups to single pixels.

        Args:
            oimg: Original image (numpy array).
            timg: Starting adversarial image (numpy array).
            olabel: Original label.
            tlabel: Target label.
            npix: Initial number of pixels per group. If float < 1.0, ratio of total.
            max_query: Maximum number of model queries.
            snapshot_interval: Queries between snapshots.

        Returns:
            Tuple of (adversarial_image, n_queries, l0_trace, snapshots).
        """
        shape = oimg.shape
        N = oimg.size
        npix = self._resolve_npix(npix, N)
        start_qry = 0
        end_qry = 0

        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        n_queries = 0
        D = np.zeros(max_query + 500, dtype=int)
        d = compute_l0(
            torch.from_numpy(oimg).cuda(),
            torch.from_numpy(x.reshape(shape)).cuda(),
        )
        snapshots = {}
        next_snap_query = snapshot_interval

        if self.verbose:
            print(f"[PW-Sched] N={N}, npix={npix}")

        # === Phase 1: Group replacement with scheduling ===
        # npix halves after each full pass (coarse -> fine)
        terminate = False
        while not terminate:
            n_groups = N // npix
            indices = list(range(N))
            random.shuffle(indices)

            for group_idx in range(n_groups):
                idx = indices[group_idx * npix:(group_idx + 1) * npix]
                old_value = x[idx].copy()
                new_value = original[idx]
                if np.abs(old_value - new_value).sum() == 0:
                    continue

                x[idx] = new_value
                n_queries += 1
                is_adv = self._check_adv_status(
                    x.reshape(shape), olabel, tlabel
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = n_queries
                    D[start_qry:end_qry] = d
                    d = compute_l0(
                        torch.from_numpy(oimg).cuda(),
                        torch.from_numpy(x.reshape(shape)).cuda(),
                    )
                    if self.verbose and n_queries % self.log_interval == 0:
                        print(
                            f"[PW-Sched Phase1] query={n_queries}/{max_query}, "
                            f"L0={d}, npix={npix}"
                        )
                else:
                    x[idx] = old_value

                if n_queries >= next_snap_query:
                    snapshots[next_snap_query] = torch.from_numpy(x.reshape(shape)).clone().float().cuda()
                    next_snap_query += snapshot_interval

                if n_queries >= max_query:
                    terminate = True
                    break
            else:
                # Full pass completed without breaking -> halve or stop
                terminate = True

            if npix >= 2:
                npix //= 2

        # === Phase 2: Group refinement with binary search ===
        # Uses the final (smallest) npix from scheduling
        n_groups = N // max(npix, 1)
        terminate = n_queries >= max_query

        if self.verbose:
            print(f"[PW-Sched] Entering refinement stage (npix={npix})")
        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)
            improved = False

            for group_idx in range(n_groups):
                idx = indices[group_idx * npix:(group_idx + 1) * npix]
                old_value = x[idx].copy()
                original_value = original[idx]
                if np.abs(old_value - original_value).sum() == 0:
                    continue

                x[idx] = original_value
                n_queries += 1
                is_adv = self._check_adv_status(
                    x.reshape(shape), olabel, tlabel
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = n_queries
                    D[start_qry:end_qry] = d
                    d = compute_l0(
                        torch.from_numpy(oimg).cuda(),
                        torch.from_numpy(x.reshape(shape)).cuda(),
                    )
                    improved = True
                else:
                    best_adv_value, nqry = self._binary_search(
                        x, idx, old_value, original_value,
                        shape, olabel, tlabel,
                        n_steps=min(10, max_query - n_queries),
                    )
                    n_queries += nqry
                    diff = old_value - best_adv_value
                    if diff.sum() != 0:
                        x[idx] = best_adv_value
                        improved = True
                        start_qry = end_qry
                        end_qry = n_queries
                        D[start_qry:end_qry] = d
                        d = compute_l0(
                            torch.from_numpy(oimg).cuda(),
                            torch.from_numpy(x.reshape(shape)).cuda(),
                        )
                        if n_queries % self.log_interval == 0:
                            print(
                                f"[PW-Sched Phase2] query={n_queries}/{max_query}, "
                                f"L0={d}"
                            )
                    else:
                        x[idx] = old_value

                if n_queries >= next_snap_query:
                    snapshots[next_snap_query] = torch.from_numpy(x.reshape(shape)).clone().float().cuda()
                    next_snap_query += snapshot_interval

                if n_queries >= max_query:
                    terminate = True
                    break

            if not improved:
                terminate = True

        # Final L0
        d = compute_l0(
            torch.from_numpy(oimg).cuda(),
            torch.from_numpy(x.reshape(shape)).cuda(),
        )
        D[end_qry:n_queries] = d

        return x, n_queries, D[:n_queries], snapshots
