"""SparseEvo: Evolutionary L0 adversarial attack for object detection.

Decision-based black-box attack that uses differential evolution to
find sparse perturbations that disrupt object detections.

Adapted from classification-based SparseEvo to work with detection
models via the MMDetModelAdapter interface.
"""

import torch
import numpy as np
from typing import Tuple, Optional

from .metrics import compute_l0


class SpaEvoAtt:
    """Sparse Evolutionary Attack.

    Uses differential evolution to optimize a binary mask that determines
    which pixels from a target (adversarial starting point) image to keep,
    minimizing L0 perturbation while maintaining attack success.

    Args:
        model: Model adapter with predict_label(x) → int interface.
        n: Number of pixels for initial random perturbation.
        pop_size: Population size for evolution.
        cr: Crossover rate.
        mu: Mutation rate.
        seed: Random seed for reproducibility.
        flag: If True, targeted attack; if False, untargeted attack.
        log_interval: Print progress every N queries.
    """

    def __init__(
        self,
        model,
        n: int = 4,
        pop_size: int = 10,
        cr: float = 0.9,
        mu: float = 0.01,
        seed: Optional[int] = None,
        flag: bool = True,
        log_interval: int = 50,
    ):
        self.model = model
        self.n_pix = n
        self.pop_size = pop_size
        self.cr = cr
        self.mu = mu
        self.seed = seed
        self.flag = flag
        self.log_interval = log_interval
        self.verbose = True

    def _convert_1d_to_2d(self, idx: int, width: int) -> Tuple[int, int]:
        """Convert 1D pixel index to 2D (row, col) coordinates."""
        row = idx // width
        col = idx - row * width
        return row, col

    def _convert_2d_to_1d(self, x: int, y: int, width: int) -> int:
        """Convert 2D (row, col) coordinates to 1D pixel index."""
        return x * width + y

    def _compute_mask(
        self, oimg: torch.Tensor, timg: torch.Tensor
    ) -> np.ndarray:
        """Find pixel indices where oimg and timg differ.

        Args:
            oimg: Original image [1, C, H, W].
            timg: Target (starting point) image [1, C, H, W].

        Returns:
            1D array of pixel indices that differ.
        """
        diff = torch.abs(oimg - timg)
        pixel_diff = torch.zeros(diff.shape[2], diff.shape[3]).bool().cuda()
        for c in range(diff.shape[1]):
            pixel_diff = pixel_diff | (diff[0, c] > 0.0).bool().cuda()

        w = oimg.shape[3]  # W dimension
        coords = np.where(pixel_diff.int().cpu().numpy() == 1)
        indices = self._convert_2d_to_1d(coords[0], coords[1], w)
        return indices

    def _init_population(
        self,
        oimg: torch.Tensor,
        timg: torch.Tensor,
        olabel: int,
        tlabel: int,
    ) -> Tuple[list, int, torch.Tensor]:
        """Initialize population using uniform random selection.

        Each individual is a binary mask: 1 = keep timg pixel, 0 = use oimg pixel.

        Returns:
            Tuple of (population list, query count, fitness tensor).
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        h = oimg.shape[2]
        w = oimg.shape[3]
        n_queries = 0

        fitness = np.full(self.pop_size, np.inf)
        population = []

        # Base mask: 1 where timg differs from oimg
        base_mask = np.zeros(h * w, dtype=int)
        diff_indices = self._compute_mask(oimg, timg)
        base_mask[diff_indices] = 1

        n_pix = min(self.n_pix, base_mask.sum())

        for i in range(self.pop_size):
            n = n_pix
            j = 0
            while True:
                mask = base_mask.copy()
                idx = np.random.choice(diff_indices, n, replace=False)
                mask[idx] = 0
                n_queries += 1
                fit = self._evaluate_fitness(mask, oimg, timg, olabel, tlabel)

                if fit < fitness[i]:
                    population.append(mask)
                    fitness[i] = fit
                    break
                elif n > 1:
                    n -= 1
                elif n == 1:
                    while j < len(diff_indices):
                        mask[diff_indices[j]] = 0
                        n_queries += 1
                        fit = self._evaluate_fitness(
                            mask, oimg, timg, olabel, tlabel
                        )
                        if fit < fitness[i]:
                            population.append(mask)
                            fitness[i] = fit
                            break
                        else:
                            j += 1
                    break

            if j == len(diff_indices) - 1:
                break

        # Fill remaining population slots with base mask
        while len(population) < self.pop_size:
            population.append(base_mask.copy())

        return population, n_queries, fitness

    def _crossover(
        self, p_best: np.ndarray, p1: np.ndarray, p2: np.ndarray
    ) -> np.ndarray:
        """Differential evolution crossover (recombination).

        Args:
            p_best: Best individual's mask.
            p1: Parent 1 mask.
            p2: Parent 2 mask.

        Returns:
            Offspring mask.
        """
        cross_points = np.random.rand(len(p1)) < self.cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, len(p1))] = True
        trial = np.where(cross_points, p1, p2).astype(int)
        trial = np.logical_and(p_best, trial).astype(int)
        return trial

    def _mutate(self, mask: np.ndarray) -> np.ndarray:
        """Mutation: randomly flip some '1' bits to '0'.

        Args:
            mask: Binary mask to mutate.

        Returns:
            Mutated mask.
        """
        result = mask.copy()
        if result.sum() == 0:
            return result

        ones = np.where(result == 1)[0]
        n_flip = max(1, int(len(ones) * self.mu))
        idx = np.random.choice(ones, n_flip, replace=False)
        result[idx] = 0
        return result

    def _apply_mask(
        self,
        mask: np.ndarray,
        oimg: torch.Tensor,
        timg: torch.Tensor,
    ) -> torch.Tensor:
        """Apply binary mask to create perturbed image.

        mask=1 → use timg pixel, mask=0 → use oimg pixel.

        Args:
            mask: Binary mask of shape [H*W].
            oimg: Original image [1, C, H, W].
            timg: Target image [1, C, H, W].

        Returns:
            Perturbed image [1, C, H, W].
        """
        w = oimg.shape[3]  # W dimension
        img = timg.clone()
        zero_indices = np.where(mask == 0)[0]
        rows, cols = self._convert_1d_to_2d(zero_indices, w)
        img[:, :, rows, cols] = oimg[:, :, rows, cols]
        return img

    def _evaluate_fitness(
        self,
        mask: np.ndarray,
        oimg: torch.Tensor,
        timg: torch.Tensor,
        olabel: int,
        tlabel: int,
    ) -> float:
        """Evaluate fitness of a candidate solution.

        fitness = L2_distance + classification_penalty
        where penalty is 0 (success) or inf (failure).

        This is a decision-based evaluation: only uses hard predictions.
        """
        perturbed = self._apply_mask(mask, oimg, timg)
        l2 = torch.norm(oimg - perturbed).cpu().numpy()
        pred_label = self.model.predict_label(perturbed)

        if self.flag:  # Targeted
            lc = 0 if pred_label == tlabel else np.inf
        else:  # Untargeted
            lc = 0 if pred_label != olabel else np.inf

        return l2 + lc

    def _select(
        self,
        mask1: np.ndarray,
        fit1: float,
        mask2: np.ndarray,
        fit2: float,
    ) -> Tuple[np.ndarray, float]:
        """Tournament selection: keep the fitter individual."""
        if fit2 < fit1:
            return mask2, fit2
        return mask1, fit1

    def evo_perturb(
        self,
        oimg: torch.Tensor,
        timg: torch.Tensor,
        olabel: int,
        tlabel: int,
        max_query: int = 1000,
        snapshot_interval: int = 0,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, dict]:
        """Run the evolutionary attack.

        Args:
            oimg: Original (benign) image [1, C, H, W].
            timg: Starting point image [1, C, H, W] (already adversarial).
            olabel: Original label (0 for detection adapter).
            tlabel: Target label (-1 for detection adapter).
            max_query: Maximum number of model queries.

        Returns:
            Tuple of:
                - Adversarial image [1, C, H, W]
                - Number of queries used
                - L0 distance trace tensor
                - Snapshots dict {query_num: adv_image_tensor}
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        D = torch.zeros(max_query + 500, dtype=int).cuda()
        width = oimg.shape[3]
        height = oimg.shape[2]

        diff_indices = self._compute_mask(oimg, timg)
        snapshots = {}  # {query_num: adv_image_tensor}

        if len(diff_indices) <= 1:
            # Too few differing pixels, return target as-is
            D[0] = 1
            return timg, 1, D[:1], snapshots

        # 1. Initialize population
        population, n_queries, fitness = self._init_population(
            oimg, timg, olabel, tlabel
        )

        if len(population) == 0:
            return timg, n_queries, D[:n_queries], snapshots

        # 2. Find best and worst
        rank = np.argsort(fitness)
        best_idx = rank[0].item()
        worst_idx = rank[-1].item()

        # Record initial L0
        D[:n_queries] = compute_l0(
            self._apply_mask(population[best_idx], oimg, timg), oimg
        )

        # 3. Evolution loop
        while True:
            # a. Crossover
            candidates = [
                i for i in range(self.pop_size) if i != best_idx
            ]
            id1, id2 = np.random.choice(candidates, 2, replace=False)
            offspring = self._crossover(
                population[best_idx], population[id1], population[id2]
            )

            # b. Mutation
            offspring = self._mutate(offspring)

            # c. Fitness evaluation
            fit_offspring = self._evaluate_fitness(
                offspring, oimg, timg, olabel, tlabel
            )

            # d. Selection (replace worst)
            population[worst_idx], fitness[worst_idx] = self._select(
                population[worst_idx], fitness[worst_idx],
                offspring, fit_offspring,
            )

            # e. Update best and worst
            rank = np.argsort(fitness)
            best_idx = rank[0].item()
            worst_idx = rank[-1].item()

            # Record L0
            D[n_queries] = compute_l0(
                self._apply_mask(population[best_idx], oimg, timg), oimg
            )
            n_queries += 1

            if self.verbose and n_queries % self.log_interval == 0:
                best_img = self._apply_mask(population[best_idx], oimg, timg)
                current_l0 = compute_l0(best_img, oimg)
                total_pixels = oimg.shape[2] * oimg.shape[3]
                sr = current_l0 / total_pixels
                print(
                    f"[SpaEvo] query={n_queries}/{max_query}, "
                    f"L0={current_l0}, SR={sr:.4f}, "
                    f"pred={self.model.predict_label(best_img)}"
                )

            # Save snapshot at interval
            if snapshot_interval > 0 and n_queries % snapshot_interval == 0:
                snap_img = self._apply_mask(population[best_idx], oimg, timg)
                snapshots[n_queries] = snap_img.clone()

            if n_queries >= max_query:
                break

        adv = self._apply_mask(population[best_idx], oimg, timg)
        return adv, n_queries, D[:n_queries], snapshots
