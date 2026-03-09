"""
Hierarchical time series forecast reconciliation.

Implements bottom-up, top-down, and MinT (Minimum Trace) reconciliation
so that forecasts at different aggregation levels of a hierarchy are
coherent (i.e. the parts add up to the whole).

Notation follows Wickramasuriya, Athanasopoulos & Hyndman (2019).
Let **y** be the vector of all series in the hierarchy and **S** the
summing matrix such that  y = S b  where b is the vector of bottom-level
series.  Reconciled forecasts are  y_tilde = S G y_hat  where G is
chosen according to the reconciliation method.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HierarchicalReconciler:
    """
    Reconcile hierarchical forecasts for coherency.

    Parameters
    ----------
    hierarchy : dict[str, list[str]]
        Mapping from parent series to its children.
        Example::

            {
                "total": ["region_A", "region_B"],
                "region_A": ["store_1", "store_2"],
                "region_B": ["store_3"],
            }

    method : str
        Default reconciliation method (``"bottom_up"``, ``"top_down"``,
        ``"mint"``).
    non_negative : bool
        Clamp reconciled values to >= 0.
    """

    SUPPORTED_METHODS = {"bottom_up", "top_down", "mint"}

    def __init__(
        self,
        hierarchy: Dict[str, List[str]],
        method: str = "mint",
        non_negative: bool = True,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from {self.SUPPORTED_METHODS}"
            )

        self.hierarchy = hierarchy
        self.method = method
        self.non_negative = non_negative

        self._all_series: List[str] = []
        self._bottom_series: List[str] = []
        self._S: Optional[np.ndarray] = None

        self._build_structure()

    # ------------------------------------------------------------------
    # Structure helpers
    # ------------------------------------------------------------------

    def _build_structure(self) -> None:
        """Derive the summing matrix S from the hierarchy dict."""
        parents = set(self.hierarchy.keys())
        all_children: set[str] = set()
        for children in self.hierarchy.values():
            all_children.update(children)

        self._bottom_series = sorted(all_children - parents)
        upper_series = self._topological_sort()
        self._all_series = upper_series + self._bottom_series

        n_total = len(self._all_series)
        n_bottom = len(self._bottom_series)

        S = np.zeros((n_total, n_bottom))

        bottom_idx = {name: i for i, name in enumerate(self._bottom_series)}

        for row_idx, series_name in enumerate(self._all_series):
            leaves = self._get_leaves(series_name)
            for leaf in leaves:
                if leaf in bottom_idx:
                    S[row_idx, bottom_idx[leaf]] = 1.0

        self._S = S
        logger.info(
            "Hierarchy: %d total series, %d bottom-level, S shape %s",
            n_total,
            n_bottom,
            S.shape,
        )

    def _topological_sort(self) -> List[str]:
        """Return upper-level series in top-down order."""
        parents = set(self.hierarchy.keys())
        all_children: set[str] = set()
        for children in self.hierarchy.values():
            all_children.update(children)

        roots = parents - all_children
        ordered: List[str] = []
        visited: set[str] = set()

        def dfs(node: str) -> None:
            if node in visited or node not in self.hierarchy:
                return
            visited.add(node)
            ordered.append(node)
            for child in self.hierarchy[node]:
                if child in self.hierarchy:
                    dfs(child)

        for root in sorted(roots):
            dfs(root)

        # Add any parents not yet visited (handles disjoint sub-trees)
        for p in sorted(parents):
            if p not in visited:
                dfs(p)

        return ordered

    def _get_leaves(self, node: str) -> List[str]:
        """Recursively collect bottom-level descendants."""
        if node not in self.hierarchy:
            return [node]
        leaves: List[str] = []
        for child in self.hierarchy[node]:
            leaves.extend(self._get_leaves(child))
        return leaves

    # ------------------------------------------------------------------
    # Reconciliation methods
    # ------------------------------------------------------------------

    def bottom_up(
        self,
        forecasts: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Bottom-up reconciliation.

        Keeps bottom-level forecasts unchanged and sums them up
        through the hierarchy.
        """
        n_bottom = len(self._bottom_series)
        horizon = self._get_horizon(forecasts)

        B = np.zeros((n_bottom, horizon))
        for i, name in enumerate(self._bottom_series):
            if name in forecasts:
                B[i, :] = forecasts[name][:horizon]

        reconciled_matrix = self._S @ B  # (n_total, horizon)
        return self._matrix_to_dict(reconciled_matrix, horizon)

    def top_down(
        self,
        forecasts: Dict[str, np.ndarray],
        proportions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Top-down reconciliation.

        Distributes the top-level forecast to bottom-level series
        using historical proportions.

        Parameters
        ----------
        proportions : dict[str, float] | None
            Share of each bottom series relative to the total.
            If ``None``, proportions are derived from the mean of
            the supplied forecasts.
        """
        horizon = self._get_horizon(forecasts)

        # Identify the root(s)
        root_names = [
            s for s in self._all_series if s in self.hierarchy
            and s not in {
                c for children in self.hierarchy.values() for c in children
            }
        ]

        if not root_names:
            root_names = [self._all_series[0]]

        root_name = root_names[0]
        top_forecast = forecasts.get(root_name, np.zeros(horizon))[:horizon]

        if proportions is None:
            proportions = self._compute_proportions(forecasts)

        n_bottom = len(self._bottom_series)
        B = np.zeros((n_bottom, horizon))
        for i, name in enumerate(self._bottom_series):
            prop = proportions.get(name, 1.0 / n_bottom)
            B[i, :] = top_forecast * prop

        reconciled_matrix = self._S @ B
        return self._matrix_to_dict(reconciled_matrix, horizon)

    def mint_reconciliation(
        self,
        forecasts: Dict[str, np.ndarray],
        residuals: Optional[Dict[str, np.ndarray]] = None,
        covariance_method: str = "shrink",
    ) -> Dict[str, np.ndarray]:
        """
        Minimum Trace (MinT) reconciliation.

        Computes  G = (S' W^{-1} S)^{-1} S' W^{-1}  where W is the
        covariance matrix of base-forecast errors, estimated via
        shrinkage or the identity (OLS).
        """
        S = self._S
        n_total, n_bottom = S.shape
        horizon = self._get_horizon(forecasts)

        Y_hat = np.zeros((n_total, horizon))
        for i, name in enumerate(self._all_series):
            if name in forecasts:
                Y_hat[i, :] = forecasts[name][:horizon]

        # Estimate W
        if residuals is not None and covariance_method == "shrink":
            W = self._shrinkage_covariance(residuals)
        elif residuals is not None and covariance_method == "sample":
            W = self._sample_covariance(residuals)
        else:
            W = np.eye(n_total)

        W_inv = np.linalg.pinv(W)
        StWinv = S.T @ W_inv
        G = np.linalg.pinv(StWinv @ S) @ StWinv

        B_tilde = G @ Y_hat  # (n_bottom, horizon)
        reconciled_matrix = S @ B_tilde

        return self._matrix_to_dict(reconciled_matrix, horizon)

    def _shrinkage_covariance(
        self, residuals: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Ledoit-Wolf-style shrinkage covariance estimation.

        Falls back to diagonal if residual matrix is too small.
        """
        n_total = len(self._all_series)
        res_matrix = np.zeros((n_total, 1))

        max_len = max(
            (len(v) for v in residuals.values()), default=1
        )
        res_matrix = np.zeros((n_total, max_len))

        for i, name in enumerate(self._all_series):
            if name in residuals:
                r = residuals[name][:max_len]
                res_matrix[i, : len(r)] = r

        if max_len < 3:
            return np.diag(np.var(res_matrix, axis=1) + 1e-8)

        sample_cov = np.cov(res_matrix)
        if sample_cov.ndim < 2:
            sample_cov = np.array([[sample_cov]])

        target = np.diag(np.diag(sample_cov))
        shrinkage_alpha = 0.5
        return (1 - shrinkage_alpha) * sample_cov + shrinkage_alpha * target

    def _sample_covariance(
        self, residuals: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Plain sample covariance of residuals."""
        n_total = len(self._all_series)
        max_len = max((len(v) for v in residuals.values()), default=1)
        res_matrix = np.zeros((n_total, max_len))

        for i, name in enumerate(self._all_series):
            if name in residuals:
                r = residuals[name][:max_len]
                res_matrix[i, : len(r)] = r

        cov = np.cov(res_matrix)
        if cov.ndim < 2:
            cov = np.array([[cov]])
        return cov + np.eye(cov.shape[0]) * 1e-8

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def reconcile(
        self,
        forecasts: Dict[str, np.ndarray],
        residuals: Optional[Dict[str, np.ndarray]] = None,
        method: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Reconcile forecasts using the specified (or default) method.

        Parameters
        ----------
        forecasts : dict[str, ndarray]
            Base forecasts keyed by series name.
        residuals : dict[str, ndarray] | None
            In-sample residuals for covariance estimation (MinT).
        method : str | None
            Override for :attr:`self.method`.
        """
        chosen = method or self.method

        if chosen == "bottom_up":
            result = self.bottom_up(forecasts)
        elif chosen == "top_down":
            result = self.top_down(forecasts)
        elif chosen == "mint":
            result = self.mint_reconciliation(
                forecasts, residuals=residuals
            )
        else:
            raise ValueError(f"Unknown reconciliation method: {chosen}")

        if self.non_negative:
            result = {
                k: np.maximum(v, 0.0) for k, v in result.items()
            }

        logger.info(
            "Reconciled %d series using '%s'", len(result), chosen
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_proportions(
        self, forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Derive bottom-level proportions from forecast means."""
        bottom_means = {}
        for name in self._bottom_series:
            if name in forecasts:
                bottom_means[name] = float(np.mean(np.abs(forecasts[name])))
            else:
                bottom_means[name] = 1.0

        total = sum(bottom_means.values())
        if total == 0:
            total = 1.0
        return {k: v / total for k, v in bottom_means.items()}

    def _get_horizon(self, forecasts: Dict[str, np.ndarray]) -> int:
        """Determine the forecast horizon from the supplied arrays."""
        lengths = [len(v) for v in forecasts.values() if len(v) > 0]
        return min(lengths) if lengths else 1

    def _matrix_to_dict(
        self, matrix: np.ndarray, horizon: int
    ) -> Dict[str, np.ndarray]:
        """Convert a reconciled matrix back to a named dict."""
        result: Dict[str, np.ndarray] = {}
        for i, name in enumerate(self._all_series):
            result[name] = matrix[i, :horizon]
        return result

    @property
    def summing_matrix(self) -> np.ndarray:
        """Return the S matrix."""
        if self._S is None:
            raise RuntimeError("Hierarchy has not been built")
        return self._S.copy()

    @property
    def all_series(self) -> List[str]:
        """All series names in hierarchical order."""
        return list(self._all_series)

    @property
    def bottom_series(self) -> List[str]:
        """Bottom-level series names."""
        return list(self._bottom_series)
