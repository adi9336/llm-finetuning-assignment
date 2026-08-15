"""train_loop_safeguard — quarantine and halt logic for the training loop (M6).

Pure-logic safety net that wraps the poison detector's streaming anomaly
scores during training (M3 consumes this; no ML dependencies here):

  * quarantine_rows()  — split a batch into (active, quarantined): rows whose
                         anomaly score >= quarantine_threshold are pulled out of
                         the training batch so they never contribute gradients.
  * halt_on_anomaly()  — track a rolling anomaly rate over the last
                         window_size steps; when the rate exceeds
                         max_anomaly_rate the safeguard raises SafeguardHalt so
                         the training loop can stop before the model is damaged.

Both paths are configurable and unit-provable with synthetic score streams.

Usage:
    from src.train_loop_safeguard import Safeguard, SafeguardConfig
    sg = Safeguard(SafeguardConfig(max_anomaly_rate=0.05, window_size=100))
    active, quarantined = sg.quarantine_rows(batch, score_fn)
    for score in score_stream:
        state = sg.step(score)          # raises SafeguardHalt on runaway rate
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, List, Sequence, Tuple

ScoreFn = Callable[[Dict[str, Any]], float]


class SafeguardConfig:
    """Tunable safeguard parameters (validated by validate(); see below)."""

    def __init__(
        self,
        max_anomaly_rate: float = 0.05,
        window_size: int = 100,
        halt_on_anomaly: bool = True,
        quarantine_threshold: float = 0.50,
        quarantine_limit: int | None = None,
    ) -> None:
        self.max_anomaly_rate = max_anomaly_rate
        self.window_size = window_size
        self.halt_on_anomaly = halt_on_anomaly
        self.quarantine_threshold = quarantine_threshold
        self.quarantine_limit = quarantine_limit

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not 0.0 <= self.max_anomaly_rate <= 1.0:
            errors.append(f"max_anomaly_rate must be in [0, 1], got {self.max_anomaly_rate}")
        if self.window_size < 1:
            errors.append(f"window_size must be >= 1, got {self.window_size}")
        if not 0.0 <= self.quarantine_threshold <= 1.0:
            errors.append(f"quarantine_threshold must be in [0, 1], got {self.quarantine_threshold}")
        if self.quarantine_limit is not None and self.quarantine_limit < 0:
            errors.append(f"quarantine_limit must be >= 0 or None, got {self.quarantine_limit}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_anomaly_rate": self.max_anomaly_rate,
            "window_size": self.window_size,
            "halt_on_anomaly": self.halt_on_anomaly,
            "quarantine_threshold": self.quarantine_threshold,
            "quarantine_limit": self.quarantine_limit,
        }


class SafeguardHalt(Exception):
    """Raised when the rolling anomaly rate exceeds max_anomaly_rate.

    Carries the state that triggered it so the training loop can log and
    checkpoint before shutting down.
    """

    def __init__(self, rate: float, threshold: float, step: int, quarantined: int) -> None:
        super().__init__(
            f"anomaly rate {rate:.2%} exceeds threshold {threshold:.2%} "
            f"at step {step} ({quarantined} rows quarantined so far)"
        )
        self.rate = rate
        self.threshold = threshold
        self.step = step
        self.quarantined = quarantined


class Safeguard:
    """Streaming quarantine + halt watchdog over per-row anomaly scores."""

    def __init__(self, config: SafeguardConfig | None = None) -> None:
        self.config = config or SafeguardConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError("invalid SafeguardConfig: " + "; ".join(errors))
        self._window: Deque[bool] = deque(maxlen=self.config.window_size)
        self._step = 0
        self._quarantined_total = 0

    # -- quarantine path -------------------------------------------------- #
    def quarantine_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        score_fn: ScoreFn,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split `rows` into (active, quarantined) by anomaly score.

        A row is quarantined when score_fn(row) >= quarantine_threshold.
        quarantine_limit caps how many rows are pulled out per call (None =
        unlimited); rows beyond the cap stay in the active batch.
        """
        active: List[Dict[str, Any]] = []
        quarantined: List[Dict[str, Any]] = []
        limit = self.config.quarantine_limit
        for row in rows:
            if score_fn(row) >= self.config.quarantine_threshold and (
                limit is None or len(quarantined) < limit
            ):
                quarantined.append(row)
            else:
                active.append(row)
        self._quarantined_total += len(quarantined)
        return active, quarantined

    # -- halt path -------------------------------------------------------- #
    def observe(self, score: float) -> Dict[str, Any]:
        """Record one streaming anomaly score; return the rolling state."""
        is_anomaly = score >= self.config.quarantine_threshold
        self._window.append(is_anomaly)
        self._step += 1
        rate = sum(self._window) / len(self._window)
        return {
            "step": self._step,
            "window_size": len(self._window),
            "anomalies": sum(self._window),
            "rate": round(rate, 6),
        }

    def halt(self, state: Dict[str, Any]) -> None:
        """Raise SafeguardHalt when the observed rate exceeds the threshold."""
        if (
            self.config.halt_on_anomaly
            and state["rate"] > self.config.max_anomaly_rate
        ):
            raise SafeguardHalt(
                rate=state["rate"],
                threshold=self.config.max_anomaly_rate,
                step=state["step"],
                quarantined=self._quarantined_total,
            )

    def step(self, score: float) -> Dict[str, Any]:
        """observe() + halt() in one call — the training-loop entry point."""
        state = self.observe(score)
        self.halt(state)
        return state

    @property
    def quarantined_total(self) -> int:
        return self._quarantined_total
