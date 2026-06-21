"""
Logger utility for DG-HMCF.

Provides a unified Logger class that combines:
  - Python standard logging (file + console handlers)
  - In-memory metric history tracking
  - Optional WandB integration
"""

import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


class Logger:
    """
    Experiment logger with metric tracking and optional WandB support.

    Parameters
    ----------
    name     : str  – logger name (usually the experiment/run name)
    log_dir  : str  – directory to save log file
    use_wandb: bool – whether to log metrics to WandB
    wandb_project : str – WandB project name
    wandb_config  : dict – config dict to log to WandB
    level    : int  – logging level (default logging.INFO)
    """

    def __init__(
        self,
        name: str = "dg-hmcf",
        log_dir: str = "logs",
        use_wandb: bool = False,
        wandb_project: str = "dg-hmcf-depression",
        wandb_config: Optional[Dict] = None,
        level: int = logging.INFO,
    ) -> None:
        self.name = name
        self.log_dir = log_dir
        self.use_wandb = use_wandb

        os.makedirs(log_dir, exist_ok=True)

        # ---- Standard Python logger -------------------------------------
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # File handler
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # ---- Metric history ---------------------------------------------
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._step: int = 0

        # ---- WandB ------------------------------------------------------
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=name, config=wandb_config or {})
                self.wandb = wandb
            except Exception as e:
                self.logger.warning("WandB init failed: %s", e)
                self.use_wandb = False

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def info(self, msg: str, *args, **kwargs) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.logger.debug(msg, *args, **kwargs)

    # ------------------------------------------------------------------
    # Metric tracking
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log a dict of metrics.

        Parameters
        ----------
        metrics : dict of metric_name → float value
        step    : global step counter; if None, auto-incremented
        prefix  : optional prefix to add to all metric names
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1

        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            self._metrics[full_key].append(float(value))

        # Log to file
        metric_str = " | ".join(
            f"{(prefix + '/' + k if prefix else k)}={v:.4f}"
            for k, v in metrics.items()
        )
        self.logger.info("Step %d | %s", self._step, metric_str)

        # WandB
        if self.wandb is not None:
            self.wandb.log(
                {(f"{prefix}/{k}" if prefix else k): v for k, v in metrics.items()},
                step=self._step,
            )

    def get_metric_history(self, key: str) -> List[float]:
        """Return logged history for a given metric key."""
        return self._metrics.get(key, [])

    def get_best_metric(self, key: str, mode: str = "max") -> float:
        """
        Return the best value for a metric.

        Parameters
        ----------
        key  : metric name (with prefix if applicable)
        mode : "max" or "min"
        """
        history = self._metrics.get(key, [])
        if not history:
            return float("-inf") if mode == "max" else float("inf")
        return float(np.max(history)) if mode == "max" else float(np.min(history))

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters as a structured info message."""
        self.logger.info("Hyperparameters:")
        for k, v in params.items():
            self.logger.info("  %-30s = %s", k, v)
        if self.wandb is not None:
            self.wandb.config.update(params, allow_val_change=True)

    def log_model_summary(self, model_name: str, n_params: int) -> None:
        """Log model name and parameter count."""
        self.logger.info(
            "Model: %s | Trainable parameters: %s",
            model_name,
            f"{n_params:,}",
        )

    def finish(self) -> None:
        """Finalise logging (flush handlers, close WandB run)."""
        for handler in self.logger.handlers:
            handler.flush()
        if self.wandb is not None:
            self.wandb.finish()
