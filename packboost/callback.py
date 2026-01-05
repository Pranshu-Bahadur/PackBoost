import time
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from typing import Optional, Callable, Dict, Any, Literal

Q30 = 1 << 30

def corr_metric(a, b):
    # Helper to calc correlation on Q30 preds vs float targets
    if a is None or b is None: return 0.0
    # Decode Q30
    if torch.is_tensor(a): a = a.detach().cpu().float().numpy()
    a = a.ravel() / Q30

    # Targets
    if torch.is_tensor(b): b = b.detach().cpu().float().numpy()
    b = b.ravel()

    # Trim
    n = min(len(a), len(b))
    if n == 0: return 0.0
    return np.corrcoef(a[:n], b[:n])[0, 1]

class EarlyStoppingCallback:
    """
    Callback for early stopping during model training based on validation metrics.
    
    Monitors a validation metric and stops training when the metric stops improving
    for a specified number of rounds (patience). Optionally keeps track of the best
    model state for restoration after training.
    
    Args:
        patience: Number of rounds without improvement before stopping training.
        keep_best: If True, stores the best model state for later restoration.
        metric_fn: Custom metric function that takes (y_true, y_pred) and returns a score.
                   If None, uses MSE (mean squared error) as the default metric.
        mode: Either "min" (minimize metric) or "max" (maximize metric).
        eval_every: Evaluate the callback every N training rounds.
        verbose: If True, prints progress messages during training.
    """
    
    def __init__(
        self,
        patience: int = 5,
        keep_best: bool = True,
        metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        mode: Literal["min", "max"] = "max",
        eval_every: int = 100,
        verbose: bool = True,
       
    ) -> None:
        self.patience = int(patience)
        self.keep_best = bool(keep_best)
        assert mode in ("max", "min")
        self.mode = mode
        self.verbose = bool(verbose)
        self.eval_every = max(1, int(eval_every))

        self.inv_scale = 1.0 / float(1 << 30)

        if metric_fn is None:
            # Default metric: Correlation (higher is better)
            def metric_fn_numpy(y_true, y_pred):
                return corr_metric(y_true, y_pred)
            self.metric_fn = metric_fn_numpy
        else:
            self.metric_fn = metric_fn

        self.best_score = float("-inf") if self.mode == "max" else float("inf")
        self.no_improve = 0
        self.best_tree_set = 0
        self.best_state = None
        self.early_stopped = False

    def is_better(self, cur: float, best: float) -> bool:
        """
        Compare current score with best score based on optimization mode.
        
        Args:
            cur: Current metric score.
            best: Best metric score so far.
            
        Returns:
            True if current score is better than best score, False otherwise.
        """
        return (cur > best) if self.mode == "max" else (cur < best)

    @torch.no_grad()
    def __call__(self, model: Any) -> None:
        """
        Evaluate the model and check for early stopping conditions.
        
        This method is called during training to monitor validation performance.
        It computes the validation metric, tracks improvement, and triggers early
        stopping if patience is exceeded.
        
        Args:
            model: The model being trained. Expected to have attributes:
                   - Yv: Validation labels.
                   - Pv_: Validation predictions in Q30 fixed-point format.
                   - tree_set: Current training round number.
                   - V, I: Model parameters to save when keep_best=True.
        """
        try:
            # Optional short-circuit if already stopped
            if self.early_stopped:
                return

            # Only evaluate every `eval_every` rounds
            cur_round = int(getattr(model, "tree_set", 0))
            if cur_round % self.eval_every != 0:
                return

            # Require validation labels/preds
            if not (hasattr(model, "Yv") and hasattr(model, "Pv_")):
                return

            # Get validation labels and predictions
            Yv = model.Yv
            Pv = model.Pv_
            
            # Handle slice if needed
            N_val = getattr(model, "val_N", None)
            if N_val is not None:
                if Pv is not None: Pv = Pv[:N_val]
                if Yv is not None: Yv = Yv[:N_val]

            # Compute score (corr_metric handles Q30 decoding internally)
            current_score = self.metric_fn(Pv, Yv)
            if isinstance(current_score, torch.Tensor):
                current_score = float(current_score.item())

            # Compare and possibly update best
            if self.is_better(current_score, self.best_score):
                self.best_score = current_score
                self.no_improve = 0
                self.best_tree_set = int(getattr(model, "tree_set", 0))

                if self.keep_best:
                    tree_set = int(getattr(model, "tree_set", 0))
                    self.best_state = {
                        "V": model.V.detach().cpu().clone()[:tree_set],
                        "I": model.I.detach().cpu().clone()[:tree_set],
                        "tree_set": tree_set,
                    }

                if self.verbose:
                    print(f"[Round {model.tree_set}] improved -> score={self.best_score:.6e}")
            else:
                self.no_improve += 1
                if self.verbose:
                    print(f"[Round {model.tree_set}] no improve {self.no_improve}/{self.patience} | {self.best_score}")

            # Early stopping condition
            if self.no_improve >= self.patience:
                self.early_stopped = True
                model.stop_training = True
                if self.verbose:
                    print(f"✓ Early stopping: best_round={self.best_tree_set}, best_score={self.best_score:.6e}")

        except Exception as e:
            print(e)
            raise

            
    def restore_best(self, model: Any) -> None:
        """
        Restore the model to its best state found during training.
        
        Loads the saved model parameters from the round with the best validation
        score. Only works if keep_best=True was set during initialization.
        
        Args:
            model: The model to restore. Must have V, I, and tree_set attributes.
        """
        if not self.keep_best or self.best_state is None:
            if self.verbose:
                print("No best state to restore (keep_best=False or no best found).")
            return
    
        device = model.V.device
    
        # Stored tensors are CPU tensors → just move them back to the model's device
        model.V = self.best_state["V"].to(device)
        model.I = self.best_state["I"].to(device)
        model.tree_set = self.best_state["tree_set"]
    
        if self.verbose:
            print(f"✓ Restored best model from round {self.best_tree_set}")


class LoggingCallback:
    def __init__(self, frequency=500):
        self.frequency = frequency
        self.t0 = time.time()
        self.t = self.t0

    def _corr(self, a, b):
        return corr_metric(a, b)

    def __call__(self, booster):
        r = getattr(booster, "tree_set", 0) + 1
        if r % self.frequency != 0 and r != 1: return

        # Train Corr
        P = getattr(booster, "P_", None)
        Y = getattr(booster, "Y", None) or getattr(booster, "dY", None)
        # Handle slice
        N_tr = getattr(booster, "train_N", None)
        if N_tr and P is not None: P = P[:N_tr]
        if N_tr and Y is not None: Y = Y[:N_tr]
        tr_corr = self._corr(P, Y)

        # Val Corr
        Pv = getattr(booster, "Pv_", None)
        Yv = getattr(booster, "Yv", None)
        # Handle slice
        N_val = getattr(booster, "val_N", None)
        if N_val and Pv is not None: Pv = Pv[:N_val]
        val_corr = self._corr(Pv, Yv)

        dt = time.time() - self.t
        print(f"Round {r}: Train {tr_corr:.5f} | Val {val_corr:.5f} | {dt:.2f}s")
        self.t = time.time()