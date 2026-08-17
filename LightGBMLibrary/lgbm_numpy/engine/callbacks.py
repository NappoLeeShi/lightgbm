import time
from typing import Optional, List, Dict, Callable


class CallbackEnv:
    def __init__(self):
        self.iteration: int = 0
        self.train_loss: float = 0.0
        self.valid_losses: Dict[str, float] = {}
        self.best_iteration: int = 0
        self.best_score: float = float("inf")
        self.model = None


class Callback:
    def __init__(self):
        pass

    def __call__(self, env: CallbackEnv):
        pass

    def init(self, env: CallbackEnv):
        pass


class EarlyStopping(Callback):
    def __init__(self, stopping_rounds: int = 50, metric_name: str = "valid_0", verbose: bool = True):
        super().__init__()
        self.stopping_rounds = stopping_rounds
        self.metric_name = metric_name
        self.verbose = verbose
        self.best_score = float("inf")
        self.best_iter = 0

    def init(self, env: CallbackEnv):
        self.best_score = float("inf")
        self.best_iter = 0

    def __call__(self, env: CallbackEnv):
        score = env.valid_losses.get(self.metric_name, env.train_loss)
        if score < self.best_score:
            self.best_score = score
            self.best_iter = env.iteration
            env.best_iteration = env.iteration
            env.best_score = score
        elif env.iteration - self.best_iter >= self.stopping_rounds:
            if self.verbose:
                print(f"Early stopping at round {env.iteration}, best iteration: {self.best_iter}, best score: {self.best_score:.6f}")
            return True
        return False


class LogCallback(Callback):
    def __init__(self, period: int = 1):
        super().__init__()
        self.period = period

    def __call__(self, env: CallbackEnv):
        if env.iteration % self.period == 0:
            msg = f"[{env.iteration:5d}] train loss: {env.train_loss:.6f}"
            for k, v in env.valid_losses.items():
                msg += f" | {k}: {v:.6f}"
            print(msg)


class LearningRateScheduler(Callback):
    def __init__(self, decay_factor: float = 0.99, min_lr: float = 1e-5):
        super().__init__()
        self.decay_factor = decay_factor
        self.min_lr = min_lr

    def __call__(self, env: CallbackEnv):
        if env.model is not None and hasattr(env.model, 'learning_rate'):
            env.model.learning_rate = max(
                env.model.learning_rate * self.decay_factor,
                self.min_lr,
            )
