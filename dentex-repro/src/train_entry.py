"""
Training launcher for the reproduction.

Run as a *subprocess* (see ``train_utils.launch_training``) rather than in the
notebook kernel, for three reasons: ``detectron2.engine.launch`` forks worker
processes for multi-GPU, a CUDA OOM must not take the notebook kernel with it,
and a Kaggle session kill during training must leave a resumable checkpoint
rather than a wedged kernel.

It reuses the repo's own ``Trainer`` (``train_net_patched.Trainer``) unchanged
and only adds hooks this study needs:

* :class:`TimeBudgetHook` — stops at a wall-clock budget, so the hour cap in
  ``micro`` mode holds by construction on any hardware.
* :class:`TrajectoryHook` — snapshots at fixed fractions of the run, for the
  "is the ablation ordering stable across training budget?" figure.
* :class:`HeartbeatHook` — periodic JSON so partial results survive a kill.

The two ablation switches are *not* implemented here. Weight transfer is
``MODEL.WEIGHTS``; noisy-box manipulation is the ``NOISY_BOX_TRAIN`` /
``NOISY_BOX_VAL`` environment variables the patched dataset mapper reads. Both
are set by the caller, so no upstream file needs a variant-specific branch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
for _path in (_PROJECT_ROOT, _REPO_ROOT):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, _REPO_ROOT)      # vendored detectron2 / pycocotools win
sys.path.insert(1, _PROJECT_ROOT)   # `src` package

from src import setup_env  # noqa: E402
from src.registration import TEST_DATASET, TRAIN_DATASET  # noqa: E402


class TimeBudgetExceeded(Exception):
    """Raised inside the training loop when the wall-clock budget is spent."""


def _hooks_module():
    from detectron2.engine import hooks

    return hooks


def build_hook_classes():
    """
    Built lazily: ``detectron2.engine.hooks`` must be imported after the
    vendored path is in place, which happens at module import time above, but
    keeping the class bodies in a function also keeps this file importable for
    ``--help`` on a machine with no torch.
    """
    from detectron2.engine.train_loop import HookBase
    from detectron2.utils import comm

    class TimeBudgetHook(HookBase):
        """Stop training once ``budget_seconds`` of wall clock has elapsed."""

        def __init__(self, budget_seconds: float):
            self.budget_seconds = float(budget_seconds)
            self.started = None

        def before_train(self):
            self.started = time.time()

        def after_step(self):
            if self.budget_seconds <= 0 or self.started is None:
                return
            elapsed = time.time() - self.started
            if elapsed >= self.budget_seconds:
                raise TimeBudgetExceeded(
                    "wall-clock budget of {:.0f}s reached at iteration {}"
                    .format(self.budget_seconds, self.trainer.iter)
                )

    class TrajectoryHook(HookBase):
        """
        Save an extra, *named* checkpoint at each requested iteration.

        Named rather than relying on the periodic checkpointer so the eval
        notebook can address "this variant at 1/3 of its budget" without
        guessing which numbered checkpoint that was.
        """

        def __init__(self, checkpointer, iteration_to_name):
            self.checkpointer = checkpointer
            self.iteration_to_name = {int(k): v for k, v in iteration_to_name.items()}

        def after_step(self):
            if not comm.is_main_process():
                return
            name = self.iteration_to_name.get(self.trainer.iter + 1)
            if name:
                self.checkpointer.save(name)

    class HeartbeatHook(HookBase):
        """Periodic JSON: iteration, throughput, latest losses, ETA."""

        def __init__(self, path: str, period: int, max_iter: int,
                     budget_seconds: float = 0.0):
            self.path = path
            self.period = max(1, int(period))
            self.max_iter = int(max_iter)
            self.budget_seconds = float(budget_seconds)
            self.started = None

        def before_train(self):
            self.started = time.time()

        def _write(self, final: bool = False):
            if not comm.is_main_process():
                return
            elapsed = time.time() - self.started
            done = self.trainer.iter - self.trainer.start_iter + 1
            rate = done / elapsed if elapsed > 0 else 0.0
            losses = {}
            try:
                history = self.trainer.storage.latest()
                losses = {k: float(v[0]) for k, v in history.items()
                          if k.startswith("loss") or k == "total_loss"}
            except Exception:                          # noqa: BLE001 - heartbeat must never kill a run
                losses = {}
            payload = {
                "iteration": self.trainer.iter,
                "start_iteration": self.trainer.start_iter,
                "max_iter": self.max_iter,
                "elapsed_seconds": round(elapsed, 1),
                "iters_per_second": round(rate, 4),
                "eta_seconds": round((self.max_iter - self.trainer.iter) / rate, 1) if rate else None,
                "budget_seconds": self.budget_seconds,
                "losses": losses,
                "final": final,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w") as handle:
                json.dump(payload, handle, indent=2)

        def after_step(self):
            if (self.trainer.iter + 1) % self.period == 0:
                self._write()

        def after_train(self):
            self._write(final=True)

    class RateProbeHook(HookBase):
        """Record per-iteration timestamps so calibration can measure it/s."""

        def __init__(self, path: str, warmup: int):
            self.path = path
            self.warmup = int(warmup)
            self.timestamps = []

        def after_step(self):
            self.timestamps.append(time.time())

        def after_train(self):
            if not comm.is_main_process():
                return
            usable = self.timestamps[self.warmup:]
            rate = None
            if len(usable) >= 2:
                rate = (len(usable) - 1) / (usable[-1] - usable[0])
            payload = {
                "iterations_timed": len(self.timestamps),
                "warmup_skipped": self.warmup,
                "iters_per_second": round(rate, 5) if rate else None,
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w") as handle:
                json.dump(payload, handle, indent=2)

    return TimeBudgetHook, TrajectoryHook, HeartbeatHook, RateProbeHook


def build_config(args):
    from detectron2.config import get_cfg
    from hierarchialdet import add_diffusiondet_config
    from hierarchialdet.util.model_ema import add_model_ema_configs

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    from detectron2.engine import default_setup
    from detectron2.utils import comm

    # Registration must happen inside main(): with --num-gpus 2 the workers are
    # spawned processes that never execute this module's __main__ block, so a
    # registration done only in the parent would be invisible to them.
    register_datasets_from_env()

    cfg = build_config(args)
    default_setup(cfg, args)
    setup_env.seed_everything(cfg.SEED, deterministic=False)

    from train_net_patched import Trainer

    TimeBudgetHook, TrajectoryHook, HeartbeatHook, RateProbeHook = build_hook_classes()

    trajectory = json.loads(args.trajectory or "{}")
    budget_seconds = float(args.budget_seconds or 0.0)
    heartbeat_path = args.heartbeat or os.path.join(cfg.OUTPUT_DIR, "heartbeat.json")

    class ReproTrainer(Trainer):
        def build_hooks(self):
            hooks_list = super().build_hooks()
            extra = [HeartbeatHook(heartbeat_path, args.heartbeat_period,
                                   cfg.SOLVER.MAX_ITER, budget_seconds)]
            if trajectory:
                extra.append(TrajectoryHook(self.checkpointer, trajectory))
            if budget_seconds > 0:
                extra.append(TimeBudgetHook(budget_seconds))
            if args.rate_probe:
                extra.append(RateProbeHook(args.rate_probe, args.rate_probe_warmup))
            return hooks_list + extra

    trainer = ReproTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    started = time.time()
    stopped_early = False
    try:
        trainer.train()
    except TimeBudgetExceeded as stop:
        stopped_early = True
        print("[train_entry] {}".format(stop), flush=True)
        if comm.is_main_process():
            trainer.checkpointer.save("model_final")
    wall_seconds = time.time() - started

    if comm.is_main_process():
        record = {
            "output_dir": cfg.OUTPUT_DIR,
            "config_file": os.path.abspath(args.config_file),
            "overrides": list(args.opts),
            "seed": cfg.SEED,
            "max_iter": cfg.SOLVER.MAX_ITER,
            "final_iteration": trainer.iter,
            "start_iteration": trainer.start_iter,
            "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH,
            "amp": cfg.SOLVER.AMP.ENABLED,
            "ema": cfg.MODEL_EMA.ENABLED,
            "num_gpus": args.num_gpus,
            "wall_seconds": round(wall_seconds, 1),
            "budget_seconds": budget_seconds,
            "stopped_on_time_budget": stopped_early,
            "weights_init": cfg.MODEL.WEIGHTS,
            "noisy_box_train": os.environ.get("NOISY_BOX_TRAIN"),
            "noisy_box_val": os.environ.get("NOISY_BOX_VAL"),
            "tier": os.environ.get("TIER"),
            "trajectory_checkpoints": trajectory,
            "device": "cuda" if args.num_gpus else "cpu",
        }
        with open(os.path.join(cfg.OUTPUT_DIR, "run_record.json"), "w") as handle:
            json.dump(record, handle, indent=2)
        print("[train_entry] wrote {}".format(
            os.path.join(cfg.OUTPUT_DIR, "run_record.json")), flush=True)


def register_datasets_from_env():
    from src.registration import register

    required = ("TRAIN_JSON", "TRAIN_IMG_DIR", "VAL_JSON", "VAL_IMG_DIR")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit(
            "train_entry.py requires {} in the environment (set by "
            "src.registration.training_env). Missing: {}".format(required, missing)
        )
    register(TRAIN_DATASET, os.environ["TRAIN_JSON"], os.environ["TRAIN_IMG_DIR"])
    register(TEST_DATASET, os.environ["VAL_JSON"], os.environ["VAL_IMG_DIR"])


def get_parser():
    from detectron2.engine import default_argument_parser

    parser = default_argument_parser()
    parser.add_argument("--budget-seconds", type=float, default=0.0,
                        help="hard wall-clock cap; 0 disables it")
    parser.add_argument("--trajectory", default="",
                        help='JSON object {"<iteration>": "<checkpoint name>"}')
    parser.add_argument("--heartbeat", default="",
                        help="path of the periodic progress JSON")
    parser.add_argument("--heartbeat-period", type=int, default=50)
    parser.add_argument("--rate-probe", default="",
                        help="write measured iters/second here (calibration runs)")
    parser.add_argument("--rate-probe-warmup", type=int, default=50,
                        help="iterations to discard before measuring throughput")
    return parser


if __name__ == "__main__":
    setup_env.assert_vendored()
    from detectron2.engine import launch

    cli_args = get_parser().parse_args()
    print("[train_entry] args: {}".format(cli_args), flush=True)
    register_datasets_from_env()
    launch(
        main,
        cli_args.num_gpus,
        num_machines=cli_args.num_machines,
        machine_rank=cli_args.machine_rank,
        dist_url=cli_args.dist_url,
        args=(cli_args,),
    )
