"""
Measure what it costs to run HierarchicalDet, so a smaller clinic or research
group can tell whether the method is usable on their hardware.

Reports: parameter count and on-disk checkpoint size, peak GPU memory during
inference, steady-state throughput (images/minute) on GPU and CPU, and how both
scale with the number of diffusion steps. Also prints the host's GPU and RAM so
the numbers can be read against a known machine (e.g. a Kaggle T4 or a Colab
free-tier session).

Runtime numbers here exclude data loading on purpose -- they time the forward
pass over an already-loaded image, so they are a property of the model rather
than of the disk. tools/dump_predictions.py reports the end-to-end
per-image time that includes loading.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from repro_common import DEFAULT_CONFIG, build_eval_model, register_dataset, setup_cfg  # noqa: E402

from detectron2.data.detection_utils import read_image  # noqa: E402


def host_info():
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["gpu"] = properties.name
        info["gpu_total_memory_gb"] = round(properties.total_memory / (1024 ** 3), 2)
        info["cuda"] = torch.version.cuda
    try:
        import psutil

        info["host_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        pass
    return info


def time_forward(model, batched_inputs, tier, repeats, device):
    times = []
    with torch.no_grad():
        for index in range(repeats + 2):  # 2 warmup passes
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(batched_inputs, k=tier)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= 2:
                times.append(elapsed)
    return times


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output", required=True, help="results JSON to write")
    p.add_argument("--config-file", default=DEFAULT_CONFIG)
    p.add_argument("--tier", type=int, default=2, choices=(0, 1, 2))
    p.add_argument("--devices", nargs="*", default=["cuda", "cpu"])
    p.add_argument("--steps", nargs="*", type=int, default=[1, 4])
    p.add_argument("--repeats", type=int, default=5, help="timed forward passes per condition")
    p.add_argument("--dataset-name", default="benchmark_split")
    args = p.parse_args()

    with open(args.json) as f:
        images = json.load(f)["images"]
    sample = images[0]
    image_path = os.path.join(args.image_dir, sample["file_name"])

    results = {
        "host": host_info(),
        "weights": os.path.abspath(args.weights),
        "checkpoint_size_mb": round(os.path.getsize(args.weights) / (1024 ** 2), 1),
        "sample_image": sample["file_name"],
        "sample_resolution": [sample["width"], sample["height"]],
        "tier": args.tier,
        "runs": [],
    }

    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            print("skipping cuda: not available on this host")
            continue
        for step in args.steps:
            cfg = setup_cfg(args.config_file, ["MODEL.DiffusionDet.SAMPLE_STEP", str(step)],
                            weights=args.weights, device=device)
            register_dataset(args.dataset_name, args.json, args.image_dir)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model = build_eval_model(cfg)

            if "num_parameters" not in results:
                results["num_parameters"] = sum(p_.numel() for p_ in model.parameters())
                results["num_trainable_parameters"] = sum(
                    p_.numel() for p_ in model.parameters() if p_.requires_grad)

            image = read_image(image_path, format=cfg.INPUT.FORMAT)
            height, width = image.shape[:2]
            tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            batched_inputs = [{
                "image": tensor, "height": height, "width": width,
                "image_id": sample["id"], "file_name": image_path,
            }]

            times = time_forward(model, batched_inputs, args.tier, args.repeats, device)
            run = {
                "device": device,
                "sample_step": step,
                "seconds_per_image_mean": round(float(np.mean(times)), 4),
                "seconds_per_image_std": round(float(np.std(times)), 4),
                "images_per_minute": round(60.0 / float(np.mean(times)), 2),
            }
            if device == "cuda":
                run["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
                run["peak_gpu_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1)
            results["runs"].append(run)
            print(json.dumps(run))

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote {}".format(args.output))


if __name__ == "__main__":
    main()
