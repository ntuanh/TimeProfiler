from ultralytics.nn.tasks import DetectionModel
import torch
import time
import pandas as pd
from src.Utils import write_partial


class LayerProfiler:
    def __init__(self, config):
        self.yaml_path = config["model"]               # path to yolov8.yaml
        self.num_runs = config["time_layer"]["num_round"]
        self.input_shape = config["input_shape"]

        # Build model from YAML (no pretrained weights)
        self.model = DetectionModel(self.yaml_path, nc=80, verbose=False)
        self.model.eval()

        # Random input tensor
        self.x = torch.randn(*self.input_shape)

        # Storage
        self.layer_start = {}
        self.layer_times = {}

        # Register hooks
        for m in self.model.model:
            m.register_forward_pre_hook(self._pre_hook)
            m.register_forward_hook(self._hook_fn)

    def _pre_hook(self, m, inp):
        self.layer_start[m.i] = time.perf_counter()

    def _hook_fn(self, m, inp, out):
        elapsed = (time.perf_counter() - self.layer_start[m.i]) * 1e6  # μs
        name = f"{m.__class__.__name__}_{m.i}"
        if name not in self.layer_times:
            self.layer_times[name] = []
        self.layer_times[name].append(elapsed)

    def run(self, filename="layer_times.csv"):
        # Warm-up run
        with torch.no_grad():
            self.model(self.x)

        # Benchmark runs
        for _ in range(self.num_runs):
            with torch.no_grad():
                self.model(self.x)

        # Compute average per-layer times
        avg_times = {
            name: round(sum(times) / len(times), 2)
            for name, times in self.layer_times.items()
        }

        # Total time
        total_time = round(sum(avg_times.values()), 2)
        avg_times["total_time"] = total_time

        # Save results
        headers = list(avg_times.keys())
        row = list(avg_times.values())
        write_partial("machine", "device", headers, row, filename=filename)

        print(f"[Num_runs] {self.num_runs}")
        print(f"Results saved to {filename}")
        return avg_times
