"""Profiling callback for RF3 training benchmarks.

Collects per-step metrics including:
- Throughput (samples/sec, tokens/sec)
- GPU memory (peak allocated, peak reserved, current allocated)
- GPU utilization via pynvml polling
- Step timing breakdown (complements TimingCallback)

Outputs a CSV summary at the end of training.
"""

import csv
import os
import threading
import time
from pathlib import Path

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only

from foundry.callbacks.callback import BaseCallback


class ProfilingCallback(BaseCallback):
    """Collects GPU performance metrics for benchmarking.

    Args:
        log_every_n: Log aggregated metrics every N optimizer steps.
        output_csv: Path to write per-step metrics CSV. If None, writes to
            ``{output_dir}/profiling_metrics.csv``.
        gpu_poll_interval: Seconds between GPU utilization polls. Default 0.1.
        num_tokens_per_example: Number of tokens per example (for tokens/sec). Default 384.
        num_atoms_per_example: Number of atoms per example (for atoms/sec). Default 3072.
    """

    def __init__(
        self,
        log_every_n: int = 10,
        output_csv: str | None = None,
        gpu_poll_interval: float = 0.1,
        num_tokens_per_example: int = 384,
        num_atoms_per_example: int = 3072,
    ):
        super().__init__()
        self.log_every_n = log_every_n
        self.output_csv = output_csv
        self.gpu_poll_interval = gpu_poll_interval
        self.num_tokens_per_example = num_tokens_per_example
        self.num_atoms_per_example = num_atoms_per_example

        # State
        self._step_records: list[dict] = []
        self._batch_start_time: float | None = None
        self._optimizer_start_time: float | None = None
        self._epoch_start_time: float | None = None

        # GPU utilization polling
        self._nvml_handle = None
        self._poll_thread: threading.Thread | None = None
        self._poll_stop_event = threading.Event()
        self._gpu_util_samples: list[int] = []
        self._gpu_mem_util_samples: list[int] = []

    def _init_nvml(self):
        """Initialize NVIDIA Management Library for GPU utilization polling."""
        try:
            import pynvml

            pynvml.nvmlInit()
            # Get handle for the current GPU
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            return True
        except Exception:
            return False

    def _poll_gpu_utilization(self):
        """Background thread that polls GPU utilization."""
        import pynvml

        while not self._poll_stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                self._gpu_util_samples.append(util.gpu)
                self._gpu_mem_util_samples.append(util.memory)
            except Exception:
                pass
            self._poll_stop_event.wait(self.gpu_poll_interval)

    def _start_polling(self):
        """Start the GPU utilization polling thread."""
        if self._nvml_handle is not None:
            self._poll_stop_event.clear()
            self._gpu_util_samples.clear()
            self._gpu_mem_util_samples.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_gpu_utilization, daemon=True
            )
            self._poll_thread.start()

    def _stop_polling(self):
        """Stop the GPU utilization polling thread."""
        if self._poll_thread is not None:
            self._poll_stop_event.set()
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

    def _get_gpu_memory_stats(self) -> dict[str, float]:
        """Get current GPU memory statistics in GB."""
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        stats = torch.cuda.memory_stats(device)
        return {
            "gpu_mem/allocated_gb": stats.get("allocated_bytes.all.current", 0) / 1e9,
            "gpu_mem/peak_allocated_gb": stats.get("allocated_bytes.all.peak", 0) / 1e9,
            "gpu_mem/reserved_gb": stats.get("reserved_bytes.all.current", 0) / 1e9,
            "gpu_mem/peak_reserved_gb": stats.get("reserved_bytes.all.peak", 0) / 1e9,
        }

    def _get_avg_gpu_util(self) -> dict[str, float]:
        """Get average GPU utilization from polling samples."""
        result = {}
        if self._gpu_util_samples:
            result["gpu_util/sm_pct"] = sum(self._gpu_util_samples) / len(
                self._gpu_util_samples
            )
            self._gpu_util_samples.clear()
        if self._gpu_mem_util_samples:
            result["gpu_util/mem_pct"] = sum(self._gpu_mem_util_samples) / len(
                self._gpu_mem_util_samples
            )
            self._gpu_mem_util_samples.clear()
        return result

    @rank_zero_only
    def on_fit_start(self, trainer, **kwargs):
        nvml_ok = self._init_nvml()
        if nvml_ok:
            self._start_polling()
        else:
            print(
                "[ProfilingCallback] pynvml not available; GPU utilization polling disabled."
            )

        # Determine output CSV path
        if self.output_csv is None:
            output_dir = getattr(trainer, "output_dir", ".")
            self.output_csv = str(Path(output_dir) / "profiling_metrics.csv")

        # Reset peak memory stats for clean measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @rank_zero_only
    def on_train_epoch_start(self, trainer, **kwargs):
        self._epoch_start_time = time.perf_counter()

    @rank_zero_only
    def on_train_batch_start(self, trainer, batch, batch_idx, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_start_time = time.perf_counter()

    @rank_zero_only
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_end_time = time.perf_counter()

        if self._batch_start_time is not None:
            step_time = batch_end_time - self._batch_start_time
            record = {
                "global_step": trainer.state.get("global_step", 0),
                "batch_idx": batch_idx,
                "step_time_s": step_time,
            }
            # GPU memory
            record.update(self._get_gpu_memory_stats())
            self._step_records.append(record)

    @rank_zero_only
    def on_before_optimizer_step(self, trainer, optimizer, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._optimizer_start_time = time.perf_counter()

    @rank_zero_only
    def on_after_optimizer_step(self, optimizer, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._optimizer_start_time is not None and self._step_records:
            opt_time = time.perf_counter() - self._optimizer_start_time
            self._step_records[-1]["optimizer_step_time_s"] = opt_time

    @rank_zero_only
    def optimizer_step(self, trainer, optimizer):
        step = trainer.state.get("global_step", 0)

        if step > 0 and step % self.log_every_n == 0:
            # Compute aggregated metrics over recent steps
            recent = self._step_records[-self.log_every_n :]
            if not recent:
                return

            avg_step_time = sum(r["step_time_s"] for r in recent) / len(recent)
            samples_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
            # Multiply by world_size since each rank processes one sample per step
            world_size = trainer.fabric.world_size
            total_samples_per_sec = samples_per_sec * world_size
            tokens_per_sec = total_samples_per_sec * self.num_tokens_per_example
            atoms_per_sec = total_samples_per_sec * self.num_atoms_per_example

            metrics = {
                "profiling/samples_per_sec": total_samples_per_sec,
                "profiling/tokens_per_sec": tokens_per_sec,
                "profiling/atoms_per_sec": atoms_per_sec,
                "profiling/avg_step_time_s": avg_step_time,
            }

            # Add GPU utilization if available
            metrics.update(self._get_avg_gpu_util())

            # Add latest memory stats
            metrics.update(
                {f"profiling/{k}": v for k, v in self._get_gpu_memory_stats().items()}
            )

            # Add optimizer step time if available
            opt_times = [
                r["optimizer_step_time_s"]
                for r in recent
                if "optimizer_step_time_s" in r
            ]
            if opt_times:
                metrics["profiling/avg_optimizer_step_time_s"] = sum(opt_times) / len(
                    opt_times
                )

            trainer.fabric.log_dict(metrics, step=step)

            if trainer.fabric.is_global_zero:
                self._print_summary(step, metrics)

    def _print_summary(self, step: int, metrics: dict):
        """Print a concise profiling summary."""
        print(f"\n--- Profiling (step {step}) ---")
        for k, v in sorted(metrics.items()):
            if "per_sec" in k:
                print(f"  {k}: {v:,.1f}")
            elif "time" in k:
                print(f"  {k}: {v:.4f}s")
            elif "gb" in k.lower():
                print(f"  {k}: {v:.2f} GB")
            elif "pct" in k:
                print(f"  {k}: {v:.1f}%")
            else:
                print(f"  {k}: {v}")
        print("---")

    @rank_zero_only
    def on_train_epoch_end(self, trainer, **kwargs):
        if self._epoch_start_time is not None:
            epoch_time = time.perf_counter() - self._epoch_start_time
            epoch = trainer.state.get("current_epoch", 0)
            print(f"[ProfilingCallback] Epoch {epoch} completed in {epoch_time:.2f}s")

    @rank_zero_only
    def on_fit_end(self, trainer, **kwargs):
        self._stop_polling()

        # Write all per-step records to CSV
        if self._step_records:
            self._write_csv()

        # Print final summary
        self._print_final_summary(trainer)

        # Cleanup nvml
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            pass

    def _write_csv(self):
        """Write all collected step records to a CSV file."""
        if not self._step_records:
            return

        # Ensure output directory exists
        output_path = Path(self.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Gather all column names from records
        all_keys = set()
        for record in self._step_records:
            all_keys.update(record.keys())
        fieldnames = sorted(all_keys)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._step_records:
                writer.writerow(record)

        print(
            f"[ProfilingCallback] Wrote {len(self._step_records)} step records to {self.output_csv}"
        )

    def _print_final_summary(self, trainer):
        """Print a final summary of the training run."""
        if not self._step_records:
            print("[ProfilingCallback] No step records collected.")
            return

        # Skip first 5 steps (warmup)
        warmup = 5
        records = (
            self._step_records[warmup:]
            if len(self._step_records) > warmup
            else self._step_records
        )

        step_times = [r["step_time_s"] for r in records]
        world_size = trainer.fabric.world_size

        avg_time = sum(step_times) / len(step_times)
        min_time = min(step_times)
        max_time = max(step_times)
        median_time = sorted(step_times)[len(step_times) // 2]
        samples_per_sec = world_size / avg_time if avg_time > 0 else 0

        print("\n" + "=" * 60)
        print("PROFILING SUMMARY (excluding first 5 warmup steps)")
        print("=" * 60)
        print(f"  Total steps recorded: {len(self._step_records)}")
        print(f"  Steps used for stats: {len(records)}")
        print(f"  World size: {world_size}")
        print(f"  Avg step time: {avg_time:.4f}s")
        print(f"  Min step time: {min_time:.4f}s")
        print(f"  Max step time: {max_time:.4f}s")
        print(f"  Median step time: {median_time:.4f}s")
        print(f"  Throughput: {samples_per_sec:.2f} samples/sec")
        print(
            f"  Throughput: {samples_per_sec * self.num_tokens_per_example:.0f} tokens/sec"
        )

        # Memory stats from last record
        last = self._step_records[-1]
        if "gpu_mem/peak_allocated_gb" in last:
            print(
                f"  Peak GPU memory allocated: {last['gpu_mem/peak_allocated_gb']:.2f} GB"
            )
            print(
                f"  Peak GPU memory reserved: {last['gpu_mem/peak_reserved_gb']:.2f} GB"
            )

        print(f"  Metrics CSV: {self.output_csv}")
        print("=" * 60)
