"""Monkey-patches for PyTorch ops that fail on Blackwell (SM 10.3) GPUs.

PyTorch's NVRTC (runtime CUDA compiler) does not support SM 10.3 in versions
up to at least 2.12.0.dev. This causes ops that use JIT-compiled kernels
(erfinv, prod, and potentially others) to fail with:
    "nvrtc: error: invalid value for --gpu-architecture (-arch)"

This module wraps affected ops to fall back to CPU computation when the GPU
kernel fails. Import this module before any model code:

    import rf3.blackwell_compat  # patches torch ops for B300
"""

import functools

import torch


def _make_cpu_fallback(original_fn, op_name):
    """Create a wrapper that falls back to CPU when NVRTC JIT fails."""

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        try:
            return original_fn(*args, **kwargs)
        except RuntimeError as e:
            if "nvrtc" in str(e).lower() or "gpu-architecture" in str(e).lower():
                # Move tensor args to CPU, compute, move back
                device = None
                cpu_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        if device is None:
                            device = arg.device
                        cpu_args.append(arg.cpu())
                    else:
                        cpu_args.append(arg)
                cpu_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        if device is None:
                            device = v.device
                        cpu_kwargs[k] = v.cpu()
                    else:
                        cpu_kwargs[k] = v
                result = original_fn(*cpu_args, **cpu_kwargs)
                if isinstance(result, torch.Tensor) and device is not None:
                    return result.to(device)
                elif isinstance(result, tuple):
                    return tuple(
                        r.to(device)
                        if isinstance(r, torch.Tensor) and device is not None
                        else r
                        for r in result
                    )
                return result
            raise

    return wrapper


def patch():
    """Apply all Blackwell compatibility patches."""
    _patched = getattr(patch, "_applied", False)
    if _patched:
        return

    # Patch known NVRTC-broken ops
    torch.erfinv = _make_cpu_fallback(torch.erfinv, "erfinv")

    # torch.prod - used in various reduction contexts
    _original_prod = torch.prod
    torch.prod = _make_cpu_fallback(_original_prod, "prod")

    # Also patch the Tensor method versions
    _original_tensor_erfinv = torch.Tensor.erfinv
    torch.Tensor.erfinv = _make_cpu_fallback(_original_tensor_erfinv, "Tensor.erfinv")

    _original_tensor_prod = torch.Tensor.prod
    torch.Tensor.prod = _make_cpu_fallback(_original_tensor_prod, "Tensor.prod")

    patch._applied = True


# Auto-patch on import if running on a Blackwell GPU
if torch.cuda.is_available():
    try:
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 10:  # Blackwell is SM 10.x
            patch()
    except Exception:
        pass
