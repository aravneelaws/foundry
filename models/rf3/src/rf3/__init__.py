"""RF3 - RosettaFold3 model implementation."""

__version__ = "0.1.0"

# Apply Blackwell (SM 10.3) compatibility patches for NVRTC-broken ops
import rf3.blackwell_compat  # noqa: F401
