# Derived from model_server/utils.py and model_server/constants.py.
"""Host introspection: what accelerator is there, and how much CPU may we use."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GPUStatus:
    CUDA = "cuda"
    MAC_MPS = "mps"
    NONE = "none"


def get_gpu_type() -> str:
    import torch

    if torch.cuda.is_available():
        return GPUStatus.CUDA
    if torch.backends.mps.is_available():
        return GPUStatus.MAC_MPS
    return GPUStatus.NONE


CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_CPU_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_CPU_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def get_cgroup_cpu_limit(
    v2_cpu_max: Path = CGROUP_V2_CPU_MAX,
    v1_cpu_quota: Path = CGROUP_V1_CPU_QUOTA,
    v1_cpu_period: Path = CGROUP_V1_CPU_PERIOD,
) -> int | None:
    """Cores this container may actually use, per its cgroup CFS quota.

    torch and the OpenMP/BLAS runtimes underneath it size their thread pools from
    the *host's* core count and know nothing about cgroups. A two-core container
    on a ninety-six-core node therefore starts ninety-six threads, thrashes, and
    gets CFS-throttled. Reading the quota lets the caller cap the pool.

    None means no quota is set, or the cgroup files are missing or unreadable.
    Quotas floor to whole cores so the cap never exceeds the real budget.
    """
    # cgroup v2: one file, "<quota> <period>", where "max" means unlimited. When
    # the v2 file exists it is authoritative and v1 is never consulted: a hybrid
    # host can carry stale v1 quota files that do not reflect the real limit.
    try:
        quota_str, period_str = v2_cpu_max.read_text().split()
        if quota_str == "max":
            return None
        period = int(period_str)
        return max(1, int(quota_str) // period) if period > 0 else None
    except FileNotFoundError:
        pass  # not a cgroup v2 host; fall through to v1
    except (OSError, ValueError) as exc:
        logger.debug("Could not parse cgroup v2 cpu.max (%s): %s", v2_cpu_max, exc)
        return None

    # cgroup v1: separate quota and period files; quota <= 0 means unlimited.
    try:
        quota = int(v1_cpu_quota.read_text())
        period = int(v1_cpu_period.read_text())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except FileNotFoundError:
        pass  # no cpu controller; treat as unlimited
    except (OSError, ValueError) as exc:
        logger.debug("Could not parse cgroup v1 cpu quota/period: %s", exc)

    return None
