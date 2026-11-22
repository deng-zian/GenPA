"""Infrastructure isolation and supervision; no scene, instruction or score logic."""

from pathlib import Path
import re


CACHE_VARIABLES = {
    "WARP_CACHE_PATH": "warp",
    "TORCHINDUCTOR_CACHE_DIR": "torch_inductor",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
}

FATAL_PATTERNS = (
    r"CUDA kernel build failed",
    r"Warp NVRTC compilation error",
    r"CUDA out of memory",
    r"torch\.OutOfMemoryError",
    r"CUDA error:.*(?:illegal memory access|device-side assert|invalid device)",
    r"AttributeError:.*(?:left_planner|right_planner)",
    r"(?:ImportError|ModuleNotFoundError):",
    r"RuntimeError: Error building extension",
    r"No space left on device",
)


def compiler_environment(root, slot):
    """Reuse compilations sequentially in a slot, never between active workers."""
    directory = Path(root) / f"gpu{slot[0]}_worker{slot[1]}"
    result = {}
    for variable, child in CACHE_VARIABLES.items():
        path = directory / child
        path.mkdir(parents=True, exist_ok=True)
        result[variable] = str(path)
    # Torch extensions retain their versioned cache and built-in FileBaton lock.
    # A first build must not consume every CPU core on the evaluation host.
    result["MAX_JOBS"] = "1"
    return result


class WorkerWatchdog:
    def __init__(self, attempt, started):
        self.attempt = Path(attempt)
        self.started = started
        self.offset = 0
        self.tail = ""

    def fatal_error(self):
        log = self.attempt / "worker.log"
        if not log.exists():
            return None
        with log.open("rb") as stream:
            stream.seek(self.offset)
            text = stream.read(1 << 20).decode(errors="replace")
            self.offset = stream.tell()
        text = self.tail + text
        self.tail = text[-4096:]
        for pattern in FATAL_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return "Infrastructure failure: " + match.group(0)
        return None

    def stalled(self, now, seconds):
        # Stdout is NOT progress: upstream can print the same exception forever.
        progress = self.started
        for name in ("episode_end_states.jsonl", "policy_metrics.jsonl"):
            path = self.attempt / name
            if path.exists():
                progress = max(progress, path.stat().st_mtime)
        return now - progress > seconds

