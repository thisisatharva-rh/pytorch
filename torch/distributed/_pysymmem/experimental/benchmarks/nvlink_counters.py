"""Read cumulative endpoint payload counters without resetting them."""

import re
import subprocess
import time


def read_counters():
    started = time.monotonic_ns()
    raw = subprocess.check_output(
        ["nvidia-smi", "nvlink", "--getthroughput", "d"], text=True, timeout=30
    )
    gpus = {}
    gpu = None
    for line in raw.splitlines():
        match = re.match(r"GPU (\d+):.*UUID: ([^)]+)", line)
        if match:
            gpu, uuid = match.groups()
            gpus[gpu] = {"uuid": uuid, "links": {}}
        match = re.search(r"Link (\d+): Data (Tx|Rx): (\d+) KiB", line)
        if match:
            if gpu is None:
                raise RuntimeError("NVLink counter has no GPU identity")
            link, direction, value = match.groups()
            gpus[gpu]["links"].setdefault(link, {})[direction.lower()] = (
                int(value) * 1024
            )
    if not gpus or any(
        not g["links"] or any(set(v) != {"tx", "rx"} for v in g["links"].values())
        for g in gpus.values()
    ):
        raise RuntimeError(f"Incomplete NVLink endpoint counters: {raw}")
    return {"started_ns": started, "finished_ns": time.monotonic_ns(), "gpus": gpus}


def counter_delta(before, after):
    if set(before["gpus"]) != set(after["gpus"]):
        raise RuntimeError("NVLink GPU set changed")
    result = {}
    for gpu, initial in before["gpus"].items():
        final = after["gpus"][gpu]
        if initial["uuid"] != final["uuid"] or set(initial["links"]) != set(
            final["links"]
        ):
            raise RuntimeError("NVLink device or link identity changed")
        links = {
            link: {
                direction: final["links"][link][direction] - values[direction]
                for direction in ("tx", "rx")
            }
            for link, values in initial["links"].items()
        }
        if any(value < 0 for values in links.values() for value in values.values()):
            raise RuntimeError("NVLink counter reset or wrap")
        result[gpu] = {
            "uuid": initial["uuid"],
            "links": links,
            "tx_bytes": sum(v["tx"] for v in links.values()),
            "rx_bytes": sum(v["rx"] for v in links.values()),
        }
    return result
