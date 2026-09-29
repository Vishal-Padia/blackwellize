"""Pull a fixed set of metrics out of every .ncu-rep in the ncu-profiles volume.

    modal run experiments/extract_metrics.py

Writes results/ncu_metrics.json as {report_name: {metric: value}}. No GPU needed.
"""

import csv
import io
import json
import pathlib
import re

import modal

REPO = pathlib.Path(__file__).parent.parent
profiles = modal.Volume.from_name("ncu-profiles", create_if_missing=True)
image = modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
app = modal.App("ncu-extract", image=image)

KEEP = re.compile(
    r"^(gpu__time_duration\.sum|launch__grid_size|launch__waves_per_multiprocessor|"
    r"sm__throughput\.avg\.pct_of_peak_sustained_elapsed|"
    r"gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed|"
    r"dram__throughput\.avg\.pct_of_peak_sustained_elapsed|"
    r"l1tex__throughput\..*|"
    r"lts__throughput\.avg\.pct_of_peak_sustained_elapsed|"
    r"lts__t_sector_hit_rate\.pct|dram__bytes_(read|write)\.sum|"
    r"sm__warps_active\.avg\.pct_of_peak_sustained_active|"
    r"sm__cycles_active\.avg|smsp__cycles_active\.avg|"
    r"l1tex__data_bank_conflicts_pipe_lsu_mem_shared\.sum|"
    r"l1tex__data_pipe_lsu_wavefronts_mem_shared\.sum|"
    r"sm__pipe_tensor.*pct_of_peak_sustained_(active|elapsed)|"
    r"sm__inst_executed_pipe_tensor.*pct_of_peak_sustained_(active|elapsed)|"
    r"smsp__average_warp_latency_issue_stalled_.*|"
    r"smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio|"
    r"smsp__cycles_per_warp_active.*|"
    r"smsp__warps_issue_stalled_.*_per_warp_active\.pct|"
    r"gpu__dram_throughput.*|"
    r"launch__registers_per_thread|launch__shared_mem_per_block.*|"
    r"sm__maximum_warps_per_active_cycle_pct)"
)


@app.function(volumes={"/profiles": profiles}, timeout=1800)
def extract():
    import subprocess

    out = {}
    for f in sorted(pathlib.Path("/profiles").glob("*.ncu-rep")):
        p = subprocess.run(
            f"ncu --import {f} --page raw --csv", shell=True, capture_output=True, text=True
        )
        lines = p.stdout.splitlines()
        # ncu prepends non-csv banner lines; csv starts at the header row
        start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
        if start is None:
            print(f.name, "no csv", p.stderr[-200:])
            continue
        rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
        header = rows[0]
        # rows[1] is the units row; data rows follow
        data = rows[2:] if len(rows) > 2 else []
        m = {}
        for r in data:
            name_col = header.index("Kernel Name")
            m["kernel"] = r[name_col]
            for h, v in zip(header, r):
                if KEEP.match(h):
                    m[h] = v
            break
        out[f.name.removesuffix(".ncu-rep")] = m
        print(f.name, len(m), flush=True)
    return out


@app.local_entrypoint()
def main():
    res = extract.remote()
    path = REPO / "results" / "ncu_metrics.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path} ({len(res)} reports)")
