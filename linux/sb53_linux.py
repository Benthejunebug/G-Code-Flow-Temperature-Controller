#!/usr/bin/env python3
"""G-Code Flow Temperature Controller (SB53, Linux): flow-based nozzle temperature for sliced G-code (headless).

A Linux / command-line port of the method used by the SB53 G-Code Flow &
Temperature Controller (https://github.com/sb53systems/G-Code-Flow-Temperature-Controller,
MIT, BELAYEL Salim). Runs after OrcaSlicer, no GUI.

What it does:
  1. Runs the G-code through klipper_estimator `dump-moves` to get every move's
     real time (acceleration and cornering included) and flow (mm^3/s).
  2. Bins flow per second (blend of average and peak), smooths it.
  3. Maps flow -> temperature with a few points per filament.
  4. Plans the temperature so it never changes faster than the hotend can
     heat or cool. Rises are planned *backwards*, so heating starts early
     enough to be at temperature when the fast section begins (the original
     only limited rises going forwards, which made the temperature lag).
  5. Writes M104 changes into the G-code at the matching point (by filament
     used), and removes the slicer's own body-temperature M104 lines.

Not changed: first-layer temperatures (layers at first-layer height, including
the first layer of later objects in sequential prints), start and end G-code.

Usage:
  sb53_linux.py --config CONFIG.json input.gcode            # edit in place
  sb53_linux.py --config CONFIG.json input.gcode -o out.gcode --plot plan.png --report plan.json
OrcaSlicer post-processing (the slicer appends the G-code path):
  /usr/bin/python3 /path/sb53_linux.py --config /path/CONFIG.json;
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "0.1.0"
MARKER = "; Edited by G-Code Flow Temperature Controller (SB53, Linux)"
OLD_MARKERS = ("; Edited by SB53-Linux",)
TAG = "; Flow Temp Controller"
FILAMENT_AREA = math.pi * (1.75 / 2) ** 2  # mm^2, 1.75 mm filament


# ---------------------------------------------------------------- config
@dataclass
class Config:
    estimator: str
    machine: dict                      # klipper_estimator config (printer limits)
    flow_points: list                  # [[flow_mm3s, temp_C], ...] ascending flow
    min_temp: float
    max_temp: float
    hard_max_temp: float               # refuse configs above this (hotend safety)
    rise_c_per_s: float                # how fast the hotend can heat while printing
    fall_c_per_s: float                # how fast it cools
    peak_weight: float = 0.3           # 0 = average flow per second, 1 = peak flow
    smooth_s: int = 5                  # centred moving-average window (seconds)
    lead_s: float = 2.0                # extra lead for heater/PID lag
    step_c: float = 1.0                # round targets to this step
    min_change_c: float = 1.0          # don't emit changes smaller than this
    extra: dict = field(default_factory=dict)

    @staticmethod
    def load(path: Path) -> "Config":
        raw = json.loads(path.read_text())
        known = {f for f in Config.__dataclass_fields__ if f != "extra"}
        cfg = Config(**{k: v for k, v in raw.items() if k in known},
                     extra={k: v for k, v in raw.items() if k not in known})
        pts = cfg.flow_points
        if len(pts) < 2 or any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
            raise SystemExit("config: flow_points needs >=2 points with increasing flow")
        if cfg.max_temp > cfg.hard_max_temp:
            raise SystemExit(f"config: max_temp {cfg.max_temp} is above hard_max_temp {cfg.hard_max_temp}")
        if not (cfg.rise_c_per_s > 0 and cfg.fall_c_per_s > 0):
            raise SystemExit("config: rise/fall rates must be > 0")
        return cfg


def temp_for_flow(cfg: Config, flow: float) -> float:
    pts = cfg.flow_points
    if flow <= pts[0][0]:
        t = pts[0][1]
    elif flow >= pts[-1][0]:
        t = pts[-1][1]
    else:
        i = bisect.bisect_right([p[0] for p in pts], flow) - 1
        (f0, t0), (f1, t1) = pts[i], pts[i + 1]
        t = t0 + (t1 - t0) * (flow - f0) / (f1 - f0)
    return min(cfg.max_temp, max(cfg.min_temp, t))


# ---------------------------------------------------------------- estimator
def strip_notes(obj):
    """Drop "_comment" style keys (leading underscore) before handing config to the estimator."""
    if isinstance(obj, dict):
        return {k: strip_notes(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_notes(v) for v in obj]
    return obj


FLOW_RE = re.compile(r"^Flow = (?:Some\((-?[0-9.eE+-]+)\)|None)")
TIME_RE = re.compile(r"^Time = ([0-9.eE+-]+)")


def run_estimator(cfg: Config, gcode_path: Path, workdir: Path) -> list[tuple[float, float]]:
    """Return [(duration_s, flow_mm3s), ...] per planned move (flow 0 for non-extrusion/retracts)."""
    exe = shutil.which(cfg.estimator) or cfg.estimator
    if not Path(exe).exists():
        raise SystemExit(f"klipper_estimator not found: {cfg.estimator}")
    mcfg = workdir / "machine.json"
    mcfg.write_text(json.dumps(strip_notes(cfg.machine)))
    res = subprocess.run([exe, "--config_file", str(mcfg), "dump-moves", str(gcode_path)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"klipper_estimator failed ({res.returncode}): {res.stderr[-2000:]}")
    moves, flow = [], 0.0
    for line in res.stdout.splitlines():
        m = FLOW_RE.match(line)
        if m:
            flow = float(m.group(1)) if m.group(1) else 0.0
            flow = flow if flow > 0 else 0.0
            continue
        m = TIME_RE.match(line)
        if m:
            moves.append((float(m.group(1)), flow))
            flow = 0.0
    if not moves:
        raise SystemExit("klipper_estimator returned no moves")
    return moves


# ---------------------------------------------------------------- planning
def per_second(moves, peak_weight):
    """Per-second blended flow and cumulative filament (mm) at the end of each second."""
    total = sum(d for d, _ in moves)
    n = int(math.ceil(total)) + 1
    vol = [0.0] * n   # mm^3 extruded in each second
    peak = [0.0] * n
    t = 0.0
    for dur, flow in moves:
        end = t + dur
        while t < end - 1e-12:
            i = int(t)
            seg = min(end, i + 1) - t
            vol[i] += flow * seg
            if flow > peak[i]:
                peak[i] = flow
            t += seg
        t = end
    blended = [(1 - peak_weight) * v + peak_weight * p for v, p in zip(vol, peak)]
    cum, acc = [], 0.0
    for v in vol:
        acc += v / FILAMENT_AREA
        cum.append(acc)
    return blended, cum, total


def smooth(xs, window):
    if window <= 1:
        return xs[:]
    h = window // 2
    pref = [0.0]
    for x in xs:
        pref.append(pref[-1] + x)
    out = []
    for i in range(len(xs)):
        a, b = max(0, i - h), min(len(xs), i + h + 1)
        out.append((pref[b] - pref[a]) / (b - a))
    return out


def plan(cfg: Config, flow_s):
    want = [temp_for_flow(cfg, f) for f in flow_s]
    n = len(want)
    # backward pass: be at temperature when needed -> start heating early enough
    t = want[:]
    for i in range(n - 2, -1, -1):
        t[i] = max(t[i], t[i + 1] - cfg.rise_c_per_s)
    # forward pass: can't cool faster than the hotend cools
    for i in range(1, n):
        t[i] = max(t[i], t[i - 1] - cfg.fall_c_per_s)
    # extra lead for heater / PID lag: use the target from lead_s seconds later
    lead = int(round(cfg.lead_s))
    if lead > 0:
        t = [max(t[i:i + lead + 1]) for i in range(n)]
    step = cfg.step_c
    t = [min(cfg.max_temp, max(cfg.min_temp, round(x / step) * step)) for x in t]
    return want, t


# ---------------------------------------------------------------- g-code rewrite
NUM = r"(-?\d*\.?\d+)"
E_RE = re.compile(r"\bE" + NUM)
S_RE = re.compile(r"\bS" + NUM)


def next_move_first_layer(lines, first_layer_z):
    """For each line: is the next extrusion move on a first-layer-height layer?

    The slicer writes the first-layer temperature *before* the layer-change
    marker (e.g. for the second object in a sequential print), so M104 lines
    are judged by the layer of the next extrusion move, not the current layer.
    """
    n = len(lines)
    first = [False] * n
    cur = True
    for i, ln in enumerate(lines):
        if ln.startswith(";Z:"):
            try:
                cur = abs(float(ln[3:]) - first_layer_z) < 1e-3
            except ValueError:
                pass
        first[i] = cur
    out = [False] * n
    nxt = True
    for i in range(n - 1, -1, -1):
        s = lines[i].split(";", 1)[0].strip()
        if s[:3] in ("G1 ", "G2 ", "G3 ") and E_RE.search(s):
            nxt = first[i]
        out[i] = nxt
    return out


def rewrite(lines, cum, temps, first_layer_z):
    """Insert M104 changes by filament position; neutralise slicer body M104s.

    Returns (new_lines, stats).
    """
    out = []
    absolute_e, e_pos, used = True, 0.0, 0.0
    in_print, done = False, False
    cur_z, layer_is_first = None, True
    current = None            # last temperature we commanded (or slicer's, in first layers)
    inserted, neutralised = 0, 0
    first_inserted_at = None
    n = len(cum)
    upcoming_first = next_move_first_layer(lines, first_layer_z)
    for idx, ln in enumerate(lines):
        s = ln.split(";", 1)[0].strip()
        if ln.startswith(";LAYER_CHANGE"):
            in_print = True
        if ln.startswith(";Z:"):
            try:
                cur_z = float(ln[3:])
                layer_is_first = abs(cur_z - first_layer_z) < 1e-3
            except ValueError:
                pass
        if s.startswith("M104") and in_print and not done:
            m = S_RE.search(s)
            val = float(m.group(1)) if m else None
            if val == 0:
                done = True                      # end G-code: leave it alone
            elif not upcoming_first[idx]:
                out.append(TAG + " removed slicer temp: " + ln.rstrip("\n") + "\n")
                neutralised += 1
                continue
            else:
                current = val                    # first-layer temps are kept as-is
        if s.startswith("M82"):
            absolute_e = True
        elif s.startswith("M83"):
            absolute_e = False
        elif s.startswith("G92"):
            m = E_RE.search(s)
            if m:
                e_pos = float(m.group(1))
        elif s[:3] in ("G0 ", "G1 ", "G2 ", "G3 ") or s in ("G0", "G1"):
            m = E_RE.search(s)
            if m:
                e = float(m.group(1))
                de = e - e_pos if absolute_e else e
                if absolute_e:
                    e_pos = e
                if de > 0:
                    if in_print and not done and not layer_is_first:
                        i = min(bisect.bisect_left(cum, used), n - 1)
                        target = temps[i]
                        if current is None or abs(target - current) >= 1e-9:
                            out.append(f"M104 S{target:g} {TAG}\n")
                            inserted += 1
                            if first_inserted_at is None:
                                first_inserted_at = cur_z
                            current = target
                    used += de
        out.append(ln)
    return out, {"m104_inserted": inserted, "slicer_m104_removed": neutralised,
                 "filament_mm_gcode": round(used, 1), "first_change_z": first_inserted_at}


def first_layer_height(lines):
    for ln in lines:
        if ln.startswith(";Z:"):
            return float(ln[3:])
    raise SystemExit("no ;Z: layer markers found (OrcaSlicer G-code expected)")


def collapse_small_changes(temps, min_change):
    """Drop changes smaller than min_change so we don't spam M104 lines."""
    out, cur = [], None
    for t in temps:
        if cur is None or abs(t - cur) >= min_change:
            cur = t
        out.append(cur)
    return out


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gcode", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--estimator", help="path to klipper_estimator (overrides the config)")
    ap.add_argument("-o", "--output", type=Path, help="write here instead of editing in place")
    ap.add_argument("--report", type=Path, help="write a JSON report of the plan")
    ap.add_argument("--plot", type=Path, help="write a PNG graph of flow and temperature (needs matplotlib)")
    ap.add_argument("--dry-run", action="store_true", help="plan and report only, don't write G-code")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    if a.estimator:
        cfg.estimator = a.estimator
    src = a.gcode.read_text()
    if MARKER in src or any(m in src for m in OLD_MARKERS):
        raise SystemExit("already edited by the flow temperature controller; re-slice first")
    lines = src.splitlines(keepends=True)
    fl_z = first_layer_height(lines)

    with tempfile.TemporaryDirectory(prefix="sb53linux-") as td:
        moves = run_estimator(cfg, a.gcode, Path(td))
    flow_s, cum, total = per_second(moves, cfg.peak_weight)
    flow_s = smooth(flow_s, cfg.smooth_s)
    want, temps = plan(cfg, flow_s)
    temps = collapse_small_changes(temps, cfg.min_change_c)

    new_lines, stats = rewrite(lines, cum, temps, fl_z)
    est_fil, gc_fil = cum[-1], stats["filament_mm_gcode"]
    mismatch = abs(est_fil - gc_fil) / max(gc_fil, 1e-9)
    if mismatch > 0.02:
        raise SystemExit(f"filament mismatch estimator {est_fil:.1f} mm vs g-code {gc_fil:.1f} mm "
                         f"({mismatch:.1%}); refusing to write")
    inserted = [float(S_RE.search(l).group(1)) for l in new_lines if l.endswith(TAG + "\n")]
    if any(t < cfg.min_temp or t > cfg.max_temp for t in inserted):
        raise SystemExit("internal error: planned temperature outside limits")

    header = (f"{MARKER} v{VERSION} ({a.config.name}): {stats['m104_inserted']} temperature changes, "
              f"{min(inserted, default=0):g}-{max(inserted, default=0):g} C\n")
    report = {
        "version": VERSION, "config": str(a.config), "gcode": str(a.gcode),
        "estimated_time_s": round(total, 1), "filament_mm_estimator": round(est_fil, 1),
        **stats, "filament_mismatch": round(mismatch, 4),
        "flow_max_mm3s": round(max(flow_s), 2),
        "flow_mean_printing_mm3s": round(sum(f for f in flow_s if f > 0) / max(1, sum(1 for f in flow_s if f > 0)), 2),
        "temp_min": min(inserted, default=None), "temp_max": max(inserted, default=None),
        "seconds_at_temp": {f"{t:g}": temps.count(t) for t in sorted(set(temps))},
    }
    if a.report:
        a.report.write_text(json.dumps(report, indent=2))
    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            x = [i / 60 for i in range(len(flow_s))]
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(x, flow_s, lw=0.6, color="tab:blue", label="flow (mm³/s, smoothed)")
            ax.set_xlabel("minutes"); ax.set_ylabel("flow mm³/s")
            ax2 = ax.twinx()
            ax2.plot(x, want, lw=0.6, color="tab:orange", alpha=0.6, label="wanted temp")
            ax2.plot(x, temps, lw=1.2, color="tab:red", label="planned temp")
            ax2.set_ylabel("°C")
            fig.legend(loc="upper right")
            ax.set_title(f"Flow temperature plan: {a.gcode.name}")
            fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        except ImportError:
            print("matplotlib not installed; skipping plot", file=sys.stderr)

    if not a.dry_run:
        dst = a.output or a.gcode
        body = "".join(new_lines)
        body = header + body
        tmp = dst.with_name(dst.name + ".sb53tmp")
        tmp.write_text(body)
        os.replace(tmp, dst)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
