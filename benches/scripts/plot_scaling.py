"""Parses `SCALE|...` lines from `scaling_bench` RTT logs (see
`src/bin/scaling_bench.rs` in the companion `ferrite-embedded` firmware
repo) and plots the two axes it isolates: RAM vs resolution at fixed
channel count, and compute-bound FPS ceiling vs channel count.

Each SCALE line, after the defmt/probe-rs prefix and file:line suffix:
    SCALE|name|C|W|cycles_per_row|us_per_row|cycles_per_mac|ram_bytes|fps_120|fps_240|fps_480|fps_720

Usage:
    python3 plot_scaling.py scaling_bench.log
"""

import re
import sys

import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else "scaling_bench.log"

PATTERN = re.compile(
    r"SCALE\|([a-z0-9_]+)\|(\d+)\|(\d+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)\|"
    r"(-?[\d.eE+-]+)\|(\d+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)"
)

rows = []
with open(LOG) as f:
    for line in f:
        m = PATTERN.search(line)
        if m:
            name, c, w, cyc_row, us_row, cyc_mac, ram, f120, f240, f480, f720 = m.groups()
            rows.append(
                dict(
                    name=name,
                    C=int(c),
                    W=int(w),
                    ram_bytes=int(ram),
                    fps_120=float(f120),
                    fps_240=float(f240),
                    fps_480=float(f480),
                    fps_720=float(f720),
                )
            )

if not rows:
    print("no SCALE lines found in", LOG)
    sys.exit(1)

# --- Plot 1: RAM vs resolution, C=1 sweep (space axis) ---
c1 = sorted([r for r in rows if r["C"] == 1], key=lambda r: r["W"])

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(
    [r["W"] for r in c1],
    [r["ram_bytes"] / 1024 for r in c1],
    marker="o",
    color="#1f77b4",
)
ax.set_title("Streaming convolution: RAM vs sensor width (C=1, STM32F446RE)")
ax.set_xlabel("sensor width W (pixels)")
ax.set_ylabel("RAM (KB)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/ram_vs_resolution.png", dpi=140)
print("wrote plots/ram_vs_resolution.png")

# --- Plot 2: FPS ceiling at native resolutions vs channel count ---
widths = sorted({r["W"] for r in rows if r["name"].startswith("c") and "_w" in r["name"]} & {160, 320, 640, 1280})
targets = [("120p", "fps_120"), ("240p", "fps_240"), ("480p", "fps_480"), ("720p", "fps_720")]

fig, axes = plt.subplots(1, len(widths), figsize=(4.2 * len(widths), 5), sharey=True)
if len(widths) == 1:
    axes = [axes]

for ax, w in zip(axes, widths):
    subset = sorted([r for r in rows if r["W"] == w], key=lambda r: r["C"])
    for label, key in targets:
        ax.plot(
            [r["C"] for r in subset],
            [r[key] for r in subset],
            marker="o",
            label=label,
        )
    ax.axhline(100, color="red", linestyle="--", linewidth=1, alpha=0.7, label="100 fps")
    ax.set_title(f"W={w}")
    ax.set_xlabel("channels (C)")
    ax.set_xticks(sorted({r["C"] for r in subset}))
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

axes[0].set_ylabel("compute-bound FPS ceiling (log scale)")
axes[-1].legend(loc="upper right", fontsize=8)
fig.suptitle("Streaming convolution: FPS ceiling vs channel count, per native resolution (STM32F446RE)")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("plots/fps_vs_channels.png", dpi=140)
print("wrote plots/fps_vs_channels.png")

# --- Console summary: is space/speed proportional to channels/resolution? ---
c1_slope = (c1[-1]["ram_bytes"] - c1[0]["ram_bytes"]) / (c1[-1]["W"] - c1[0]["W"])
print(f"RAM vs W (C=1): slope = {c1_slope:.3f} bytes/pixel-width")

for w in widths:
    subset = sorted([r for r in rows if r["W"] == w], key=lambda r: r["C"])
    ram_c1 = next(r["ram_bytes"] for r in subset if r["C"] == 1)
    ram_c3 = next((r["ram_bytes"] for r in subset if r["C"] == 3), None)
    if ram_c3 is not None:
        print(f"W={w}: RAM ratio C=3 / C=1 = {ram_c3 / ram_c1:.3f} (exactly 3.0 = perfectly proportional)")
