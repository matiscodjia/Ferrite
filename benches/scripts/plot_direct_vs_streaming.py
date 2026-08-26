"""Compares the two convolution strategies at the same C=1/C=3 regimes:
direct (full-frame `tensordot_3`+im2col, `regime_bench` firmware,
`REGIME|...` lines) vs streaming (`ConvStreaming`, `scaling_bench`
firmware, `SCALE|...` lines). Direct is faster per MAC where it runs at
all, but has an early RAM wall (the full output tensor plus im2col buffer
scale with H, not just W); streaming trades some speed for RAM that never
depends on frame height.

Usage:
    python3 plot_direct_vs_streaming.py regime_bench.log scaling_bench.log
"""

import re
import sys

import matplotlib.pyplot as plt

REGIME_LOG, SCALE_LOG = sys.argv[1], sys.argv[2]

REGIME_PATTERN = re.compile(
    r"REGIME\|([a-z0-9_]+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|"
    r"(OK|SKIP)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)"
)
SCALE_PATTERN = re.compile(
    r"SCALE\|([a-z0-9_]+)\|(\d+)\|(\d+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)\|"
    r"(-?[\d.eE+-]+)\|(\d+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)\|(-?[\d.eE+-]+)"
)
SYSCLK_HZ = 168_000_000.0

direct = []
with open(REGIME_LOG) as f:
    for line in f:
        m = REGIME_PATTERN.search(line)
        if m:
            name, h, w, c, k, kh, kw, macs, ram, status, cyc_iter, cyc_mac, time_us, pct_tick = m.groups()
            if not name.startswith("grid_c"):
                continue  # keep only the systematic resolution sweep, not the one-off real-sensor regimes
            direct.append(
                dict(
                    name=name,
                    W=int(w),
                    C=int(c),
                    status=status,
                    cycles_per_mac=float(cyc_mac) if status == "OK" else None,
                    time_us=float(time_us) if status == "OK" else None,
                )
            )

stream = []
with open(SCALE_LOG) as f:
    for line in f:
        m = SCALE_PATTERN.search(line)
        if m:
            name, c, w, cyc_row, us_row, cyc_mac, ram, f120, f240, f480, f720 = m.groups()
            stream.append(
                dict(name=name, W=int(w), C=int(c), cycles_per_row=float(cyc_row), cycles_per_mac=float(cyc_mac))
            )

# --- Plot 1: FPS vs resolution -- the direct-conv RAM wall ---
d_c1 = sorted([r for r in direct if r["C"] == 1], key=lambda r: r["W"])
s_c1 = sorted([r for r in stream if r["C"] == 1], key=lambda r: r["W"])

fig, ax = plt.subplots(figsize=(9, 6))

d_ok = [r for r in d_c1 if r["status"] == "OK"]
d_skip_w = [r["W"] for r in d_c1 if r["status"] == "SKIP"]
ax.plot(
    [r["W"] for r in d_ok],
    [1e6 / r["time_us"] for r in d_ok],
    marker="o",
    color="#d62728",
    label="direct (full-frame tensordot_3 + im2col)",
)
if d_skip_w:
    wall_x = min(d_skip_w)
    ax.axvline(wall_x, color="#d62728", linestyle=":", linewidth=1.8)
    ax.annotate(
        f"RAM WALL ({wall_x}px)",
        xy=(wall_x, 1),
        xycoords=("data", "axes fraction"),
        xytext=(8, -14),
        textcoords="offset points",
        color="#d62728",
        fontsize=10,
        fontweight="bold",
        va="top",
    )

ax.plot(
    [r["W"] for r in s_c1],
    [SYSCLK_HZ / (r["W"] * r["cycles_per_row"]) for r in s_c1],
    marker="o",
    color="#1f77b4",
    label="streaming (ConvStreaming, RAM independent of height)",
)

ax.set_yscale("log")
ax.set_title("Direct vs streaming convolution: FPS vs frame size (square frame, C=1, STM32F446RE)")
ax.set_xlabel("frame width = height (pixels)")
ax.set_ylabel("FPS (log scale)")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("plots/fps_direct_vs_resolution_wall.png", dpi=140)
print("wrote plots/fps_direct_vs_resolution_wall.png")

# --- Plot 2: cycles/MAC, direct vs streaming, as channel count grows ---
def avg_cycles_per_mac(rows, c):
    vals = [r["cycles_per_mac"] for r in rows if r["C"] == c and r.get("cycles_per_mac") is not None]
    return sum(vals) / len(vals) if vals else None


channels = sorted({r["C"] for r in direct if r["status"] == "OK"} | {r["C"] for r in stream})
direct_vals = [avg_cycles_per_mac(direct, c) for c in channels]
stream_vals = [avg_cycles_per_mac(stream, c) for c in channels]

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(channels, direct_vals, marker="o", color="#d62728", label="direct (full-frame)")
ax.plot(channels, stream_vals, marker="o", color="#1f77b4", label="streaming (ConvStreaming)")
ax.set_title("Cost per MAC vs channel count: direct vs streaming (STM32F446RE)")
ax.set_xlabel("channels (C)")
ax.set_ylabel("cycles / MAC (lower is better)")
ax.set_xticks(channels)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/cycles_per_mac_direct_vs_streaming.png", dpi=140)
print("wrote plots/cycles_per_mac_direct_vs_streaming.png")

# --- Console summary, the one point comparable at equal resolution (W=80, C=1) ---
d80 = next((r for r in d_c1 if r["W"] == 80 and r["status"] == "OK"), None)
s80 = next((r for r in s_c1 if r["W"] == 80), None)
if d80 and s80:
    fps_d = 1e6 / d80["time_us"]
    fps_s = SYSCLK_HZ / (80 * s80["cycles_per_row"])
    print(f"At 80x80, C=1 (the only point both strategies can reach):")
    print(f"  direct    : {fps_d:.1f} FPS, {d80['cycles_per_mac']:.2f} cycles/MAC")
    print(f"  streaming : {fps_s:.1f} FPS, {s80['cycles_per_mac']:.2f} cycles/MAC")
