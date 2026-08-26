"""Parses the RTT log of `adaline_anc` (in the companion `frugal_ml-embedded`
firmware repo) and plots convergence + timing results.

Usage:
    probe-rs run --chip STM32F446RETx target/.../adaline_anc > adaline_anc.log
    python3 plot_adaline_anc.py adaline_anc.log

Each DATA line printed by the firmware is a CSV record:
    DATA,n,s_true,n0,d_n,y_n,e_n,err_vs_truth,cycles
where:
    n             sample index
    s_true        ground-truth clean signal (only known because this is a
                  simulation; not observable in a real deployment)
    n0            ground-truth noise (same caveat)
    d_n           primary input = s_true + n0 (what the microphone/ADC sees)
    y_n           the ADALINE's noise estimate
    e_n           d_n - y_n, the cancelled output
    err_vs_truth  e_n - s_true, only computable because s_true is known here
    cycles        DWT cycle count for that sample's forward+backward+update
"""

import re
import sys

import matplotlib.pyplot as plt
import numpy as np

LOG = sys.argv[1] if len(sys.argv) > 1 else "adaline_anc.log"
FS = 8000.0

PATTERN = re.compile(
    r"DATA,(-?\d+),(-?[\d.eE+-]+),(-?[\d.eE+-]+),(-?[\d.eE+-]+),"
    r"(-?[\d.eE+-]+),(-?[\d.eE+-]+),(-?[\d.eE+-]+),(\d+)"
)

rows = []
with open(LOG) as f:
    for line in f:
        m = PATTERN.search(line)
        if m:
            rows.append([float(x) for x in m.groups()])

data = np.array(rows)
n, s_true, n0, d_n, y_n, e_n, err_vs_truth, cycles = data.T
t = n / FS

t_raw_lo, t_raw_hi = 0.10, 0.14
t_before_lo, t_before_hi = 0.0, 0.01
t_after_lo, t_after_hi = t.max() - 0.04, t.max()

mask_raw = (t >= t_raw_lo) & (t <= t_raw_hi)
mask_before = (t >= t_before_lo) & (t <= t_before_hi)
mask_after = (t >= t_after_lo) & (t <= t_after_hi)

fig, axes = plt.subplots(5, 1, figsize=(11, 15))

ax = axes[0]
ax.plot(t[mask_raw], d_n[mask_raw], label="d[n] = primary input (signal + noise)", color="#888", linewidth=1.0)
ax.plot(t[mask_raw], s_true[mask_raw], label="s[n] = clean signal (ground truth)", color="#1f77b4", linewidth=1.6)
ax.set_title(f"1. The problem: useful signal buried in noise (zoom {t_raw_lo*1000:.0f}-{t_raw_hi*1000:.0f} ms)")
ax.set_ylabel("amplitude")
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
ax.plot(t[mask_before], y_n[mask_before], label="y[n] = ADALINE's noise estimate", color="#d62728", linewidth=1.2)
ax.plot(t[mask_before], n0[mask_before], label="n0[n] = true noise (ground truth)", color="#555", linewidth=1.6, linestyle="--", alpha=0.8)
ax.set_title(f"2. VERY EARLY ({t_before_lo*1000:.0f}-{t_before_hi*1000:.0f} ms) -- the ADALINE has barely learned anything yet")
ax.set_ylabel("amplitude")
ax.legend(loc="upper right", fontsize=8)

ax = axes[2]
ax.plot(t[mask_after], y_n[mask_after], label="y[n] = ADALINE's noise estimate", color="#d62728", linewidth=1.2)
ax.plot(t[mask_after], n0[mask_after], label="n0[n] = true noise (ground truth)", color="#555", linewidth=1.6, linestyle="--", alpha=0.8)
ax.set_title(f"3. MECHANISM, converged ({t_after_lo*1000:.0f}-{t_after_hi*1000:.0f} ms) -- the ADALINE reconstructs the noise, in phase, to subtract it")
ax.set_ylabel("amplitude")
ax.legend(loc="upper right", fontsize=8)

ax = axes[3]
ax.plot(t[mask_after], e_n[mask_after], label="e[n] = d[n]-y[n]  (cancelled output)", color="#2ca02c", linewidth=1.2)
ax.plot(t[mask_after], s_true[mask_after], label="s[n]  (ground truth)", color="#1f77b4", linewidth=1.6, alpha=0.8)
ax.set_title(f"4. RESULT, converged ({t_after_lo*1000:.0f}-{t_after_hi*1000:.0f} ms) -- d[n] minus the reconstructed noise = the clean signal")
ax.set_ylabel("amplitude")
ax.legend(loc="upper right", fontsize=8)

ax = axes[4]
cycles_us = cycles / 168.0
budget_us = 1e6 / FS
mean_us = cycles_us.mean()
margin = budget_us / mean_us
ax.plot(t, cycles_us, color="#9467bd", linewidth=1.0, label="measured wall time (DWT)")
ax.axhline(budget_us, color="red", linestyle="--", linewidth=1.2, label=f"real-time budget at {FS:.0f} Hz = {budget_us:.1f} us")
ax.annotate(
    f"{mean_us:.2f} us measured -- {margin:.1f}x margin",
    xy=(t[len(t) // 2], mean_us),
    xytext=(0, 10),
    textcoords="offset points",
    ha="center",
    color="#9467bd",
    fontsize=9,
    fontweight="bold",
)
ax.set_title("5. Cost per training step (forward+backward+update) vs real-time budget, full run")
ax.set_xlabel("time (s)")
ax.set_ylabel("microseconds (log scale)")
ax.set_yscale("log")
ax.legend(loc="center right", fontsize=8)
ax.set_ylim(mean_us * 0.5, budget_us * 1.8)

fig.suptitle("ADALINE / LMS on STM32F446RE -- adaptive noise cancellation (real hardware measurements)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("adaline_anc_results.png", dpi=140)
print("wrote adaline_anc_results.png")

abs_err = np.abs(err_vs_truth)
print(f"corr(y,n0) VERY EARLY window  : {np.corrcoef(y_n[mask_before], n0[mask_before])[0,1]:.3f}")
print(f"corr(y,n0) CONVERGED window   : {np.corrcoef(y_n[mask_after], n0[mask_after])[0,1]:.3f}")
print(f"RMS |e-s| VERY EARLY window   : {np.sqrt(np.mean(abs_err[mask_before]**2)):.4f}")
print(f"RMS |e-s| CONVERGED window    : {np.sqrt(np.mean(abs_err[mask_after]**2)):.4f}")
print(f"cycles/sample: mean={cycles.mean():.1f} ({cycles.mean()/168:.3f} us) max={cycles.max():.0f}")
print(f"real-time budget at {FS:.0f}Hz: {budget_us:.2f} us -> margin x{budget_us/(cycles.mean()/168):.1f}")
