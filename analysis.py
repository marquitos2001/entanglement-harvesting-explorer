
import os
import re
import glob
import time
from collections import defaultdict

import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from scipy.special import eval_hermite, factorial
from scipy.signal import find_peaks, peak_widths
from joblib import Parallel, delayed


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_DIR     = r"/your/directory/"
FILE_PATTERN = "data_n*_k*.h5"

N_MAX        = 150          # max n for asymptotic plots (k=0 only)
N_DEEP       = 50           # the only n that loads multiple k-values
K_DEEP_MAX   = 10           # for n=N_DEEP, load k=0..K_DEEP_MAX
K_SELECT     = 0            # for all other n, load only k=0
OMEGA_MAX    = 20.0
THETA_RES    = 100
N_JOBS       = 24
T_RANGE      = np.linspace(-6, 6, 200)

DELTA_RATIO_THRESHOLD = 0.05
PEAK_MIN_PROMINENCE   = 0.05
PEAK_ZOOM_FACTOR      = 8

# Layout
SINGLE_COL = 3.375
DOUBLE_COL = 7.0

CBAR_TRAJ = [0.15, 1.05, 0.72, 0.025]
CBAR_SCAN = [0.15, 1.00, 0.72, 0.020]
CBAR_POS  = [0.10, 1.02, 0.88, 0.040]
LABEL_XY  = (-0.10, 1.04)


# ==========================================================
# RC PRESETS
# ==========================================================

RC_SMALL = {
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'lines.linewidth': 0.8, 'lines.markersize': 4,
}
RC_LARGE = {
    'font.size': 19, 'axes.titlesize': 19, 'axes.labelsize': 19,
    'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 15,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'lines.linewidth': 2.0, 'lines.markersize': 6,
}


# ==========================================================
# FORMATTERS
# ==========================================================

def int_formatter(x, pos):
    return r"$%d$" % int(x)

def float_formatter(x, pos):
    if x == int(x):
        return r"$%d$" % int(x)
    return r"$%.1f$" % x

def sci_formatter(x, pos):
    if x == 0:
        return r"$0$"
    exp   = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exp
    if abs(coeff - round(coeff)) < 0.05:
        return r"$%d\!\times\!10^{%d}$" % (round(coeff), exp)
    return r"$%.1f\!\times\!10^{%d}$" % (coeff, exp)


# ==========================================================
# IO
# ==========================================================

def parse_nk(fp):
    m = re.search(r"_n(\d+)_k(\d+)", fp)
    return int(m.group(1)), int(m.group(2))

def load_file(filepath):
    with h5py.File(filepath, "r") as f:
        base = f["Export"] if "Export" in f else f
        return dict(
            omegas = np.array(base["omegas"]),
            G      = np.array(base["G_real"])     + 1j * np.array(base["G_imag"]),
            W      = np.array(base["W_real"])     + 1j * np.array(base["W_imag"]),
            Delta  = np.array(base["Delta_real"]) + 1j * np.array(base["Delta_imag"]),
            n      = int(np.array(base["n"])[0]),
            T      = float(np.array(base["Tval"] if "Tval" in base else base["Tn"])[0]),
            ki     = int(np.array(base["ki"])[0]) if "ki" in base else -1,
        )


# ==========================================================
# CORE COMPUTATIONS
# ==========================================================

def get_hermite_function_val(p, t, T):
    norm = 1.0 / np.sqrt((2**p) * factorial(p) * np.sqrt(np.pi) * T)
    return norm * eval_hermite(p, t / T) * np.exp(-t**2 / (2 * T**2))

def calculate_trajectory(t_vals, c_opt, n, T):
    f_vals = np.zeros_like(t_vals, dtype=np.complex128)
    for p in range(n + 1):
        f_vals += c_opt[p] * get_hermite_function_val(p, t_vals, T)
    return np.real(f_vals)

def negativity_for_c(c, ReG_j, ImG_j, ReW_j, Delta_j):
    A = float(np.real(c @ ReG_j @ c))
    B = float(np.real(c @ ImG_j @ c))
    C = float(np.real(c @ ReW_j @ c))
    d = complex(c @ Delta_j @ c)
    neg = max(0.0, 0.5 * np.sqrt(A**2 + B**2 + abs(d)**2) - C)
    return neg, 0.5 * abs(d)

def _scan_one_omega(Rg, Ig, Rw, thetas):
    cos_t = np.cos(thetas)[:, None, None]
    sin_t = np.sin(thetas)[:, None, None]
    M_stack = 0.5 * (Rg * cos_t + Ig * sin_t) - Rw
    evals_all = np.linalg.eigvalsh(M_stack)
    max_per_theta = np.max(evals_all, axis=1)
    idx_best = np.argmax(max_per_theta)
    best_theta = thetas[idx_best]
    M_best = 0.5 * (Rg * np.cos(best_theta) + Ig * np.sin(best_theta)) - Rw
    _, evecs = np.linalg.eigh(M_best)
    return max_per_theta[idx_best], best_theta, evecs[:, -1]

def solve_one(data, store_raw=False):
    omegas = data["omegas"]
    ReG, ImG = np.real(data["G"]), np.imag(data["G"])
    ReW, Delta = np.real(data["W"]), data["Delta"]
    thetas = np.linspace(0, 2 * np.pi, THETA_RES)

    scan = Parallel(n_jobs=N_JOBS, batch_size="auto")(
        delayed(_scan_one_omega)(ReG[i], ImG[i], ReW[i], thetas)
        for i in range(len(omegas))
    )
    best_thetas = np.array([r[1] for r in scan])
    best_cvecs  = [r[2] for r in scan]

    max_negs   = np.zeros(len(omegas))
    delta_vals = np.zeros(len(omegas))
    for i in range(len(omegas)):
        max_negs[i], delta_vals[i] = negativity_for_c(
            best_cvecs[i], ReG[i], ImG[i], ReW[i], Delta[i])

    gi = int(np.argmax(max_negs))
    opt_val = max_negs[gi]
    ratio = delta_vals[gi] / opt_val if opt_val > 0 else np.inf

    result = dict(
        n=data["n"], T=data["T"], ki=data["ki"],
        opt_omega=omegas[gi], opt_theta=best_thetas[gi],
        max_neg=opt_val, delta_at_opt=delta_vals[gi], ratio=ratio,
        c_opt=best_cvecs[gi], omegas=omegas,
        max_negs=max_negs, delta_vals=delta_vals,
    )
    if store_raw:
        result.update(ReG=ReG, ImG=ImG, ReW=ReW, Delta=Delta, best_cvecs=best_cvecs)
    return result

def frozen_c_sweep(c, ReG, ImG, ReW, Delta):
    n_omega = ReG.shape[0]
    nf = np.zeros(n_omega)
    df = np.zeros(n_omega)
    for j in range(n_omega):
        nf[j], df[j] = negativity_for_c(c, ReG[j], ImG[j], ReW[j], Delta[j])
    return nf, df

def scan_negativity_simple(omegas, ReG, ImG, ReW):
    thetas = np.linspace(0, 2 * np.pi, THETA_RES)
    max_negs = np.zeros(len(omegas))
    for i in range(len(omegas)):
        cos_t = np.cos(thetas)[:, None, None]
        sin_t = np.sin(thetas)[:, None, None]
        M_stack = 0.5 * (ReG[i] * cos_t + ImG[i] * sin_t) - ReW[i]
        max_negs[i] = np.max(np.linalg.eigvalsh(M_stack))
    return np.maximum(max_negs, 0)


# ==========================================================
# COLORBAR HELPER
# ==========================================================

def add_T0inv_colorbar(fig, cmap, cnorm, T_vals, cbar_pos):
    sm = cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cax  = fig.add_axes(cbar_pos)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.set_ticks(T_vals)
    cbar.ax.set_xticklabels(
        [r"$%.1f$" % (1.0 / t) for t in T_vals],
        fontsize=plt.rcParams['xtick.labelsize'])
    cbar.ax.set_title(r"$T_0^{-1}\,T$",
                      fontsize=plt.rcParams['axes.titlesize'], pad=8)
    return cbar


# ==========================================================
# DEEP PLOTS — only for n = N_DEEP (k = 0..K_DEEP_MAX)
# ==========================================================

def plot_peaks_for_n(n_val, best, L=5.0):
    omegas = best["omegas"]
    max_negs = best["max_negs"]
    ReG, ImG, ReW = best["ReG"], best["ImG"], best["ReW"]
    Delta = best["Delta"]
    best_cvecs = best["best_cvecs"]
    d_omega = omegas[1] - omegas[0]

    peak_idxs, _ = find_peaks(max_negs, prominence=PEAK_MIN_PROMINENCE * max_negs.max())
    gi = int(np.argmax(max_negs))
    if gi not in peak_idxs:
        peak_idxs = np.sort(np.append(peak_idxs, gi))
    widths_idx, _, _, _ = peak_widths(max_negs, peak_idxs, rel_height=0.5)

    all_nf, all_df = [], []
    for p in peak_idxs:
        nf, df = frozen_c_sweep(best_cvecs[p], ReG, ImG, ReW, Delta)
        all_nf.append(nf)
        all_df.append(df)

    peak_vals = [max_negs[p] for p in peak_idxs]
    best_rank = int(np.argmax(peak_vals))
    best_nf, best_df = all_nf[best_rank], all_df[best_rank]

    n_peaks = len(peak_idxs)
    n_cols = int(np.ceil(np.sqrt(n_peaks)))
    n_rows = int(np.ceil(n_peaks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(DOUBLE_COL * n_cols, DOUBLE_COL * 0.8 * n_rows),
                             squeeze=False)

    for rank, (p, w) in enumerate(zip(peak_idxs, widths_idx)):
        ax = axes[rank // n_cols][rank % n_cols]
        op = omegas[p]
        hw = PEAK_ZOOM_FACTOR * w * d_omega
        ax.plot(omegas, all_nf[rank], color="steelblue", lw=2.0,
                label=r"$\lambda^{-2}\,\widetilde{\mathcal{N}}_{\mathrm{max}}$")
        ax.plot(omegas, all_df[rank], color="tomato", lw=2.0,
                label=r"$\frac{1}{2}\, \lambda^{-2}\, |\Delta_{\mathrm{max}}|$")
        ax.axvline(op, color="gray", ls=":", lw=1.5)
        ax.set_xlim(max(omegas[0], op - hw), min(omegas[-1], op + hw))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(float_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.set_xlabel(r"$\Omega \, T_0$")
        ax.grid(True, alpha=0.3)
        if rank == 0:
            ax.legend()

    for idx in range(n_peaks, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.show()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))
    ax.plot(omegas, best_nf, color="steelblue", lw=2.4,
            label=r"$\lambda^{-2}\,\widetilde{\mathcal{N}}_{\mathrm{max}}$")
    ax.plot(omegas, best_df, color="tomato", lw=2.4,
            label=r"$\frac{1}{2}\, \lambda^{-2}\, |\Delta_{\mathrm{max}}|$")
    ax.set_xlim(5, 7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(float_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_xlabel(r"$\Omega \, T_0$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.5)
    plt.show()


def plot_overview_for_n(n_val, k_results, L=5.0):
    n_k = len(k_results)
    T_vals = np.array([r["T"] for r in k_results])
    cmap, cnorm = cm.viridis, mcolors.Normalize(vmin=T_vals.min(), vmax=T_vals.max())
    colors = [cmap(cnorm(r["T"])) for r in k_results]
    tick_idx = np.round(np.linspace(0, n_k - 1, min(n_k, 6))).astype(int)
    T_ticks = T_vals[tick_idx]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))
    gauss = np.exp(-T_RANGE**2 / 2)
    for sgn, off in [(1, 0), (-1, 0), (-1, L), (1, L)]:
        ax.plot(sgn * gauss + off, T_RANGE, 'k', alpha=0.4, lw=1.5)
    ax.plot(L/2 + T_RANGE, T_RANGE, color='orange', lw=1, alpha=0.5)
    ax.plot(L/2 - T_RANGE, T_RANGE, color='orange', lw=1, alpha=0.5)
    for res, col in zip(k_results, colors):
        f_cur = calculate_trajectory(T_RANGE, res["c_opt"], n_val, res["T"])
        alpha = np.sqrt(np.sqrt(np.pi) / np.sum(np.abs(res["c_opt"])**2))
        mode = alpha * f_cur
        ax.plot(mode, T_RANGE, color=col, lw=1.5)
        ax.plot(mode + L, T_RANGE, color=col, lw=1.5)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(float_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(float_formatter))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.5)
    add_T0inv_colorbar(fig, cmap, cnorm, T_ticks, CBAR_TRAJ)
    plt.show()

    fig, (ax_neg, ax_del) = plt.subplots(
        2, 1, figsize=(DOUBLE_COL, DOUBLE_COL * 1.2),
        sharex=True, gridspec_kw={"hspace": 0.35})
    for res, col in zip(k_results, colors):
        ax_neg.plot(res["omegas"], res["max_negs"], color=col, lw=1.5)
        ax_del.plot(res["omegas"], res["delta_vals"], color=col, lw=1.5)
    for ax in (ax_neg, ax_del):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(float_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.grid(True, alpha=0.3)
    ax_del.set_xlabel(r"$\Omega \, T_0$")
    ax_neg.text(-0.1, 1.04, r"$\lambda^{-2}\,\widetilde{\mathcal{N}}_{\mathrm{max}}$",
                transform=ax_neg.transAxes, fontsize=22, ha="left", va="bottom")
    ax_del.text(-0.1, 1.04, r"$\frac{1}{2}\, \lambda^{-2}\, |\Delta_{\mathrm{max}}|$",
                transform=ax_del.transAxes, fontsize=22, ha="left", va="bottom")
    plt.tight_layout(pad=0.5)
    add_T0inv_colorbar(fig, cmap, cnorm, T_ticks, CBAR_SCAN)
    plt.show()


def select_best_k(k_results):
    valid = [r for r in k_results if r["ratio"] <= DELTA_RATIO_THRESHOLD]
    if not valid:
        valid = [min(k_results, key=lambda r: r["ratio"])]
    return max(valid, key=lambda r: r["max_neg"])


def plot_best_for_n(n_val, best, L=5.0):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))
    gauss = np.exp(-T_RANGE**2 / 2)
    for sgn, off in [(1, 0), (-1, 0), (-1, L), (1, L)]:
        ax.plot(sgn * gauss + off, T_RANGE, 'k', alpha=0.4, lw=1.5)
    ax.plot(L/2 + T_RANGE, T_RANGE, color='orange', lw=1, alpha=0.6)
    ax.plot(L/2 - T_RANGE, T_RANGE, color='orange', lw=1, alpha=0.6)
    f_cur = calculate_trajectory(T_RANGE, best["c_opt"], n_val, best["T"])
    alpha = np.sqrt(np.sqrt(np.pi) / np.sum(np.abs(best["c_opt"])**2))
    mode = alpha * f_cur
    ax.plot(mode, T_RANGE, 'green', lw=2.0)
    ax.plot(mode + L, T_RANGE, 'purple', lw=2.0)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(float_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(float_formatter))
    ax.set_xlim(-2, 6.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(r"$x/T_0$")
    ax.set_ylabel(r"$t/T_0$")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.5)
    plt.show()

    fig, (ax_neg, ax_del) = plt.subplots(
        2, 1, figsize=(DOUBLE_COL, DOUBLE_COL * 1.2),
        sharex=True, gridspec_kw={"hspace": 0.35})
    ax_neg.plot(best["omegas"], best["max_negs"], color="steelblue", lw=2.0,
                label=r"$\lambda^{-2}\,\widetilde{\mathcal{N}}_{\mathrm{max}}$")
    ax_del.plot(best["omegas"], best["delta_vals"], color="tomato", lw=2.0,
                label=r"$\frac{1}{2}\, \lambda^{-2}\, |\Delta_{\mathrm{max}}|$")
    for ax in (ax_neg, ax_del):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(float_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax_del.set_xlabel(r"$\Omega \, T_0$")
    plt.tight_layout(pad=0.5)
    plt.show()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))
    ax.plot(best["omegas"], best["max_negs"], color="steelblue", lw=2.0,
            label=r"$\lambda^{-2}\,\widetilde{\mathcal{N}}_{\mathrm{max}}$")
    ax.plot(best["omegas"], best["delta_vals"], color="tomato", lw=2.0,
            label=r"$\frac{1}{2}\, \lambda^{-2}\, |\Delta_{\mathrm{max}}|$")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(float_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_xlabel(r"$\Omega \, T_0$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.5)
    plt.show()


# ==========================================================
# ASYMPTOTIC: MAX NEGATIVITY VS N (k=0 only)
# ==========================================================

def make_evolution_plot(k0_results):
    plt.rcParams.update(RC_SMALL)
    if not k0_results:
        return

    ns = np.array(sorted(k0_results.keys()))
    max_negs = np.array([k0_results[n]["max_neg"] for n in ns])

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 3/4))
    ax.plot(ns, max_negs, 'o-', color="steelblue", markersize=4, lw=0.8,
            label=r"$\lambda^{-2}\, \mathrm{max}_\Omega \,\,"
                  r"\widetilde{\mathcal{N}}^+_{\mathrm{max}}$")
    ax.set_xlim(ns[0] - 1, N_MAX + 1)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(int_formatter))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.set_xlabel(r"$N$")
    ax.legend(loc="lower right", frameon=True, fancybox=True,
              framealpha=0.9, edgecolor="0.6", fontsize=9)
    plt.tight_layout(pad=0.4)
    plt.show()

# ==========================================================
# NEGATIVITY VS OMEGA FOR ALL N (k=0 only) — reuses k0_results
# ==========================================================

def make_negativity_vs_omega_plot(k0_results):
    plt.rcParams.update(RC_SMALL)
    if not k0_results:
        return

    results = []
    for n_val in sorted(k0_results):
        res = k0_results[n_val]
        omegas = res["omegas"]
        mask = omegas <= OMEGA_MAX
        results.append((n_val, omegas[mask], res["max_negs"][mask]))

    n_vals = [r[0] for r in results]
    norm = mcolors.Normalize(vmin=min(n_vals), vmax=max(n_vals))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 1.05))
    for n_val, om, mn in results:
        ax.plot(om, mn, color=cmap(norm(n_val)))
    ax.set_xlim(0, OMEGA_MAX)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(int_formatter))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlabel(r"$\Omega \,T_0$")
    ax.text(LABEL_XY[0], LABEL_XY[1],
            r"$\lambda^{-2}\,\widetilde{\mathcal{N}}^+_{\mathrm{max}}$",
            transform=ax.transAxes, fontsize=11, ha="left", va="bottom")

    plt.tight_layout(pad=0.4)
    cax = fig.add_axes(CBAR_POS)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(int_formatter))
    cbar.ax.tick_params(labelsize=8)
    cax.text(1.03, 0.5, r"$N$", transform=cax.transAxes,
             fontsize=10, ha="left", va="center")
    plt.show()

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    all_files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
    if not all_files:
        raise FileNotFoundError(f"No files matched: {FILE_PATTERN}")

    # Filter: keep k=0 for all n<=N_MAX, plus k=0..K_DEEP_MAX for n=N_DEEP
    selected = []
    for fp in all_files:
        n, k = parse_nk(fp)
        if n > N_MAX:
            continue
        if n == N_DEEP and 0 <= k <= K_DEEP_MAX:
            selected.append(fp)
        elif n != N_DEEP and k == K_SELECT:
            selected.append(fp)

    selected.sort(key=parse_nk)
    print(f"Selected {len(selected)} files")

    by_n = defaultdict(list)
    for fp in selected:
        by_n[parse_nk(fp)[0]].append(fp)

    plt.rcParams.update(RC_LARGE)

    k0_results = {}     # n -> result dict (k=0 only) for asymptotic plot
    k0_data    = {}     # n -> raw data dict (k=0 only) for omega-vs-N plot

    for n_val in sorted(by_n.keys()):
        k_files = sorted(by_n[n_val], key=lambda fp: parse_nk(fp)[1])
        print(f"\nn={n_val} ({len(k_files)} k-values)")

        if n_val == N_DEEP:
            # Solve all k = 0..K_DEEP_MAX for this n
            k_results = []
            for fp in k_files:
                n, k = parse_nk(fp)
                t0 = time.time()
                data = load_file(fp)
                if k == K_SELECT:
                    k0_data[n_val] = data
                res = solve_one(data, store_raw=False)
                print(f"  k={res['ki']} T={res['T']:.4f} "
                      f"MaxNeg={res['max_neg']:.4e} ({time.time()-t0:.1f}s)")
                k_results.append(res)
                if k == K_SELECT:
                    k0_results[n_val] = res

            plot_overview_for_n(n_val, k_results)
            best_lite = select_best_k(k_results)
            best_fp = [fp for fp in k_files
                       if parse_nk(fp)[1] == best_lite["ki"]][0]
            best = solve_one(load_file(best_fp), store_raw=True)
            plot_best_for_n(n_val, best)
            plot_peaks_for_n(n_val, best)

        else:
            # Only k=0 file for this n
            for fp in k_files:
                n, k = parse_nk(fp)
                t0 = time.time()
                data = load_file(fp)
                k0_data[n_val] = data
                res = solve_one(data, store_raw=False)
                print(f"  k={res['ki']} T={res['T']:.4f} "
                      f"MaxNeg={res['max_neg']:.4e} ({time.time()-t0:.1f}s)")
                k0_results[n_val] = res

    make_evolution_plot(k0_results)
    make_negativity_vs_omega_plot(k0_results)

    print("\nAll done.")