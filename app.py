"""
UDW Entanglement Harvesting Explorer - Streamlit version
Run with:  python -m streamlit run app.py
"""

import numpy as np
import h5py
import streamlit as st
import plotly.graph_objects as go
import ast
import operator
import time

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Entanglement Harvesting Explorer",
    layout="wide",
)

# ================================================================
# SAFE MATH EXPRESSION PARSER
# ================================================================

ALLOWED_FUNCTIONS = {
    'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'sech': lambda x: 1.0 / np.cosh(x),
    'arctan': np.arctan, 'arcsin': np.arcsin, 'arccos': np.arccos,
    'heaviside': lambda x: np.heaviside(x, 0.5),
    'step': lambda x: np.heaviside(x, 0.5),
    'sign': np.sign,
    'pi': np.pi,
    'e': np.e,
}

ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


class SafeMathParser:
    def __init__(self, expr_string):
        if len(expr_string) > 500:
            raise ValueError("Expression too long (max 500 characters)")
        self.expr = expr_string
        self.tree = ast.parse(expr_string, mode='eval')
        self._validate(self.tree.body)
    
    def _validate(self, node):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants allowed")
        elif isinstance(node, ast.Name):
            if node.id not in ('t',) and node.id not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Unknown name: '{node.id}'. Use 't' and math functions.")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in ALLOWED_OPS:
                raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in ALLOWED_OPS:
                raise ValueError("Unary op not allowed")
            self._validate(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            if node.func.id not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Function not allowed: '{node.func.id}'")
            for arg in node.args:
                self._validate(arg)
        else:
            raise ValueError(f"Not allowed: {type(node).__name__}")
    
    def evaluate(self, t_grid):
        return self._eval_node(self.tree.body, t_grid)
    
    def _eval_node(self, node, t):
        if isinstance(node, ast.Constant):
            return float(node.value)
        elif isinstance(node, ast.Name):
            if node.id == 't':
                return t
            return ALLOWED_FUNCTIONS[node.id]
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, t)
            right = self._eval_node(node.right, t)
            return ALLOWED_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, t)
            return ALLOWED_OPS[type(node.op)](operand)
        elif isinstance(node, ast.Call):
            func = ALLOWED_FUNCTIONS[node.func.id]
            args = [self._eval_node(arg, t) for arg in node.args]
            return func(*args)


# ================================================================
# LOAD DATA
# ================================================================
@st.cache_resource
def load_engine():
    NMAX = 150
    T0 = 1.0
    n_basis = NMAX + 1
    H5_PATH = "TOOL_data_n_150_small.h5"
    
    with h5py.File(H5_PATH, 'r') as f:
        omega_grid = f['omegas'][:]
        H_real = f['Hreal'][:]
        H_imag = f['Himag'][:]
        W_real = f['Wreal'][:]
        D_real = f['Dreal'][:]
        D_imag = f['Dimag'][:]
        T_basis = float(np.array(f['T']).flat[0])
        L_val = float(np.array(f['L']).flat[0])
    
    n_omega = len(omega_grid)
    HRealT = H_real.transpose(1, 2, 0)
    HImagT = H_imag.transpose(1, 2, 0)
    WRealT = W_real.transpose(1, 2, 0)
    DRealT = D_real.transpose(1, 2, 0)
    DImagT = D_imag.transpose(1, 2, 0)
    
    n_quad = 16000
    t_half = max(20, 8 * T_basis * np.sqrt(2.0 * NMAX + 1))
    t_grid = np.linspace(-t_half, t_half, n_quad)
    dt_val = t_grid[1] - t_grid[0]
    x_grid = t_grid / T_basis
    
    h_basis = np.zeros((n_basis, n_quad))
    h_basis[0] = np.pi**(-0.25) * np.exp(-x_grid**2 / 2.0) / np.sqrt(T_basis)
    if NMAX >= 1:
        h_basis[1] = np.sqrt(2.0) * x_grid * h_basis[0]
    for nn in range(2, NMAX + 1):
        h_basis[nn] = (np.sqrt(2.0 / nn) * x_grid * h_basis[nn - 1]
                       - np.sqrt((nn - 1.0) / nn) * h_basis[nn - 2])
    
    return {
        'omega_grid': omega_grid,
        'HRealT': HRealT, 'HImagT': HImagT,
        'WRealT': WRealT, 'DRealT': DRealT, 'DImagT': DImagT,
        'T_basis': T_basis, 'L_val': L_val,
        'n_omega': n_omega, 'n_basis': n_basis, 'NMAX': NMAX, 'T0': T0,
        't_grid': t_grid, 'dt': dt_val, 'h_basis': h_basis,
    }

E = load_engine()

# ================================================================
# CORE COMPUTATION
# ================================================================

def contract(MT, c_hat):
    intermediate = np.tensordot(MT, c_hat, axes=([1], [0]))
    return c_hat @ intermediate

def analyze(chi_func):
    chi_vals = chi_func(E['t_grid'])
    chi_norm_sq = np.sum(chi_vals**2) * E['dt']
    c_tilde = E['h_basis'] @ (chi_vals * E['dt'])
    c_vec = np.sqrt(E['T0']) * c_tilde
    captured = np.sum(c_tilde**2) / chi_norm_sq if chi_norm_sq > 0 else 0.0
    
    c_norm = np.linalg.norm(c_vec)
    if c_norm == 0:
        z = np.zeros(E['n_omega'])
        return z, z, z, captured, c_tilde, 0.0, 0.0
    
    c_hat = c_vec / c_norm
    
    h_re = contract(E['HRealT'], c_hat)
    h_im = contract(E['HImagT'], c_hat)
    w_re = contract(E['WRealT'], c_hat)
    d_re = contract(E['DRealT'], c_hat)
    d_im = contract(E['DImagT'], c_hat)
    
    gf_re = (h_re - d_im) / 2.0
    gf_im = (h_im + d_re) / 2.0
    abs_gf = np.sqrt(gf_re**2 + gf_im**2)
    
    negativity = np.maximum(0.0, abs_gf - w_re)
    half_delta = np.sqrt(d_re**2 + d_im**2) / 2.0
    
    neg_max = float(np.max(negativity))
    omega_at_max = float(E['omega_grid'][np.argmax(negativity)]) if neg_max > 0 else 0.0
    
    return negativity, half_delta, w_re, captured, c_tilde, neg_max, omega_at_max


# ================================================================
# SWITCHING FUNCTION FAMILIES
# ================================================================

def make_chi(family, params):
    if family == "Gaussian":
        t0 = params['t0']; sig = params['sigma']
        return lambda t: np.exp(-(t - t0)**2 / (2 * sig**2))
    elif family == "Gaussian pair":
        sig = params['sigma']; sep = params['separation']
        return lambda t: (np.exp(-(t - sep)**2 / (2 * sig**2))
                         + np.exp(-(t + sep)**2 / (2 * sig**2)))
    elif family == "Bump":
        T0b = params['T0']
        def bump(t):
            mask = np.abs(t) < T0b
            result = np.zeros_like(t, dtype=float)
            t_safe = np.clip(t[mask]**2, 0, T0b**2 - 1e-15)
            result[mask] = np.exp(-T0b**2 / (T0b**2 - t_safe))
            return result
        return bump
    elif family == "Cos² window":
        T0c = params['T0']
        return lambda t: np.where(np.abs(t) < T0c,
                                   np.cos(np.pi * t / (2 * T0c))**2, 0.0)
    elif family == "Poly bump":
        T0p = params['T0']; nP = params['n']
        return lambda t: np.where(np.abs(t) < T0p,
                                   (1 - np.clip((t/T0p)**2, 0, 1))**nP, 0.0)
    elif family == "Top hat":
        T0th = params['T0']
        return lambda t: np.where(np.abs(t) <= T0th, 1.0, 0.0)
    elif family == "Smooth step":
        T0s = params['T0']; alpha = params['alpha']
        return lambda t: (0.25 * (1 + np.tanh(alpha * (t + T0s)))
                              * (1 - np.tanh(alpha * (t - T0s))))
    elif family == "Gauss × cos":
        sig = params['sigma']; freq = params['freq']
        return lambda t: np.exp(-t**2 / (2 * sig**2)) * np.cos(freq * t)
    elif family == "Lorentzian":
        gamma = params['gamma']
        return lambda t: 1.0 / (1.0 + (t / gamma)**2)
    elif family == "Sech":
        w = params['width']
        return lambda t: 1.0 / np.cosh(t / w)
    else:
        return lambda t: np.exp(-t**2 / 2)


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.header("χ(t) Switching Function")
    
    mode = st.radio("Input mode", ["Preset functions", "Custom function"],
                     horizontal=True)
    
    st.markdown("---")
    
    chi_func = None
    func_label = ""
    
    if mode == "Preset functions":
        family = st.selectbox("Family", [
            "Gaussian", "Gaussian pair", "Bump", "Cos² window",
            "Poly bump", "Top hat", "Smooth step", "Gauss × cos",
            "Lorentzian", "Sech"
        ])
        func_label = family
        params = {}
        
        if family == "Gaussian":
            params['t0'] = st.slider("t₀", -3.0, 3.0, 0.0, 0.1)
            params['sigma'] = st.slider("σ", 0.3, 3.0, 1.0, 0.05)
        elif family == "Gaussian pair":
            params['sigma'] = st.slider("σ", 0.3, 3.0, 1.0, 0.05)
            params['separation'] = st.slider("separation", 0.1, 3.0, 1.0, 0.1)
        elif family == "Bump":
            params['T0'] = st.slider("T₀", 0.3, 3.0, 1.0, 0.05)
        elif family == "Cos² window":
            params['T0'] = st.slider("T₀", 0.3, 3.0, 1.0, 0.05)
        elif family == "Poly bump":
            params['T0'] = st.slider("T₀", 0.3, 3.0, 1.0, 0.05)
            params['n'] = st.slider("n (power)", 1, 8, 3, 1)
        elif family == "Top hat":
            params['T0'] = st.slider("T₀", 0.3, 3.0, 1.0, 0.05)
        elif family == "Smooth step":
            params['T0'] = st.slider("T₀", 0.3, 3.0, 1.0, 0.05)
            params['alpha'] = st.slider("α (steepness)", 1.0, 20.0, 5.0, 0.5)
        elif family == "Gauss × cos":
            params['sigma'] = st.slider("σ", 0.3, 3.0, 1.0, 0.05)
            params['freq'] = st.slider("ω_cos", 0.5, 15.0, 3.0, 0.5)
        elif family == "Lorentzian":
            params['gamma'] = st.slider("γ", 0.3, 3.0, 1.0, 0.05)
        elif family == "Sech":
            params['width'] = st.slider("width", 0.3, 3.0, 1.0, 0.05)
        
        chi_func = make_chi(family, params)
    
    else:  # Custom function
        st.markdown("Enter χ(t) as a function of `t`.")
        st.markdown("The function will be **truncated** to the window ±T.")
        
        expr = st.text_input("χ(t) =", value="exp(-t**2 / 2)")
        trunc = st.slider("Truncation window (±T)", 0.5, 5.0, 2.5, 0.1)
        
        st.caption("**Allowed:** `+  -  *  /  **  ( )`")
        st.caption("**Functions:** `exp, log, sqrt, abs, sin, cos, tan, "
                   "sinh, cosh, tanh, sech, arctan, arcsin, arccos, "
                   "heaviside, step, sign`")
        st.caption("**Constants:** `pi, e`")
        
        func_label = f"Custom: {expr}"
        
        try:
            parser = SafeMathParser(expr)
            def chi_func_custom(t, _parser=parser, _trunc=trunc):
                raw = _parser.evaluate(t)
                raw = np.atleast_1d(np.asarray(raw, dtype=float))
                if raw.shape != t.shape:
                    raw = np.broadcast_to(raw, t.shape).copy()
                raw[np.abs(t) > _trunc] = 0.0
                return raw
            chi_func = chi_func_custom
        except ValueError as err:
            st.error(f"⚠️ {err}")
            chi_func = None
        except Exception as err:
            st.error(f"⚠️ Could not parse: {err}")
            chi_func = None
    
    # ---- About section ----
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "**Entanglement Harvesting Explorer** (Alpha version)\n\n"
        "by Marcos Morote-Balboa & T. Rick Perche\n\n"
        "Source code and Mathematica (.wl) files (coming soon): "
        "[GitHub](https://github.com/marquitos2001/entanglement-harvesting-explorer)"
    )
    st.markdown("#### References")
    st.markdown(
        "1. M. Morote-Balboa, T. R. Perche, "
        "\"Optimization of entanglement harvesting with arbitrary temporal profiles:the limit of second order perturbation theory,\" "
        "[arXiv:2604.06303](https://doi.org/10.48550/arXiv.2604.06303)\n"
        "3. M. Morote-Balboa, T. R. Perche, "
        "[arXiv:2604.06303](https://doi.org/10.48550/arXiv.2604.06303) — SER\n"
        "2. E. Tjoa, E. Martín-Martínez, "
        "\"When entanglement harvesting is not really harvesting,\" "
        "[Phys. Rev. D **104**, 125005 (2021)](https://doi.org/10.1103/PhysRevD.104.125005) — CMEE\n"
    )


# ================================================================
# MAIN AREA
# ================================================================

st.title("Entanglement Harvesting Explorer")
st.caption(
    f"Hermite basis: N_max = {E['NMAX']}  |  "
    f"T = {E['T_basis']}  |  L = {E['L_val']} |  "
    f"ω points = {E['n_omega']}"
)

if chi_func is None:
    st.warning("Enter a valid function to see results.")
    st.stop()

# ---- COMPUTE ----
neg, half_delta, w_val, captured, c_tilde, neg_max, omega_at_max = analyze(chi_func)

# Reconstruct chi from basis coefficients (this is what the computation actually uses)
chi_reconstructed_full = c_tilde @ E['h_basis']   # on the fine t_grid
# Interpolate to the plotting grid
t_plot = np.linspace(-6, 6, 1000)
chi_reconstructed = np.interp(t_plot, E['t_grid'], chi_reconstructed_full)

# Compute CMEE and SER at peak (safe: 0 if no entanglement)
if neg_max > 0:
    idx_peak = np.argmax(neg)
    abs_delta_peak = 2.0 * half_delta[idx_peak]
    w_peak = w_val[idx_peak]
    neg_peak = neg[idx_peak]
    cmee = max(0.0, (abs_delta_peak - w_peak) / neg_peak)
    ser = abs_delta_peak / neg_peak
else:
    cmee = 0.0
    ser = 0.0

# ---- METRICS ROW ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Max negativity", f"{neg_max:.3e}")
col2.metric("at Ω =", f"{omega_at_max:.4f}")
col3.metric("CMEE  I[ρ]", f"{cmee:.4f}" if neg_max > 0 else "0")
col4.metric("SER  Θ[ρ]", f"{ser:.4f}" if neg_max > 0 else "0")

# ---- PLOTS ----
try:
    chi_plot = chi_func(t_plot)
except Exception as err:
    st.error(f"Error evaluating function: {err}")
    st.stop()

# Top row: original chi(t) and reconstructed comparison
top_left, top_right = st.columns(2)

with top_left:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=t_plot, y=chi_plot,
        mode='lines', line=dict(color='green', width=2),
        name='χ(t) input',
    ))
    fig1.update_layout(
        title=f"χ(t) — {func_label}",
        xaxis_title="t", yaxis_title="χ(t)",
        height=350, margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig1, use_container_width=True)

with top_right:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=t_plot, y=chi_plot,
        mode='lines', line=dict(color='green', width=2, dash='dot'),
        name='χ(t) input',
    ))
    fig2.add_trace(go.Scatter(
        x=t_plot, y=chi_reconstructed,
        mode='lines', line=dict(color='blue', width=2),
        name='χ(t) reconstructed',
    ))
    fig2.update_layout(
        title=f"Basis reconstruction",
        xaxis_title="t", yaxis_title="χ'(t)",
        height=350, margin=dict(l=50, r=20, t=40, b=40),
        #legend=dict(x=0.55, y=0.95, font=dict(size=11)),
    )
    st.plotly_chart(fig2, use_container_width=True)
    
# Bottom: negativity (full width)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=E['omega_grid'].tolist(), y=neg.tolist(),
    mode='lines',
    name=r'λ⁻² 𝒩',
    line=dict(color='blue', width=2),
))
fig3.add_trace(go.Scatter(
    x=E['omega_grid'].tolist(), y=half_delta.tolist(),
    mode='lines',
    name=r'½ λ⁻² |Δ|',
    line=dict(color='red', width=2),
))

# Determine x-range
if neg_max > 0:
    nonzero = np.where(neg > 0)[0]
    om_max = min(E['omega_grid'][nonzero[-1]] * 1.5, E['omega_grid'][-1])
else:
    om_max = E['omega_grid'][-1]

# Determine y-range: peak sits at ~75% of axis height
if neg_max > 0:
    visible_mask = (E['omega_grid'] >= 0) & (E['omega_grid'] <= om_max)
    neg_vis = neg[visible_mask]
    vis_neg_peak = float(np.max(neg_vis))
    y_top = vis_neg_peak / 0.70
else:
    y_top = 1.0

fig3.update_layout(
    title=(
        f"Negativity — max = {neg_max:.3e}  at Ω = {omega_at_max:.4f}"
        f"   |   CMEE = {cmee:.4f}   |   SER = {ser:.4f}"
        if neg_max > 0
        else "No entanglement detected"
    ),
    xaxis_title="Ω",
    xaxis=dict(range=[0, float(om_max)]),
    yaxis=dict(
        range=[0, y_top],
        autorange=False,            # KEY: prevent plotly from overriding
        exponentformat='power',
        showexponent='all',
    ),
    height=400,
    margin=dict(l=60, r=20, t=40, b=40),
    legend=dict(x=0.75, y=0.95, font=dict(size=14)),
)
st.plotly_chart(fig3, use_container_width=True)
