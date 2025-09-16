import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Canonical parameter options

def piecewise_grid(start=0.0, split=0.9, stop=1.0, step1=0.1, step2=0.01, decimals=6, include_stop=True):
    # segment 1: start → split (inclusive)
    n1 = int(round((split - start) / step1)) + 1
    seg1 = np.linspace(start, split, n1, endpoint=True)

    # segment 2: split → stop (drop the split to avoid duplicate)
    n2 = int(round((stop - split) / step2)) + 1
    seg2 = np.linspace(split, stop, n2, endpoint=include_stop)[1:]

    grid = np.concatenate([seg1, seg2])
    return np.round(grid, decimals)

# examples
CL_OPTIONS = piecewise_grid(0.0, 0.8, 1.0, step1=0.1, step2=0.01)   # 0.0, 0.1, ..., 0.9, 0.91, ..., 1.0
R_OPTIONS = piecewise_grid(0.0, 0.8, 1.0, step1=0.1, step2=0.01)

def snap_defaults(vals, options, tol=1e-9):
    """Return vals that exactly match one of options within tol (avoid 0.9900000000000001 issues)."""
    snapped = []
    for v in vals:
        for o in options:
            if abs(float(v) - float(o)) <= tol:
                snapped.append(o)
                break
    # keep order, deduplicate
    out = []
    for x in snapped:
        if x not in out:
            out.append(x)
    return out

st.set_page_config(page_title="Reliability Verification based on success-run tests (Zero-failure)", layout="wide")

# ---------- Header ----------
st.markdown("""
<style>
  .block-container { padding-top: 3rem !important; }  /* tweak 3rem as needed */
</style>
""", unsafe_allow_html=True)
st.title("Reliability Verification based on Success-Run Tests")
st.caption("Zero-failure (success-run) relationships:  R = (1−CL)^(1/n),  n = ceil(ln(1−CL)/ln(R)),  CL = 1 − R^n")

# ---------- Styling (dark theme + compact tables) ----------
st.markdown("""
<style>
.stApp, .stApp > header, .stApp > iframe { background-color: #111827 !important; }
section[data-testid="stSidebar"] { background-color: #0f172a !important; }
.block-container { padding-top: 1.25rem; }
h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stCaption { color: #e5e7eb !important; }
.stAlert { background: #111827; border: 1px solid #374151; }
.stMarkdown a { color: #93c5fd !important; }
.stDataFrame table { font-size: 12px; }
.stDataFrame td, .stDataFrame th { padding: 2px 6px !important; }
div[data-testid="stVerticalBlock"] > div { background-color: #1f2937; border-radius: 12px; padding: 0.5rem 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers (zero-failure model) ----------
def r_from_cl_n(CL: float, n: int) -> float:
    CL = float(np.clip(CL, 1e-12, 1-1e-12))
    n = max(1, int(n))
    return (1.0 - CL) ** (1.0 / n)

def n_from_r_cl(R: float, CL: float) -> int:
    R = float(np.clip(R, 1e-12, 1-1e-12))
    CL = float(np.clip(CL, 1e-12, 1-1e-12))
    return max(1, math.ceil(math.log(1.0 - CL) / math.log(R)))

def cl_from_r_n(R: float, n: int) -> float:
    R = float(np.clip(R, 1e-12, 1-1e-12))
    n = max(1, int(n))
    return 1.0 - (R ** n)

# ---------- Sidebar (layout + chart controls) ----------
with st.sidebar:
    st.header("Layout")
    table_h = st.slider("Table height (px)", 140, 360, 210, 10)
    plot_h  = st.slider("Plot height (px)", 400, 1200, 700, 10)
    st.markdown("---")
    st.header("Chart options")
    max_n   = st.slider("Max n on charts", 20, 2000, 400, 10)
    r_range = st.slider("R range for charts", 0.80, 0.999, (0.90, 0.999))
    # Presets for parameters
    default_cls = [0.80, 0.90, 0.95, 0.99]
    default_rs  = [0.90, 0.95, 0.975, 0.99]

# ---------- Main layout: left inputs & results | right dynamic chart ----------
left, right = st.columns([1, 1])

with left:
    st.subheader("Calculator")

    calc_mode = st.radio(
        "Select calculation",
        ["Reliability R from (CL, n)",
         "Samples n from (R, CL)",
         "Confidence CL from (R, n)"],
        horizontal=False
    )

    if calc_mode == "Reliability R from (CL, n)":
        c1, c2 = st.columns(2)
        CL = c1.number_input("Confidence level CL (0–1)", min_value=0.0, max_value=0.999999, value=0.9, step=0.01, format="%.6f")
        n  = c2.number_input("Number of samples n (≥1)", min_value=1, max_value=100000, value=20, step=1)
        R  = r_from_cl_n(CL, n)

        st.markdown(f"**Result**:  **R** = (1 − CL)\u2071⁄ⁿ = **{R:.6f}**")

        # Nearby table
        n_grid = np.arange(max(1, n-3), n+4)
        df = pd.DataFrame({
            "n": n_grid,
            f"R at CL={CL:.3f}": np.round([(1-CL)**(1.0/x) for x in n_grid], 6)
        })
        st.dataframe(df, use_container_width=True, height=table_h)

    elif calc_mode == "Samples n from (R, CL)":
        c1, c2 = st.columns(2)
        R  = c1.number_input("Target reliability R (0–1)", min_value=0.5, max_value=0.999999, value=0.95, step=0.01, format="%.6f")
        CL = c2.number_input("Confidence level CL (0–1)", min_value=0.5, max_value=0.999999, value=0.90, step=0.01, format="%.6f")
        n  = n_from_r_cl(R, CL)

        st.markdown(f"**Result**:  **n** ≥ ceil( ln(1−CL) / ln(R) ) = **{n}**  (0 failures)")

        # Nearby table
        R_vals = np.array(sorted(set([0.90, 0.95, 0.975, 0.99, float(R)])))
        CL_vals = np.array([0.80, 0.90, 0.95, 0.99, float(CL)])
        rows = []
        for rv in R_vals:
            for clv in CL_vals:
                rows.append({"R": rv, "CL": clv, "n (ceil)": n_from_r_cl(rv, clv)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=table_h)

    else:  # "Confidence CL from (R, n)"
        c1, c2 = st.columns(2)
        R = c1.number_input("Reliability R (0–1)", min_value=0.5, max_value=0.999999, value=0.95, step=0.01, format="%.6f")
        n = c2.number_input("Number of samples n (≥1)", min_value=1, max_value=100000, value=20, step=1)
        CL = cl_from_r_n(R, n)

        st.markdown(f"**Result**:  **CL** = 1 − Rⁿ = **{CL:.6f}**")

        # Nearby table
        n_grid = np.arange(max(1, n-3), n+4)
        df = pd.DataFrame({
            "n": n_grid,
            f"CL at R={R:.3f}": np.round([1.0 - (R ** x) for x in n_grid], 6)
        })
        st.dataframe(df, use_container_width=True, height=table_h)

with right:
    st.subheader("Chart")

    fig = go.Figure()

    if calc_mode == "Reliability R from (CL, n)":
        # Y: R, X: n, Parameter: CL (multiple lines)
        # pick CL lines; ensure current CL is included
        cl_defaults = snap_defaults(
            (default_cls + ([float(CL)] if 'CL' in locals() else [])),
            CL_OPTIONS
        )
        cl_choices = st.multiselect(
            "Confidence levels to plot (parameter)",
            options=CL_OPTIONS,
            default=cl_defaults
        )
        n_space = np.arange(1, max_n + 1)
        for cl in sorted(set(cl_choices)):
            R_vals = (1.0 - cl) ** (1.0 / n_space)
            fig.add_trace(go.Scatter(
                x=n_space, y=R_vals, mode="lines",
                name=f"CL={cl:.3f}",
                line=dict(width=4),
                hovertemplate="n: %{x}<br>R: %{y:.6f}<extra></extra>"
            ))
        fig.update_xaxes(
            title_text="Samples n",
            showgrid=True,
            title_font=dict(size=18),
            tickfont=dict(size=14)
        )
        fig.update_yaxes(
            title_text="Reliability R",
            range=[0, 1],
            showgrid=True,
            title_font=dict(size=18),
            tickfont=dict(size=14)
        )

    elif calc_mode == "Samples n from (R, CL)":
        # Y: n, X: R, Parameter: CL (multiple lines)
        cl_defaults = snap_defaults(
            (default_cls + ([float(CL)] if 'CL' in locals() else [])),
            CL_OPTIONS
        )
        cl_choices = st.multiselect(
            "Confidence levels to plot (parameter)",
            options=CL_OPTIONS,
            default=cl_defaults
        )
        R_min, R_max = r_range
        R_space = np.linspace(R_min, R_max, 400)
        R_space = np.clip(R_space, 1e-12, 1-1e-12)
        for cl in sorted(set(cl_choices)):
            n_vals = np.log(1.0 - cl) / np.log(R_space)
            fig.add_trace(go.Scatter(
                x=R_space, y=np.ceil(n_vals), mode="lines",
                name=f"CL={cl:.3f}",
                line=dict(width=4),
                hovertemplate="R: %{x:.6f}<br>n (ceil): %{y:.0f}<extra></extra>"
            ))
        fig.update_xaxes(
            title_text="Reliability R",
            showgrid=True,
            range=[R_min, R_max],
            title_font=dict(size=18),
            tickfont=dict(size=14)
        )
        fig.update_yaxes(
            title_text="Samples n",
            showgrid=True,
            title_font=dict(size=18),
            tickfont=dict(size=14)
        )

    else:  # "Confidence CL from (R, n)"
        # Y: CL, X: n, Parameter: R (multiple lines)
        r_defaults = snap_defaults(
            (default_rs + ([float(R)] if 'R' in locals() else [])),
            R_OPTIONS
        )
        r_choices = st.multiselect(
            "Reliability values to plot (parameter)",
            options=R_OPTIONS,
            default=r_defaults
        )
        n_space = np.arange(1, max_n + 1)
        for rv in sorted(set(r_choices)):
            CL_vals = 1.0 - (np.clip(rv, 1e-12, 1-1e-12) ** n_space)
            fig.add_trace(go.Scatter(
                x=n_space, y=CL_vals, mode="lines",
                name=f"R={rv:.3f}",
                line=dict(width=4),
                hovertemplate="n: %{x}<br>CL: %{y:.6f}<extra></extra>"
            ))
        fig.update_xaxes(
            title_text="Samples n",
            showgrid=True,
            title_font=dict(size=18),
            tickfont=dict(size=14)
        )
        fig.update_yaxes(
            title_text="Confidence level CL",
            range=[0, 1],
            showgrid=True,
            title_font=dict(size=18),
            tickfont=dict(size=14)
        )

    fig.update_layout(
        # title="Design curves",
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.65)",
            bordercolor="black", borderwidth=1,
            font=dict(size=14)
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=plot_h,
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font=dict(color="#e5e7eb")
    )
    st.plotly_chart(fig, use_container_width=True)
