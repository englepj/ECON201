# === CBC Econ Simulation Lobby ===
# Streamlit app with sidebar game selection and CBC branding

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random
import pandas as pd
import getpass
import datetime

# ----------------------------
# Helpers
# ----------------------------
def export_csv_button(label: str, df: pd.DataFrame, prefix: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{getpass.getuser()}_{timestamp}.csv"
    st.download_button(label, data=df.to_csv(index=False), file_name=filename, mime="text/csv")


def triangle_area(base: float, height: float) -> float:
    return 0.5 * base * height


# ----------------------------
# CBC Color Palette
# ----------------------------
cbc_blue = "#00457C"
cbc_lightblue = "#dbeefd"
cbc_gold = "#fce58b"
cbc_orange = "#fbb885"
cbc_darkorange = "#f28a6c"
cbc_darkblue = "#00457c"

st.set_page_config(page_title="ECON&201 Simulations", page_icon="📊", layout="wide")

# ----------------------------
# Global Styling
# ----------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora&display=swap');

html, body, [class*="css"] {{
    font-family: 'Lora', serif !important;
    background-color: {cbc_lightblue};
}}

h1, h2, h3 {{
    color: {cbc_blue};
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}}

.block-container {{
    padding: 2rem 2rem 2rem 2rem !important;
}}

.element-container {{
    margin-bottom: 1.5rem !important;
}}

button[kind="primary"] {{
    border-radius: 10px;
    background-color: {cbc_blue};
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
}}

button[kind="primary"]:hover {{
    background-color: #002d52;
}}

.main-container {{
    background-color: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin: 1rem;
}}

.main-container.alt {{
    background-color: #f8fbff;
}}

.fade-in {{
    animation: fadeIn 0.8s ease-in;
}}

@keyframes fadeIn {{
    0% {{opacity: 0;}}
    100% {{opacity: 1;}}
}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Header Branding
# ----------------------------
st.markdown(
    f"""
<div id="top" style='background-color:{cbc_lightblue}; padding: 10px; border-radius: 10px;'>
    <h1 style='text-align: center;'>CBC Economics Simulations/Problems</h1>
    <h2 style='text-align: center;'> for Fall 2026 ECON&201 with Prof Engle </h2>
    <p style='text-align: center; color: black;'>Choose a simulation / problem set to explore key economic concepts.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Pages
# ----------------------------

import streamlit as st
import numpy as np
import plotly.graph_objects as go


def page_price_discrimination_segmentation():
    import numpy as np, streamlit as st, plotly.graph_objects as go

    try: BL, OR, GD, GR = cbc_blue, cbc_darkorange, cbc_gold, cbc_gray
    except: BL, OR, GD, GR = "#4E79A7", "#F28E2B", "#EDC948", "#8C8C8C"

    st.markdown("## 🏷️ Price Discrimination Builder (1st, 2nd, 3rd Degree)")
    st.write("Same demand/MC sliders feed all graphs; only the **labels/story** change by degree.")

    # ---- tiny helpers ----
    ht = lambda n: f"Q=%{{x:.0f}}<br>{n}=$%{{y:,.0f}}<extra></extra>"
    def add(fig,x,y,n,c,dash=None,shape=None,mode="lines"):
        fig.add_trace(go.Scatter(x=x,y=y,mode=mode,name=n,
                                 line=dict(color=c,dash=dash,shape=shape),
                                 hovertemplate=ht(n)))
    def pt(fig,q,p,txt,c):
        fig.add_trace(go.Scatter(x=[q],y=[p],mode="markers+text",text=[txt],textposition="top right",
                                 marker=dict(size=9,color=c),showlegend=False,
                                 hovertemplate="Q=%{x:.0f}<br>P=$%{y:,.0f}<extra></extra>"))
    def lay(fig,t,ytitle="$ per unit"):
        fig.update_layout(title=t,template="simple_white",
                          xaxis_title="Quantity (Q)",yaxis_title=ytitle,
                          xaxis=dict(dtick=5,rangemode="tozero"),
                          legend=dict(orientation="h",y=1.12,x=0),
                          margin=dict(t=90,b=60,l=50,r=20))

    curve = lambda P,A,B: (lambda Q:(Q[np.argsort(Q)],P[np.argsort(Q)]))(np.maximum(0,(A-P)/B))
    mc = lambda q,a,b: a + b*q
    qstar = lambda A,B,a,b,k: max(0,(A-a)/(((2*B) if k=="monopoly" else B)+b)) if (((2*B) if k=="monopoly" else B)+b)>0 else 0

    # ---- shared sliders (apply to ALL tabs) ----
    st.markdown("### ⚙️ Shared Market Settings (apply to ALL graphs)")
    c1,c2,c3 = st.columns(3)
    with c1:
        A_inel = st.slider("Inelastic demand intercept (A)", 50, 400, 300, 5)
        B_inel = st.slider("Inelastic demand slope (B)", 1, 80, 40, 1)
    with c2:
        A_el = st.slider("Elastic demand intercept (A)", 50, 400, 150, 5)
        B_el = st.slider("Elastic demand slope (B)", 1, 80, 15, 1)
    with c3:
        mc0 = st.slider("MC intercept", 0, 200, 70, 1)
        mc1 = st.slider("MC slope", 0.0, 20.0, 0.0, 0.5)
        Pmax = st.slider("Max price for graph", 100, 500, 350, 10)

    # ---- demand curves (same everywhere) ----
    P = np.linspace(0, Pmax, 400)
    QI, PI = curve(P, A_inel, B_inel)
    QE, PE = curve(P, A_el, B_el)
    QtP = np.maximum(0,(A_inel-P)/B_inel) + np.maximum(0,(A_el-P)/B_el)
    idx = np.argsort(QtP); QW, PW = QtP[idx], P[idx]

    # ---- 2nd degree (Option C): Average Price per Unit (AP) schedule ----
    st.markdown("### 🥈 2nd-Degree (Option C): Average price per unit falls when you buy more")
    q0 = 3
    avgWTP = (A_inel + A_el)/2
    p_first = int(np.clip(round(0.70*avgWTP), 1, Pmax))   # “small purchase” per-unit price
    p_more  = int(np.clip(round(0.55*p_first), 1, Pmax))  # “bulk” per-unit price
    st.info(
        f"Rule: Units **1–{q0}** cost **${p_first} each**; units **{q0+1}+** cost **${p_more} each**. "
        "We graph **Average Price per Unit** = (Total Bill / Q)."
    )

    # ---- figures ----
    def fig_first(title, A, B, Qc, Pc, whole=False):
        fig = go.Figure()
        add(fig, Qc, Pc, "Demand", BL)
        q = np.linspace(0, float(np.max(Qc)), 250)
        add(fig, q, mc(q, mc0, mc1), "MC", OR)

        if whole:
            p = np.interp(q, Qc, Pc); tr = p*q
            mr = np.gradient(tr, q); add(fig, q, mr, "MR (approx)", GR, dash="dash")
            qs = float(q[np.argmin(abs(p - mc(q, mc0, mc1)))])
        else:
            add(fig, q, A - 2*B*q, "MR", GR, dash="dash")
            qs = float(qstar(A, B, mc0, mc1, "efficient"))

        pt(fig, qs, float(np.interp(qs, Qc, Pc)), "P=MC", GD)
        lay(fig, title); return fig

    def fig_second_AP(title, Qc, Pc):
        fig = go.Figure()
        add(fig, Qc, Pc, "Demand", BL)
        qmax = int(max(10, np.ceil(np.max(Qc)))) if len(Qc) else 10

        # Average price per unit under the discount rule
        Qb = np.arange(1, qmax + 1)
        bill = np.where(Qb <= q0, p_first*Qb, p_first*q0 + p_more*(Qb - q0))
        ap = bill / Qb

        add(fig, Qb, ap, "Average Price per Unit (Total Bill / Q)", GD, mode="lines+markers")
        add(fig, Qb, np.full_like(Qb, p_more, dtype=float), f"Bulk per-unit price = ${p_more}", GR, dash="dash")
        add(fig, Qb, np.full_like(Qb, p_first, dtype=float), f"Small-batch per-unit price = ${p_first}", GR, dash="dot")

        q = np.linspace(0, qmax, 250)
        add(fig, q, mc(q, mc0, mc1), "MC", OR)

        fig.add_vline(x=q0, line_width=1, line_dash="dot", opacity=0.6)
        fig.add_annotation(
            x=0.01, y=0.99, xref="paper", yref="paper", showarrow=False, align="left",
            text=(f"<b>Read this (2nd-degree PD)</b><br>"
                  f"• Buying more lowers your <b>average</b> price per unit<br>"
                  f"• After Q={q0}, each extra unit is cheaper (${p_more})<br>"
                  f"• Price-sensitive shoppers (CostCo-type) buy enough to drop AP"),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1
        )
        lay(fig, title); return fig

    def fig_third(title, A, B, Qc, Pc, whole=False):
        fig = go.Figure()
        add(fig, Qc, Pc, "Demand", BL)
        q = np.linspace(0, float(np.max(Qc)), 250)

        if whole:
            p = np.interp(q, Qc, Pc); tr = p*q
            mr = np.gradient(tr, q); add(fig, q, mr, "MR (approx)", GR)
            add(fig, q, mc(q, mc0, mc1), "MC", OR)
            ok = mr >= mc(q, mc0, mc1)
            qs = float(q[np.where(ok)[0][-1]]) if ok.any() else 0.0
        else:
            add(fig, q, A - 2*B*q, "MR", GR)
            add(fig, q, mc(q, mc0, mc1), "MC", OR)
            qs = float(qstar(A, B, mc0, mc1, "monopoly"))

        pt(fig, qs, float(np.interp(qs, Qc, Pc)), "MR=MC", GD)
        lay(fig, title); return fig

    # ---- tabs ----
    t1, t2, t3 = st.tabs(["🥇 1st Degree (Perfect PD)", "🥈 2nd Degree (Average Price)", "🥉 3rd Degree (Group Pricing)"])

    with t1:
        cols = st.columns(3)
        labs = ["I Have ESP, Inc's Product (inelastic)", "I Have ESP, Inc's Product (elastic)", "Whole Market"]
        for col, lab, A, B, Qc, Pc, is_whole in zip(
            cols, labs,
            [A_inel, A_el, None], [B_inel, B_el, None],
            [QI, QE, QW], [PI, PE, PW],
            [False, False, True]
        ):
            col.plotly_chart(fig_first(lab, A, B, Qc, Pc, whole=is_whole), use_container_width=True)

    with t2:
        cols = st.columns(3)
        labs = ["Safeway (inelastic)", "CostCo (elastic)", "Whole Market"]
        for col, lab, Qc, Pc in zip(cols, labs, [QI, QE, QW], [PI, PE, PW]):
            col.plotly_chart(fig_second_AP(lab, Qc, Pc), use_container_width=True)

    with t3:
        cols = st.columns(3)
        labs = ["Business Travel (inelastic)", "Leisure Travel (elastic)", "Whole Market"]
        for col, lab, A, B, Qc, Pc, is_whole in zip(
            cols, labs,
            [A_inel, A_el, None], [B_inel, B_el, None],
            [QI, QE, QW], [PI, PE, PW],
            [False, False, True]
        ):
            col.plotly_chart(fig_third(lab, A, B, Qc, Pc, whole=is_whole), use_container_width=True)

def page_supply_shock_elasticity():
    st.markdown("## 📉 Supply Shock & Demand Elasticity (Supply-only)")

    # --- Scenario constants ---
    P0, Q0 = 10.0, 100.0
    supply_slope, supply_shock = 8.0, 50.0  # shock fixed at +50 (right shift)

    # --- CBC-ish colors ---
    cbc_light_blue, cbc_yellow = "#7EC8FF", "#F2C94C"
    cbc_gold, cbc_blue = "#C9A227", "#0B3D91"

    # --- Control: target elasticity at (P0,Q0) ---
    e = st.slider("Select |PED| at the initial equilibrium (P0,Q0)", 0.2, 5.0, 1.2, 0.1)

    # --- Linear demand calibrated so point elasticity at (P0,Q0) equals e ---
    # Qd(P) = a - bP,  |PED| at (P0,Q0) = b*(P0/Q0)  =>  b = e*Q0/P0,  a = Q0 + bP0
    b = e * Q0 / P0
    a = Q0 + b * P0
    Qd = lambda P: np.maximum(0.0, a - b * np.asarray(P, float))

    # --- Linear supply (and shifted supply) passing through (P0,Q0) ---
    s_int = Q0 - supply_slope * P0
    Qs0 = lambda P: s_int + supply_slope * np.asarray(P, float)
    Qs1 = lambda P: (s_int + supply_shock) + supply_slope * np.asarray(P, float)

    # --- Closed-form equilibria (linear vs linear) ---
    denom = supply_slope + b
    P_eq0 = (a - s_int) / denom
    P_eq1 = (a - (s_int + supply_shock)) / denom
    Q_eq0, Q_eq1 = float(Qd(P_eq0)), float(Qd(P_eq1))

    # --- Plot grids ---
    P_grid = np.linspace(1, 30, 300)
    Qd_grid, Qs0_grid, Qs1_grid = Qd(P_grid), Qs0(P_grid), Qs1(P_grid)

    def add_curve(fig, Q, P, name, color, dash="solid", w=3):
        fig.add_trace(go.Scatter(x=Q, y=P, mode="lines", name=name, line=dict(color=color, dash=dash, width=w)))

    def add_eq(fig, Q, P, name, color, pos="top right"):
        fig.add_trace(go.Scatter(
            x=[Q], y=[P], mode="markers+text", name=name,
            marker=dict(size=10, color=color),
            text=[f"{name}<br>P={P:.2f}<br>Q={Q:.1f}"], textposition=pos
        ))

    # ----------------------------
    # Graph 1: Market equilibrium before/after
    # ----------------------------
    fig1 = go.Figure()
    add_curve(fig1, Qd_grid, P_grid, "Demand (linear)", cbc_blue)
    add_curve(fig1, Qs0_grid, P_grid, "Supply (S0)", "#444444")
    add_curve(fig1, Qs1_grid, P_grid, "Supply after shock (S1 = S0 + 50)", cbc_gold, "dash")
    add_eq(fig1, Q_eq0, P_eq0, "E0", "#111111")
    add_eq(fig1, Q_eq1, P_eq1, "E1", cbc_gold)
    fig1.update_layout(
        title="Market: Supply Shock (+50) with Linear Demand (calibrated to |PED| at P0,Q0)",
        xaxis_title="Quantity (Q)", yaxis_title="Price (P)", legend_title="Curves", height=520
    )
    fig1.update_xaxes(rangemode="tozero")
    fig1.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig1, use_container_width=True)

    # ----------------------------
    # Graph 2: Equilibrium movement
    # ----------------------------
    dP, dQ = P_eq1 - P_eq0, Q_eq1 - Q_eq0
    fig2 = go.Figure()
    add_curve(fig2, Qd_grid, P_grid, "Demand (linear)", cbc_blue)
    add_curve(fig2, Qs0_grid, P_grid, "Supply (S0)", "#444444")
    add_curve(fig2, Qs1_grid, P_grid, "Supply after shock (S1)", cbc_gold, "dash")
    fig2.add_trace(go.Scatter(
        x=[Q_eq0, Q_eq1], y=[P_eq0, P_eq1], mode="lines+markers",
        name="Equilibrium shift", line=dict(width=2), marker=dict(size=8)
    ))
    fig2.add_annotation(
        x=Q_eq1, y=P_eq1, text=f"ΔP = {dP:+.2f}<br>ΔQ = {dQ:+.1f}",
        showarrow=True, arrowhead=2, ax=40, ay=-40
    )
    fig2.update_layout(title="Before vs After: Equilibrium Movement", xaxis_title="Quantity (Q)", yaxis_title="Price (P)", height=520)
    fig2.update_xaxes(rangemode="tozero")
    fig2.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # Graph 3: Demand-only + pick two points + compute PED (arc) + revenue boxes
    # ----------------------------
    st.markdown("### 📌 Demand Elasticity Calculator (Pick Two Points)")

    colA, colB = st.columns(2)
    with colA:
        P1 = st.slider("Point 1: Price (P₁)", 1.0, 30.0, float(np.clip(P_eq0, 1, 30)), 0.5)
    with colB:
        P2 = st.slider("Point 2: Price (P₂)", 1.0, 30.0, float(np.clip(P_eq1, 1, 30)), 0.5)

    Q1, Q2 = float(Qd(P1)), float(Qd(P2))
    avgQ, avgP = (Q1 + Q2) / 2, (P1 + P2) / 2
    dQ_pts, dP_pts = (Q2 - Q1), (P2 - P1)
    ped_arc = np.nan if abs(dP_pts) < 1e-9 else (dQ_pts / avgQ) / (dP_pts / avgP)

    fig3 = go.Figure()
    add_curve(fig3, Qd_grid, P_grid, "Demand (linear)", cbc_blue)

    # Revenue rectangles (TR = P*Q) at each point
    for Q, P, fc in [(Q1, P1, cbc_light_blue), (Q2, P2, cbc_yellow)]:
        fig3.add_shape(type="rect", x0=0, y0=0, x1=Q, y1=P, fillcolor=fc, opacity=0.25, line_width=0)

    fig3.add_trace(go.Scatter(
        x=[Q1], y=[P1], mode="markers+text", name="Point 1",
        marker=dict(size=10, color=cbc_light_blue),
        text=[f"P₁={P1:.2f}<br>Q₁={Q1:.1f}<br>TR₁={P1*Q1:,.0f}"], textposition="top right"
    ))
    fig3.add_trace(go.Scatter(
        x=[Q2], y=[P2], mode="markers+text", name="Point 2",
        marker=dict(size=10, color=cbc_yellow),
        text=[f"P₂={P2:.2f}<br>Q₂={Q2:.1f}<br>TR₂={P2*Q2:,.0f}"], textposition="bottom right"
    ))

    fig3.update_layout(
        title="Demand Only (Linear): Pick Two Points, Compute PED, and Compare Revenue Boxes",
        xaxis_title="Quantity (Q)", yaxis_title="Price (P)", height=560
    )
    fig3.update_xaxes(rangemode="tozero")
    fig3.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig3, use_container_width=True)

    if np.isnan(ped_arc):
        st.warning("PED is undefined because P₁ and P₂ are the same (ΔP = 0).")
    else:
        st.markdown("**Arc Price Elasticity of Demand (between the two points):**")
        st.latex(r"PED \;=\; \frac{\Delta Q / \overline{Q}}{\Delta P / \overline{P}}")
        st.write(f"ΔQ = {Q2:.2f} − {Q1:.2f} = **{dQ_pts:.2f}**  |  Q̄ = **{avgQ:.2f}**")
        st.write(f"ΔP = {P2:.2f} − {P1:.2f} = **{dP_pts:.2f}**  |  P̄ = **{avgP:.2f}**")
        st.write(f"PED = **{ped_arc:.3f}**  (|PED| = **{abs(ped_arc):.3f}**)")

    with st.expander("Show equilibrium summary"):
        st.write(f"Initial equilibrium (E0): P = {P_eq0:.2f}, Q = {Q_eq0:.1f}")
        st.write(f"After supply shock (E1): P = {P_eq1:.2f}, Q = {Q_eq1:.1f}")
        st.write(f"Supply shock fixed at: +{supply_shock} units (right shift)")
        st.write("Note: With linear demand, elasticity varies by point; the slider calibrates |PED| specifically at (P0,Q0).")


# ============================================================
# Marginal Revenue - Pricing Power Builder
# ------------------------------------------------------------
# Everything is DISCRETE (whole units, Q = 0..10) so every number on a
# graph can be traced to a row in the table:
#   MC(q)  = cost of the q-th unit        TC(Q) = FC + sum of MC
#   MR(q)  = TR(q) - TR(q-1)              (never the continuous formula)
#   Q*     = the Q with the largest profit (same as "keep making units
#            while MR >= MC"; includes Q = 0, so shutdown is handled)
# ============================================================
from plotly.subplots import make_subplots

PP_QMAX = 10

# Chart colors (CBC blue kept for demand/TR; checked for colorblind separation)
PP_BLUE = "#00457C"    # demand / price / TR
PP_ORANGE = "#D9603B"  # MC / TC
PP_PURPLE = "#9B59B6"  # MR (dashed)
PP_GRAY = "#6B6B6B"    # ATC (dotted)
PP_GOLD = "#B8860B"    # profit
PP_INK = "#1F2933"

PP_COST_DEFAULTS = {"pp_FC": 60, "pp_mc0": 5, "pp_mc1": 4.0, "pp_mc2": 0.8}
PP_PRESETS = {
    "Situation 1 (default values)": {"pp_P": 75, "pp_a": 250, "pp_b": 25, "pp_dial": 25},
    "Situation 2 (demand shifts up: intercept 350)": {"pp_P": 75, "pp_a": 350, "pp_b": 25, "pp_dial": 25},
    "Situation 3 (flatter demand: intercept 145, slope 3)": {"pp_P": 75, "pp_a": 145, "pp_b": 3, "pp_dial": 25},
}
for _v in PP_PRESETS.values():
    _v.update(PP_COST_DEFAULTS)


def pp_costs(FC, mc0, mc1, mc2, qmax=PP_QMAX):
    Q = np.arange(0, qmax + 1)
    MC = np.where(Q == 0, np.nan, mc0 + mc1 * Q + mc2 * Q**2)
    TC = FC + np.concatenate([[0.0], np.cumsum(MC[1:])])
    with np.errstate(divide="ignore", invalid="ignore"):
        ATC = np.where(Q == 0, np.nan, TC / Q)
    return Q, MC, TC, ATC


def pp_demand_prices(a, b, qmax=PP_QMAX):
    """Linear demand P = a - bQ, floored at $0."""
    Q = np.arange(0, qmax + 1)
    return np.maximum(0.0, a - b * Q).astype(float)


def pp_schedule(prices, costs):
    """Full table for a given price schedule (index 0 = Q of 0)."""
    Q, MC, TC, ATC = costs
    P = np.asarray(prices, dtype=float)
    TR = P * Q
    MR = np.concatenate([[np.nan], np.diff(TR)])
    # MR split into: price of the new unit  -  price cut on all earlier units
    cut = np.concatenate([[np.nan], (P[:-1] - P[1:]) * Q[:-1]])
    return pd.DataFrame({
        "Q": Q, "P": P, "TR": TR, "MR": MR, "PriceCut": cut,
        "MC": MC, "TC": TC, "ATC": ATC, "Profit": TR - TC, "Gap": P - MC,
    })


def pp_best_q(df):
    """Profit-maximizing Q (ties -> the larger Q, i.e. make the unit when MR = MC)."""
    best = df["Profit"].max()
    return int(df.loc[np.isclose(df["Profit"], best), "Q"].max())


def pp_competitive_q(df):
    """Market outcome where demand meets MC: keep going while P >= MC."""
    q = 0
    for _, r in df.iloc[1:].iterrows():
        if r["P"] >= r["MC"]:
            q = int(r["Q"])
        else:
            break
    return q


def pp_dwl(df, q_mono, q_comp):
    lost = df[(df["Q"] > q_mono) & (df["Q"] <= q_comp)]
    return float(lost["Gap"].sum())


def pp_dial_demand(P, b, q_anchor):
    """Demand that passes through the price-taker outcome (q_anchor, P) with slope b."""
    return P + b * q_anchor, b


def pp_dial_max(P, q_anchor, qmax=PP_QMAX):
    """Steepest slope that keeps demand at or above $0 through Q = qmax."""
    if q_anchor >= qmax:
        return 60
    return max(1, int(P // (qmax - q_anchor)))


def pp_outcomes(costs, P, a, b):
    """Every number the problem set asks for, from the same functions the page uses."""
    pt = pp_schedule(np.full(PP_QMAX + 1, float(P)), costs)
    dm = pp_schedule(pp_demand_prices(a, b), costs)
    qpt, qm, qc = pp_best_q(pt), pp_best_q(dm), pp_competitive_q(dm)
    row = lambda d, q: d.loc[d["Q"] == q].iloc[0]
    return {
        "pt": (qpt, float(P), row(pt, qpt)["TR"], row(pt, qpt)["Profit"]),
        "comp": (qc, row(dm, qc)["P"], row(dm, qc)["TR"], row(dm, qc)["Profit"]),
        "mono": (qm, row(dm, qm)["P"], row(dm, qm)["TR"], row(dm, qm)["Profit"]),
        "dwl": pp_dwl(dm, qm, qc),
    }


# ---------------- chart builders (pure: return a plotly Figure) ----------------
def _pp_money(v):
    return "—" if pd.isna(v) else f"${v:,.2f}"


def _pp_unit_hover(label):
    return f"Unit %{{x}}<br>{label} = $%{{y:,.2f}}<extra></extra>"


def _pp_axes(fig, rows, per_unit_row, top):
    for r in rows:
        fig.update_xaxes(range=[-0.4, PP_QMAX + 0.4], dtick=1, showgrid=True,
                         gridcolor="#ECEFF3", row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#ECEFF3", zeroline=True,
                         zerolinecolor="#B8C0CC", tickprefix="$", separatethousands=True,
                         row=r, col=1)
    if per_unit_row:
        fig.update_yaxes(range=[-130, top], row=per_unit_row, col=1)
    fig.update_xaxes(title_text="Quantity (Q)", row=rows[-1], col=1)


def _pp_line(fig, x, y, name, color, dash=None, row=None, hover=None, width=2.5, group=None):
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=name, legendgroup=group or name,
        line=dict(color=color, width=width, dash=dash), marker=dict(size=8, color=color),
        hovertemplate=hover or _pp_unit_hover(name), connectgaps=False,
    ), row=row, col=1 if row else None)


def _pp_guide(fig, q, rows, color=PP_INK, text=None):
    for r in rows:
        fig.add_vline(x=q, line=dict(color=color, width=1.5, dash="dash"), row=r, col=1)
    if text:
        fig.add_annotation(x=q, y=1.0, yref="paper", text=text, showarrow=False,
                           xanchor="left", xshift=6, yanchor="top",
                           font=dict(color=color, size=13), bgcolor="rgba(255,255,255,0.85)")


def pp_firm_figure(df, kind, show_guide, demand_label="Demand (P)"):
    """Stacked totals / profit / per-unit view for one firm. kind = 'taker' or 'maker'."""
    q_star = pp_best_q(df)
    r = df.loc[df["Q"] == q_star].iloc[0]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.36, 0.20, 0.44],
        subplot_titles=("Totals: TR and TC (Profit is the gap)",
                        "Profit at each quantity (TR − TC)",
                        "Per unit: price, MR, MC, ATC"),
    )
    Q = df["Q"]
    # Row 1: totals
    _pp_line(fig, Q, df["TR"], "Total Revenue (TR)", PP_BLUE, row=1,
             hover="Q = %{x}<br>TR = $%{y:,.2f}<extra></extra>")
    _pp_line(fig, Q, df["TC"], "Total Cost (TC)", PP_ORANGE, row=1,
             hover="Q = %{x}<br>TC = $%{y:,.2f}<extra></extra>")
    # Row 2: profit bars
    bar_colors = [PP_GOLD if (show_guide and q == q_star) else "rgba(184,134,11,0.45)" for q in Q]
    fig.add_trace(go.Bar(x=Q, y=df["Profit"], name="Profit", marker_color=bar_colors,
                         hovertemplate="Q = %{x}<br>Profit = $%{y:,.2f}<extra></extra>"), row=2, col=1)
    # Row 3: per unit
    units = df.iloc[1:]
    if kind == "taker":
        _pp_line(fig, units["Q"], units["P"], f"P = MR = d (${units['P'].iloc[0]:,.0f})", PP_BLUE, row=3,
                 hover="Unit %{x}<br>P = MR = $%{y:,.2f}<extra></extra>")
    else:
        _pp_line(fig, units["Q"], units["P"], demand_label, PP_BLUE, row=3,
                 hover="Q = %{x}<br>Price = $%{y:,.2f}<extra></extra>")
        _pp_line(fig, units["Q"], units["MR"], "Marginal Revenue (MR)", PP_PURPLE, dash="dash", row=3)
    _pp_line(fig, units["Q"], units["MC"], "Marginal Cost (MC)", PP_ORANGE, row=3)
    _pp_line(fig, units["Q"], units["ATC"], "Average Total Cost (ATC)", PP_GRAY, dash="dot", row=3,
             width=2, hover="Q = %{x}<br>ATC = $%{y:,.2f}<extra></extra>")

    if show_guide and q_star > 0:
        _pp_guide(fig, q_star, [1, 2, 3], text=f"Q* = {q_star}")
        # profit rectangle (P* - ATC*) x Q*
        y0, y1 = float(r["ATC"]), float(r["P"])
        fig.add_shape(type="rect", x0=0, x1=q_star, y0=min(y0, y1), y1=max(y0, y1),
                      fillcolor="rgba(184,134,11,0.30)", line=dict(color=PP_GOLD, width=1.5), layer="below", row=3, col=1)
        fig.add_trace(go.Scatter(
            x=[q_star], y=[r["P"]], mode="markers", name=f"P* = {_pp_money(r['P'])}",
            marker=dict(size=13, color=PP_BLUE, line=dict(color="white", width=2)),
            hovertemplate=f"Q* = {q_star}<br>P* = {_pp_money(r['P'])}<br>ATC = {_pp_money(r['ATC'])}"
                          f"<br>Profit = {_pp_money(r['Profit'])}<extra></extra>"), row=3, col=1)
    elif show_guide:
        fig.add_annotation(x=0, y=1.0, yref="paper", text="Q* = 0 (shut down)", showarrow=False,
                           xanchor="left", font=dict(color=PP_INK, size=13))

    top = max(370.0, float(np.nanmax(df["P"])) + 20)
    tot_top = max(1300.0, float(np.nanmax(df["TR"])) * 1.05)
    _pp_axes(fig, [1, 2, 3], 3, top)
    fig.update_yaxes(range=[0, tot_top], row=1, col=1)
    fig.update_layout(
        template="simple_white", height=900, bargap=0.35, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        margin=dict(t=110, b=50, l=70, r=20), font=dict(size=13),
    )
    return fig


def pp_mr_split_figure(df):
    """Why MR < P: each extra unit = (price of the new unit) - (price cut on the earlier units)."""
    u = df.iloc[1:]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=u["Q"], y=u["P"], name="Price of the new unit", marker_color=PP_BLUE,
                         hovertemplate="Unit %{x}<br>Sells for $%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=u["Q"], y=-u["PriceCut"], name="Price cut on all the earlier units",
                         marker_color=PP_PURPLE, opacity=0.75,
                         customdata=np.stack([u["Q"] - 1, u["PriceCut"]], axis=-1),
                         hovertemplate="Unit %{x}<br>Give up $%{customdata[1]:,.2f} "
                                       "on the first %{customdata[0]} units<extra></extra>"))
    fig.add_trace(go.Scatter(x=u["Q"], y=u["MR"], mode="markers", name="MR (what's left)",
                             marker=dict(symbol="diamond", size=12, color=PP_INK,
                                         line=dict(color="white", width=1.5)),
                             hovertemplate="Unit %{x}<br>MR = $%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=u["Q"], y=u["MC"], mode="lines+markers", name="Marginal Cost (MC)",
                             line=dict(color=PP_ORANGE, width=2.5), marker=dict(size=8),
                             hovertemplate=_pp_unit_hover("MC")))
    fig.update_layout(
        template="simple_white", barmode="relative", height=430, bargap=0.3,
        title="Why MR is less than Price: to sell one more, you cut the price on every unit",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        margin=dict(t=90, b=50, l=70, r=20), font=dict(size=13),
    )
    fig.update_xaxes(title_text="Unit number", dtick=1, range=[0.4, PP_QMAX + 0.6])
    fig.update_yaxes(tickprefix="$", zeroline=True, zerolinecolor="#8A94A3", gridcolor="#ECEFF3",
                     showgrid=True)
    return fig


def pp_compare_figure(df, show_guide, show_dwl, title):
    """Same market demand: competitive outcome (D = MC) vs monopoly (MR = MC)."""
    qm, qc = pp_best_q(df), pp_competitive_q(df)
    u = df.iloc[1:]
    fig = make_subplots(rows=1, cols=1)
    _pp_line(fig, u["Q"], u["P"], "Demand (P)", PP_BLUE, row=1,
             hover="Q = %{x}<br>Price = $%{y:,.2f}<extra></extra>")
    _pp_line(fig, u["Q"], u["MR"], "Marginal Revenue (MR)", PP_PURPLE, dash="dash", row=1)
    _pp_line(fig, u["Q"], u["MC"], "Marginal Cost (MC)", PP_ORANGE, row=1)
    if show_dwl and qc > qm:
        lost = df[(df["Q"] > qm) & (df["Q"] <= qc)]
        for _, r in lost.iterrows():
            fig.add_shape(type="rect", x0=r["Q"] - 0.5, x1=r["Q"] + 0.5, y0=r["MC"], y1=r["P"],
                          fillcolor="rgba(155,89,182,0.35)", line=dict(color="rgba(155,89,182,0.7)", width=1),
                          layer="below")
        fig.add_trace(go.Scatter(  # legend entry only
            x=[None], y=[None], mode="markers", name=f"Deadweight loss = {_pp_money(pp_dwl(df, qm, qc))}",
            marker=dict(size=14, color="rgba(155,89,182,0.35)", symbol="square")))
        fig.add_trace(go.Scatter(  # invisible hover targets in the middle of each lost unit
            x=lost["Q"], y=(lost["P"] + lost["MC"]) / 2, mode="markers", showlegend=False,
            marker=dict(size=30, color="rgba(0,0,0,0)", symbol="square"),
            customdata=lost["Gap"],
            hovertemplate="Unit %{x} is not made<br>Lost value (P − MC) = $%{customdata:,.2f}<extra></extra>"))
    if show_guide:
        rc, rm = df.loc[df["Q"] == qc].iloc[0], df.loc[df["Q"] == qm].iloc[0]
        for q, r, c, lab in [(qc, rc, PP_BLUE, "Competitive"), (qm, rm, PP_PURPLE, "Monopoly")]:
            if q == 0:
                continue
            fig.add_vline(x=q, line=dict(color=c, width=1.5, dash="dash"))
            fig.add_trace(go.Scatter(
                x=[q], y=[r["P"]], mode="markers", name=f"{lab}: Q = {q}, P = {_pp_money(r['P'])}",
                marker=dict(size=14, color=c, line=dict(color="white", width=2)),
                hovertemplate=f"{lab}<br>Q = {q}<br>P = {_pp_money(r['P'])}<extra></extra>"))
    top = max(370.0, float(np.nanmax(df["P"])) + 20)
    fig.update_layout(
        template="simple_white", height=580, hovermode="closest",
        title=dict(text=title, y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=140, b=50, l=70, r=20), font=dict(size=13),
    )
    _pp_axes(fig, [1], 1, top)
    fig.update_xaxes(range=[0.4, PP_QMAX + 0.6])
    return fig


def _pp_table(df, cols, highlight=None, lost=None):
    names = {"Q": "Q", "P": "Price (P)", "TR": "Total Revenue (TR)", "MR": "Marginal Revenue (MR)",
             "MC": "Marginal Cost (MC)", "TC": "Total Cost (TC)", "ATC": "Avg Total Cost (ATC)",
             "Profit": "Profit (TR − TC)", "Gap": "P − MC"}
    out = df[cols].rename(columns=names)

    def paint(row):
        q = row["Q"]
        if highlight is not None and q == highlight:
            return ["background-color: #FCE58B"] * len(row)
        if lost is not None and lost[0] < q <= lost[1]:
            return ["background-color: #EBDDF2"] * len(row)
        return [""] * len(row)

    money = {names[c]: _pp_money for c in cols if c != "Q"}
    return out.style.apply(paint, axis=1).format(money).hide(axis="index")


# ---------------- the page ----------------
def _pp_apply_preset():
    choice = st.session_state.get("pp_situation")
    if choice in PP_PRESETS:
        for k, v in PP_PRESETS[choice].items():
            st.session_state[k] = v


def _pp_mark_custom():
    st.session_state["pp_situation"] = "Custom"


def _pp_instructor_view(costs):
    with st.expander("🔒 Instructor view (answer key)"):
        try:
            code = st.secrets.get("INSTRUCTOR_PASSCODE", None)
        except Exception:
            code = None
        if not code:
            st.caption("To turn this on, add INSTRUCTOR_PASSCODE to this app's Secrets in Streamlit Cloud.")
            return
        if st.text_input("Passcode", type="password", key="pp_passcode") != code:
            return
        rows = []
        for name, v in PP_PRESETS.items():
            o = pp_outcomes(pp_costs(v["pp_FC"], v["pp_mc0"], v["pp_mc1"], v["pp_mc2"]),
                            v["pp_P"], v["pp_a"], v["pp_b"])
            rows.append({"Situation": name,
                         "Price-taker Q / TR / Profit": f"{o['pt'][0]} / {_pp_money(o['pt'][2])} / {_pp_money(o['pt'][3])}",
                         "Competitive Q / P": f"{o['comp'][0]} / {_pp_money(o['comp'][1])}",
                         "Competitive TR / Profit": f"{_pp_money(o['comp'][2])} / {_pp_money(o['comp'][3])}",
                         "Monopoly Q / P": f"{o['mono'][0]} / {_pp_money(o['mono'][1])}",
                         "Monopoly TR / Profit": f"{_pp_money(o['mono'][2])} / {_pp_money(o['mono'][3])}",
                         "Deadweight loss": _pp_money(o["dwl"])})
        st.markdown("**Presets**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        P = st.session_state["pp_P"]
        q_anchor = pp_best_q(pp_schedule(np.full(PP_QMAX + 1, float(P)), costs))
        dial_rows = []
        for b in range(0, pp_dial_max(P, q_anchor) + 1):
            a, _ = pp_dial_demand(P, b, q_anchor)
            o = pp_outcomes(costs, P, a, b)
            dial_rows.append({"Dial": b, "Demand": f"P = {a:g} − {b}Q",
                              "Monopoly Q": o["mono"][0], "Monopoly P": _pp_money(o["mono"][1]),
                              "Monopoly profit": _pp_money(o["mono"][3]), "Deadweight loss": _pp_money(o["dwl"])})
        st.markdown(f"**Pricing power dial** (current costs, anchored at Q = {q_anchor}, P = ${P})")
        st.dataframe(pd.DataFrame(dial_rows), hide_index=True, use_container_width=True)


def page_market_power_cost_revenue_builder():
    for k, v in PP_PRESETS["Situation 1 (default values)"].items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("pp_situation", "Situation 1 (default values)")

    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 🏛️ Marginal Revenue – Pricing Power Builder")
    st.write("Same firm, same costs. In **① Price-Taker** the market sets the price. In **② Price-Maker** "
             "the firm faces the whole demand curve, so selling one more unit means cutting the price. "
             "**③ Compare** puts both outcomes on one market, and **④ Pricing Power Dial** turns pricing power up and down.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.selectbox("Situation (sets the sliders for the problem set)", ["Custom"] + list(PP_PRESETS),
                     key="pp_situation", on_change=_pp_apply_preset)
    with c2:
        st.write("")
        show_guide = st.toggle("✅ Check my answer (show Q*, P*, and profit)", key="pp_guide", value=False)

    with st.expander("⚙️ Cost settings (same firm in every tab)"):
        k1, k2, k3, k4 = st.columns(4)
        k1.slider("Fixed Cost (FC)", 0, 300, step=5, key="pp_FC", on_change=_pp_mark_custom)
        k2.slider("MC of the 1st unit, base", 0, 80, step=1, key="pp_mc0", on_change=_pp_mark_custom)
        k3.slider("MC slope", 0.0, 20.0, step=0.5, key="pp_mc1", on_change=_pp_mark_custom)
        k4.slider("MC curvature", 0.0, 5.0, step=0.1, key="pp_mc2", on_change=_pp_mark_custom)
    ss = st.session_state
    costs = pp_costs(ss["pp_FC"], ss["pp_mc0"], ss["pp_mc1"], ss["pp_mc2"])

    match = [n for n, v in PP_PRESETS.items()
             if all(np.isclose(ss[k], v[k]) for k in v if k != "pp_dial")]
    st.caption(f"Settings match **{match[0].split(' (')[0]}**." if match
               else "Custom settings (you've moved a slider away from the Situations).")

    t1, t2, t3, t4 = st.tabs(["① Price-Taker", "② Price-Maker", "③ Compare", "④ Pricing Power Dial"])
    col_firm = ["Q", "P", "TR", "MR", "MC", "TC", "ATC", "Profit"]

    # ---- ① Price-taker ----
    with t1:
        st.slider("Market price (the firm takes it: P = MR)", 5, 150, step=1, key="pp_P", on_change=_pp_mark_custom)
        P = ss["pp_P"]
        pt = pp_schedule(np.full(PP_QMAX + 1, float(P)), costs)
        st.plotly_chart(pp_firm_figure(pt, "taker", show_guide), use_container_width=True)
        st.caption("Marginal values sit at the unit they belong to: the point at Q = 6 is the 6th unit. "
                   "Keep making units while MR ≥ MC.")
        st.dataframe(_pp_table(pt, col_firm, highlight=pp_best_q(pt) if show_guide else None),
                     use_container_width=True, hide_index=True)

    # ---- ② Price-maker ----
    with t2:
        d1, d2 = st.columns(2)
        d1.slider("Demand intercept (P at Q = 0)", 50, 350, step=5, key="pp_a", on_change=_pp_mark_custom)
        d2.slider("Demand slope (price drop per +1 Q)", 1, 60, step=1, key="pp_b", on_change=_pp_mark_custom)
        a, b = ss["pp_a"], ss["pp_b"]
        dm = pp_schedule(pp_demand_prices(a, b), costs)
        st.caption(f"Demand: P = {a} − {b}Q" + (" (floored at $0)" if a - b * PP_QMAX < 0 else ""))
        st.plotly_chart(pp_firm_figure(dm, "maker", show_guide), use_container_width=True)
        st.caption("MR = the change in TR from one more unit. Profit peaks (top two panels) at the same Q "
                   "where MR stops being ≥ MC (bottom panel). Then go UP to demand for the price.")
        st.dataframe(_pp_table(dm, col_firm, highlight=pp_best_q(dm) if show_guide else None),
                     use_container_width=True, hide_index=True)
        st.plotly_chart(pp_mr_split_figure(dm), use_container_width=True)
        st.caption("Each blue bar is what the new unit sells for. Each purple bar is what you give up by "
                   "lowering the price on every unit you were already selling. The diamond is MR.")

    # ---- ③ Compare ----
    with t3:
        st.write(f"Same market demand as ② (P = {a} − {b}Q). **Perfect competition** produces where "
                 "demand meets MC (P ≥ MC). **Monopoly** produces where MR = MC and charges what demand will bear.")
        show_dwl = st.toggle("Show deadweight loss", key="pp_dwl", value=False)
        st.plotly_chart(pp_compare_figure(dm, show_guide, show_dwl, "Competition vs. monopoly in the same market"),
                        use_container_width=True)
        qm, qc = pp_best_q(dm), pp_competitive_q(dm)
        st.dataframe(_pp_table(dm, ["Q", "P", "MR", "MC", "Gap", "TR", "TC", "Profit"],
                               lost=(qm, qc) if show_dwl else None),
                     use_container_width=True, hide_index=True)
        if show_guide:
            o = pp_outcomes(costs, P, a, b)
            st.dataframe(pd.DataFrame({
                "": ["Perfect competition (P = MC)", "Monopoly (MR = MC)"],
                "Q": [o["comp"][0], o["mono"][0]],
                "Price": [_pp_money(o["comp"][1]), _pp_money(o["mono"][1])],
                "TR": [_pp_money(o["comp"][2]), _pp_money(o["mono"][2])],
                "Profit": [_pp_money(o["comp"][3]), _pp_money(o["mono"][3])],
            }), hide_index=True, use_container_width=True)
            st.caption(f"Deadweight loss = {_pp_money(o['dwl'])}: the P − MC on each unit the monopolist doesn't make.")

    # ---- ④ Pricing power dial ----
    with t4:
        q_anchor = pp_best_q(pt)
        if q_anchor == 0:
            st.warning("At this market price the price-taker shuts down (Q* = 0), so there's no outcome to "
                       "pivot around. Raise the market price in ① to use the dial.")
        else:
            dmax = pp_dial_max(P, q_anchor)
            ss["pp_dial"] = int(min(ss.get("pp_dial", 0), dmax))
            st.slider("Pricing power (how steep demand is; 0 = price-taker)", 0, dmax, step=1, key="pp_dial")
            ad, bd = pp_dial_demand(P, ss["pp_dial"], q_anchor)
            dd = pp_schedule(pp_demand_prices(ad, bd), costs)
            st.caption(f"Demand: P = {ad:g} − {bd}Q. It always passes through the price-taker outcome "
                       f"from ① (Q = {q_anchor}, P = ${P}), so the competitive outcome stays put while the "
                       "firm's pricing power changes.")
            st.plotly_chart(pp_compare_figure(dd, show_guide, True, "Turn up pricing power: watch MR peel away from demand"),
                            use_container_width=True)
            qmd, qcd = pp_best_q(dd), pp_competitive_q(dd)
            st.dataframe(_pp_table(dd, ["Q", "P", "MR", "MC", "Gap", "TR", "TC", "Profit"],
                                   highlight=qmd if show_guide else None, lost=(qmd, qcd)),
                         use_container_width=True, hide_index=True)

    _pp_instructor_view(costs)
    st.markdown("</div>", unsafe_allow_html=True)



def page_instructions():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## Welcome to ECON&201 Micro Simulations and Visualization w/ Prof. Engle")

    st.markdown(
        f"""
        <div style='background-color:{cbc_lightblue}; padding: 20px; border-radius: 10px;'>
            <p style='font-size: 16px; color: black;'>
            This little app is a hands-on and specific problem, charting companion to your Microeconomics course with Prof. Engle.
            Use the simulations and problems and projects to explore core economic concepts through interactive visualizations and data export tools.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📚 How to Use This App")
    with st.expander("📘 Step-by-Step Instructions"):
        st.markdown(
            """
1. Use the sidebar to select a topic or problem set for the week if this is in your assignment.  Check your assignment!
2. Adjust the charts using the slider tools (as instructed).
3. Observe how the graphs and data tables respond in real-time and use these for your charts and problems exercises.
4. Download CSV files to use in Excel or assignments (if needed).
"""
        )

    with st.expander("🎯 What You'll Learn"):
        st.markdown(
            """
- Basic games of coordination, conflict, and different rational choice scenarios in 2x2 games (zero-sum, positive-sum, and negative-sum)
- How consumers make choices under different constraints
- How producers make choices under different cost structures
- How markets reach equilibrium between consumers and producers
- How to interpret supply/demand shifts
- How to interpret gain/loss of consumer and producer surplus, equity, and efficiency under conditions of:
    - Need / Want (Elasticity)
    - Market Power (Ability for Seller or Buyer to Change Price)
    - Market Power and Segmentation (Ability for Seller or Buyer to Change Price for specific people)
    - Market Power and Business Tactics (Ability for Seller or Buyer to Change Price in other situations)
- How to interpret gain/loss of consumer and producer surplus, equity, and efficiency in
    - Markets for Work
    - Markets for Money
"""
        )

    st.info("Pick a game from the left sidebar to get started!")
    st.markdown("</div>", unsafe_allow_html=True)


def page_fixed_variable_cost():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 🏭 Total Fixed and Variable Cost Simulator")

    st.write("Choose your fixed and variable costs and see how they shape your total cost.")

    fixed_cost = st.slider("Fixed Cost ($)", 0, 1000, 200)
    variable_cost_per_unit = st.slider("Variable Cost per Unit ($)", 1, 100, 10)
    max_quantity = st.slider("Pick Quantity to Produce", 1, 50, 10)

    quantities = np.arange(1, max_quantity + 1)
    total_costs = fixed_cost + variable_cost_per_unit * quantities

    col1, col2 = st.columns(2)

    with col1:
        cost_fig = go.Figure()
        cost_fig.add_trace(
            go.Scatter(
                x=quantities,
                y=total_costs,
                mode="lines+markers",
                name="Total Cost",
                line=dict(color=cbc_blue),
            )
        )
        cost_fig.update_layout(
            title="Total Cost and Quantity",
            xaxis_title="Quantity Produced",
            yaxis_title="Total Cost ($)",
            template="simple_white",
            xaxis=dict(range=[0, max_quantity]),
            yaxis=dict(range=[0, None]),
        )
        st.plotly_chart(cost_fig, use_container_width=True)

    with col2:
        variable_total = variable_cost_per_unit * max_quantity
        bar_fig = go.Figure()
        bar_fig.add_trace(
            go.Bar(
                x=["Fixed Cost", "Variable Cost"],
                y=[fixed_cost, variable_total],
                marker_color=[cbc_orange, cbc_gold],
                text=[f"{fixed_cost}", f"{variable_total}"],
                textposition="auto",
            )
        )
        bar_fig.update_layout(title="Cost Contribution Breakdown", yaxis_title="Cost ($)", template="simple_white")
        st.plotly_chart(bar_fig, use_container_width=True)

    df_cost = pd.DataFrame(
        {
            "Quantity": quantities,
            "Fixed Cost": [fixed_cost] * len(quantities),
            "Variable Cost per Unit": [variable_cost_per_unit] * len(quantities),
            "Total Cost": total_costs,
        }
    )
    export_csv_button("📥 Download Cost CSV", df_cost, "cost_sim")
    st.markdown("</div>", unsafe_allow_html=True)


def page_marginal_utility_demand():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 📈 Marginal Utility ($) & Demand (Individual → Market)")

    st.write(
        "Use the **sliders** to set a consumer’s 5 marginal utilities (MU1..MU5). "
        "Or click **New Random Utilities** to generate a new consumer (values can be **0 to 20**, "
        "and are **non-increasing** as quantity rises). "
        "Click **Submit and Add** to add that consumer into the market."
    )

    # ----------------------------
    # Helpers
    # ----------------------------
    def generate_decreasing_mus(n=5, low=0, high=20):
        """Random MU values between low..high, sorted to be non-increasing MU1..MUn."""
        vals = [random.randint(low, high) for _ in range(n)]
        vals.sort(reverse=True)
        return vals

    def demand_from_mu(mu_list):
        """
        Convert MU list (WTP per unit) into an individual demand schedule:
        At price p, quantity demanded = count of MU >= p.
        """
        prices = list(range(20, -1, -1))  # 20 down to 0
        qd = [sum(1 for u in mu_list if u >= p) for p in prices]
        return prices, qd

    # ----------------------------
    # Session state (market accumulation)
    # ----------------------------
    st.session_state.setdefault("mu_current", generate_decreasing_mus())  # now allows zeros
    st.session_state.setdefault("market_by_price", {p: 0 for p in range(0, 21)})  # 0..20
    st.session_state.setdefault("market_submissions", [])  # store each consumer's MU list
    st.session_state.setdefault("first_market_curve", None)
    # ----------------------------
    # Buttons ABOVE graphs
    # ----------------------------
    colA, colB, colC = st.columns([1.2, 1.2, 2.6])

    with colA:
        if st.button("🎲 New Random Utilities"):
            st.session_state.mu_current = generate_decreasing_mus(low=0, high=20)
            # keep slider defaults in sync
            st.session_state["mu1"] = st.session_state.mu_current[0]
            st.session_state["mu2"] = st.session_state.mu_current[1]
            st.session_state["mu3"] = st.session_state.mu_current[2]
            st.session_state["mu4"] = st.session_state.mu_current[3]
            st.session_state["mu5"] = st.session_state.mu_current[4]
            st.toast("New randomized MU values generated.", icon="🎲")

    with colB:
        if st.button("✅ Submit and Add"):
            mu_list = st.session_state.mu_current
            prices_desc, qd = demand_from_mu(mu_list)

            # Store FIRST consumer's demand curve only once
            if st.session_state.first_market_curve is None:
                st.session_state.first_market_curve = {
                    "prices": prices_desc.copy(),
                    "quantities": qd.copy()
                }
            
            # Add this consumer into the market (horizontal sum)
            for p, q in zip(prices_desc, qd):
                st.session_state.market_by_price[p] += q

            st.session_state.market_submissions.append(
                {
                    "Submission": len(st.session_state.market_submissions) + 1,
                    "MU1": mu_list[0],
                    "MU2": mu_list[1],
                    "MU3": mu_list[2],
                    "MU4": mu_list[3],
                    "MU5": mu_list[4],
                }
            )

            # After submit, auto-generate next consumer
            st.session_state.mu_current = generate_decreasing_mus(low=0, high=20)
            st.session_state["mu1"] = st.session_state.mu_current[0]
            st.session_state["mu2"] = st.session_state.mu_current[1]
            st.session_state["mu3"] = st.session_state.mu_current[2]
            st.session_state["mu4"] = st.session_state.mu_current[3]
            st.session_state["mu5"] = st.session_state.mu_current[4]
            st.success("Added one consumer to the market demand curve.")

    with colC:
        if st.button("🔄 Reset Market"):
            st.session_state.market_by_price = {p: 0 for p in range(0, 21)}
            st.session_state.market_submissions = []
            st.session_state.mu_current = generate_decreasing_mus(low=0, high=20)
            st.session_state["mu1"] = st.session_state.mu_current[0]
            st.session_state["mu2"] = st.session_state.mu_current[1]
            st.session_state["mu3"] = st.session_state.mu_current[2]
            st.session_state["mu4"] = st.session_state.mu_current[3]
            st.session_state["mu5"] = st.session_state.mu_current[4]
            st.warning("Market reset (all submissions cleared).")

    st.divider()

    # ----------------------------
    # Sliders (5) for student input (safe when bounds hit 0)
    # ----------------------------
    st.markdown("### 🎚️ Set Marginal Utilities (Student Input)")

    # Initialize slider keys from current MU once
    st.session_state.setdefault("mu1", st.session_state.mu_current[0])
    st.session_state.setdefault("mu2", st.session_state.mu_current[1])
    st.session_state.setdefault("mu3", st.session_state.mu_current[2])
    st.session_state.setdefault("mu4", st.session_state.mu_current[3])
    st.session_state.setdefault("mu5", st.session_state.mu_current[4])

    def bounded_slider(label: str, upper: int, key: str, current: int) -> int:
        """
        Streamlit sliders cannot have min==max (e.g., 0..0).
        If upper==0, show a disabled 0..1 slider and force value to 0.
        """
        if upper <= 0:
            st.slider(label, 0, 1, 0, key=key, disabled=True)
            return 0
        return st.slider(label, 0, int(upper), int(min(current, upper)), key=key)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        mu1 = st.slider("MU1", 0, 20, int(st.session_state["mu1"]), key="mu1")

    with c2:
        mu2 = bounded_slider("MU2", int(mu1), "mu2", int(st.session_state["mu2"]))

    with c3:
        mu3 = bounded_slider("MU3", int(mu2), "mu3", int(st.session_state["mu3"]))

    with c4:
        mu4 = bounded_slider("MU4", int(mu3), "mu4", int(st.session_state["mu4"]))

    with c5:
        mu5 = bounded_slider("MU5", int(mu4), "mu5", int(st.session_state["mu5"]))

    # Update current MU list from sliders (always non-increasing)
    st.session_state.mu_current = [mu1, mu2, mu3, mu4, mu5]

    st.divider()

    # ----------------------------
    # Current consumer MU (display)
    # ----------------------------
    mu = st.session_state.mu_current
    quantities = np.arange(1, 6)
    total_utility = np.cumsum(mu)
    individual_prices = mu  # WTP per unit

    # ----------------------------
    # Market demand curve from accumulated submissions
    # ----------------------------
    market_prices_desc = list(range(20, -1, -1))  # 20..0
    market_q_desc = [st.session_state.market_by_price[p] for p in market_prices_desc]

    # ----------------------------
    # Graphs
    # ----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Bar(
                x=quantities,
                y=total_utility,
                marker_color=cbc_gold,
                text=total_utility,
                textposition="auto",
                name="Total Utility",
            )
        )
        fig1.update_layout(
            title="Current Consumer: Total Utility by Quantity",
            xaxis_title="Quantity Consumed",
            yaxis_title="Total Utility ($)",
            template="simple_white",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=quantities,
                y=individual_prices,
                mode="lines+markers",
                name="Individual Demand (MU)",
                line=dict(color=cbc_blue),
            )
        )
        fig2.update_layout(
            title="Current Consumer: Demand Curve (WTP per Unit)",
            xaxis_title="Quantity",
            yaxis_title="Willingness to Pay ($)",
            template="simple_white",
            yaxis=dict(range=[0, 21]),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = go.Figure()

        # ---- Market demand (accumulated) ----
        fig3.add_trace(
            go.Scatter(
                x=market_q_desc,
                y=market_prices_desc,
                mode="lines+markers",
                name="Market Demand (All Consumers)",
                line=dict(color=cbc_darkorange, shape="hv"),
            )
        )

        # ---- First consumer demand (fixed reference) ----
        if st.session_state.first_market_curve is not None:
            fig3.add_trace(
                go.Scatter(
                    x=st.session_state.first_market_curve["quantities"],
                    y=st.session_state.first_market_curve["prices"],
                    mode="lines",
                    name="First Consumer Only",
                    line=dict(
                        color=cbc_blue,
                        width=2,
                        dash="dash",   # visually distinct
                        shape="hv"
                    ),
                )
            )

        fig3.update_layout(
            title="Market Demand Curve (Original vs Growing Market)",
            xaxis_title="Total Quantity Demanded (Market)",
            yaxis_title="Price ($)",
            template="simple_white",
            yaxis=dict(range=[0, 21]),
        )

        st.plotly_chart(fig3, use_container_width=True)


    # ----------------------------
    # Tables + Export
    # ----------------------------
    st.markdown("### 📋 Current Consumer Utility Table")
    df_current = pd.DataFrame(
        {
            "Quantity": quantities,
            "Marginal Utility (MU)": mu,
            "Total Utility": total_utility,
            "WTP ($)": individual_prices,
        }
    )
    st.dataframe(df_current, use_container_width=True)
    export_csv_button("📥 Download Current Consumer CSV", df_current, "utility_demand_current")

    st.markdown("### 🧑‍🤝‍🧑 Market Submissions (Each Click = One Consumer)")
    if st.session_state.market_submissions:
        df_subs = pd.DataFrame(st.session_state.market_submissions)
        st.dataframe(df_subs, use_container_width=True)
        export_csv_button("📥 Download Submissions CSV", df_subs, "utility_demand_submissions")

        st.markdown("### 🏪 Market Demand Schedule (Price → Quantity)")
        df_market = pd.DataFrame(
            {"Price ($)": market_prices_desc, "Market Quantity Demanded": market_q_desc}
        )
        st.dataframe(df_market, use_container_width=True)
        export_csv_button("📥 Download Market Demand CSV", df_market, "market_demand_schedule")
    else:
        st.info("No submissions yet. Click **Submit and Add** to start building the market demand curve.")

    st.markdown("</div>", unsafe_allow_html=True)



def page_budget_constraint():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 💰 Budget Constraint Simulator")
 
    income = st.sidebar.slider("Weekly Budget ($)", 10, 200, 100)
    price_coffee = st.sidebar.slider("Price of Coffee ($)", 1, 10, 5)
    price_sandwich = st.sidebar.slider("Price of Sandwich ($)", 1, 10, 5)
 
    max_coffee = income / price_coffee        # x-intercept
    max_sandwich = income / price_sandwich    # y-intercept
    slope = price_coffee / price_sandwich     # sandwiches given up per coffee
 
    # Baseline line so students can SEE shifts (income) and pivots (prices).
    current = {"income": income, "pc": price_coffee, "ps": price_sandwich}
    st.session_state.setdefault("bc_ref", current)
    if st.sidebar.button("📌 Set current line as baseline", key="bc_set_ref"):
        st.session_state.bc_ref = current
    ref = st.session_state.bc_ref
    ref_max_c, ref_max_s = ref["income"] / ref["pc"], ref["income"] / ref["ps"]
 
    # Keep the student's coffee choice when the budget changes (clamp instead of reset).
    cmax = int(max_coffee)
    st.session_state.setdefault("bc_coffee", cmax // 2)
    st.session_state.bc_coffee = min(int(st.session_state.bc_coffee), cmax)
    coffee_choice = st.slider("Coffees to Buy", 0, cmax, key="bc_coffee")
    sandwich_choice = (income - coffee_choice * price_coffee) / price_sandwich
 
    st.markdown(f"☕ Coffees: **{coffee_choice}**  | 🥪 Sandwiches: **{sandwich_choice:.1f}**")
    st.caption(
        f"Intercepts: {max_coffee:.1f} coffees (all on coffee) or {max_sandwich:.1f} sandwiches (all on sandwiches). "
        f"Slope = −P_coffee / P_sandwich = −{slope:.2f}, so each coffee costs {slope:.2f} sandwiches."
    )
 
    col1, col2 = st.columns(2)
    with col1:
        xmax = 1.1 * max(max_coffee, ref_max_c)
        ymax = 1.1 * max(max_sandwich, ref_max_s)
        bc_fig = go.Figure()
        bc_fig.add_trace(go.Scatter(
            x=[0, ref_max_c], y=[ref_max_s, 0], mode="lines", name="Baseline",
            line=dict(color="gray", dash="dash"),
        ))
        bc_fig.add_trace(go.Scatter(
            x=[0, max_coffee], y=[max_sandwich, 0], mode="lines", name="Budget Line",
            line=dict(color=cbc_blue, width=3), fill="tozeroy", fillcolor="rgba(0,69,124,0.08)",
        ))
        bc_fig.add_trace(go.Scatter(
            x=[coffee_choice], y=[sandwich_choice], mode="markers", name="Your Choice",
            marker=dict(size=12, color=cbc_orange),
        ))
        bc_fig.update_layout(
            title="Budget Constraint (shaded = affordable)",
            xaxis_title="Coffees",
            yaxis_title="Sandwiches",
            template="simple_white",
            xaxis=dict(range=[0, xmax]),   # fixed axes: shifts/pivots are visible
            yaxis=dict(range=[0, ymax]),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(bc_fig, use_container_width=True)
 
    with col2:
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=["Coffees", "Sandwiches"],
            y=[coffee_choice, sandwich_choice],
            marker_color=[cbc_orange, cbc_blue],
            text=[f"{coffee_choice}", f"{sandwich_choice:.1f}"],
            textposition="auto",
        ))
        bar_fig.update_layout(title="Your Spending Allocation", yaxis_title="Quantity Chosen",
                              xaxis_title="Goods", template="simple_white")
        st.plotly_chart(bar_fig, use_container_width=True)
 
    df_budget = pd.DataFrame({
        "Good": ["Coffees", "Sandwiches"],
        "Quantity": [coffee_choice, sandwich_choice],
        "Price ($)": [price_coffee, price_sandwich],
        "Spending ($)": [coffee_choice * price_coffee, sandwich_choice * price_sandwich],
    })
    export_csv_button("📥 Download Budget CSV", df_budget, "budget_allocation")
    st.markdown("</div>", unsafe_allow_html=True)
 
 
def page_ppf_rabbits_berries():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 🐇🍓 Production Possibilities Frontier (PPF)")
 
    resources = st.sidebar.slider("Total Labor Hours", 10, 100, 50)
    # With rabbits = (L/c)^0.5, c is exactly the hours for the FIRST rabbit; later ones take more.
    cost_rabbit = st.sidebar.slider("Hours for 1st Rabbit (each extra takes longer)", 1, 5, 2)
    cost_berry = st.sidebar.slider("Hours for 1st Berry (each extra takes longer)", 1, 5, 1)
 
    def produce_rabbits(labor, c):
        return (labor / c) ** 0.5
 
    def produce_berries(labor, c):
        return (labor / c) ** 0.6
 
    def frontier(R, cr, cb):
        L = np.linspace(0, R, 200)
        return produce_rabbits(L, cr), produce_berries(R - L, cb)
 
    # Baseline PPF so growth (outward shift) and tech changes are visible.
    current = {"R": resources, "cr": cost_rabbit, "cb": cost_berry}
    st.session_state.setdefault("ppf_ref", current)
    if st.sidebar.button("📌 Set current PPF as baseline", key="ppf_set_ref"):
        st.session_state.ppf_ref = current
    ref = st.session_state.ppf_ref
 
    r_output, b_output = frontier(resources, cost_rabbit, cost_berry)
    r_ref, b_ref = frontier(ref["R"], ref["cr"], ref["cb"])
 
    # Keep the labor choice when total hours change (clamp instead of reset).
    st.session_state.setdefault("ppf_rlabor", resources // 2)
    st.session_state.ppf_rlabor = min(int(st.session_state.ppf_rlabor), resources)
    r_labor = st.slider("Labor for Rabbits", 0, resources, key="ppf_rlabor")
    b_labor = resources - r_labor
 
    r_choice = float(produce_rabbits(r_labor, cost_rabbit))
    b_choice = float(produce_berries(b_labor, cost_berry))
 
    st.markdown(f"🐇 Rabbits: **{r_choice:.1f}**  |  🍓 Berries: **{b_choice:.1f}**")
    st.caption(
        f"Diminishing returns: N rabbits take {cost_rabbit}×N² hours; N berries take {cost_berry}×N^1.67 hours. "
        f"Max possible: {r_output.max():.1f} rabbits or {b_output.max():.1f} berries."
    )
 
    # Opportunity cost = slope of the PPF = MP_berries / MP_rabbits (berries given up per extra rabbit)
    if 0 < r_labor < resources:
        mp_r = 0.5 / cost_rabbit * (r_labor / cost_rabbit) ** -0.5
        mp_b = 0.6 / cost_berry * (b_labor / cost_berry) ** -0.4
        st.caption(
            f"Opportunity cost at your point: 1 more rabbit ≈ **{mp_b / mp_r:.2f} berries**. "
            "It rises as you shift labor toward rabbits — that's why the PPF bows outward."
        )
 
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r_ref, y=b_ref, mode="lines", name="Baseline PPF",
                                 line=dict(color="gray", dash="dash")))
        fig.add_trace(go.Scatter(x=r_output, y=b_output, mode="lines", name="PPF Frontier",
                                 line=dict(color=cbc_darkorange, width=3)))
        fig.add_trace(go.Scatter(x=[r_choice], y=[b_choice], mode="markers", name="Your Allocation",
                                 marker=dict(size=12, color=cbc_gold)))
        fig.update_layout(
            title="Production Possibilities Frontier",
            xaxis_title="Rabbits",
            yaxis_title="Berries",
            template="simple_white",
            xaxis=dict(range=[0, 1.1 * max(r_output.max(), r_ref.max())]),
            yaxis=dict(range=[0, 1.1 * max(b_output.max(), b_ref.max())]),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with col2:
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=["Rabbits", "Berries"],
            y=[r_choice, b_choice],
            marker_color=[cbc_gold, cbc_blue],
            text=[f"{r_choice:.1f}", f"{b_choice:.1f}"],
            textposition="auto",
        ))
        bar_fig.update_layout(title="Your Output Allocation", yaxis_title="Units Produced",
                              xaxis_title="Goods", template="simple_white")
        st.plotly_chart(bar_fig, use_container_width=True)
 
    df_ppf = pd.DataFrame({
        "Good": ["Rabbits", "Berries"],
        "Labor Hours": [r_labor, b_labor],
        "Output": [r_choice, b_choice],
    })
    export_csv_button("📥 Download CSV", df_ppf, "ppf_allocation")
    st.markdown("</div>", unsafe_allow_html=True)


def page_coordination_conflict():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 🛠️ Coordination / Conflict Games")

    st.write(
        "Choose an action and see what happens against the computer. "
        "Pick a game type to explore coordination vs conflict under different payoff structures."
    )

    # ----------------------------
    # Controls: choose game FIRST
    # ----------------------------
    game_type = st.radio(
        "Choose game type:",
        [
            "Coordination (Positive Sum)",
            "Coordination (Negative Sum)",
            "Conflict (Zero Sum)",
            "Conflict (Negative Sum)",
            "Conflict (Positive Sum)",
        ],
        horizontal=True,
    )

    user_choice = st.radio("Your Choice:", ["Cooperate", "Not Cooperate"], horizontal=True)

    # ----------------------------
    # Show ONLY the relevant payoff matrix
    # ----------------------------
    show_matrix = st.checkbox("📊 Show payoff matrix for this game", value=True)

    payoff_tables = {
        "Coordination (Positive Sum)": {
            "": ["You: Cooperate", "You: Not Cooperate"],
            "Computer: Cooperate": ["You +2 / Comp +3", "You 0 / Comp 0"],
            "Computer: Not Cooperate": ["You 0 / Comp 0", "You +2 / Comp +3"],
        },
        "Coordination (Negative Sum)": {
            "": ["You: Cooperate", "You: Not Cooperate"],
            "Computer: Cooperate": ["You -1 / Comp -2", "You -3 / Comp -4"],
            "Computer: Not Cooperate": ["You -3 / Comp -4", "You -1 / Comp -2"],
        },
        "Conflict (Zero Sum)": {
            "": ["You: Cooperate", "You: Not Cooperate"],
            "Computer: Cooperate": ["You +1 / Comp -1", "You -1 / Comp +1"],
            "Computer: Not Cooperate": ["You -1 / Comp +1", "You +1 / Comp -1"],
        },
        "Conflict (Negative Sum)": {
            "": ["You: Cooperate", "You: Not Cooperate"],
            "Computer: Cooperate": ["You -2 / Comp -2", "You -1 / Comp -4"],
            "Computer: Not Cooperate": ["You -4 / Comp -1", "You -3 / Comp -3"],
        },
        "Conflict (Positive Sum)": {
            "": ["You: Cooperate", "You: Not Cooperate"],
            "Computer: Cooperate": ["You +2 / Comp +2", "You +4 / Comp +1"],
            "Computer: Not Cooperate": ["You +1 / Comp +4", "You +2 / Comp +2"],
        },
    }

    if show_matrix:
        st.markdown("### 📊 Payoff Matrix (You / Computer)")
        st.markdown(f"#### {game_type}")
        st.table(payoff_tables[game_type])

    # ----------------------------
    # Session state
    # ----------------------------
    st.session_state.setdefault("coord_user_total", 0)
    st.session_state.setdefault("coord_comp_total", 0)
    st.session_state.setdefault("coord_history", [])

    # ----------------------------
    # Play Round
    # ----------------------------
    if st.button("Play Round"):
        comp_choice = random.choice(["Cooperate", "Not Cooperate"])
        st.write(f"🤖 Computer chose: **{comp_choice}**")

        # ---- Payoff logic ----
        if game_type == "Coordination (Positive Sum)":
            if user_choice == comp_choice:
                user_score, comp_score = 2, 3
                outcome = "✅ You coordinated! Both gain (positive-sum)."
            else:
                user_score, comp_score = 0, 0
                outcome = "⚪ No coordination. No gains."

        elif game_type == "Coordination (Negative Sum)":
            if user_choice == comp_choice:
                user_score, comp_score = -1, -2
                outcome = "😬 You coordinated, but the environment is harsh. Both lose a little."
            else:
                user_score, comp_score = -3, -4
                outcome = "💥 You failed to coordinate in a negative-sum environment. Bigger losses."

        elif game_type == "Conflict (Zero Sum)":
            # Consistent rule: if you match, you win; if mismatch, computer wins.
            if user_choice == comp_choice:
                user_score, comp_score = 1, -1
                outcome = "🏆 You won this round (zero-sum)."
            else:
                user_score, comp_score = -1, 1
                outcome = "🤖 Computer won this round (zero-sum)."

        elif game_type == "Conflict (Negative Sum)":
            # Conflict (Negative Sum) — Dominant strategy = Not Cooperate
            if user_choice == "Cooperate" and comp_choice == "Cooperate":
                user_score, comp_score = -2, -2
                outcome = "😬 You both try to cooperate, but the structure of the conflict still hurts everyone."

            elif user_choice == "Not Cooperate" and comp_choice == "Cooperate":
                user_score, comp_score = -1, -4
                outcome = "⚔️ You protect yourself while the other side cooperates — you lose less, they lose more."

            elif user_choice == "Cooperate" and comp_choice == "Not Cooperate":
                user_score, comp_score = -4, -1
                outcome = "⚔️ You cooperate while the other side defects — you take the bigger hit."

            else:  # Both Not Cooperate
                user_score, comp_score = -3, -3
                outcome = "💥 Mutual defection: both sides act defensively and both lose. (Nash equilibrium)"

        else:  # Conflict (Positive Sum)
            # Everyone gains something, but mismatch determines who gains more.
            if user_choice == comp_choice:
                user_score, comp_score = 2, 2
                outcome = "🤝 You matched — both gain evenly (positive-sum conflict)."
            else:
                if user_choice == "Cooperate" and comp_choice == "Not Cooperate":
                    user_score, comp_score = 4, 1
                    outcome = "😈 Conflict outcome: you gained more this time (still positive-sum overall)."
                else:
                    user_score, comp_score = 1, 4
                    outcome = "😈 Conflict outcome: computer gained more this time (still positive-sum overall)."

        # Update totals
        st.session_state.coord_user_total += user_score
        st.session_state.coord_comp_total += comp_score

        # Record history
        st.session_state.coord_history.append(
            {
                "Round": len(st.session_state.coord_history) + 1,
                "Game Type": game_type,
                "User Choice": user_choice,
                "Computer Choice": comp_choice,
                "User": user_score,
                "Computer": comp_score,
                "Total User": st.session_state.coord_user_total,
                "Total Computer": st.session_state.coord_comp_total,
            }
        )

        st.info(outcome)

    # ----------------------------
    # Charts + Export
    # ----------------------------
    if st.session_state.coord_history:
        history_df = pd.DataFrame(st.session_state.coord_history)

        col1, col2 = st.columns(2)
        with col1:
            cumulative_fig = go.Figure()
            cumulative_fig.add_trace(
                go.Bar(
                    x=["You", "Computer"],
                    y=[st.session_state.coord_user_total, st.session_state.coord_comp_total],
                    marker_color=[cbc_orange, cbc_lightblue],
                    text=[st.session_state.coord_user_total, st.session_state.coord_comp_total],
                    textposition="auto",
                )
            )
            cumulative_fig.update_layout(
                title="Cumulative Payoffs",
                yaxis_title="Total Points",
                xaxis_title="Player",
                template="simple_white",
            )
            st.plotly_chart(cumulative_fig, use_container_width=True)

        with col2:
            # Pass Python lists (avoids occasional plotly/narwhals dtype issues)
            rounds = history_df["Round"].to_list()
            total_user = history_df["Total User"].to_list()
            total_comp = history_df["Total Computer"].to_list()

            line_fig = go.Figure()
            line_fig.add_trace(go.Scatter(x=rounds, y=total_user, mode="lines+markers", name="You", line=dict(color=cbc_gold)))
            line_fig.add_trace(go.Scatter(x=rounds, y=total_comp, mode="lines+markers", name="Computer", line=dict(color=cbc_blue)))
            line_fig.update_layout(
                title="Payoff Over Time",
                xaxis_title="Round",
                yaxis_title="Cumulative Score",
                template="simple_white",
            )
            st.plotly_chart(line_fig, use_container_width=True)

        export_csv_button("📅 Download Round History", history_df, "coordination_conflict_history")

    # ----------------------------
    # Reset
    # ----------------------------
    if st.button("🔄 Reset Scores"):
        st.session_state.coord_user_total = 0
        st.session_state.coord_comp_total = 0
        st.session_state.coord_history = []
        st.success("Scores and history reset.")

    st.markdown("</div>", unsafe_allow_html=True)


def page_cost_curve_explorer():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 🍔 Wendy’s Cost Curve Explorer — Stylized (Textbook MC)")
    st.write("Two inputs: **Fixed cost** and a **labor cost level** (interpreted as the minimum of AVC/MC at the efficient scale).")

    # ----------------------------
    # Inputs (ONLY these two)
    # ----------------------------
    c1, c2 = st.columns(2)
    fixed_cost = c1.slider("Weekly Fixed Cost ($)", 0, 25000, 9000, 250)
    labor_level = c2.slider("Labor Cost Level ($ per burger at efficient scale)", 0.50, 12.00, 4.00, 0.25)

    # Optional price (keeps your Q* / shading logic)
    st.markdown("### 💵 Price → Output (Profit/Loss Shading)")
    price = st.slider("Burger Price ($)", 0.50, 20.00, 5.00, 0.10)

    # ----------------------------
    # Textbook cost functional form (NOT tied to MP/MPL)
    # VC(q) = a1*q + a2*q^2 + a3*q^3
    # => AVC(q)=a1 + a2*q + a3*q^2  (U-shaped if a2<0, a3>0)
    # => MC(q)=a1 + 2*a2*q + 3*a3*q^2 (U-shaped)
    # And MC intersects AVC at min AVC.
    # ----------------------------
    QMAX, N = 6000, 400
    q = np.linspace(1, QMAX, N)

    q_eff = 1200.0  # "efficient scale" where AVC is minimized (stylized)
    a3 = 0.5 * labor_level / (q_eff**2)     # curvature (crowding/complexity)
    a2 = -2 * a3 * q_eff                    # ensures min of AVC at q_eff
    a1 = labor_level - (a2*q_eff + a3*q_eff**2)  # ensures AVC(q_eff)=labor_level

    vc = a1*q + a2*q**2 + a3*q**3
    fc = np.full_like(q, fixed_cost, dtype=float)
    tc = fc + vc

    afc = fc / q
    avc = vc / q
    atc = tc / q
    mc = a1 + 2*a2*q + 3*a3*q**2

    # Enforce textbook-ish "MC doesn't decline after its minimum" (optional but nice)
    m = int(np.nanargmin(mc))
    mc[m:] = np.maximum.accumulate(mc[m:])

    # ----------------------------
    # Visual cap: do NOT plot per-unit costs above $20 (view window only)
    # (This extends x further until MC hits 20, while keeping it readable.)
    # ----------------------------
    MC_CAP = 20.0
    ok = np.where(np.isfinite(mc) & (mc <= MC_CAP))[0]
    i_max = int(ok[-1]) if len(ok) else len(q) - 1

    qP, tcP, fcP, vcP = (arr[:i_max+1] for arr in (q, tc, fc, vc))
    afcP, avcP, atcP, mcP = (arr[:i_max+1] for arr in (afc, avc, atc, mc))

    # ----------------------------
    # Choose Q* (shutdown: P < min AVC => 0)
    # ----------------------------
    avc_min = float(np.nanmin(avcP))
    if price < avc_min:
        idx_star, q_star, revenue, profit, atc_star = None, 0.0, 0.0, 0.0, np.nan
    else:
        feas = np.where(np.isfinite(mcP) & (mcP <= price))[0]
        idx_star = int(feas[-1]) if len(feas) else None
        q_star = float(qP[idx_star]) if idx_star is not None else 0.0
        revenue = price * q_star if q_star > 0 else 0.0
        atc_star = float(atcP[idx_star]) if idx_star is not None else np.nan
        profit = revenue - float(tcP[idx_star]) if idx_star is not None else 0.0

    # ----------------------------
    # Plot 1: Levels
    # ----------------------------
    levels = go.Figure()
    for y, name, col in [(tcP, "Total Cost (TC)", cbc_blue), (fcP, "Fixed Cost (FC)", cbc_lightblue), (vcP, "Variable Cost (VC)", cbc_orange)]:
        levels.add_trace(go.Scatter(x=qP, y=y, mode="lines", name=name, line=dict(color=col)))
    levels.update_layout(
        title="Weekly Cost Levels vs Quantity",
        xaxis_title="Quantity (burgers/week)",
        yaxis_title="Cost ($/week)",
        template="simple_white",
    )

    # ----------------------------
    # Plot 2: Per-unit curves + price + shading (y capped at $20)
    # ----------------------------
    curves = go.Figure()
    for y, n, c, d, w in [(afcP, "AFC", cbc_darkblue, "dot", 2), (avcP, "AVC", cbc_orange, None, 2), (atcP, "ATC", "black", None, 3), (mcP, "MC", cbc_gold, None, 3)]:
        curves.add_trace(go.Scatter(x=qP, y=y, mode="lines", name=n, line=dict(color=c, dash=d, width=w)))

    curves.add_hline(y=price, line_dash="dash", annotation_text=f"Price = ${price:,.2f}", annotation_position="top left")

    shapes = []
    if idx_star is not None and q_star > 0:
        curves.add_vline(x=q_star, line_dash="dash", annotation_text=f"Q* ≈ {q_star:,.0f}", annotation_position="bottom right")
        shapes.append(dict(type="rect", xref="x", yref="y", x0=0, x1=q_star, y0=0, y1=price,
                           fillcolor="rgba(120,120,120,0.18)", line=dict(width=0), layer="below"))
        if np.isfinite(atc_star):
            y0, y1, fill = (atc_star, price, "rgba(0,160,0,0.22)") if profit >= 0 else (price, atc_star, "rgba(200,0,0,0.22)")
            shapes.append(dict(type="rect", xref="x", yref="y", x0=0, x1=q_star, y0=y0, y1=y1,
                               fillcolor=fill, line=dict(width=0), layer="below"))
        st.write(f"**Q\*** ≈ **{q_star:,.0f} burgers/week**  |  **Revenue:** ${revenue:,.0f}  |  **Profit:** ${profit:,.0f}")
    else:
        st.warning("At this price, the firm produces **0 burgers** (price is below AVC or MC never reaches price in view).")

    curves.update_layout(
        title="Per-Unit Cost Curves (Capped at $20 for readability)",
        xaxis_title="Quantity (burgers/week)",
        yaxis_title="Cost ($/burger)",
        template="simple_white",
        yaxis=dict(range=[0, 20]),
        shapes=shapes,
    )

    # ----------------------------
    # Layout
    # ----------------------------
    a, b = st.columns(2)
    a.plotly_chart(levels, use_container_width=True)
    b.plotly_chart(curves, use_container_width=True)

    # ----------------------------
    # Export
    # ----------------------------
    df_cost = pd.DataFrame({
        "Quantity (burgers/week)": q,
        "Fixed Cost (weekly $)": fc,
        "Variable Cost (weekly $)": vc,
        "Total Cost (weekly $)": tc,
        "AFC ($/burger)": afc,
        "AVC ($/burger)": avc,
        "ATC ($/burger)": atc,
        "MC ($/burger)": mc,
        "Fixed Cost input ($/week)": np.full_like(q, fixed_cost, dtype=float),
        "Labor cost level input ($/burger at efficient scale)": np.full_like(q, labor_level, dtype=float),
        "Selected Price ($/burger)": np.full_like(q, price, dtype=float),
    })
    export_csv_button("📥 Download Cost Curves (CSV)", df_cost, "wendys_cost_curves_stylized")

    st.markdown("</div>", unsafe_allow_html=True)



def page_supply_demand_shock():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 📉📈 Supply & Demand Shock Simulator")
    st.write("Explore shifts in supply/demand and visualize ceilings/floors with a chosen price line.")

    # ---------- helpers (condensed) ----------
    def fmt(x):
        if x is None or (isinstance(x, (int, float)) and (np.isnan(x) or np.isinf(x))):
            return "N/A"
        return f"{float(x):.1f}"  # no scientific notation

    def eq(P0d, sd, P0s, ss):
        den = sd - ss
        if den == 0:
            return None, None
        Q = (P0s - P0d) / den
        P = P0d + sd * Q
        return (Q, P) if (np.isfinite(Q) and np.isfinite(P)) else (None, None)

    def xint(P0, m): return None if m == 0 else -P0 / m
    def q_at_p(P0, m, Pbar): return None if m == 0 else (Pbar - P0) / m
    def clipq(q): return None if (q is None or (not np.isfinite(q)) or q < 0) else float(q)

    def add_pt(fig, q, p, name, sym="circle", sz=9, col=cbc_gold):
        if q is None or p is None or (not np.isfinite(q)) or (not np.isfinite(p)) or q < 0:
            return
        fig.add_trace(go.Scatter(x=[q], y=[p], mode="markers", name=name, marker=dict(size=sz, symbol=sym, color=col)))

    # ---------- top controls ----------
    top = st.container()
    with top:
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
        with c1:
            d_shift = st.slider("Demand Shock (ΔQ; + right)", -50, 50, 0, key="sd_dshift")
            s_shift = st.slider("Supply Shock (ΔQ; + right)", -50, 50, 0, key="sd_sshift")
        with c3:
            show_init = st.checkbox("Show initial curves on shock chart", True, key="sd_show_init")
        pbar_slot = c2.empty()

    # ---------- chart placeholders ----------
    colL, colR = st.columns(2)
    with colL:
        chartL, textL = st.empty(), st.empty()
    with colR:
        chartR, textR = st.empty(), st.empty()

    # ---------- bottom sliders ----------
    st.markdown("---")
    st.markdown("### Curve parameters (slopes & intercepts)")
    b1, b2 = st.columns(2)
    with b1:
        P0d = float(st.slider("Demand Intercept (P when Q=0)", 10, 120, 60, key="sd_P0d"))
        sd = float(st.slider("Demand Slope (negative)", -12, -1, -5, key="sd_sd"))
    with b2:
        P0s = float(st.slider("Supply Intercept (P when Q=0)", 0, 120, 20, key="sd_P0s"))
        ss = float(st.slider("Supply Slope (positive)", 1, 12, 5, key="sd_ss"))

    # ---------- compute ----------
    P0d_n, P0s_n = P0d - sd * d_shift, P0s - ss * s_shift
    q0, p0 = eq(P0d, sd, P0s, ss)
    q1, p1 = eq(P0d_n, sd, P0s_n, ss)

    default_pbar = int(np.clip(round(p1), 0, 120)) if (p1 is not None and np.isfinite(p1)) else 40
    if "sd_Pbar" not in st.session_state:
        st.session_state.sd_Pbar = default_pbar
    Pbar = float(pbar_slot.slider("Selected Price (P̄) — ceiling/floor reference", 0, 120, int(st.session_state.sd_Pbar), key="sd_Pbar"))

    # shared axes + center equilibria
    q_center = float(np.mean([q for q in [q0, q1] if q is not None and np.isfinite(q)])) if any(q is not None for q in [q0, q1]) else 25.0
    p_center = float(np.mean([p for p in [p0, p1] if p is not None and np.isfinite(p)])) if any(p is not None for p in [p0, p1]) else float(Pbar)

    q_cands = [q_center]
    for P0_, m_ in [(P0d, sd), (P0s, ss), (P0d_n, sd), (P0s_n, ss)]:
        q_cands += [xint(P0_, m_), q_at_p(P0_, m_, Pbar)]
    q_cands += [q0, q1]
    q_cands = [q for q in q_cands if q is not None and np.isfinite(q)]
    q_half = max(q_center - min(q_cands), max(q_cands) - q_center, 10.0) * 1.25
    qmin, qmax = max(0.0, q_center - q_half), min(220.0, q_center + q_half)
    if qmax - qmin < 20:
        qmax = min(220.0, qmin + 20.0)

    Q = np.linspace(qmin, qmax, 650)
    D0, S0 = P0d + sd * Q, P0s + ss * Q
    D1, S1 = P0d_n + sd * Q, P0s_n + ss * Q

    p_cands = [p_center, Pbar, float(np.nanmin(D0)), float(np.nanmax(D0)), float(np.nanmin(S0)), float(np.nanmax(S0)),
               float(np.nanmin(D1)), float(np.nanmax(D1)), float(np.nanmin(S1)), float(np.nanmax(S1))]
    p_half = max(p_center - min(p_cands), max(p_cands) - p_center, 10.0) * 1.25
    pmin, pmax = max(0.0, p_center - p_half), min(220.0, p_center + p_half)
    if pmax - pmin < 10:
        pmax = min(220.0, pmin + 10.0)

    def ref(P0d_, P0s_):
        qeq, peq = eq(P0d_, sd, P0s_, ss)
        dx, sx = clipq(xint(P0d_, sd)), clipq(xint(P0s_, ss))
        qd, qs = clipq(q_at_p(P0d_, sd, Pbar)), clipq(q_at_p(P0s_, ss, Pbar))
        return {"eq": (clipq(qeq), peq), "dy": (0.0, P0d_), "sy": (0.0, P0s_), "dx": (dx, 0.0), "sx": (sx, 0.0), "qd": (qd, Pbar), "qs": (qs, Pbar)}

    R0, R1 = ref(P0d, P0s), ref(P0d_n, P0s_n)

    # ---------- plot styling fixes (TRY AGAIN) ----------
    # Fix (1): Put legend BELOW plot, horizontal, and increase bottom margin so it won't collide with x-axis title.
    # Also trim legend bloat: hide marker entries from legend (so legend is mostly curves + price line).
    def apply_layout(fig, title):
        fig.update_layout(
            title=title,
            template="simple_white",
            xaxis_title="Quantity",
            yaxis_title="Price",
            xaxis=dict(range=[qmin, qmax]),
            yaxis=dict(range=[pmin, pmax]),
            margin=dict(l=55, r=20, t=60, b=125),
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        )
        fig.add_trace(go.Scatter(x=[qmin, qmax], y=[Pbar, Pbar], mode="lines",
                                 name="Selected Price", line=dict(color="darkgreen", dash="dash")))

    # Fix (2): Use actual Plotly marker symbols in the TEXT labels (no emoji).
    def sym_txt(sym):
        return {"diamond": "◆", "circle": "●", "x": "✕", "triangle-up": "▲"}.get(sym, "•")

    figL = go.Figure()
    figL.add_trace(go.Scatter(x=Q, y=D0, mode="lines", name="Demand", line=dict(color=cbc_blue)))
    figL.add_trace(go.Scatter(x=Q, y=S0, mode="lines", name="Supply", line=dict(color=cbc_orange)))
    apply_layout(figL, "Initial Market")

    add_pt(figL, *R0["eq"], "Equilibrium", sym="diamond", sz=11, col=cbc_gold)
    add_pt(figL, *R0["dy"], "Demand y-int", sym="circle", sz=8, col=cbc_blue)
    add_pt(figL, *R0["sy"], "Supply y-int", sym="circle", sz=8, col=cbc_orange)
    add_pt(figL, *R0["dx"], "Demand x-int", sym="x", sz=10, col=cbc_blue)
    add_pt(figL, *R0["sx"], "Supply x-int", sym="x", sz=10, col=cbc_orange)
    add_pt(figL, *R0["qd"], "Qd @ P̄", sym="triangle-up", sz=10, col=cbc_blue)
    add_pt(figL, *R0["qs"], "Qs @ P̄", sym="triangle-up", sz=10, col=cbc_orange)

    # Hide marker traces from legend to reduce crowding (legend should be curves + price line)
    for tr in figL.data:
        if getattr(tr, "mode", "") == "markers":
            tr.showlegend = False

    figR = go.Figure()
    if show_init:
        figR.add_trace(go.Scatter(x=Q, y=D0, mode="lines", name="Initial Demand", line=dict(color=cbc_blue, dash="dash")))
        figR.add_trace(go.Scatter(x=Q, y=S0, mode="lines", name="Initial Supply", line=dict(color=cbc_orange, dash="dash")))
    figR.add_trace(go.Scatter(x=Q, y=D1, mode="lines", name="New Demand", line=dict(color=cbc_blue)))
    figR.add_trace(go.Scatter(x=Q, y=S1, mode="lines", name="New Supply", line=dict(color=cbc_orange)))
    apply_layout(figR, "Market After Shocks")

    add_pt(figR, *R1["eq"], "New Equilibrium", sym="diamond", sz=11, col=cbc_gold)
    add_pt(figR, *R1["dy"], "New demand y-int", sym="circle", sz=8, col=cbc_blue)
    add_pt(figR, *R1["sy"], "New supply y-int", sym="circle", sz=8, col=cbc_orange)
    add_pt(figR, *R1["dx"], "New demand x-int", sym="x", sz=10, col=cbc_blue)
    add_pt(figR, *R1["sx"], "New supply x-int", sym="x", sz=10, col=cbc_orange)
    add_pt(figR, *R1["qd"], "New Qd @ P̄", sym="triangle-up", sz=10, col=cbc_blue)
    add_pt(figR, *R1["qs"], "New Qs @ P̄", sym="triangle-up", sz=10, col=cbc_orange)

    for tr in figR.data:
        if getattr(tr, "mode", "") == "markers":
            tr.showlegend = False

    chartL.plotly_chart(figL, use_container_width=True)
    chartR.plotly_chart(figR, use_container_width=True)

    qeq0, peq0 = R0["eq"]
    textL.markdown(
        "**Initial reference points (P, Q):**\n"
        f"- {sym_txt('diamond')} Equilibrium: P={fmt(peq0)}, Q={fmt(qeq0)}\n"
        f"- {sym_txt('circle')} Demand intercepts: y={fmt(R0['dy'][1])}, x={fmt(R0['dx'][0])}\n"
        f"- {sym_txt('circle')} Supply intercepts: y={fmt(R0['sy'][1])}, x={fmt(R0['sx'][0])}\n"
        f"- {sym_txt('triangle-up')} At selected price P̄={fmt(Pbar)}: Qd={fmt(R0['qd'][0])}, Qs={fmt(R0['qs'][0])}"
    )

    qeq1, peq1 = R1["eq"]
    textR.markdown(
        "**New reference points (P, Q):**\n"
        f"- {sym_txt('diamond')} Equilibrium: P={fmt(peq1)}, Q={fmt(qeq1)}\n"
        f"- {sym_txt('circle')} Demand intercepts: y={fmt(R1['dy'][1])}, x={fmt(R1['dx'][0])}\n"
        f"- {sym_txt('circle')} Supply intercepts: y={fmt(R1['sy'][1])}, x={fmt(R1['sx'][0])}\n"
        f"- {sym_txt('triangle-up')} At selected price P̄={fmt(Pbar)}: Qd={fmt(R1['qd'][0])}, Qs={fmt(R1['qs'][0])}"
    )

    df_market = pd.DataFrame(
        {
            "Scenario": ["Initial", "New"],
            "Equilibrium Price": [p0, p1],
            "Equilibrium Quantity": [q0, q1],
            "Demand Shift (ΔQ)": [0, d_shift],
            "Supply Shift (ΔQ)": [0, s_shift],
            "Selected Price (P̄)": [Pbar, Pbar],
            "Qd @ P̄": [R0["qd"][0], R1["qd"][0]],
            "Qs @ P̄": [R0["qs"][0], R1["qs"][0]],
        }
    )
    export_csv_button("📥 Download Market Summary CSV", df_market, "supply_demand_shock")
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Sidebar + Routing
# ----------------------------
st.sidebar.title("🧭 Simulation Lobby")

pages = {
    "📘 Instructions / Start Here": page_instructions,
    "Coordination / Conflict Games": page_coordination_conflict,
    "Budget Constraint": page_budget_constraint,
    "PPF: Rabbits vs Berries": page_ppf_rabbits_berries,
    "Marginal Utility & Demand": page_marginal_utility_demand,
    "Cost Curve Explorer": page_cost_curve_explorer,
    "Supply & Demand Shock": page_supply_demand_shock,
    "Supply Shock and Demand Elasticity": page_supply_shock_elasticity,
    "Marginal Revenue - Pricing Power Builder": page_market_power_cost_revenue_builder,
    "Market Segmentation": page_price_discrimination_segmentation
}

game_choice = st.sidebar.radio("Select a Game:", list(pages.keys()), key="selected_game")

# ----------------------------
# Sidebar controls
# ----------------------------
#st.sidebar.markdown("### 🔧 Model Parameters")
#
#A_tmp = st.sidebar.slider("A: Baseline Productivity", 50.0, 600.0, 300.0, 10.0)
#B_tmp = st.sidebar.slider("B: Early Gains", 0.0, 800.0, 40.0, 5.0)
#D_tmp = st.sidebar.slider("D: Diminishing Returns", 1, 60, 20, 1)
#X3_tmp = st.sidebar.slider("X3: Peak / Inflection", 1, 400, 8, 1)

# ----------------------------
# Initialize session state ONCE
# ----------------------------
if "params" not in st.session_state:
    st.session_state.params = {
        "A": 180.0,
        "B": 335.0,
        "D": 20,
        "X3": 8
    }

# ----------------------------
# Apply button (must come AFTER sliders)
# ----------------------------
#if st.sidebar.button("➕ Add / Apply Curve"):
#    st.session_state.params["A"] = A_tmp
#    st.session_state.params["B"] = B_tmp
#    st.session_state.params["D"] = D_tmp
#    st.session_state.params["X3"] = X3_tmp
    

# Render selected page
pages[game_choice]()

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
<hr style='border-top: 1px solid #aaa;'>
<p style='text-align: center;'>Microeconomics Simulations</p>
""",
    unsafe_allow_html=True,
)
































































































