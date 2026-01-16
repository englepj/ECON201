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
    <h2 style='text-align: center;'> for Winter 2026 ECON&201 with Prof Engle </h2>
    <p style='text-align: center; color: black;'>Choose a simulation / problem set to explore key economic concepts.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Pages
# ----------------------------

def page_price_setting_marginal_revenue_game():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 🏷️ Price-Setting Game: Demand, Total Revenue, and Marginal Revenue")

    st.write(
        "You are a company with **price-setting ability**. Each round, you choose a price. "
        "The market responds with a quantity demanded. We track **Total Revenue (TR)** and **Marginal Revenue (MR)**."
    )

    # ----------------------------
    # Session State
    # ----------------------------
    st.session_state.setdefault("mr_game_history", [])
    st.session_state.setdefault("mr_game_best_tr", None)  # store dict row

    # ----------------------------
    # Demand Setup (Linear Demand)
    # Q = max(0, A - B*P)
    # ----------------------------
    st.markdown("### ⚙️ Market Demand Settings")
    colA, colB, colC = st.columns(3)
    with colA:
        A = st.slider("Demand intercept A (max Q when P=0)", 10, 200, 80, step=5)
    with colB:
        B = st.slider("Demand slope B (how fast Q falls as P rises)", 1, 10, 2, step=1)
    with colC:
        max_price = st.slider("Max price students can set", 5, 100, 40, step=1)

    def quantity_demanded(price: float) -> int:
        q = A - B * price
        return int(max(0, round(q)))

    st.divider()

    # ----------------------------
    # Controls (Game Round)
    # ----------------------------
    st.markdown("### 🎮 Play a Round")
    col1, col2, col3 = st.columns([1.2, 1.2, 2.6])
    with col1:
        price = st.slider("Set your price ($)", 0, max_price, int(max_price * 0.5), step=1)
    with col2:
        if st.button("✅ Submit Round"):
            q = quantity_demanded(price)
            tr = price * q

            # MR is ΔTR/ΔQ between this round and previous round (if any)
            if st.session_state.mr_game_history:
                prev = st.session_state.mr_game_history[-1]
                dq = q - prev["Q"]
                dtr = tr - prev["TR"]
                mr = None if dq == 0 else dtr / dq
            else:
                mr = None  # undefined on first round

            row = {
                "Round": len(st.session_state.mr_game_history) + 1,
                "P": float(price),
                "Q": int(q),
                "TR": float(tr),
                "MR (ΔTR/ΔQ)": None if mr is None else float(mr),
            }
            st.session_state.mr_game_history.append(row)

            # Track best TR
            best = st.session_state.mr_game_best_tr
            if (best is None) or (row["TR"] > best["TR"]):
                st.session_state.mr_game_best_tr = row

            st.success(f"Round added: Price ${price}, Quantity {q}, Total Revenue ${tr}")

    with col3:
        if st.button("🔄 Reset Game"):
            st.session_state.mr_game_history = []
            st.session_state.mr_game_best_tr = None
            st.warning("Game reset.")

    # ----------------------------
    # Demand Curve + TR + MR Curves (theory view)
    # ----------------------------
    st.divider()
    st.markdown("### 📉📈 Visuals: Demand, Total Revenue, and Marginal Revenue")

    prices = np.arange(0, max_price + 1)
    qs = np.array([quantity_demanded(p) for p in prices])
    trs = prices * qs

    # Approximate MR as discrete ΔTR/ΔQ stepping along the demand schedule as Q changes
    # Build MR aligned to Q levels (when Q drops by 1 unit, MR is change in TR from q to q-1)
    # We'll compute MR over unique Q values decreasing.
    q_unique = np.unique(qs)[::-1]  # descending
    mr_vals = []
    q_for_mr = []
    # For each q (except the smallest), find the max TR at that q and at next q-1
    # Using the demand schedule mapping.
    q_to_tr = {}
    for p, q, tr in zip(prices, qs, trs):
        # for each q, keep the highest TR observed (ties occur with rounding)
        q_to_tr[q] = max(q_to_tr.get(q, -1), tr)

    for i in range(len(q_unique) - 1):
        q_now = int(q_unique[i])
        q_next = int(q_unique[i + 1])
        if q_now == q_next:
            continue
        tr_now = q_to_tr.get(q_now, None)
        tr_next = q_to_tr.get(q_next, None)
        if tr_now is None or tr_next is None:
            continue
        dq = q_next - q_now
        dtr = tr_next - tr_now
        mr_vals.append(dtr / dq)
        q_for_mr.append(q_next)  # place MR at the "new" quantity level

    g1, g2, g3 = st.columns(3)

    with g1:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=qs, y=prices, mode="lines", name="Demand (P vs Q)", line=dict(color=cbc_blue)))
        fig_d.update_layout(
            title="Demand Curve (Price vs Quantity)",
            xaxis_title="Quantity (Q)",
            yaxis_title="Price (P)",
            template="simple_white",
            yaxis=dict(range=[0, max_price]),
        )
        st.plotly_chart(fig_d, use_container_width=True)

    with g2:
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(x=qs, y=trs, mode="lines", name="Total Revenue", line=dict(color=cbc_darkorange)))
        fig_tr.update_layout(
            title="Total Revenue (TR) vs Quantity",
            xaxis_title="Quantity (Q)",
            yaxis_title="Total Revenue (TR)",
            template="simple_white",
        )
        st.plotly_chart(fig_tr, use_container_width=True)

    with g3:
        fig_mr = go.Figure()
        if len(q_for_mr) > 0:
            fig_mr.add_trace(
                go.Scatter(
                    x=q_for_mr,
                    y=mr_vals,
                    mode="lines",
                    name="Marginal Revenue (Approx)",
                    line=dict(color=cbc_gold, shape="hv"),
                )
            )
        fig_mr.update_layout(
            title="Marginal Revenue (MR) vs Quantity (Step)",
            xaxis_title="Quantity (Q)",
            yaxis_title="Marginal Revenue (MR)",
            template="simple_white",
        )
        st.plotly_chart(fig_mr, use_container_width=True)

    # ----------------------------
    # History table + highlight best TR
    # ----------------------------
    st.divider()
    st.markdown("### 🧾 Your Rounds (History)")
    if st.session_state.mr_game_history:
        hist_df = pd.DataFrame(st.session_state.mr_game_history)

        best = st.session_state.mr_game_best_tr
        if best is not None:
            st.info(f"🏆 Best Total Revenue so far: **${best['TR']:.0f}** at **P=${best['P']:.0f}**, **Q={best['Q']}**")

        st.dataframe(hist_df, use_container_width=True)
        export_csv_button("📥 Download MR Game History CSV", hist_df, "mr_game_history")
    else:
        st.info("No rounds yet. Choose a price and click **Submit Round**.")

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
        fig3.add_trace(
            go.Scatter(
                x=market_q_desc,
                y=market_prices_desc,
                mode="lines+markers",
                name="Market Demand (Step)",
                line=dict(color=cbc_darkorange, shape="hv"),  # stair-step
            )
        )
        fig3.update_layout(
            title="Market Demand Curve (After Submissions)",
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


def page_ppf_rabbits_berries():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 🐇🍓 Production Possibilities Frontier (PPF)")

    resources = st.sidebar.slider("Total Labor Hours", 10, 100, 50)
    cost_rabbit = st.sidebar.slider("Labor per Rabbit", 1, 5, 2)
    cost_berry = st.sidebar.slider("Labor per Berry", 1, 5, 1)

    def produce_rabbits(labor):
        return (labor / cost_rabbit) ** 0.5

    def produce_berries(labor):
        return (labor / cost_berry) ** 0.6

    labor_range = np.linspace(0, resources, 100)
    r_output = produce_rabbits(labor_range)
    b_output = produce_berries(resources - labor_range)

    r_labor = st.slider("Labor for Rabbits", 0, resources, int(resources / 2))
    b_labor = resources - r_labor

    r_choice = float(produce_rabbits(r_labor))
    b_choice = float(produce_berries(b_labor))

    st.markdown(f"🐇 Rabbits: **{r_choice:.1f}**  |  🍓 Berries: **{b_choice:.1f}**")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r_output, y=b_output, mode="lines", name="PPF Frontier", line=dict(color=cbc_darkorange)))
        fig.add_trace(
            go.Scatter(x=[r_choice], y=[b_choice], mode="markers", name="Your Allocation", marker=dict(size=12, color=cbc_gold))
        )
        fig.update_layout(
            title="Production Possibilities Frontier",
            xaxis_title="Rabbits",
            yaxis_title="Berries",
            template="simple_white",
            legend=dict(x=0.8, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bar_fig = go.Figure()
        bar_fig.add_trace(
            go.Bar(
                x=["Rabbits", "Berries"],
                y=[r_choice, b_choice],
                marker_color=[cbc_gold, cbc_blue],
                text=[f"{r_choice:.1f}", f"{b_choice:.1f}"],
                textposition="auto",
            )
        )
        bar_fig.update_layout(title="Your Output Allocation", yaxis_title="Units Produced", xaxis_title="Goods", template="simple_white")
        st.plotly_chart(bar_fig, use_container_width=True)

    df_ppf = pd.DataFrame({"Good": ["Rabbits", "Berries"], "Output": [r_choice, b_choice]})
    export_csv_button("📥 Download CSV", df_ppf, "ppf_allocation")
    st.markdown("</div>", unsafe_allow_html=True)


def page_budget_constraint():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 💰 Budget Constraint Simulator")

    income = st.sidebar.slider("Weekly Budget ($)", 10, 200, 100)
    price_coffee = st.sidebar.slider("Price of Coffee ($)", 1, 10, 5)
    price_sandwich = st.sidebar.slider("Price of Sandwich ($)", 1, 10, 5)

    max_coffee = income / price_coffee
    coffee_range = np.linspace(0, max_coffee, 100)
    sandwich_range = (income - coffee_range * price_coffee) / price_sandwich

    coffee_choice = st.slider("Coffees to Buy", 0, int(max_coffee), int(max_coffee // 2))
    sandwich_choice = (income - coffee_choice * price_coffee) / price_sandwich

    st.markdown(f"☕ Coffees: **{coffee_choice}**  | 🥪 Sandwiches: **{sandwich_choice:.1f}**")

    col1, col2 = st.columns(2)
    with col1:
        bc_fig = go.Figure()
        bc_fig.add_trace(go.Scatter(x=coffee_range, y=sandwich_range, mode="lines", name="Budget Line", line=dict(color=cbc_blue)))
        bc_fig.add_trace(
            go.Scatter(x=[coffee_choice], y=[sandwich_choice], mode="markers", name="Your Choice", marker=dict(size=12, color=cbc_orange))
        )
        bc_fig.update_layout(
            title="Budget Constraint",
            xaxis_title="Coffees",
            yaxis_title="Sandwiches",
            template="simple_white",
            legend=dict(x=0.75, y=0.95),
        )
        st.plotly_chart(bc_fig, use_container_width=True)

    with col2:
        bar_fig = go.Figure()
        bar_fig.add_trace(
            go.Bar(
                x=["Coffees", "Sandwiches"],
                y=[coffee_choice, sandwich_choice],
                marker_color=[cbc_orange, cbc_blue],
                text=[f"{coffee_choice}", f"{sandwich_choice:.1f}"],
                textposition="auto",
            )
        )
        bar_fig.update_layout(title="Your Spending Allocation", yaxis_title="Quantity Chosen", xaxis_title="Goods", template="simple_white")
        st.plotly_chart(bar_fig, use_container_width=True)

    df_budget = pd.DataFrame({"Good": ["Coffees", "Sandwiches"], "Quantity": [coffee_choice, sandwich_choice]})
    export_csv_button("📥 Download Budget CSV", df_budget, "budget_allocation")
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
    st.markdown("## 📊 Average and Marginal Cost Curve Explorer")

    st.write("Choose your fixed and variable costs and see how they shape your cost structure and curves.")

    fixed_cost = st.slider("Fixed Cost ($)", 0, 1000, 200)
    variable_cost_per_unit = st.slider("Variable Cost per Unit ($)", 1, 100, 10)
    max_quantity = st.slider("Quantity to Produce", 1, 50, 10)

    quantities = np.arange(1, max_quantity + 1)
    total_costs = fixed_cost + variable_cost_per_unit * quantities

    atc = total_costs / quantities
    avc = variable_cost_per_unit * np.ones_like(quantities)
    afc = fixed_cost / quantities
    mc = np.gradient(total_costs, quantities)

    col1, col2 = st.columns(2)

    with col1:
        cost_fig = go.Figure()
        cost_fig.add_trace(go.Scatter(x=quantities, y=atc, mode="lines+markers", name="ATC", line=dict(color=cbc_blue)))
        cost_fig.add_trace(go.Scatter(x=quantities, y=mc, mode="lines+markers", name="MC", line=dict(color=cbc_gold)))
        cost_fig.update_layout(
            title="Cost vs Quantity",
            xaxis_title="Quantity Produced",
            yaxis_title="Cost per Unit ($)",
            template="simple_white",
            xaxis=dict(range=[0, max_quantity]),
            yaxis=dict(range=[0, None]),
        )
        st.plotly_chart(cost_fig, use_container_width=True)

    with col2:
        curve_fig = go.Figure()
        curve_fig.add_trace(go.Scatter(x=quantities, y=avc, mode="lines+markers", name="AVC", line=dict(color=cbc_orange)))
        curve_fig.add_trace(go.Scatter(x=quantities, y=afc, mode="lines+markers", name="AFC", line=dict(color=cbc_lightblue)))
        curve_fig.update_layout(
            title="Cost Curves (Fixed and Variable)",
            xaxis_title="Quantity Produced",
            yaxis_title="Cost per Unit ($)",
            template="simple_white",
            yaxis=dict(range=[0, None]),
        )
        st.plotly_chart(curve_fig, use_container_width=True)

    df_cost = pd.DataFrame(
        {
            "Quantity": quantities,
            "Fixed Cost": [fixed_cost] * len(quantities),
            "Variable Cost per Unit": [variable_cost_per_unit] * len(quantities),
            "Total Cost": total_costs,
            "ATC": atc,
            "AVC": avc,
            "AFC": afc,
            "MC": mc,
        }
    )
    export_csv_button("📥 Download Cost Curve CSV", df_cost, "cost_curve")
    st.markdown("</div>", unsafe_allow_html=True)


def page_supply_demand_shock():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## 📉📈 Supply & Demand Shock Simulator")

    st.write("Explore how shifts in supply or demand affect equilibrium price and quantity, and how surplus changes.")

    initial_demand_intercept = st.slider("Initial Demand Intercept (P when Q=0)", 10, 100, 60)
    demand_slope = st.slider("Demand Slope (negative)", -10, -1, -5)
    initial_supply_intercept = st.slider("Initial Supply Intercept (P when Q=0)", 10, 100, 20)
    supply_slope = st.slider("Supply Slope (positive)", 1, 10, 5)

    demand_shift = st.slider("Demand Shift (horizontal)", -50, 50, 0)
    supply_shift = st.slider("Supply Shift (horizontal)", -50, 50, 0)

    quantity = np.linspace(0, 50, 500)
    initial_demand = initial_demand_intercept + demand_slope * quantity
    initial_supply = initial_supply_intercept + supply_slope * quantity

    new_demand = initial_demand_intercept + demand_slope * (quantity - demand_shift)
    new_supply = initial_supply_intercept + supply_slope * (quantity - supply_shift)

    def find_equilibrium(d_line, s_line):
        diff = np.abs(d_line - s_line)
        idx = int(np.argmin(diff))
        return float(quantity[idx]), float(d_line[idx])

    q_eq_init, p_eq_init = find_equilibrium(initial_demand, initial_supply)
    q_eq_new, p_eq_new = find_equilibrium(new_demand, new_supply)

    max_price = float(max(initial_demand[0], initial_supply[0], new_demand[0], new_supply[0]))

    cs_init = triangle_area(q_eq_init, float(initial_demand[0] - p_eq_init))
    ps_init = triangle_area(q_eq_init, float(p_eq_init - initial_supply[0]))

    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=quantity, y=initial_demand, mode="lines", name="Demand", line=dict(color=cbc_blue)))
        fig1.add_trace(go.Scatter(x=quantity, y=initial_supply, mode="lines", name="Supply", line=dict(color=cbc_orange)))
        fig1.add_trace(go.Scatter(x=[q_eq_init], y=[p_eq_init], mode="markers", name="Equilibrium", marker=dict(size=10, color=cbc_gold)))

        fig1.add_trace(
            go.Scatter(
                x=[0, q_eq_init, 0],
                y=[p_eq_init, p_eq_init, float(initial_demand[0])],
                fill="toself",
                mode="none",
                fillcolor="rgba(0, 138, 230, 0.2)",
                name="Consumer Surplus",
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=[0, q_eq_init, 0],
                y=[p_eq_init, p_eq_init, float(initial_supply[0])],
                fill="toself",
                mode="none",
                fillcolor="rgba(255, 165, 0, 0.2)",
                name="Producer Surplus",
            )
        )

        fig1.update_layout(title="Initial Market", xaxis_title="Quantity", yaxis_title="Price", template="simple_white", yaxis=dict(range=[0, max_price]))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"**Initial Consumer Surplus:** ${cs_init:.2f}")
        st.markdown(f"**Initial Producer Surplus:** ${ps_init:.2f}")

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=quantity, y=initial_demand, mode="lines", name="Initial Demand", line=dict(color=cbc_blue, dash="dash")))
        fig2.add_trace(go.Scatter(x=quantity, y=initial_supply, mode="lines", name="Initial Supply", line=dict(color=cbc_orange, dash="dash")))
        fig2.add_trace(go.Scatter(x=quantity, y=new_demand, mode="lines", name="New Demand", line=dict(color=cbc_blue, dash="dot")))
        fig2.add_trace(go.Scatter(x=quantity, y=new_supply, mode="lines", name="New Supply", line=dict(color=cbc_orange, dash="dot")))
        fig2.add_trace(go.Scatter(x=[q_eq_new], y=[p_eq_new], mode="markers", name="New Equilibrium", marker=dict(size=10, color=cbc_gold)))

        fig2.add_trace(
            go.Scatter(
                x=[0, q_eq_new, 0],
                y=[p_eq_new, p_eq_new, float(new_demand[0])],
                fill="toself",
                mode="none",
                fillcolor="rgba(0, 138, 230, 0.2)",
                name="Consumer Surplus",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[0, q_eq_new, 0],
                y=[p_eq_new, p_eq_new, float(new_supply[0])],
                fill="toself",
                mode="none",
                fillcolor="rgba(255, 165, 0, 0.2)",
                name="Producer Surplus",
            )
        )

        fig2.update_layout(title="New Market After Shock", xaxis_title="Quantity", yaxis_title="Price", template="simple_white", yaxis=dict(range=[0, max_price]))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"**Initial Equilibrium:** Price = ${p_eq_init:.2f}, Quantity = {q_eq_init:.2f}")
    st.markdown(f"**New Equilibrium:** Price = ${p_eq_new:.2f}, Quantity = {q_eq_new:.2f}")

    df_market = pd.DataFrame(
        {
            "Scenario": ["Initial", "New"],
            "Equilibrium Price": [p_eq_init, p_eq_new],
            "Equilibrium Quantity": [q_eq_init, q_eq_new],
            "Demand Shift": [0, demand_shift],
            "Supply Shift": [0, supply_shift],
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
    "Fixed & Variable Cost Simulator": page_fixed_variable_cost,
    "Cost Curve Explorer": page_cost_curve_explorer,
    "Supply & Demand Shock": page_supply_demand_shock,
    "Marginal Revenue": page_price_setting_marginal_revenue_game,
}

game_choice = st.sidebar.radio("Select a Game:", list(pages.keys()), key="selected_game")

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









