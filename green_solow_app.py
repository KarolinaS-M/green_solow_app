import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(layout="wide")
st.title("Green Solow Model Simulator")

# === Sidebar: Model parameters ===
st.sidebar.header("Model Parameters")

alpha = st.sidebar.slider("α (capital share)", 0.1, 0.9, 0.35)
s = st.sidebar.slider("s (savings rate)", 0.0, 1.0, 0.25)
delta = st.sidebar.slider("δ (depreciation rate)", 0.0, 0.1, 0.035)
n = st.sidebar.slider("n (population growth)", 0.0, 0.05, 0.01)
g = st.sidebar.slider("g (technology growth)", 0.0, 0.05, 0.015)

g_A = st.sidebar.slider("g_A (abatement tech. growth)", 0.0, 0.1, 0.03)
eta = st.sidebar.slider("η (regeneration rate)", 0.0, 0.1, 0.03)
epsilon = st.sidebar.slider("ε (abatement elasticity)", 0.5, 5.0, 1.5)
Omega0 = st.sidebar.slider("Ω₀ (initial pollution intensity)", 0.5, 5.0, 2.5)

theta_weak = st.sidebar.slider("θ (Weak abatement)", 0.001, 0.1, 0.005)
theta_strong = st.sidebar.slider("θ (Strong abatement)", 0.005, 0.2, 0.05)

# === Time horizon ===
T = 180
t_eval = np.linspace(0, T, T + 1)

# === Functions ===
def f(k):
    return k ** alpha

def a(theta):
    return (1 - theta) ** epsilon

def green_solow(t, y, theta):
    k, X, Omega = y
    y_output = f(k) * (1 - theta)
    emissions = f(k) * Omega * a(theta)
    dkdt = s * y_output - (delta + n + g) * k
    dXdt = emissions - eta * X
    dOmegadt = -g_A * Omega
    return [dkdt, dXdt, dOmegadt]

# === Solve ===
def solve(theta):
    sol = solve_ivp(
        green_solow, (0, T), [0.01, 0.0, Omega0],
        args=(theta,), t_eval=t_eval
    )
    k, X, Omega = sol.y
    F = f(k)
    Y = F * (1 - theta)
    C = (1 - s) * Y
    FA = theta * F
    E = F * Omega * a(theta)
    logEY = np.log(E / Y)
    return {
        "t": t_eval, "k": k, "X": X, "Omega": Omega, "F": F,
        "Y": Y, "C": C, "FA": FA, "E": E, "logEY": logEY
    }

data_w = solve(theta_weak)
data_s = solve(theta_strong)

# === Plotting ===
def plot_dual_line(ax, t, y1, y2, label1, label2, color1, color2, title, ylabel):
    ax.plot(t, y1, '--', label=label1, color=color1)
    ax.plot(t, y2, '-', label=label2, color=color2)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

# === Layout ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pollution Stock and Emissions")
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plot_dual_line(ax1, data_w["t"], data_w["X"], data_s["X"],
                   "Weak A.", "Strong A.", 'skyblue', 'navy',
                   "Pollution Stock X", "Level")
    plot_dual_line(ax2, data_w["t"], data_w["E"], data_s["E"],
                   "Weak A.", "Strong A.", 'tomato', 'firebrick',
                   "Emissions E", "Level")
    st.pyplot(fig1)

with col2:
    st.subheader("Emissions and Log Intensity")
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))
    ax3.plot(data_s["t"], data_s["E"], label="Emissions E", color='seagreen')
    ax3.plot(data_s["t"], data_s["logEY"], label="log(E/Y)", color='darkorange')
    ax3.set_title("Strong Abatement")
    ax3.set_xlabel("Time")
    ax3.grid(True)
    ax3.legend()

    ax4.plot(data_w["t"], data_w["E"], label="Emissions E", color='tomato')
    ax4.plot(data_w["t"], data_w["logEY"], label="log(E/Y)", color='slategray')
    ax4.set_title("Weak Abatement")
    ax4.set_xlabel("Time")
    ax4.grid(True)
    ax4.legend()
    st.pyplot(fig2)

st.subheader("Growth and Abatement Trade-offs")
fig3, axs = plt.subplots(2, 2, figsize=(12, 8))
plot_dual_line(axs[0, 0], data_w["t"], data_w["k"], data_s["k"],
               "Weak A.", "Strong A.", 'skyblue', 'navy',
               "Capital per Eff. Labor k", "Level")
plot_dual_line(axs[0, 1], data_w["t"], data_w["Y"], data_s["Y"],
               "Weak A.", "Strong A.", 'mediumseagreen', 'seagreen',
               "Output per Eff. Labor y", "Level")
plot_dual_line(axs[1, 0], data_w["t"], data_w["C"], data_s["C"],
               "Weak A.", "Strong A.", 'peachpuff', 'darkorange',
               "Consumption per Eff. Labor c", "Level")
plot_dual_line(axs[1, 1], data_w["t"], data_w["FA"], data_s["FA"],
               "Weak A.", "Strong A.", 'gray', 'black',
               "Abatement Fᵃ = θ·F", "Level")
fig3.tight_layout()
st.pyplot(fig3)