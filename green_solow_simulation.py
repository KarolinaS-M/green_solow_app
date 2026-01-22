import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === Plot style ===
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# === Model parameters ===
alpha = 0.35
s = 0.25
delta = 0.035
n = 0.01
g = 0.015
g_A = 0.03
eta = 0.03
epsilon = 1.5

# === Time horizon ===
T = 180
t_span = (0, T)
t_eval = np.linspace(*t_span, T + 1)

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

# === Initial conditions and scenarios ===
k0, X0, Omega0 = 0.01, 0.0, 2.5  # Omega0 chosen to match the emission scale in Brock and Taylor
theta_weak = 0.005
theta_strong = 0.05

sol_weak = solve_ivp(green_solow, t_span, [k0, X0, Omega0], t_eval=t_eval, args=(theta_weak,))
sol_strong = solve_ivp(green_solow, t_span, [k0, X0, Omega0], t_eval=t_eval, args=(theta_strong,))

k_w, X_w, Omega_w = sol_weak.y
k_s, X_s, Omega_s = sol_strong.y

# === Derived variables ===
E_w = f(k_w) * Omega_w * a(theta_weak)
E_s = f(k_s) * Omega_s * a(theta_strong)

Y_w = f(k_w) * (1 - theta_weak)
Y_s = f(k_s) * (1 - theta_strong)

C_w = (1 - s) * Y_w
C_s = (1 - s) * Y_s

FA_w = theta_weak * f(k_w)
FA_s = theta_strong * f(k_s)

log_EY_w = np.log(E_w / Y_w)
log_EY_s = np.log(E_s / Y_s)

# === Plot 1: Pollution Stock and Emissions ===
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
ax1.plot(t_eval, X_w, '--', label="Weak A.", color='skyblue')
ax1.plot(t_eval, X_s, '-', label="Strong A.", color='navy')
ax1.set_title("Pollution Stock $X$")
ax1.set_xlabel("Time")
ax1.set_ylabel("Level")
ax1.legend()
ax1.grid(True)

ax2.plot(t_eval, E_w, '--', label="Weak A.", color='tomato')
ax2.plot(t_eval, E_s, '-', label="Strong A.", color='firebrick')
ax2.set_title("Emissions $E$")
ax2.set_xlabel("Time")
ax2.legend()
ax2.grid(True)

fig1.tight_layout()
fig1.savefig("pollution_and_emissions.png", dpi=300)
plt.show()

# === Plot 2: Capital, Output, Consumption, Abatement ===
fig2, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(t_eval, k_w, '--', label='Weak A.', color='skyblue')
axs[0, 0].plot(t_eval, k_s, '-', label='Strong A.', color='navy')
axs[0, 0].set_title("Capital per Effective Labor $k$")
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Level")
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(t_eval, Y_w, '--', label='Weak A.', color='mediumseagreen')
axs[0, 1].plot(t_eval, Y_s, '-', label='Strong A.', color='seagreen')
axs[0, 1].set_title("Output per Effective Labor $y$")
axs[0, 1].set_xlabel("Time")
axs[0, 1].set_ylabel("Level")
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].plot(t_eval, C_w, '--', label='Weak A.', color='peachpuff')
axs[1, 0].plot(t_eval, C_s, '-', label='Strong A.', color='darkorange')
axs[1, 0].set_title("Consumption per Effective Labor $c$")
axs[1, 0].set_xlabel("Time")
axs[1, 0].set_ylabel("Level")
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(t_eval, FA_w, '--', label='Weak A.', color='gray')
axs[1, 1].plot(t_eval, FA_s, '-', label='Strong A.', color='black')
axs[1, 1].set_title("Abatement $F^A = \theta F$")
axs[1, 1].set_xlabel("Time")
axs[1, 1].set_ylabel("Level")
axs[1, 1].legend()
axs[1, 1].grid(True)

fig2.tight_layout()
fig2.savefig("macro_variables_fourpanel.png", dpi=300)
plt.show()

# === Plot 3: Emissions and log(E/Y) ===
fig3, (ax3_left, ax3_right) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

ax3_left.plot(t_eval, E_s, label="Emissions $E$", color='seagreen')
ax3_left.plot(t_eval, log_EY_s, label=r"$\log(E/Y)$", color='darkorange')
ax3_left.set_title("Strong Abatement")
ax3_left.set_xlabel("Time")
ax3_left.set_ylabel("Level")
ax3_left.legend()
ax3_left.grid(True)

ax3_right.plot(t_eval, E_w, label="Emissions $E$", color='tomato')
ax3_right.plot(t_eval, log_EY_w, label=r"$\log(E/Y)$", color='slategray')
ax3_right.set_title("Weak Abatement")
ax3_right.set_xlabel("Time")
ax3_right.legend()
ax3_right.grid(True)

fig3.tight_layout()
fig3.savefig("log_emission_intensity.png", dpi=300)
plt.show()
