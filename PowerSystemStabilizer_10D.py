#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-Dimensional Optimal Control Solver (10D System)
---------------------------------------------------
Implements Minimum Energy Control using Simpson's Rule for Gramian calculation on a power system stabilization model.

By: Lokanshu Malur
Fall 2025
"""

import numpy as np
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

## SETUP ##

# System
N_states = 10                                                   # States (y): [delta1..delta5, omega1..omega5]
N_inputs = 3                                                    # PSS input torques on generators 1, 3, 5

# Matrix A (10x10)
A = np.zeros((N_states, N_states))

# Block 1 (Top Right 5x5)
A[0:5, 5:10] = np.eye(5)                                        # delta_dot = omega (Identity matrix)

# Block 2 (Bottom Right 5x5)
A[5:10, 5:10] = np.diag([-0.15, -0.25, -0.35, -0.20, -0.40])    # Damping terms (-D/M * omega) & Damping constants M/D ratios (negative diagonal)

# Block 3 (Bottom Left 5x5) - A highly coupled structure
# Coupling terms (-P_sync/M * delta)                        
A_21 = np.diag([-2.0, -3.0, -3.5, -2.5, -4.0])                  # Self-stabilization (Pii terms)

# Inter-area coupling (simplified symmetric coupling)
A_21[0, 1] = 0.8; A_21[1, 0] = 0.8
A_21[2, 3] = 0.5; A_21[3, 2] = 0.5
A_21[4, 0] = 0.3; A_21[0, 4] = 0.3
A[5:10, 0:5] = A_21

# Matrix B (10x3) - Control inputs (u) are torques, affecting only omega_dot (states 5-9)
B = np.zeros((N_states, N_inputs))
B[5, 0] = 1.0  # PSS 1 affects Generator 1 frequency (state y[5])
B[7, 1] = 1.0  # PSS 2 affects Generator 3 frequency (state y[7])
B[9, 2] = 1.0  # PSS 3 affects Generator 5 frequency (state y[9])

# Optimal Control Steerage Parameters
T = 8.0                                                         # Final time (longer horizon for electrical modes)

# Initial disturbance: 5 angle deviations (delta) and 5 frequency deviations (omega)
y0_angles = np.array([0.5, -0.2, 0.1, 0.0, -0.3])
y0_freqs = np.array([0.1, 0.05, -0.1, 0.02, 0.15])
y0 = np.concatenate([y0_angles, y0_freqs])
yT = np.zeros(N_states)                                         # Target State (stable operation)

# Function Definitions
def compute_gramian_simpson(A, B, T, N_intervals):              # Gramian computation using Simpson's Rule
    if N_intervals % 2 != 0: N_intervals += 1
    dt = T / N_intervals
    Q = np.zeros((A.shape[0], A.shape[0]))
    def F(t):
        expAt = expm(A * t)                                     # Matrix Exponential - Numerical Method
        return expAt @ B @ B.T @ expm(A.T * t)
    Q += F(0.0) + F(T)
    for i in range(1, N_intervals, 2): Q += 4 * F(i * dt)
    for i in range(2, N_intervals, 2): Q += 2 * F(i * dt)
    Q *= dt / 3.0
    return Q

def optimal_control(t, A, B, T, z):
    if t < 0.0 or t > T: return np.zeros(B.shape[1])
    expA_T_minus_t = expm(A.T * (T - t))
    u = - B.T @ expA_T_minus_t @ z
    return u

def optimal_closed_loop(t, y):
    u_vector = optimal_control(t, A, B, T, z)
    return A @ y + B @ u_vector

## CALCULATIONS ##

N_intervals_run = 8000
print(f"--- 10D Power System Optimal Control Calculation (T={T}s) ---")
print(f"1. Computing Gramian Q_T using Simpson's Rule (N={N_intervals_run} intervals)...")
QT = compute_gramian_simpson(A, B, T, N_intervals=N_intervals_run)
expAT = expm(A * T)
rhs = expAT @ y0 - yT
z = solve(QT, rhs)
print("2. Solved for z using stable linear solve (Decomposition method).")

## SIMULATIONS ##

t_span = (0.0, T)
t_eval = np.linspace(0.0, T, 1500)
sol_opt = solve_ivp(optimal_closed_loop, t_span, y0, t_eval=t_eval, rtol=1e-10, atol=1e-12)

# FINAL METRICS ##

U_opt_vec = np.array([optimal_control(t, A, B, T, z) for t in sol_opt.t])
U_opt_mag_sq = np.sum(U_opt_vec**2, axis=1)
J_total = np.trapz(U_opt_mag_sq, sol_opt.t)
final_state = sol_opt.y[:,-1]
error = np.linalg.norm(final_state - yT)
max_excursion = np.max(np.abs(sol_opt.y))

## PRINTS ##

print("\n3. Simulation & Physics Results (Power System):")
print(f"    Final Steerage Error ||y(T)||: {error:.4e} (Target angle/frequency stability achieved)")
print(f"    Total L2 Control Cost J: {J_total:.4f} (Total PSS energy/effort required)")
print(f"    System's Max Excursion (Max |y(t)|): {max_excursion:.3f} units (Max initial deviation from target)")
print(f"    Input U1 Max Effort (PSS on Gen 1): {np.max(np.abs(U_opt_vec[:,0])):.2f} pu (Peak control torque)")
print(f"    Input U2 Max Effort (PSS on Gen 3): {np.max(np.abs(U_opt_vec[:,1])):.2f} pu (Peak control torque)")
print(f"    Input U3 Max Effort (PSS on Gen 5): {np.max(np.abs(U_opt_vec[:,2])):.2f} pu (Peak control torque)")
print(f"    Required Control Vector z Magnitude ||z||: {np.linalg.norm(z):.2f}")


## PLOTS ##

# State Trajectories (Angles and Frequencies)
plt.figure(figsize=(10, 6))
# Angle States (y[0] and y[4])
plt.plot(sol_opt.t, sol_opt.y[0], label="$\delta_1$ (Gen 1 Angle)", linewidth=2, linestyle='-')
plt.plot(sol_opt.t, sol_opt.y[4], label="$\delta_5$ (Gen 5 Angle)", linewidth=2, linestyle='--')
# Frequency States (y[5] and y[9])
plt.plot(sol_opt.t, sol_opt.y[5], label="$\omega_1$ (Gen 1 Freq)", linewidth=1.5, linestyle=':')
plt.plot(sol_opt.t, sol_opt.y[9], label="$\omega_5$ (Gen 5 Freq)", linewidth=1.5, linestyle='-.')

plt.axhline(0, color='k', linewidth=0.8, alpha=0.7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Optimal State Trajectories ($\delta$ and $\omega$) for 10D Power System", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("State Value"); plt.legend(loc='upper right')
plt.show()

# Cumulative Control Cost
plt.figure(figsize=(8, 6))
plt.plot(sol_opt.t, np.cumsum(U_opt_mag_sq) * (sol_opt.t[1] - sol_opt.t[0]),
          label="Cumulative $L_2$ Cost", color='green', linewidth=3)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Cumulative Control Cost $J = \int_0^T ||u(t)||^2 dt$", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("Cost J"); plt.legend(loc='lower right')
plt.show()

# Control Inputs U_1, U_2, U_3
plt.figure(figsize=(10, 6))
plt.plot(sol_opt.t, U_opt_vec[:,0], label="$u_1^*(t)$ (PSS 1 on Gen 1)", color='darkblue', linewidth=2)
plt.plot(sol_opt.t, U_opt_vec[:,1], label="$u_2^*(t)$ (PSS 2 on Gen 3)", color='orange', linewidth=2, linestyle='--')
plt.plot(sol_opt.t, U_opt_vec[:,2], label="$u_3^*(t)$ (PSS 3 on Gen 5)", color='darkred', linewidth=2, linestyle='-.')
plt.axhline(0, color='k', linewidth=0.8, alpha=0.7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Optimal Control Inputs $u^*(t)$ (PSS Torques)", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("Control Torque (pu)"); plt.legend(loc='lower right')
plt.show()

## APPENDIX ##

# Numerical Verification (Simpson's Rule Convergence)
print("\n--- Numerical Verification: Simpson's Rule Convergence Study (10D) ---")
N_reference = 16000
print(f"Reference Q_T computed with N={N_reference}. Testing convergence rate...")
QT_ref = compute_gramian_simpson(A, B, T, N_intervals=N_reference)

N_tests = [1000, 2000, 4000, 8000]
errors = []

print(" N_Intervals | Frobenius Norm Error (||Q_T(N) - Q_T(Ref)||_F)")
print("-------------|-----------------------------------------------")

for N in N_tests:
    QT_N = compute_gramian_simpson(A, B, T, N_intervals=N)
    error_norm = np.linalg.norm(QT_N - QT_ref, ord='fro')
    errors.append(error_norm)
    print(f" {N:11} | {error_norm:.8e}")

# Raw Data
print("\n--- Appendix: Raw Data for Reproducibility ---")
print(f"\nSystem Matrix A ({N_states}x{N_states}):")
print(A)
print("\nInput Matrix B (10x3):")
print(B)