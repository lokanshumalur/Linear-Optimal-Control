#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-Dimensional Optimal Control Solver (8D System)
---------------------------------------------------
Implements Minimum Energy Control using Simpson's Rule for Gramian calculation on a densely coupled mechanical system.

By: Lokanshu Malur
Fall 2025
"""

import numpy as np
from scipy.linalg import expm, solve 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

## SETUP ##

# System
N_states = 8
N_inputs = 2

# Matrix A (8x8)
A = np.array([
    [-1.2, 0.8, 0.1, 0.1, 0.1, 0, 0, 0.1],
    [1.5, -2.5, 0.8, 0.1, 0, 0.1, 0, 0.1],
    [0.2, 1.5, -3.0, 0.8, 0.1, 0, 0.1, 0],
    [0, 0.2, 1.5, -4.0, 0.8, 0.1, 0, 0.1],
    [0.1, 0, 0.2, 1.5, -3.5, 0.8, 0.1, 0],
    [0, 0.1, 0, 0.2, 1.5, -2.8, 0.8, 0.1],
    [0.1, 0, 0.1, 0, 0.2, 1.5, -1.9, 0.8],
    [0.8, 0.1, 0, 0.1, 0, 0.2, 1.5, -1.0]
])

# Matrix B (8x2) - Multi-input system (actuators on states 1, 5, and 8)
B = np.zeros((N_states, N_inputs))
B[0, 0] = 1.0  # Input 1 affects state 1 strongly
B[4, 0] = 0.5  # Input 1 affects state 5 moderately
B[7, 1] = 1.0  # Input 2 affects state 8 strongly

# Optimal Control Steerage Parameters
T = 5.0                                                     # Final time 
y0 = np.array([5.0, 1.0, 0.5, 0.1, 0.1, 0.5, 1.0, 5.0])     # Initial disturbance
yT = np.zeros(N_states)                                     # Target State 

# Function Definitions
def compute_gramian_simpson(A, B, T, N_intervals):          # Gramian computation using Simpson's Rule
    if N_intervals % 2 != 0: N_intervals += 1 
    dt = T / N_intervals
    Q = np.zeros((A.shape[0], A.shape[0]))
    def F(t):
        expAt = expm(A * t)                                 # Matrix Exponential - Numerical Method
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
print(f"--- 8D Optimal Control Calculation (T={T}s) ---")
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
print("\n3. Simulation & Physics Results:")
print(f"   Final Steerage Error ||y(T)||: {error:.4e} (Optimal control goal achieved)")
print(f"   Total L2 Control Cost J: {J_total:.4f} (Total energy/effort required for steerage)")
print(f"   System's Max Excursion (Max |y(t)|): {max_excursion:.3f} units (Max displacement from equilibrium)")
print(f"   Input U1 Max Effort (Restoring Force): {np.max(np.abs(U_opt_vec[:,0])):.2f} N (Peak force on units 1 & 5)")
print(f"   Input U2 Max Effort (Restoring Force): {np.max(np.abs(U_opt_vec[:,1])):.2f} N (Peak force on unit 8)")
print(f"   Required Control Vector z Magnitude ||z||: {np.linalg.norm(z):.2f} (Measure of maneuver difficulty: large for short time T)")


## PLOTS ##

# State Trajectories
plt.figure(figsize=(8, 6))
plt.plot(sol_opt.t, sol_opt.y[0], label="$y_1$ (Actuated, High Disturbance)", linewidth=2)
plt.plot(sol_opt.t, sol_opt.y[3], label="$y_4$ (Interior, Highly Coupled)", linestyle='--')
plt.plot(sol_opt.t, sol_opt.y[7], label="$y_8$ (Actuated, High Disturbance)", linewidth=2)
plt.axhline(0, color='k', linewidth=0.8, alpha=0.7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Optimal State Trajectories for System", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("State Value (Displacement)"); plt.legend(loc='upper right')
plt.show()

# Cumulative Control Cost
plt.figure(figsize=(8, 6))
plt.plot(sol_opt.t, np.cumsum(U_opt_mag_sq) * (sol_opt.t[1] - sol_opt.t[0]), 
         label="Cumulative $L_2$ Cost", color='green', linewidth=3)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Cumulative Control Cost $J = \int_0^T ||u(t)||^2 dt$", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("Cost J"); plt.legend(loc='lower right')
plt.show()

# Control Input U_1
plt.figure(figsize=(8, 6))
plt.plot(sol_opt.t, U_opt_vec[:,0], label="$u_1^*(t)$ (Input to $y_1, y_5$)", color='darkblue', linewidth=3)
plt.axhline(0, color='k', linewidth=0.8, alpha=0.7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Optimal Control Input $u_1^*(t)$ (Restoring Force on Units 1 & 5)", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("Control Force (N)"); plt.legend(loc='lower right')
plt.show()

# Control Input U_2
plt.figure(figsize=(8, 6))
plt.plot(sol_opt.t, U_opt_vec[:,1], label="$u_2^*(t)$ (Input to $y_8$)", color='darkred', linewidth=3)
plt.axhline(0, color='k', linewidth=0.8, alpha=0.7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Optimal Control Input $u_2^*(t)$ (Restoring Force on Unit 8)", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("Control Force (N)"); plt.legend(loc='upper right')
plt.show()

## APPENDIX ##

# Numerical Verification (Simpson's Rule Convergence)
print("\n--- Numerical Verification: Simpson's Rule Convergence Study (8D) ---")
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

print("\nError Reduction Factor (should be ~16 for Simpson's Rule):")
for i in range(1, len(errors)):
    factor = errors[i-1] / errors[i]
    print(f"Reduction from N={N_tests[i-1]} to N={N_tests[i]} (Factor of 2): {factor:.2f}")

final_run_error = np.linalg.norm(QT - QT_ref, ord='fro')
print(f"\nQ_T (N={N_intervals_run}) Frobenius Norm Error: {final_run_error:.8e}")

# Raw Data
print("\n--- Appendix: Raw Data ---")
print("\nSystem Matrix A (8x8):")
print(A)
print("\nControllability Gramian Q_T (8x8, Simpson's Rule Result):")
print(QT)
print("\nVector z (Result of the Q_T⁻¹ solve):")
print(z)


