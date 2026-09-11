"""
Showcase smoothing length iteration algorithlm
==============================================
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock


def compute_sums(pmass, id_a, h_a, W, dhW, positions: np.ndarray):
    rho_sum = 0
    sumdWdh = 0

    for j in range(positions.shape[0]):
        dr = positions[id_a, :] - positions[j, :]
        rab2 = dr.dot(dr)

        rab = np.sqrt(rab2)
        rho_sum += pmass * W(rab, h_a)
        sumdWdh += pmass * dhW(rab, h_a)

    return rho_sum, sumdWdh


def W(r, h):
    return shamrock.math.sphkernel.M4_W3d(r, h)


def dhW(r, h):
    return shamrock.math.sphkernel.M4_dhW3d(r, h)


def rho_h(m, h, hfact):
    return m * (hfact / h) * (hfact / h) * (hfact / h)


hfact = 1.2  # shamrock.math.sphkernel.hfactd

H_EVOL_MPI_MAX = 1.1


def f_df(rho_ha, rho_sum, sumdWdh, h_a):
    f_iter = rho_sum - rho_ha
    df_iter = sumdWdh + 3 * rho_ha / h_a
    return f_iter, df_iter


def f_kernel(q):
    return shamrock.math.sphkernel.M4_f(q)


def df_kernel(q):
    return shamrock.math.sphkernel.M4_df(q)


def plot_f_df_kernel():
    q = np.linspace(0, 4, 1000)

    f_values = np.array([f_kernel(x) for x in q])
    df_values = np.array([df_kernel(x) for x in q])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(q, f_values, label=r"$f(q)$")
    ax.plot(q, df_values, label=r"$df(q)$")
    ax.plot(q, f_values + df_values * q / 3, label=r"$f(q) + df(q) \cdot q / 3$")
    ax.set_xlabel(r"$q$")
    ax.legend()
    plt.show()


def newton_iterate_new_h(h_a, positions, state_vars: dict, h_max_evol_m=0.9, h_max_evol_p=1.1):
    if "ha_0" not in state_vars:
        state_vars["ha_0"] = h_a

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, W, dhW, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)
    new_h = h_a - f_iter / df_iter

    print(
        f"new_h = {new_h}, h_a = {h_a}, f_iter = {f_iter}, df_iter = {df_iter}, lim m = {h_a * h_max_evol_m}, lim p = {h_a * h_max_evol_p}"
    )
    new_h = max(new_h, h_a * h_max_evol_m)

    new_h = min(new_h, h_a * h_max_evol_p)

    eps = abs(new_h - h_a) / state_vars["ha_0"]
    is_done = eps < 1e-6
    return new_h, is_done


def newton_iterate_new_h_lim(h_a, positions, state_vars: dict, h_max_evol_m=0.9, h_max_evol_p=1.1):
    if "ha_0" not in state_vars:
        state_vars["ha_0"] = h_a

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, W, dhW, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)
    new_h = h_a - f_iter / df_iter

    print(
        f"new_h = {new_h}, h_a = {h_a}, f_iter = {f_iter}, df_iter = {df_iter}, lim m = {h_a * h_max_evol_m}, lim p = {h_a * h_max_evol_p}"
    )
    new_h = max(new_h, h_a * h_max_evol_m)

    new_h = min(new_h, h_a * h_max_evol_p)

    if f_iter > 0 and f_iter / df_iter < 0:
        new_h = h_a * h_max_evol_m

    eps = abs(new_h - h_a) / state_vars["ha_0"]
    is_done = eps < 1e-6
    return new_h, is_done


def analyse_h_convergence(positions: np.ndarray, id_a: int, pmass: float, iterate_new_h):

    h_a_test = np.logspace(-2, 2, 1000)

    f_values = np.zeros(h_a_test.shape)
    df_values = np.zeros(h_a_test.shape)

    rho_sum_values = np.zeros(h_a_test.shape)
    rho_h_values = np.zeros(h_a_test.shape)

    for i in range(h_a_test.shape[0]):
        rho_ha = rho_h(pmass, h_a_test[i], hfact)
        rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a_test[i], W, dhW, positions)
        rho_sum_values[i] = rho_sum
        rho_h_values[i] = rho_ha
        f_values[i], df_values[i] = f_df(rho_ha, rho_sum, sumdWdh, h_a_test[i])

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(
        h_a_test, f_values, label=r"$f(h_a) = \sum_b m_b W(r_{ab}, h_a) - \rho_h(m_a, h_a)$"
    )
    axs[0].plot(
        h_a_test,
        df_values,
        label=r"$f'(h_a) = \sum_b m_b \frac{\partial W}{\partial h}(r_{ab}, h_a) + 3 \rho_h(m_a, h_a) / h_a$",
    )
    axs[0].plot(h_a_test, rho_h_values, label=r"$\rho_h(m_a, h_a)$")
    axs[0].plot(h_a_test, rho_sum_values, label=r"$\rho_sum(m_a, h_a)$")

    # plt.ylim(-10, 10)
    axs[0].set_yscale("symlog", linthresh=1e-4)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("h_a")
    axs[0].legend()

    # sample 10 equally spaced values in h_a_test indexes
    test_h_values = h_a_test[np.linspace(0, h_a_test.shape[0] - 1, 4).astype(int)]

    for init_h_a in test_h_values:
        h_a = init_h_a
        history_h_a = [h_a]
        state_vars = {}
        for i in range(100):
            h_a_prev = h_a
            h_a, is_done = iterate_new_h(h_a, positions, state_vars)
            assert h_a <= H_EVOL_MPI_MAX * h_a_prev, (
                f"h_a = {h_a} is larger than H_EVOL_MPI_MAX * h_a_prev = {H_EVOL_MPI_MAX * h_a_prev}"
            )
            history_h_a.append(h_a)
            if is_done:
                break
        axs[1].plot(history_h_a, label=f"init_h_a = {init_h_a}")

    axs[1].set_yscale("log")
    axs[1].set_xlabel("iteration count")
    axs[1].set_ylabel("h_a")
    axs[1].legend()

    plt.show()


positions = []

id_a = 0

for ix in range(20):
    for iy in range(20):
        for iz in range(20):
            positions.append((ix, iy, iz))
            # positions.append(np.random.rand(3))

            if ix == 10 and iy == 10 and iz == 10:
                id_a = len(positions) - 1

pmass = 1.0 / 1000.0

positions = np.array(positions)

plot_f_df_kernel()
analyse_h_convergence(positions, id_a, pmass, newton_iterate_new_h)
analyse_h_convergence(positions, id_a, pmass, newton_iterate_new_h_lim)
