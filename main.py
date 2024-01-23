import numpy as np
import scipy
import matplotlib.pyplot as plt


# from scipy.special import fresnel


def main():
    N = 200
    fc = 100e9
    wl = 3e8 / fc
    d = 1 / 2
    yn = np.arange(N) * d - (N - 1) * d / 2

    # target
    AoA = 45 * np.pi / 180
    r = 1 / wl
    nu = np.pi * d ** 2 * np.cos(AoA) ** 2 / r
    omega = 2 * np.pi * d * np.sin(AoA)
    gamma = omega + (N - 1) * nu

    ###############
    G_array = np.zeros(N, dtype=np.float64)
    for n in range(N):
        omega_n = 2 * np.pi * n / N
        G = calc_fresnel_integral(a=nu, b=omega_n - gamma, c=0, N=N)
        G_array[n] = np.abs(G)
    # plt.plot(G_array, "x-")
    # plt.show()
    ###############

    # DFT
    a_vec = array_factor_ula_nf(r, AoA, yn) / np.sqrt(N)
    a_vec_fr = array_factor_ula_nf_fresnel(r, AoA, yn) / np.sqrt(N)
    a_vec_fr_2 = array_factor_ula_nf_fresnel_2(r, AoA, d, N) / np.sqrt(N)

    eee = np.abs(a_vec_fr - a_vec_fr_2)

    D = np.fft.fft(np.eye(N)) / np.sqrt(N)  # DFT
    Da = np.conj(D) @ a_vec
    Da_fr = np.conj(D) @ a_vec_fr

    plt.plot(np.abs(Da), label="Da")
    plt.plot(np.abs(Da_fr), label="Da_fr")
    plt.plot(G_array, "x-", label="Fresnel integral approx.")
    plt.legend()
    plt.show()


def array_factor_ula_nf(r_v, theta_v, yn):
    N = len(yn)
    if type(r_v) != np.ndarray: r_v = np.array([r_v])
    if type(theta_v) != np.ndarray: theta_v = np.array([theta_v])
    R = np.tile(r_v[None, :], (N, 1))
    Rn = np.sqrt(R ** 2 + yn[:, None] ** 2 - 2 * R * (yn[:, None] @ np.sin(theta_v[None, :])))
    A_nf = np.exp(-1j * 2 * np.pi * (Rn - R))

    return A_nf  # A_nf = [a(θ1,r1), ..., a(θL, rL)]  (N, L)


def array_factor_ula_nf_fresnel(r_v, theta_v, yn):
    # Fresnel approximation
    N = yn.shape[0]
    if type(r_v) != np.ndarray: r_v = np.array([r_v])
    if type(theta_v) != np.ndarray: theta_v = np.array([theta_v])
    omega = 2 * np.pi * yn[:, None] * np.sin(theta_v[None, :])  # (N,L)
    nu = np.pi * yn[:, None] ** 2 / r_v[None, :] * np.cos(theta_v[None, :]) ** 2
    A_nf_fresnel = np.exp(1j * (omega - nu))
    return A_nf_fresnel


# yn = n*d - (N-1)/2 の場合
def array_factor_ula_nf_fresnel_2(r_v, theta_v, d, N):
    # Fresnel approximation
    if type(r_v) != np.ndarray: r_v = np.array([r_v])
    if type(theta_v) != np.ndarray: theta_v = np.array([theta_v])
    omega = 2 * np.pi * d * np.sin(theta_v)  # (N,L)
    nu = np.pi * d ** 2 / r_v * np.cos(theta_v) ** 2
    gamma = omega + (N - 1) * nu
    n_vec = np.arange(N)
    Phi_1 = nu[None, :] * n_vec[:, None] ** 2 - gamma[None, :] * n_vec[:, None]
    phi_2 = omega * (N - 1) / 2 + nu * (N - 1) ** 2 / 4
    A_nf_fresnel = np.exp(-1j * (Phi_1 + phi_2[None, :]))
    return A_nf_fresnel


def calc_fresnel_integral(a, b, c, N):
    beta = np.sqrt(2 * a / np.pi) * N
    alpha = np.sqrt(2 * a / np.pi) * b / (2 * a)
    C1, S1 = scipy.special.fresnel(beta + alpha)
    C2, S2 = scipy.special.fresnel(alpha)
    G = (C1 - C2 + 1j * (S1 - S2)) * np.exp(1j * (c - a * b ** 2)) / beta
    return G


def plot_fresnel():
    # フレネル積分
    x = np.linspace(0, 10, 101)
    C = scipy.special.fresnel(x)[0]
    S = scipy.special.fresnel(x)[1]
    G = np.abs(C + 1j * S) / np.abs(x)
    plt.plot(C, "-x", label="C(x)")
    plt.plot(S, "-o", label="S(x)")
    plt.plot(G, "-s", label="|C(x)+jS(x)|/|x|")
    plt.legend()
    plt.show()


main()
