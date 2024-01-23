import numpy as np
import scipy
import matplotlib.pyplot as plt


# from scipy.special import fresnel

######################
# # DFT行列の注意点
# 時間信号から周波数信号に変換するときは、fの成分が見たいから、exp(-j2pi*f*t)を内積する。
# 行列表現で
# x(f) = D^H x(t) と表現する場合、Dの中身はプラスの符号（exp(+j2pi*f*t)）で作成しておく必要。
# Numpyで以下のコードでDFT行列作成したとき、各要素の位相はマイナス（exp(-j2pi*f*t)）になっているので、注意（この場合 x(f) = D x(t) でオッケー（共役転置不要））
# D = np.fft.fft(np.eye(N)) / np.sqrt(N)  # DFT
######################


def main():
    N = 200
    fc = 100e9
    wl = 3e8 / fc
    d = 1 / 2
    yn = np.arange(N) * d - (N - 1) * d / 2

    # target
    AoA = 20 * np.pi / 180
    r = 1 / wl
    nu = np.pi * d ** 2 * np.cos(AoA) ** 2 / r
    omega = 2 * np.pi * d * np.sin(AoA)
    gamma = omega + (N - 1) * nu

    # Est

    HPW_ULA = np.arcsin((2.783 / N) / np.pi) * 180/np.pi

    AoA_error = 0.1 * np.pi / 180
    r_error = 0.1 / wl
    AoA_est = AoA + AoA_error
    r_est = r + r_error
    nu_est = np.pi * d ** 2 * np.cos(AoA_est) ** 2 / r_est
    omega_est = 2 * np.pi * d * np.sin(AoA_est)
    gamma_est = omega_est + (N - 1) * nu_est

    ###############
    G_array = np.zeros(N, dtype=np.float64)
    G_array_est = np.zeros(N, dtype=np.float64)
    G_diff_array = np.zeros(N, dtype=np.float64)
    for n in range(N):
        omega_n = 2 * np.pi * n / N
        G = calc_fresnel_integral(a=nu, b=omega_n - gamma, c=0, N=N)
        G_est = calc_fresnel_integral(a=nu_est, b=omega_n - gamma_est, c=0, N=N)
        G_array[n] = np.abs(G)
        G_array_est[n] = np.abs(G_est)
        G_diff_array[n] = np.abs(G - G_est)
        # G_diff_array[n] = np.abs(G) - np.abs(G_est)
    # plt.plot(G_array, "x-")
    # plt.show()
    ###############

    # DFT
    a_vec = array_factor_ula_nf(r, AoA, yn) / np.sqrt(N)
    a_vec_fr = array_factor_ula_nf_fresnel(r, AoA, yn) / np.sqrt(N)
    a_vec_fr_2 = array_factor_ula_nf_fresnel_2(r, AoA, d, N) / np.sqrt(N)

    eee = np.abs(a_vec_fr - a_vec_fr_2)

    D = np.fft.fft(np.eye(N)) / np.sqrt(N)  # DFT
    D = np.conj(D)
    Da = np.conj(D) @ a_vec
    Da_fr = np.conj(D) @ a_vec_fr

    # plt.plot(np.abs(Da), label="Da")
    # plt.plot(np.abs(Da_fr), "x--", label="Da_fr")
    plt.plot(G_array, "-", label="G (Fresnel approx.)")
    plt.plot(G_array_est, "-", label="G_est (Fresnel approx.)")
    plt.plot(G_diff_array, "-", label="G_diff (Fresnel approx.)")
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
