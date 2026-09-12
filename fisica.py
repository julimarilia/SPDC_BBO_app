import numpy as np
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt

def bboind(lamb):
    l_um = lamb * 1e6
    n_o = np.sqrt(2.7359 + 0.01878 / (l_um**2 - 0.01822) - 0.01354 * l_um**2)
    n_e = np.sqrt(2.3753 + 0.01224 / (l_um**2 - 0.01667) - 0.01516 * l_um**2)
    return n_o, n_e

def get_indices(s_crystal, n_o, n_e):
    sx, sy, sz = s_crystal
    B = sx**2 * (1/n_o**2 + 1/n_e**2) + sy**2 * (1/n_o**2 + 1/n_e**2) + sz**2 * (2/n_o**2)
    C = sx**2 / (n_o**2 * n_e**2) + sy**2 / (n_o**2 * n_e**2) + sz**2 / (n_o**4)

    discriminante = np.maximum(0, B**2 - 4*C)
    n_fast = np.sqrt(2 / (B + np.sqrt(discriminante)))
    n_slow = np.sqrt(2 / (B - np.sqrt(discriminante)))
    return n_fast, n_slow

def transform_lab_a_cristal(s_lab, theta_p, phi_p):
    R = np.array([
        [np.cos(theta_p)*np.cos(phi_p), -np.sin(phi_p), np.sin(theta_p)*np.cos(phi_p)],
        [np.cos(theta_p)*np.sin(phi_p),  np.cos(phi_p), np.sin(theta_p)*np.sin(phi_p)],
        [-np.sin(theta_p),              0,              np.cos(theta_p)]
    ])
    return R.dot(s_lab)

def phasematch_NIST_T2(x, cut, lp, ls, phi_s, W, L, tipo='II'):
    theta_s, theta_i = x
    theta_p = cut * np.pi / 180
    phi_p = 0.0

    s_lab_pump = np.array([0.0, 0.0, 1.0])
    s_lab_signal = np.array([np.sin(theta_s)*np.cos(phi_s), np.sin(theta_s)*np.sin(phi_s), np.cos(theta_s)])

    phi_i = phi_s + np.pi
    s_lab_idler = np.array([np.sin(theta_i)*np.cos(phi_i), np.sin(theta_i)*np.sin(phi_i), np.cos(theta_i)])

    s_crist_pump = transform_lab_a_cristal(s_lab_pump, theta_p, phi_p)
    s_crist_signal = transform_lab_a_cristal(s_lab_signal, theta_p, phi_p)
    s_crist_idler = transform_lab_a_cristal(s_lab_idler, theta_p, phi_p)

    n_o_p, n_e_p = bboind(lp)
    nf_pump, _ = get_indices(s_crist_pump, n_o_p, n_e_p)

    n_o_s, n_e_s = bboind(ls)
    nf_signal, ns_signal = get_indices(s_crist_signal, n_o_s, n_e_s)
    n_signal = ns_signal if tipo == 'I' else nf_signal

    li = 1 / (1/lp - 1/ls)
    n_o_i, n_e_i = bboind(li)
    _, ns_idler = get_indices(s_crist_idler, n_o_i, n_e_i)


    k_pump = (nf_pump * 2 * np.pi / lp) * s_lab_pump
    k_signal = (n_signal * 2 * np.pi / ls) * s_lab_signal
    k_idler = (ns_idler * 2 * np.pi / li) * s_lab_idler

    Dk = k_pump - k_signal - k_idler
    Dkx, Dky, Dkz = Dk[0], Dk[1], Dk[2]

    sinc_arg = 0.5 * L * Dkz
    sinc_term = 1.0 if sinc_arg == 0 else (np.sin(sinc_arg) / sinc_arg)**2
    Phi = np.exp(-0.5 * W**2 * (Dkx**2 + Dky**2)) * sinc_term

    return np.real(1.0 - Phi)


def phasematch_NIST_T2_robusto(x, cut, lp, ls, phi_s, w, L, incl=0.0, tipo='II'):
    theta_s, theta_i = x
    theta_p = cut * np.pi / 180 + incl_externa_a_interna(incl, lp, cut)
    phi_p = 0.0

    s_lab_pump = np.array([0.0, 0.0, 1.0])
    s_lab_signal = np.array([np.sin(theta_s)*np.cos(phi_s), np.sin(theta_s)*np.sin(phi_s), np.cos(theta_s)])

    phi_i = phi_s + np.pi
    s_lab_idler = np.array([np.sin(theta_i)*np.cos(phi_i), np.sin(theta_i)*np.sin(phi_i), np.cos(theta_i)])

    s_crist_pump = transform_lab_a_cristal(s_lab_pump, theta_p, phi_p)
    s_crist_signal = transform_lab_a_cristal(s_lab_signal, theta_p, phi_p)
    s_crist_idler = transform_lab_a_cristal(s_lab_idler, theta_p, phi_p)


    n_o_p, n_e_p = bboind(lp)
    nf_pump, _ = get_indices(s_crist_pump, n_o_p, n_e_p)
    n_o_s, n_e_s = bboind(ls)
    nf_signal, ns_signal = get_indices(s_crist_signal, n_o_s, n_e_s)
    n_signal = ns_signal if tipo == 'I' else nf_signal

    li = 1 / (1/lp - 1/ls)
    n_o_i, n_e_i = bboind(li)
    _, ns_idler = get_indices(s_crist_idler, n_o_i, n_e_i)

    k_pump = (nf_pump * 2 * np.pi / lp) * s_lab_pump
    k_signal = (n_signal * 2 * np.pi / ls) * s_lab_signal
    k_idler = (ns_idler * 2 * np.pi / li) * s_lab_idler

    Dk = k_pump - k_signal - k_idler

    error_Dk = (Dk[0] * w)**2 + (Dk[1] * w)**2 + (Dk[2] * L)**2

    return error_Dk

def calcular_Phi_NIST(theta_s, theta_i, cut, lp, ls, phi_s, W, L, tipo='II'):
    theta_p = cut * np.pi / 180
    phi_p = 0.0

    s_lab_pump = np.array([0.0, 0.0, 1.0])
    s_lab_signal = np.array([np.sin(theta_s)*np.cos(phi_s), np.sin(theta_s)*np.sin(phi_s), np.cos(theta_s)])

    phi_i = phi_s + np.pi
    s_lab_idler = np.array([np.sin(theta_i)*np.cos(phi_i), np.sin(theta_i)*np.sin(phi_i), np.cos(theta_i)])

    s_crist_pump = transform_lab_a_cristal(s_lab_pump, theta_p, phi_p)
    s_crist_signal = transform_lab_a_cristal(s_lab_signal, theta_p, phi_p)
    s_crist_idler = transform_lab_a_cristal(s_lab_idler, theta_p, phi_p)

    n_o_p, n_e_p = bboind(lp)
    nf_pump, _ = get_indices(s_crist_pump, n_o_p, n_e_p)

    n_o_s, n_e_s = bboind(ls)
    nf_signal, ns_signal = get_indices(s_crist_signal, n_o_s, n_e_s)
    n_signal = ns_signal if tipo == 'I' else nf_signal

    li = 1 / (1/lp - 1/ls)
    n_o_i, n_e_i = bboind(li)
    _, ns_idler = get_indices(s_crist_idler, n_o_i, n_e_i)

    k_pump = (nf_pump * 2 * np.pi / lp) * s_lab_pump
    k_signal = (n_signal * 2 * np.pi / ls) * s_lab_signal
    k_idler = (ns_idler * 2 * np.pi / li) * s_lab_idler

    Dk = k_pump - k_signal - k_idler
    Dkx, Dky, Dkz = Dk[0], Dk[1], Dk[2]

    sinc_arg = 0.5 * L * Dkz
    sinc_term = 1.0 if sinc_arg == 0 else (np.sin(sinc_arg) / sinc_arg)**2
    Phi = np.exp(-0.5 * W**2 * (Dkx**2 + Dky**2)) * sinc_term

    return np.real(Phi)


def phasematch_T2_lorentziana(x, cut, lp, ls, phi_s, W, L, tipo='II'):
    theta_s, theta_i = x
    theta_p = cut * np.pi / 180
    phi_p = 0.0

    s_lab_pump = np.array([0.0, 0.0, 1.0])
    s_lab_signal = np.array([np.sin(theta_s)*np.cos(phi_s), np.sin(theta_s)*np.sin(phi_s), np.cos(theta_s)])

    phi_i = phi_s + np.pi
    s_lab_idler = np.array([np.sin(theta_i)*np.cos(phi_i), np.sin(theta_i)*np.sin(phi_i), np.cos(theta_i)])

    s_crist_pump = transform_lab_a_cristal(s_lab_pump, theta_p, phi_p)
    s_crist_signal = transform_lab_a_cristal(s_lab_signal, theta_p, phi_p)
    s_crist_idler = transform_lab_a_cristal(s_lab_idler, theta_p, phi_p)

    n_o_p, n_e_p = bboind(lp)
    nf_pump, _ = get_indices(s_crist_pump, n_o_p, n_e_p)

    n_o_s, n_e_s = bboind(ls)
    nf_signal, ns_signal = get_indices(s_crist_signal, n_o_s, n_e_s)
    n_signal = ns_signal if tipo == 'I' else nf_signal

    li = 1 / (1/lp - 1/ls)
    n_o_i, n_e_i = bboind(li)
    _, ns_idler = get_indices(s_crist_idler, n_o_i, n_e_i)

    k_pump = (nf_pump * 2 * np.pi / lp) * s_lab_pump
    k_signal = (n_signal * 2 * np.pi / ls) * s_lab_signal
    k_idler = (ns_idler * 2 * np.pi / li) * s_lab_idler

    Dk = k_pump - k_signal - k_idler
    Dkx, Dky, Dkz = Dk[0], Dk[1], Dk[2]

    term_transversal = np.exp(-0.5 * W**2 * (Dkx**2 + Dky**2))
    term_longitudinal = (1 / (1 + (L * Dkz / 2)**2))**2

    delta = 1.0 - (term_transversal * term_longitudinal)

    return np.real(delta)

def calcular_thetas_fuera(theta_s, theta_i, cut, lp, ls, phi_s, tipo='II'):
    theta_p = cut * np.pi / 180
    phi_p = 0.0

    s_lab_signal = np.array([np.sin(theta_s)*np.cos(phi_s), np.sin(theta_s)*np.sin(phi_s), np.cos(theta_s)])
    phi_i = phi_s + np.pi
    s_lab_idler = np.array([np.sin(theta_i)*np.cos(phi_i), np.sin(theta_i)*np.sin(phi_i), np.cos(theta_i)])

    s_crist_signal = transform_lab_a_cristal(s_lab_signal, theta_p, phi_p)
    s_crist_idler = transform_lab_a_cristal(s_lab_idler, theta_p, phi_p)

    n_o_s, n_e_s = bboind(ls)
    nf_signal, ns_signal = get_indices(s_crist_signal, n_o_s, n_e_s)
    n_signal = ns_signal if tipo == 'I' else nf_signal

    li = 1 / (1/lp - 1/ls)
    n_o_i, n_e_i = bboind(li)
    _, ns_idler = get_indices(s_crist_idler, n_o_i, n_e_i)

    n_aire = 1.000293
    theta_s_out = np.arcsin(n_signal * np.sin(theta_s) / n_aire)
    theta_i_out = np.arcsin(ns_idler * np.sin(theta_i) / n_aire)

    return theta_s_out, theta_i_out

def proyectar_a_camara(theta_rad, phi, delta_theta):
    r_px = theta_rad / delta_theta
    return r_px * np.cos(phi), r_px * np.sin(phi)


def incl_externa_a_interna(incl, lp, cut):
    if incl == 0.0:
        return 0.0
    n_o, n_e = bboind(lp)
    th = cut * np.pi / 180
    n_p = 1.0 / np.sqrt(np.cos(th)**2 / n_o**2 + np.sin(th)**2 / n_e**2)
    return np.arcsin(np.sin(incl * np.pi / 180) / n_p)


def _snell_vectorial(d_in, n1, n2, n_hat):
    d_in = d_in / np.linalg.norm(d_in)
    n_hat = n_hat / np.linalg.norm(n_hat)
    eta = n1 / n2
    cos_i = float(np.dot(d_in, n_hat))
    sin2_t = eta**2 * (1.0 - cos_i**2)
    if sin2_t > 1.0:
        return None
    cos_t = np.sqrt(1.0 - sin2_t)
    return eta * d_in + (cos_t - eta * cos_i) * n_hat


def indices_signal_idler(theta_s, theta_i, cut, lp, ls, phi_s, incl=0.0, tipo='II'):
    theta_p = cut * np.pi / 180 + incl_externa_a_interna(incl, lp, cut)
    phi_p = 0.0

    s_lab_signal = np.array([np.sin(theta_s)*np.cos(phi_s), np.sin(theta_s)*np.sin(phi_s), np.cos(theta_s)])
    phi_i = phi_s + np.pi
    s_lab_idler = np.array([np.sin(theta_i)*np.cos(phi_i), np.sin(theta_i)*np.sin(phi_i), np.cos(theta_i)])

    s_crist_signal = transform_lab_a_cristal(s_lab_signal, theta_p, phi_p)
    s_crist_idler = transform_lab_a_cristal(s_lab_idler, theta_p, phi_p)

    n_o_s, n_e_s = bboind(ls)
    nf_signal, ns_signal = get_indices(s_crist_signal, n_o_s, n_e_s)
    n_signal = ns_signal if tipo == 'I' else nf_signal

    li = 1 / (1/lp - 1/ls)
    n_o_i, n_e_i = bboind(li)
    _, ns_idler = get_indices(s_crist_idler, n_o_i, n_e_i)

    return n_signal, ns_idler


def refractar_salida_dir(theta, phi, n_cristal, incl_int=0.0, phi_incl=0.0):
    n_aire = 1.000293
    d_in = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    n_hat = np.array([np.sin(incl_int)*np.cos(phi_incl), np.sin(incl_int)*np.sin(phi_incl), np.cos(incl_int)])
    d_out = _snell_vectorial(d_in, n_cristal, n_aire, n_hat)
    if d_out is None:
        return np.nan, np.nan
    return np.arctan2(d_out[0], d_out[2]), np.arctan2(d_out[1], d_out[2])


def calcular_fwhm(x_array, y_array):
    max_int = np.max(y_array)
    mitad_int = max_int / 2
    indices_fwhm = np.where(y_array >= mitad_int)[0]
    ancho_fwhm = x_array[indices_fwhm[-1]] - x_array[indices_fwhm[0]]
    return ancho_fwhm

def generar_pares_its_pico(cut, lp, ls, phi_s, w, L,
                            ventana=1 * np.pi / 180, puntos=150, N_pares=500,
                            x0=None, tipo='II', tope_respaldo=5.0):
    if x0 is None:
        x0 = np.array([1.0, 1.0]) * np.pi / 180
    bnds = ((1e-9, 0.5), (1e-9, 0.5))

    res = minimize(phasematch_NIST_T2_robusto, x0,
                   args=(cut, lp, ls, phi_s, w, L, 0.0, tipo),
                   method='L-BFGS-B', bounds=bnds)

    if res.fun > 1e-6:
        N_grid = 25
        theta_grid = np.linspace(0.05 * np.pi / 180,
                                 tope_respaldo * np.pi / 180, N_grid)
        Phi_grid = np.zeros((N_grid, N_grid))
        for qq in range(N_grid):
            for jj in range(N_grid):
                Phi_grid[qq, jj] = calcular_Phi_NIST(theta_grid[qq], theta_grid[jj],
                                                     cut, lp, ls, phi_s, w, L, tipo)
        if Phi_grid.max() <= 0:
            return None
        i_max, j_max = np.unravel_index(np.argmax(Phi_grid), Phi_grid.shape)
        x0_grid = np.array([theta_grid[i_max], theta_grid[j_max]])
        res = minimize(phasematch_NIST_T2_robusto, x0_grid,
                       args=(cut, lp, ls, phi_s, w, L, 0.0, tipo),
                       method='L-BFGS-B', bounds=bnds)

    ts_peak, ti_peak = res.x

    theta_s = np.linspace(max(1e-4, ts_peak - ventana), ts_peak + ventana, puntos)
    theta_i = np.linspace(max(1e-4, ti_peak - ventana), ti_peak + ventana, puntos)
    Ts_deg = theta_s * 180 / np.pi
    Ti_deg = theta_i * 180 / np.pi

    DELTA = np.zeros((puntos, puntos))
    for q in range(puntos):
        for j in range(puntos):
            DELTA[q, j] = calcular_Phi_NIST(theta_s[q], theta_i[j], cut, lp, ls, phi_s, w, L,
                                            tipo)

    if np.sum(DELTA) == 0:
        return None

    ts_samples, ti_samples, _, _ = its_2d(DELTA, Ts_deg, Ti_deg, N_pares)

    return {
        'ts_samples': ts_samples,
        'ti_samples': ti_samples,
        'residuo': float(res.fun),
        'DELTA': DELTA,
        'Ts_deg': Ts_deg,
        'Ti_deg': Ti_deg,
        'ts_peak': ts_peak,
        'ti_peak': ti_peak,
    }


def barrido_espectral_azimutal_its(cut, lp, lambdas_s, rango_phi_s, w, L,
                                    ventana=1 * np.pi / 180, puntos=150, N_pares=500,
                                    progress_callback=None, tipo='II',
                                    tope_respaldo=5.0):
    tx_s_all, ty_s_all = [], []
    tx_i_all, ty_i_all = [], []
    theta_s_all, theta_i_all = [], []
    lam_s_all, lam_i_all = [], []

    x0_base = np.array([1.0, 1.0]) * np.pi / 180
    N_phi = len(rango_phi_s)
    x0_phi = x0_base.copy()

    for idx_phi, phi_s in enumerate(rango_phi_s):
        x0_ls = x0_phi.copy()

        for ls in lambdas_s:
            li = 1 / (1/lp - 1/ls)

            resultado = generar_pares_its_pico(cut, lp, ls, phi_s, w, L,
                                               ventana=ventana, puntos=puntos,
                                               N_pares=N_pares, x0=x0_ls, tipo=tipo,
                                               tope_respaldo=tope_respaldo)
            if resultado is None:
                x0_ls = x0_base.copy()
                continue

            ts_samples = resultado['ts_samples']
            ti_samples = resultado['ti_samples']
            x0_ls = np.array([resultado['ts_peak'], resultado['ti_peak']])
            x0_phi = x0_ls.copy()

            phi_i = phi_s + np.pi
            tx_s_all.append(ts_samples * np.cos(phi_s))
            ty_s_all.append(ts_samples * np.sin(phi_s))
            tx_i_all.append(ti_samples * np.cos(phi_i))
            ty_i_all.append(ti_samples * np.sin(phi_i))

            theta_s_all.append(ts_samples)
            theta_i_all.append(ti_samples)
            lam_s_all.append(np.full(len(ts_samples), ls))
            lam_i_all.append(np.full(len(ti_samples), li))

        if progress_callback is not None:
            progress_callback(idx_phi + 1, N_phi,
                              f'phi_s {idx_phi+1}/{N_phi} = {np.degrees(phi_s):+6.1f}°')

    if not tx_s_all:
        return None

    return {
        'tx_s': np.concatenate(tx_s_all),
        'ty_s': np.concatenate(ty_s_all),
        'tx_i': np.concatenate(tx_i_all),
        'ty_i': np.concatenate(ty_i_all),
        'theta_s': np.concatenate(theta_s_all),
        'theta_i': np.concatenate(theta_i_all),
        'lambda_s': np.concatenate(lam_s_all),
        'lambda_i': np.concatenate(lam_i_all),
    }


def onset_colineal(lp, tipo='II', cut_min=15.0, cut_max=70.0):
    ls = 2 * lp
    s_lab = np.array([0.0, 0.0, 1.0])

    def dk(cut):
        s_cri = transform_lab_a_cristal(s_lab, cut * np.pi / 180, 0.0)
        nf_p, _ = get_indices(s_cri, *bboind(lp))
        nf_s, ns_s = get_indices(s_cri, *bboind(ls))
        _, ns_i = get_indices(s_cri, *bboind(ls))
        n_s = ns_s if tipo == 'I' else nf_s
        return nf_p / lp - n_s / ls - ns_i / ls

    if dk(cut_min) * dk(cut_max) > 0:
        return None
    return brentq(dk, cut_min, cut_max, xtol=1e-10)


def residuo_phase_matching(cut, lp, ls, phi_s, w, L, tipo='II'):
    bnds = ((1e-9, 0.5), (1e-9, 0.5))
    x0 = np.array([1.0, 1.0]) * np.pi / 180
    res = minimize(phasematch_NIST_T2_robusto, x0,
                   args=(cut, lp, ls, phi_s, w, L, 0.0, tipo),
                   method='L-BFGS-B', bounds=bnds)
    if res.fun > 1e-6:
        res2 = minimize(phasematch_NIST_T2_robusto, np.array([0.06, 0.06]),
                        args=(cut, lp, ls, phi_s, w, L, 0.0, tipo),
                        method='L-BFGS-B', bounds=bnds)
        return min(res.fun, res2.fun)
    return res.fun


def its_2d(DELTA, Ts_deg, Ti_deg, N_pares, seed=None):

    pdf = DELTA.ravel() / np.sum(DELTA)

    cdf = np.cumsum(pdf)

    if seed is not None:
        np.random.seed(seed)
    u = np.random.uniform(0, 1, N_pares)
    indices = np.searchsorted(cdf, u)


    filas, columnas = np.unravel_index(indices, DELTA.shape)
    ts_samples = Ts_deg[filas]
    ti_samples = Ti_deg[columnas]

    return ts_samples, ti_samples, filas, columnas


def samplear_y_refractar(DELTA, Ts_deg, Ti_deg, N_pares, cut, lp, ls, phi_s, tipo='II'):
    ts_int, ti_int, _, _ = its_2d(DELTA, Ts_deg, Ti_deg, N_pares)
    ts_ext = np.zeros_like(ts_int)
    ti_ext = np.zeros_like(ti_int)
    for k in range(N_pares):
        ts_o, ti_o = calcular_thetas_fuera(
            ts_int[k] * np.pi / 180, ti_int[k] * np.pi / 180,
            cut, lp, ls, phi_s, tipo)
        ts_ext[k] = ts_o * 180 / np.pi
        ti_ext[k] = ti_o * 180 / np.pi
    return ts_ext, ti_ext


def calcular_anillos(cut, lp, ls, w, L, pasos=200, incl=0.0, tipo='II'):
    rango_phis = np.linspace(0, 2*np.pi, pasos)

    ths_polar = np.full(pasos, np.nan)
    thi_polar = np.full(pasos, np.nan)

    x0 = np.array([1.0, 1.0]) * np.pi / 180
    bnds = ((1e-9, 0.5), (1e-9, 0.5))

    for j, phi_s_val in enumerate(rango_phis):
        res = minimize(phasematch_NIST_T2_robusto, x0,
                       args=(cut, lp, ls, phi_s_val, w, L, incl, tipo),
                       method='L-BFGS-B', bounds=bnds)

        if res.fun < 1e-6:
            ths_polar[j] = res.x[0] * 180 / np.pi
            thi_polar[j] = res.x[1] * 180 / np.pi
            x0 = res.x
        else:
            x0 = np.array([1.0, 1.0]) * np.pi / 180

    return rango_phis, ths_polar, thi_polar

def calcular_anillos_fuera(cut, lp, ls, w, L, pasos=200, tipo='II'):
    rango_phis, ths_polar, thi_polar = calcular_anillos(cut, lp, ls, w, L, pasos, tipo=tipo)

    ths_fuera = np.full(len(ths_polar), np.nan)
    thi_fuera = np.full(len(thi_polar), np.nan)
    for j in range(len(rango_phis)):
        if not np.isnan(ths_polar[j]):
            ts_out, ti_out = calcular_thetas_fuera(
                ths_polar[j] * np.pi / 180, thi_polar[j] * np.pi / 180,
                cut, lp, ls, rango_phis[j], tipo)
            ths_fuera[j] = ts_out * 180 / np.pi
            thi_fuera[j] = ti_out * 180 / np.pi
    return rango_phis, ths_fuera, thi_fuera

def calcular_anillos_camara(cut, lp, ls, w, L, delta_theta, incl=0.0, pasos=200, tipo='II'):
    if incl == 0.0:
        rango_phis, ths_fuera, thi_fuera = calcular_anillos_fuera(cut, lp, ls, w, L,
                                                                  pasos, tipo=tipo)
        xs, ys = proyectar_a_camara(np.deg2rad(ths_fuera), rango_phis, delta_theta)
        xi, yi = proyectar_a_camara(np.deg2rad(thi_fuera), rango_phis + np.pi, delta_theta)
        return xs, ys, xi, yi

    incl_int = incl_externa_a_interna(incl, lp, cut)

    rango_phis, ths_int, thi_int = calcular_anillos(cut, lp, ls, w, L, pasos, incl=incl,
                                                    tipo=tipo)

    desfase_eje = np.deg2rad(incl) - incl_int
    x_eje = -desfase_eje / delta_theta
    y_eje = 0.0

    xs = np.full(pasos, np.nan); ys = np.full(pasos, np.nan)
    xi = np.full(pasos, np.nan); yi = np.full(pasos, np.nan)
    for j in range(pasos):
        if np.isnan(ths_int[j]) or np.isnan(thi_int[j]):
            continue
        phi_s = rango_phis[j]
        phi_i = phi_s + np.pi
        ts = np.deg2rad(ths_int[j])
        ti = np.deg2rad(thi_int[j])
        n_s, n_i = indices_signal_idler(ts, ti, cut, lp, ls, phi_s, incl=incl, tipo=tipo)
        txs, tys = refractar_salida_dir(ts, phi_s, n_s, incl_int)
        txi, tyi = refractar_salida_dir(ti, phi_i, n_i, incl_int)
        xs[j] = txs / delta_theta - x_eje; ys[j] = tys / delta_theta - y_eje
        xi[j] = txi / delta_theta - x_eje; yi[j] = tyi / delta_theta - y_eje
    return xs, ys, xi, yi
