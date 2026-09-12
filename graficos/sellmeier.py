import os as _os, sys as _sys
_APP = _os.path.join(_os.getcwd(), 'app')
for _p in (_APP, _os.getcwd()):
    if _os.path.isdir(_os.path.join(_p, 'graficos')) and _p not in _sys.path:
        _sys.path.insert(0, _p)
from fisica import bboind, get_indices, transform_lab_a_cristal, phasematch_NIST_T2, phasematch_NIST_T2_robusto
from estilo import SIGNAL_COLOR, IDLER_COLOR
from estilo import ORDINARIO_COLOR, EXTRAORDINARIO_COLOR
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def plot_sellmeier(pasos=200):
    """n_o y n_e del BBO en función de la longitud de onda. Retorna fig.
    2 D Plot, n x , n y , n z = f (lambda Pump)"""
    ls_1 = np.linspace(400e-9, 1200e-9, pasos)
    n_o, n_e = bboind(ls_1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ls_1 * 1e9, n_o, color=ORDINARIO_COLOR, linewidth=2)
    ax.plot(ls_1 * 1e9, n_e, color=EXTRAORDINARIO_COLOR, linewidth=2)
    ax.set_xlabel('Longitud de onda (nm)')
    ax.set_ylabel('Índice de refracción')
    ax.grid(True, linestyle='--', alpha=0.6)

    x_label = 1000
    idx_label = np.argmin(np.abs(ls_1 * 1e9 - x_label))
    ax.text(x_label, n_o[idx_label] + 0.01, r'$n_x = n_y = n_o$',
            color=ORDINARIO_COLOR, ha='center', va='bottom', fontsize=12)
    ax.text(x_label, n_e[idx_label] + 0.01, r'$n_z = n_e$',
            color=EXTRAORDINARIO_COLOR, ha='center', va='bottom', fontsize=12)

    fig.tight_layout()
    return fig


def plot_theta_vs_wavelength(cut, lp, phi_s, w, L, pasos=50, tipo='II'):
    """Ángulos de emisión signal/idler vs longitud de onda del signal. Retorna fig."""
    pasos_l = pasos
    rango_ls = np.linspace(780e-9, 820e-9, pasos_l)

    ths_wvl = np.zeros(pasos_l)
    thi_wvl = np.zeros(pasos_l)

    x0 = np.array([1.0, 1.0]) * np.pi / 180
    for j, ls_val in enumerate(rango_ls):
        res = minimize(phasematch_NIST_T2_robusto, x0, args=(cut, lp, ls_val, phi_s, w, L, 0.0, tipo), method='Nelder-Mead', tol=1e-6)
        ths_wvl[j] = res.x[0] * 180 / np.pi
        thi_wvl[j] = res.x[1] * 180 / np.pi
        x0 = res.x

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rango_ls * 1e9, ths_wvl, '-o', color=SIGNAL_COLOR,
            markersize=4, linewidth=1.5, label=r'$\theta_{signal}$')
    ax.plot(rango_ls * 1e9, thi_wvl, '-o', color=IDLER_COLOR,
            markersize=4, linewidth=1.5, label=r'$\theta_{idler}$')
    ax.set_xlabel('Longitud de onda (nm)')
    ax.set_ylabel(r'$\theta_{signal}$, $\theta_{idler}$ (°)')
    ax.grid(True, linestyle='--')
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":

    w = 90e-6
    L = 0.002
    cut = 43
    lp = 405e-9
    ls = 810e-9
    phi_s = 0.0

if __name__ == "__main__":
    fig = plot_sellmeier()
    fig.savefig('sellmeier.svg', format='svg')
    plt.show()

if __name__ == "__main__":
    fig = plot_theta_vs_wavelength(cut, lp, phi_s, w, L)
    plt.show()
