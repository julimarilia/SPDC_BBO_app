import os as _os, sys as _sys
_APP = _os.path.join(_os.getcwd(), 'app')
for _p in (_APP, _os.getcwd()):
    if _os.path.isdir(_os.path.join(_p, 'graficos')) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize
from fisica import phasematch_T2_lorentziana, phasematch_NIST_T2_robusto
from estilo import SIGNAL_COLOR, IDLER_COLOR


def _guesses_iniciales(cut, lp, w, L, tipo):
    """Puntos de arranque del minimizador para las dos ramas.

    Tipo II conserva los valores calibrados a mano para cut = 43 grados, así la
    figura sale idéntica. Tipo I vive en otros ángulos (el cono abre mucho más
    rápido con el corte), así que se derivan resolviendo el caso degenerado: como
    el tipo I es azimutalmente simétrico, la rama negativa es el espejo exacto.
    """
    if tipo != 'I':
        return (np.array([3.4, 2.5]) * np.pi / 180,
                np.array([-8.5, -6.3]) * np.pi / 180)

    ls_deg = 2 * lp
    res = minimize(phasematch_NIST_T2_robusto, np.array([0.03, 0.03]),
                   args=(cut, lp, ls_deg, 0.0, w, L, 0.0, 'I'),
                   method='Nelder-Mead', tol=1e-12)
    pico = np.abs(res.x)
    return pico.copy(), -pico.copy()


def calcular_parabolas(cut, lp, phi_s, w, L, pasos=500, tipo='II'):
    """Calcula las dos ramas (positiva y negativa) de los ángulos de emisión
    signal/idler en función de la longitud de onda del signal.
    Retorna (rango_ls, lami, ths1, thi1, ths2, thi2, pmf_max_1, pmf_max_2)."""
    rango_ls = np.linspace(950e-9, 745e-9, pasos)
    lami = 1 / (1/lp - 1/rango_ls)

    ths1 = np.zeros(pasos)
    thi1 = np.zeros(pasos)
    ths2 = np.zeros(pasos)
    thi2 = np.zeros(pasos)
    pmf_max_1 = np.zeros(pasos)
    pmf_max_2 = np.zeros(pasos)

    x0_pos, x0_neg = _guesses_iniciales(cut, lp, w, L, tipo)

    x0 = x0_pos.copy()
    for j, ls in enumerate(rango_ls):
        res = minimize(phasematch_T2_lorentziana, x0,
                       args=(cut, lp, ls, phi_s, w, L, tipo),
                       method='Nelder-Mead', tol=1e-6)

        ths1[j] = res.x[0] * 180 / np.pi
        thi1[j] = res.x[1] * 180 / np.pi
        pmf_max_1[j] = 1.0 - res.fun
        x0 = res.x

    x0 = x0_neg.copy()
    for j, ls in enumerate(rango_ls):
        res = minimize(phasematch_T2_lorentziana, x0,
                       args=(cut, lp, ls, phi_s, w, L, tipo),
                       method='Nelder-Mead', tol=1e-6)

        ths2[j] = res.x[0] * 180 / np.pi
        thi2[j] = res.x[1] * 180 / np.pi
        pmf_max_2[j] = 1.0 - res.fun
        x0 = res.x

    return rango_ls, lami, ths1, thi1, ths2, thi2, pmf_max_1, pmf_max_2


def plot_parabolas_superpuestas(cut, lp, phi_s, w, L, pasos=500, tipo='II'):
    """Parábolas de SPDC Tipo II: theta_s vs λ_s y theta_i vs λ_i, ambas
    ramas superpuestas en el mismo gráfico. Retorna fig.
    Cada fotón se grafica contra su propia λ, así las curvas se cruzan
    en el degenerado (λ_s = λ_i = 810 nm)."""
    rango_ls, lami, ths1, thi1, ths2, thi2, _, _ = calcular_parabolas(cut, lp, phi_s, w, L, pasos)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(rango_ls * 1e9, ths1, '-', color=SIGNAL_COLOR, linewidth=2,
            label=r'$\theta_{signal}$')
    ax.plot(rango_ls * 1e9, ths2, '-', color=SIGNAL_COLOR, linewidth=2,
            label='_nolegend_')
    ax.plot(lami * 1e9, -thi1, '-', color=IDLER_COLOR, linewidth=2,
            label=r'$\theta_{idler}$')
    ax.plot(lami * 1e9, -thi2, '-', color=IDLER_COLOR, linewidth=2,
            label='_nolegend_')

    ax.axvline(810, ls='-', color='gray', alpha=0.7, linewidth=1.2)
    ax.axvline(850, ls='--', color='gray', alpha=0.7, linewidth=1.2)
    ax.axvline(774, ls='--', color='gray', alpha=0.7, linewidth=1.2)
    ax.axhline(0, ls='--', color='gray', alpha=0.4)

    ax.text(810, 11.5, '810 nm', color='gray', ha='right', va='top',
            rotation=90, fontsize=9)
    ax.text(850, 11.5, r'$\lambda_{signal}=850$ nm', color='gray', ha='right', va='top',
            rotation=90, fontsize=9)
    ax.text(774, 11.5, r'$\lambda_{idler}=774$ nm', color='gray', ha='right', va='top',
            rotation=90, fontsize=9)

    ax.set_xlim(min(rango_ls[-1]*1e9, lami[0]*1e9),
                max(rango_ls[0]*1e9, lami[-1]*1e9))
    ax.set_ylim(-12, 12)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlabel(r'Longitud de onda (nm)', fontsize=15)
    ax.set_ylabel(r'$\theta_{signal},\ \theta_{idler}$ (°)', fontsize=15)
    ax.tick_params(axis='both', labelsize=14)
    leg = ax.legend(title=rf'$\varphi_s = {np.degrees(phi_s):g}°$',
                    loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray',
                    fontsize=13)
    leg.get_title().set_fontsize(13)
    ax.grid(True, color='gray', alpha=0.6, linestyle='--', linewidth=1.0)
    ax.set_title(rf'SPDC Tipo II — Cut = {cut}°', pad=15, fontsize=16)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    lp = 405e-9
    cut = 43.0
    phi_s = 0
    w = 90e-6
    L = 0.002

if __name__ == "__main__":
    rango_ls, lami, ths1, thi1, ths2, thi2, pmf_max_1, pmf_max_2 = calcular_parabolas(cut, lp, phi_s, w, L)

if __name__ == "__main__":
    plt.figure(figsize=(8, 4))
    plt.plot(rango_ls * 1e9, pmf_max_1-1, '-b', label='Rama +', linewidth=2)
    plt.plot(rango_ls * 1e9, pmf_max_2-1, '-r', label='Rama -', linewidth=2)

    plt.title(f'Intensidad de Phase-Matching en el óptimo (Cut={cut}°)')
    plt.xlabel(r'Longitud de onda Signal $\lambda_s$ (nm)')
    plt.ylabel(r'Intensidad Máxima $\Phi$')
    plt.xlim(rango_ls[-1] * 1e9, rango_ls[0] * 1e9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    ax1.plot(rango_ls * 1e9, ths1, 'ro', markersize=1, label=r'$\theta_s$')
    ax1.plot(rango_ls * 1e9, ths2, 'ro', markersize=1)
    ax1.set_ylabel(r'$\theta_s$ salida (grados)')
    ax1.set_xlim(rango_ls[-1] * 1e9, rango_ls[0] * 1e9)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(lami * 1e9, -thi1, 'bo', markersize=1, label=r'$\theta_i$')
    ax2.plot(lami * 1e9, -thi2, 'bo', markersize=1)
    ax2.set_xlabel('Longitud de onda SPDC (nm)')
    ax2.set_ylabel(r'$\theta_i$ salida (grados)')
    ax2.set_xlim(lami[0] * 1e9, lami[-1] * 1e9)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fig = plot_parabolas_superpuestas(cut, lp, phi_s, w, L)
    fig.savefig('parabolas.svg', format='svg')
    plt.show()
