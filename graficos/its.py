"""Plots de Inverse Transform Sampling para el catalogo de la app.

Cada funcion replica una celda de los scripts de exploracion:
  plot_its_pares_2d_single, plot_its_angular_1d, plot_its_fotones_cartesianas
      <- exploracion/phasematching_2d_its.py
  plot_its_barrido_espectral
      <- exploracion/phasematching_its_espectral.py
"""
import os as _os, sys as _sys
_APP = _os.path.join(_os.getcwd(), 'app')
for _p in (_APP, _os.getcwd()):
    if _os.path.isdir(_os.path.join(_p, 'graficos')) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np
import matplotlib.pyplot as plt

from fisica import calcular_Phi_NIST, its_2d
from estilo import (SIGNAL_COLOR, IDLER_COLOR,
                    CMAP_DENSIDAD_CLARO, CMAP_DENSIDAD_PUNTOS)


def _mapa_2d(cut, lp, ls, phi_s, w, L, puntos, theta_min_deg, theta_max_deg, tipo='II'):
    """Computa el mapa 2D de phase matching en la grilla fija
    [theta_min_deg, theta_max_deg] (en grados) con puntos x puntos."""
    puntos = int(puntos)
    theta_s = np.linspace(theta_min_deg * np.pi / 180,
                          theta_max_deg * np.pi / 180, puntos)
    theta_i = np.linspace(theta_min_deg * np.pi / 180,
                          theta_max_deg * np.pi / 180, puntos)
    DELTA = np.zeros((puntos, puntos))
    for q in range(puntos):
        for j in range(puntos):
            DELTA[q, j] = calcular_Phi_NIST(theta_s[q], theta_i[j], cut, lp, ls, phi_s,
                                                w, L, tipo)
    Ts_deg = theta_s * 180 / np.pi
    Ti_deg = theta_i * 180 / np.pi
    return DELTA, Ts_deg, Ti_deg


def plot_its_pares_2d_single(cut, lp, ls, phi_s, w, L,
                              puntos=300, N_pares=50000,
                              theta_min_deg=0.0, theta_max_deg=2.0, tipo='II'):
    """Mapa analítico de phase matching vs histograma 2D de pares ITS
    para una configuración (ls, phi_s)."""
    N_pares = int(N_pares)
    DELTA, Ts_deg, Ti_deg = _mapa_2d(cut, lp, ls, phi_s, w, L,
                                       puntos, theta_min_deg, theta_max_deg, tipo)
    ts_samples, ti_samples, _, _ = its_2d(DELTA, Ts_deg, Ti_deg, N_pares)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    imagen = ax1.imshow(DELTA, extent=[Ti_deg[0], Ti_deg[-1], Ts_deg[0], Ts_deg[-1]],
                        origin='lower', aspect='auto', cmap=CMAP_DENSIDAD_CLARO)
    fig.colorbar(imagen, ax=ax1, label='Intensidad')
    ax1.set_title('Phase matching (analítico)', fontsize=14)
    ax1.set_xlabel('$\\theta_{idler}$ (grados)')
    ax1.set_ylabel('$\\theta_{signal}$ (grados)')
    ax1.set_xlim(Ti_deg[0], Ti_deg[-1])
    ax1.set_ylim(Ts_deg[0], Ts_deg[-1])

    hist, _, _ = np.histogram2d(ti_samples, ts_samples, bins=int(puntos),
                                 range=[[Ti_deg[0], Ti_deg[-1]],
                                        [Ts_deg[0], Ts_deg[-1]]])
    sim = ax2.imshow(hist.T, extent=[Ti_deg[0], Ti_deg[-1], Ts_deg[0], Ts_deg[-1]],
                     origin='lower', aspect='auto', cmap=CMAP_DENSIDAD_CLARO)
    fig.colorbar(sim, ax=ax2, label='Cuentas')
    ax2.set_title(f'Simulación ({N_pares} pares)', fontsize=14)
    ax2.set_xlabel('$\\theta_{idler}$ (grados)')
    ax2.set_ylabel('$\\theta_{signal}$ (grados)')
    ax2.set_xlim(Ti_deg[0], Ti_deg[-1])
    ax2.set_ylim(Ts_deg[0], Ts_deg[-1])

    plt.tight_layout()
    return fig


def plot_its_angular_1d(cut, lp, ls, phi_s, w, L,
                         puntos=300, N_pares=50000,
                         theta_min_deg=0.0, theta_max_deg=2.0, tipo='II'):
    """Histograma 1D de theta_s y -theta_i."""
    N_pares = int(N_pares)
    DELTA, Ts_deg, Ti_deg = _mapa_2d(cut, lp, ls, phi_s, w, L,
                                       puntos, theta_min_deg, theta_max_deg, tipo)
    ts_samples, ti_samples, _, _ = its_2d(DELTA, Ts_deg, Ti_deg, N_pares)

    fig, ax = plt.subplots()
    ax.hist(ts_samples, bins=70, alpha=0.7, label='$\\theta_{signal}$', color=SIGNAL_COLOR)
    ax.hist(-ti_samples, bins=70, alpha=0.7, label='$\\theta_{idler}$', color=IDLER_COLOR)
    ax.set_xlabel('$\\theta$ (grados)')
    ax.set_ylabel('Cuentas')
    ax.set_title(f'Distribución angular ({N_pares} pares, $\\phi_s$={np.degrees(phi_s):g}°)')
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_its_fotones_cartesianas(cut, lp, ls, phi_s, w, L,
                                   puntos=300, N_pares=50000,
                                   theta_min_deg=0.0, theta_max_deg=2.0, tipo='II'):
    """Scatter de fotones en (theta_x, theta_y) coloreado por cuentas del
    pixel del que vienen. Signal en phi_s, idler en phi_s + pi."""
    N_pares = int(N_pares)
    DELTA, Ts_deg, Ti_deg = _mapa_2d(cut, lp, ls, phi_s, w, L,
                                       puntos, theta_min_deg, theta_max_deg, tipo)
    ts_samples, ti_samples, filas, columnas = its_2d(DELTA, Ts_deg, Ti_deg, N_pares)

    phi_i = phi_s + np.pi
    tx_signal = ts_samples * np.cos(phi_s)
    ty_signal = ts_samples * np.sin(phi_s)
    tx_idler = ti_samples * np.cos(phi_i)
    ty_idler = ti_samples * np.sin(phi_i)

    cuentas_por_pixel = np.zeros_like(DELTA)
    np.add.at(cuentas_por_pixel, (filas, columnas), 1)
    c_par = cuentas_por_pixel[filas, columnas]

    tx_all = np.concatenate([tx_signal, tx_idler])
    ty_all = np.concatenate([ty_signal, ty_idler])
    c_all = np.concatenate([c_par, c_par])

    fig, ax = plt.subplots()
    sc = ax.scatter(tx_all, ty_all, c=c_all, s=1, cmap=CMAP_DENSIDAD_PUNTOS)
    fig.colorbar(sc, ax=ax, label='Cuentas')
    ax.set_xlabel('$\\theta_x$ (grados)')
    ax.set_ylabel('$\\theta_y$ (grados)')
    ax.set_title(f'Fotones en el detector ({N_pares} pares, $\\phi_s$={np.degrees(phi_s):g}°)')

    if abs(np.sin(phi_s)) < abs(np.cos(phi_s)):
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        ax.set_ylim(-0.5, 0.5)
    else:
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlim(-0.5, 0.5)

    plt.tight_layout()
    return fig


def plot_its_barrido_espectral(cut, lp, phi_s, w, L,
                                 ls_min=790e-9, ls_max=830e-9, N_lambdas=10,
                                 puntos=300, N_pares=1000,
                                 theta_min_deg=0.0, theta_max_deg=2.0,
                                 progress_callback=None, tipo='II'):
    """Histograma 1D de theta_s y -theta_i acumulado sobre un barrido en ls,
    coloreado por longitud de onda. Igual que phasematching_its_espectral.py."""
    N_lambdas = int(N_lambdas)
    N_pares = int(N_pares)
    puntos = int(puntos)

    lambdas_s = np.linspace(ls_min, ls_max, N_lambdas)

    theta_s = np.linspace(theta_min_deg * np.pi / 180,
                          theta_max_deg * np.pi / 180, puntos)
    theta_i = np.linspace(theta_min_deg * np.pi / 180,
                          theta_max_deg * np.pi / 180, puntos)
    Ts_deg = theta_s * 180 / np.pi
    Ti_deg = theta_i * 180 / np.pi

    samples_ts = []
    samples_ti = []
    lambdas_usadas = []

    for k, ls in enumerate(lambdas_s):
        DELTA = np.zeros((puntos, puntos))
        for q in range(puntos):
            for j in range(puntos):
                DELTA[q, j] = calcular_Phi_NIST(theta_s[q], theta_i[j], cut, lp, ls, phi_s,
                                                w, L, tipo)

        if np.sum(DELTA) == 0:
            if progress_callback is not None:
                progress_callback(k + 1, N_lambdas, f'ls = {ls*1e9:.1f} nm (sin PM)')
            continue

        ts_s, ti_s, _, _ = its_2d(DELTA, Ts_deg, Ti_deg, N_pares)
        samples_ts.append(ts_s)
        samples_ti.append(ti_s)
        lambdas_usadas.append(ls)

        if progress_callback is not None:
            progress_callback(k + 1, N_lambdas, f'ls = {ls*1e9:.1f} nm')

    fig, ax = plt.subplots()

    if not lambdas_usadas:
        ax.text(0.5, 0.5, 'Ninguna longitud de onda produjo phase matching',
                ha='center', va='center', fontsize=13, transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    lambdas_idler = [1 / (1/lp - 1/ls) for ls in lambdas_usadas]
    todas_lambdas = np.array(lambdas_usadas + lambdas_idler)
    cmap = plt.cm.turbo
    norm = plt.Normalize(vmin=todas_lambdas.min() * 1e9,
                         vmax=todas_lambdas.max() * 1e9)

    for i, ls in enumerate(lambdas_usadas):
        li = lambdas_idler[i]
        label_s = f'$\\theta_s$ — {ls*1e9:.0f} nm'
        label_i = f'$\\theta_i$ — {li*1e9:.0f} nm'
        ax.hist(samples_ts[i], bins=70, alpha=0.5, label=label_s,
                color=cmap(norm(ls * 1e9)), histtype='stepfilled')
        ax.hist(-samples_ti[i], bins=70, alpha=0.5, label=label_i,
                color=cmap(norm(li * 1e9)), histtype='stepfilled',
                linestyle='--', linewidth=1.5)

    ax.set_xlabel('$\\theta$ (grados)')
    ax.set_ylabel('Cuentas')
    ax.set_title(f'Distribución angular — barrido espectral\n'
                 f'({ls_min*1e9:.0f}–{ls_max*1e9:.0f} nm, {N_lambdas} $\\lambda_s$, '
                 f'{N_pares} pares c/u)')
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    return fig
