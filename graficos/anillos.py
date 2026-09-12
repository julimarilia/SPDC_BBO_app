import os as _os, sys as _sys
_APP = _os.path.join(_os.getcwd(), 'app')
for _p in (_APP, _os.getcwd()):
    if _os.path.isdir(_os.path.join(_p, 'graficos')) and _p not in _sys.path:
        _sys.path.insert(0, _p)
from fisica import (calcular_anillos, calcular_anillos_fuera,
                    calcular_anillos_camara)
from fisica import bboind, get_indices, transform_lab_a_cristal, phasematch_NIST_T2, phasematch_NIST_T2_robusto, calcular_thetas_fuera, proyectar_a_camara, incl_externa_a_interna, indices_signal_idler, refractar_salida_dir
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 110,
})

from estilo import SIGNAL_COLOR, IDLER_COLOR


def plot_anillos_polar(cut, lp, ls, w, L, pasos=200, tipo='II'):
    """Anillos SPDC Tipo II en coordenadas polares. Retorna fig."""
    rango_phis, ths_polar, thi_polar = calcular_anillos(cut, lp, ls, w, L, pasos, tipo=tipo)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
    ax.plot(rango_phis, ths_polar, '-', color=SIGNAL_COLOR, linewidth=2,
            label=r'Signal ($\theta_s$)')
    ax.plot(rango_phis + np.pi, thi_polar, '-', color=IDLER_COLOR, linewidth=2,
            label=r'Idler ($\theta_i$)')

    ax.set_title(rf'Anillo SPDC Tipo II ($\lambda_s={ls*1e9:.0f}$ nm, Cut={cut}°)',
                 va='bottom', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              frameon=True, framealpha=0.9, edgecolor='gray')
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(alpha=0.3, linestyle=':')
    fig.tight_layout()
    return fig


def plot_anillos_cartesianas(cut, lp, ls, w, L, pasos=200, tipo='II'):
    """Anillos SPDC en coordenadas cartesianas (theta_x, theta_y). Retorna fig.

    En tipo I signal e idler son los dos ordinarios, así que en degeneración caen
    EXACTAMENTE sobre el mismo círculo: el idler se dibuja punteado para que se
    vean los dos."""
    rango_phis, ths_polar, thi_polar = calcular_anillos(cut, lp, ls, w, L, pasos,
                                                        tipo=tipo)

    x_signal = ths_polar*np.cos(rango_phis)
    y_signal = ths_polar*np.sin(rango_phis)
    x_idler = thi_polar*np.cos(rango_phis+np.pi)
    y_idler = thi_polar*np.sin(rango_phis+np.pi)

    idx_phi0 = np.nanargmin(np.abs(rango_phis))
    x_signal_phi0 = x_signal[idx_phi0]
    x_idler_phi0 = x_idler[idx_phi0]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_signal, y_signal, '-', color=SIGNAL_COLOR, linewidth=2,
            label=r'Signal ($\theta_s$)')
    _superpuestos = tipo == 'I' and np.nanmax(np.abs(ths_polar - thi_polar)) < 1e-6
    ax.plot(x_idler, y_idler, '--' if _superpuestos else '-', color=IDLER_COLOR,
            linewidth=2,
            label=(r'Idler ($\theta_i$) — sobre el signal' if _superpuestos
                   else r'Idler ($\theta_i$)'))

    ax.set_title(rf'Anillo SPDC Tipo {tipo} ($\lambda_s={ls*1e9:.0f}$ nm, Cut={cut}°)',
                 pad=15)
    ax.set_xlabel(r"$\theta_x$ (grados)")
    ax.set_ylabel(r"$\theta_y$ (grados)")
    ax.axhline(0, color="gray", alpha=0.5, zorder=0,
               label=r"$\phi_i=180°,\ \phi_s=0°$")
    ax.axvline(x_idler_phi0, color=IDLER_COLOR, ls="--", alpha=0.5, zorder=0)
    ax.axvline(x_signal_phi0, color=SIGNAL_COLOR, ls="--", alpha=0.5, zorder=0)
    ax.plot(0, 0, marker='+', markersize=14, markeredgewidth=2,
            linestyle='None', color='black', label='Pump', zorder=5)
    ax.legend(loc='upper right',
              frameon=True, framealpha=0.9, edgecolor='gray')
    ax.set_aspect("equal")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.grid(True, color='gray', alpha=0.8, linestyle='--', linewidth=1.0)
    fig.tight_layout()
    return fig


def plot_anillos(cut, lp, ls, w, L, pasos=200, tipo='II',
                 donde='fuera del cristal', proyeccion='ángulos (grados)',
                 optica='con lente de Fourier', f_lente=0.15, d_camara=0.15,
                 pixel_size=15e-6):
    """Anillos de signal e idler vistos de frente, en todas las configuraciones.

    `donde`      'dentro del cristal' usa los ángulos internos; 'fuera del cristal'
                 los refracta a aire por Snell (lo que sale al laboratorio).
    `proyeccion` qué se dibuja una vez afuera:
                 - 'ángulos (grados)': (theta_x, theta_y), sin sensor de por medio.
                 - 'cámara (píxeles)': el radio sobre el sensor dividido el tamaño
                   de píxel.
                 - 'detector (mm)': el mismo radio, en milímetros. Sirve para un
                   sensor sin píxeles, o para saber qué tamaño de sensor hace falta.
    `optica`     cómo llega la luz al sensor:
                 - 'con lente de Fourier', el cristal en el foco: r = f·theta, y no
                   depende de la distancia a la cámara. APROXIMACION PARAXIAL, la
                   misma de proyectar_a_camara.
                 - 'sin lente (a distancia D)': r = D·tan(theta), exacto para una
                   fuente puntual, y proporcional a D.
                 APROXIMACION en los dos casos: los pares se toman naciendo en un
                 punto, no repartidos a lo largo de los L del cristal, así que el
                 anillo sale sin ese borroneo.

    Dentro del cristal no hay sensor, así que ahí `proyeccion` y `optica` se ignoran.
    """
    dentro = donde.startswith('dentro')
    if dentro:
        rango_phis, ths, thi = calcular_anillos(cut, lp, ls, w, L, pasos, tipo=tipo)
    else:
        rango_phis, ths, thi = calcular_anillos_fuera(cut, lp, ls, w, L, pasos,
                                                      tipo=tipo)

    con_lente = optica.startswith('con lente')
    proyecta = not dentro and not proyeccion.startswith('ángulos')
    if not proyecta:
        rs, ri = ths, thi
        eje_x, eje_y = r'$\theta_x$ (grados)', r'$\theta_y$ (grados)'
    elif proyeccion.startswith('cámara'):
        if con_lente:
            delta_theta = pixel_size / f_lente
            rs = np.deg2rad(ths) / delta_theta
            ri = np.deg2rad(thi) / delta_theta
        else:
            rs = d_camara * np.tan(np.deg2rad(ths)) / pixel_size
            ri = d_camara * np.tan(np.deg2rad(thi)) / pixel_size
        eje_x, eje_y = r'$x$ (píxeles)', r'$y$ (píxeles)'
    else:
        radio_m = ((lambda t: f_lente * np.deg2rad(t)) if con_lente
                   else (lambda t: d_camara * np.tan(np.deg2rad(t))))
        rs, ri = radio_m(ths) * 1e3, radio_m(thi) * 1e3
        eje_x, eje_y = r'$x$ (mm)', r'$y$ (mm)'

    x_signal = rs * np.cos(rango_phis)
    y_signal = rs * np.sin(rango_phis)
    x_idler = ri * np.cos(rango_phis + np.pi)
    y_idler = ri * np.sin(rango_phis + np.pi)

    idx_phi0 = np.nanargmin(np.abs(rango_phis))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_signal, y_signal, '-', color=SIGNAL_COLOR, linewidth=2,
            label=r'Signal ($\theta_s$)')
    _superpuestos = tipo == 'I' and np.nanmax(np.abs(rs - ri)) < 1e-6
    ax.plot(x_idler, y_idler, '--' if _superpuestos else '-', color=IDLER_COLOR,
            linewidth=2,
            label=(r'Idler ($\theta_i$) — sobre el signal' if _superpuestos
                   else r'Idler ($\theta_i$)'))

    ax.axhline(0, color='gray', alpha=0.5, zorder=0,
               label=r'$\phi_i=180°,\ \phi_s=0°$')
    ax.axvline(x_idler[idx_phi0], color=IDLER_COLOR, ls='--', alpha=0.5, zorder=0)
    ax.axvline(x_signal[idx_phi0], color=SIGNAL_COLOR, ls='--', alpha=0.5, zorder=0)
    ax.plot(0, 0, marker='+', markersize=14, markeredgewidth=2,
            linestyle='None', color='black', label='Pump', zorder=5)

    ax.set_title(rf'Anillos SPDC tipo {tipo} — {donde} '
                 rf'($\lambda_s={ls*1e9:.0f}$ nm, corte={cut}°)', pad=15)
    ax.set_xlabel(eje_x)
    ax.set_ylabel(eje_y)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    ax.set_aspect('equal')
    borde = np.nanmax(np.abs(np.concatenate([rs, ri])))
    if np.isfinite(borde) and borde > 0:
        ax.set_xlim(-1.15 * borde, 1.15 * borde)
        ax.set_ylim(-1.15 * borde, 1.15 * borde)
    ax.grid(True, color='gray', alpha=0.5, linestyle='--', linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_anillos_cartesianas_comparacion(cut, lp, ls1, ls2, w, L, pasos=200, tipo='II'):

    rango_phis_1, ths_polar_1, thi_polar_1 = calcular_anillos(cut, lp, ls1, w, L, pasos,
                                                              tipo=tipo)
    rango_phis_2, ths_polar_2, thi_polar_2 = calcular_anillos(cut, lp, ls2, w, L, pasos,
                                                              tipo=tipo)

    x_signal_1 = ths_polar_1 * np.cos(rango_phis_1)
    y_signal_1 = ths_polar_1 * np.sin(rango_phis_1)
    x_idler_1 = thi_polar_1 * np.cos(rango_phis_1 + np.pi)
    y_idler_1 = thi_polar_1 * np.sin(rango_phis_1 + np.pi)

    x_signal_2 = ths_polar_2 * np.cos(rango_phis_2)
    y_signal_2 = ths_polar_2 * np.sin(rango_phis_2)
    x_idler_2 = thi_polar_2 * np.cos(rango_phis_2 + np.pi)
    y_idler_2 = thi_polar_2 * np.sin(rango_phis_2 + np.pi)

    li1 = 1 / abs(1/lp - 1/ls1)
    li2 = 1 / abs(1/lp - 1/ls2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_signal_1, y_signal_1, '-', color=SIGNAL_COLOR, linewidth=2,
            label=rf'Signal $\lambda_s={ls1*1e9:.0f}$ nm')
    ax.plot(x_idler_1, y_idler_1, '-', color=IDLER_COLOR, linewidth=2,
            label=rf'Idler $\lambda_i={li1*1e9:.0f}$ nm')
    ax.plot(x_signal_2, y_signal_2, '--', color=SIGNAL_COLOR, linewidth=2,
            label=rf'Signal $\lambda_s={ls2*1e9:.0f}$ nm')
    ax.plot(x_idler_2, y_idler_2, '--', color=IDLER_COLOR, linewidth=2,
            label=rf'Idler $\lambda_i={li2*1e9:.0f}$ nm')

    ax.axhline(0, color='gray', alpha=0.6, linewidth=1, zorder=1)
    ax.text(-14, 0.3, r'$\varphi=0$', color='gray', fontsize=11,
            ha='left', va='bottom', zorder=2)

    ax.plot(0, 0, marker='+', markersize=14, markeredgewidth=2,
            linestyle='None', color="#800080ff", label='Pump', zorder=5)

    ax.set_title(rf'Anillos SPDC Tipo II — comparación de $\lambda_s$ (Cut={cut}°)',
                 pad=15)
    ax.set_xlabel(r"$\theta_x$ (°)")
    ax.set_ylabel(r"$\theta_y$ (°)")
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    ax.set_aspect("equal")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-10, 10)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, color='gray', alpha=0.8, linestyle='--', linewidth=1.0)
    fig.tight_layout()
    return fig


def plot_anillos_polar_fuera(cut, lp, ls, w, L, pasos=200, tipo='II'):
    """Anillos SPDC Tipo II en polar con los ángulos refractados a aire. Retorna fig."""
    rango_phis, ths_fuera, thi_fuera = calcular_anillos_fuera(cut, lp, ls, w, L, pasos,
                                                              tipo=tipo)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
    ax.plot(rango_phis, ths_fuera, '-', color=SIGNAL_COLOR, linewidth=2,
            label=r'Signal ($\theta_s$)')
    ax.plot(rango_phis + np.pi, thi_fuera, '-', color=IDLER_COLOR, linewidth=2,
            label=r'Idler ($\theta_i$)')

    ax.set_title(rf'Anillo SPDC Tipo {tipo} fuera del cristal '
                 rf'($\lambda_s={ls*1e9:.0f}$ nm, Cut={cut}°)',
                 va='bottom', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              frameon=True, framealpha=0.9, edgecolor='gray')
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(alpha=0.3, linestyle=':')
    fig.tight_layout()
    return fig


def plot_anillos_camara(cut, lp, ls, w, L, delta_theta, pasos=200, tipo='II'):
    """Anillos SPDC proyectados sobre la cámara, en pixeles.

    Usa los ángulos fuera del cristal (refractados por Snell) y los proyecta
    asumiendo el cristal a la distancia focal de la lente.
    delta_theta = tamaño_pixel / distancia_focal (rad/pixel)."""
    rango_phis, ths_fuera, thi_fuera = calcular_anillos_fuera(cut, lp, ls, w, L, pasos,
                                                              tipo=tipo)

    ths_rad = np.deg2rad(ths_fuera)
    thi_rad = np.deg2rad(thi_fuera)

    x_signal, y_signal = proyectar_a_camara(ths_rad, rango_phis, delta_theta)
    x_idler, y_idler = proyectar_a_camara(thi_rad, rango_phis + np.pi, delta_theta)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_signal, y_signal, '-', color=SIGNAL_COLOR, linewidth=2,
            label=r'Signal ($\theta_s$)')
    _superpuestos = tipo == 'I' and np.nanmax(np.abs(ths_fuera - thi_fuera)) < 1e-6
    ax.plot(x_idler, y_idler, '--' if _superpuestos else '-', color=IDLER_COLOR,
            linewidth=2,
            label=(r'Idler ($\theta_i$) — sobre el signal' if _superpuestos
                   else r'Idler ($\theta_i$)'))
    ax.plot(0, 0, marker='+', markersize=14, markeredgewidth=2,
            linestyle='None', color='black', label='Pump', zorder=5)

    ax.set_title(rf'Anillos sobre cámara — tipo {tipo} '
                 rf'($\Delta\theta={delta_theta*1e6:.2f}$ μrad/px, '
                 rf'$\lambda_s={ls*1e9:.0f}$ nm, Cut={cut}°)', pad=15)
    ax.set_xlabel(r'$x$ (pixeles)')
    ax.set_ylabel(r'$y$ (pixeles)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3, linestyle=':')
    fig.tight_layout()
    return fig


def plot_anillos_cartesianas_fuera(cut, lp, ls, w, L, pasos=200, tipo='II'):
    """Anillos SPDC Tipo II en cartesianas (theta_x, theta_y) con los
    ángulos refractados a aire. Retorna fig."""
    rango_phis, ths_fuera, thi_fuera = calcular_anillos_fuera(cut, lp, ls, w, L, pasos,
                                                              tipo=tipo)

    x_signal = ths_fuera * np.cos(rango_phis)
    y_signal = ths_fuera * np.sin(rango_phis)
    x_idler = thi_fuera * np.cos(rango_phis + np.pi)
    y_idler = thi_fuera * np.sin(rango_phis + np.pi)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_signal, y_signal, '-', color=SIGNAL_COLOR, linewidth=2,
            label=r'Signal ($\theta_s$)')
    _superpuestos = tipo == 'I' and np.nanmax(np.abs(ths_fuera - thi_fuera)) < 1e-6
    ax.plot(x_idler, y_idler, '--' if _superpuestos else '-', color=IDLER_COLOR,
            linewidth=2,
            label=(r'Idler ($\theta_i$) — sobre el signal' if _superpuestos
                   else r'Idler ($\theta_i$)'))
    ax.plot(0, 0, marker='+', markersize=14, markeredgewidth=2,
            linestyle='None', color='black', label='Pump', zorder=5)

    ax.set_title(rf'Anillo SPDC Tipo {tipo} fuera del cristal '
                 rf'($\lambda_s={ls*1e9:.0f}$ nm, Cut={cut}°)', pad=15)
    ax.set_xlabel(r"$\theta_x$ (grados)")
    ax.set_ylabel(r"$\theta_y$ (grados)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5, zorder=0)
    ax.axvline(1.7, color="gray", ls="--", alpha=0.5, zorder=0)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linestyle=':')
    fig.tight_layout()
    return fig


if __name__ == "__main__":

    w = 90e-6
    L = 0.002
    cut = 43
    lp = 405e-9
    ls = 850e-9
    phi_s = 0.0

    li = 1/abs(1/lp-1/ls)
    print(li)

if __name__ == "__main__":
    fig = plot_anillos_polar(cut, lp, ls, w, L)
    plt.show()

if __name__ == "__main__":
    fig = plot_anillos_cartesianas(cut, lp, ls, w, L)
    plt.show()

if __name__ == "__main__":
    ls1 = 810e-9
    ls2 = 850e-9
    fig = plot_anillos_cartesianas_comparacion(cut, lp, ls1, ls2, w, L)
    fig.savefig('anillos.svg', format='svg')
    plt.show()

if __name__ == "__main__":
    fig = plot_anillos_polar_fuera(cut, lp, ls, w, L)
    plt.show()

if __name__ == "__main__":
    fig = plot_anillos_cartesianas_fuera(cut, lp, ls, w, L)
    plt.show()

if __name__ == "__main__":
    f_lente = 0.15
    pixel_size = 15e-6
    delta_theta = pixel_size / f_lente
    fig = plot_anillos_camara(cut, lp, ls, w, L, delta_theta)
    plt.show()

if __name__ == "__main__":

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    pasos = 200
    rango_phis = np.linspace(0, 2*np.pi, pasos)
    cuts = np.linspace(41.9, 43, 10)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 8))

    cmap = plt.cm.viridis
    norm = Normalize(vmin=cuts.min(), vmax=cuts.max())

    for idx, i in enumerate(cuts):
        ths_polar = np.zeros(pasos)
        thi_polar = np.zeros(pasos)

        x0 = np.array([1.0, 1.0]) * np.pi / 180
        bnds = ((1e-4, 0.5), (1e-4, 0.5))

        for j, phi_s in enumerate(rango_phis):
            res = minimize(phasematch_NIST_T2_robusto, x0,
                           args=(i, lp, ls, phi_s, w, L, 0.0, 'II'),
                           method='L-BFGS-B', bounds=bnds)

            ths_polar[j] = res.x[0] * 180 / np.pi
            thi_polar[j] = res.x[1] * 180 / np.pi
            x0 = res.x

        color = cmap(norm(i))
        ax.plot(rango_phis, ths_polar, '-', color=color, linewidth=1.5)
        ax.plot(rango_phis + np.pi, thi_polar, '--', color=color, linewidth=1.5)


    ax.set_title(rf'Evolución del Anillo SPDC Tipo II ($\lambda_s={ls*1e9:.0f}$ nm)',
                 va='bottom', pad=15, fontsize=14)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.7)
    cbar.set_label('Ángulo de corte (°)')

    legend_elems = [
        Line2D([0], [0], color='k', linestyle='-', label='Signal'),
        Line2D([0], [0], color='k', linestyle='--', label='Idler'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', bbox_to_anchor=(1.15, 1.1),
              frameon=True, framealpha=0.9, edgecolor='gray')

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.show()
