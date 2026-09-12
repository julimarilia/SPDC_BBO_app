import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st

_FRONTEND_BG = "#FAFAFA"
plt.rcParams["figure.facecolor"] = _FRONTEND_BG
plt.rcParams["savefig.facecolor"] = _FRONTEND_BG

from graficos.sellmeier import plot_sellmeier, plot_theta_vs_wavelength
from graficos.anillos import plot_anillos
from graficos.parabolas import plot_parabolas_superpuestas
from graficos.its import (
    plot_its_pares_2d_single,
    plot_its_angular_1d,
    plot_its_fotones_cartesianas,
    plot_its_barrido_espectral,
)

PLOT_REGISTRY = {}


def register_plot(name, params, description, param_overrides=None, category="Determinístico",
                  soporta_tipo=False, figsize=None, escala_color=False,
                  opciones=None, params_opcionales=None):
    def decorator(func):
        PLOT_REGISTRY[name] = {
            "func": func,
            "params": params,
            "description": description,
            "param_overrides": param_overrides or {},
            "category": category,
            "figsize": figsize,
            "soporta_tipo": soporta_tipo,
            "escala_color": escala_color,
            "opciones": opciones or {},
            "params_opcionales": params_opcionales or {},
        }
        return func
    return decorator


@register_plot(
    name="Índices de refracción del BBO",
    figsize=(7, 3.0),
    params=["pasos"],
    description=(
        'Índice ordinario (n_o) y extraordinario (n_e) del BBO en función de '
        'la longitud de onda, por las ecuaciones de Sellmeier. Es el punto de '
        'partida: de la diferencia entre esos dos índices sale todo el phase '
        'matching.'),
)
def _sellmeier(pasos):
    return plot_sellmeier(pasos=int(pasos))


@register_plot(
    name="Ángulo de emisión cerca de la degeneración",
    figsize=(7, 3.0),
    params=["cut", "lp", "phi_s", "w", "L", "pasos"],
    description=(
        'A qué ángulo salen el signal y el idler para cada longitud de onda '
        'del signal, en una ventana angosta (780–820 nm) alrededor de la '
        'degeneración.'),
    soporta_tipo=True,
)
def _theta_vs_wavelength(cut, lp, phi_s, w, L, pasos, tipo='II'):
    return plot_theta_vs_wavelength(cut, lp, phi_s, w, L, pasos=int(pasos), tipo=tipo)


@register_plot(
    name="Anillos de emisión",
    figsize=(4.5, 4.5),
    params=["cut", "lp", "ls", "w", "L", "pasos"],
    description=(
        'Los conos de emisión vistos de frente. Se puede mirar el ángulo '
        'dentro del cristal o ya refractado al aire por la ley de Snell —que '
        'es lo que sale al laboratorio— y, en ese caso, proyectarlo sobre un '
        'sensor: en píxeles de una cámara o en milímetros sobre un detector. '
        'La luz llega al sensor con una lente de Fourier (el cristal en el '
        'foco, radio = f·θ) o sin lente, con el sensor a una distancia D '
        '(radio = D·tan θ). En tipo II son dos anillos corridos uno respecto '
        'del otro; en tipo I es un único anillo circular.'),
    opciones={
        "donde": {
            "label": "Ángulos",
            "valores": ["fuera del cristal", "dentro del cristal"],
            "help": "Dentro son los ángulos de emisión en el bulk; fuera, esos "
                    "mismos ángulos ya refractados al aire por Snell.",
        },
        "proyeccion": {
            "label": "Proyección",
            "valores": ["ángulos (grados)", "cámara (píxeles)", "detector (mm)"],
            "solo_si": ("donde", "fuera del cristal"),
            "help": "Sin proyectar se ven los ángulos de emisión. Sobre una cámara, "
                    "el radio del anillo en píxeles. Sobre un detector, ese mismo "
                    "radio en milímetros: sirve para un sensor sin píxeles, o para "
                    "saber de qué tamaño tiene que ser el sensor.",
        },
        "optica": {
            "label": "Cómo llega al sensor",
            "valores": ["con lente de Fourier", "sin lente (a distancia D)"],
            "solo_si": ("proyeccion", ["cámara (píxeles)", "detector (mm)"]),
            "help": "Con el cristal en el foco de una lente el radio sobre el "
                    "sensor es f·θ y no depende de dónde esté el sensor. Sin "
                    "lente, a distancia D, el radio es D·tan(θ) y sí crece con D.",
        },
    },
    params_opcionales={
        "f_lente": {"proyeccion": ["cámara (píxeles)", "detector (mm)"],
                    "optica": "con lente de Fourier"},
        "d_camara": {"proyeccion": ["cámara (píxeles)", "detector (mm)"],
                     "optica": "sin lente (a distancia D)"},
        "pixel_size": {"proyeccion": "cámara (píxeles)"},
    },
    soporta_tipo=True,
)
def _anillos(cut, lp, ls, w, L, pasos, tipo='II',
             donde='fuera del cristal', proyeccion='ángulos (grados)',
             optica='con lente de Fourier',
             f_lente=0.15, d_camara=0.15, pixel_size=15e-6):
    return plot_anillos(cut, lp, ls, w, L, pasos=int(pasos), tipo=tipo,
                        donde=donde, proyeccion=proyeccion, optica=optica,
                        f_lente=f_lente, d_camara=d_camara, pixel_size=pixel_size)


@register_plot(
    name="Curvas de sintonía (las dos ramas)",
    figsize=(7, 3.0),
    params=["cut", "lp", "phi_s", "w", "L", "pasos"],
    description=(
        'Ángulo de emisión contra longitud de onda sobre todo el rango '
        '(745–950 nm) y para las dos ramas del cono, en el corte φ=0. La '
        'forma de parábola es la firma del phase matching.'),
    param_overrides={
        "lp": {"min_value": 380.0, "max_value": 430.0},
        "cut": {"II": {"min_value": 41.0, "max_value": 45.0}},
    },
    soporta_tipo=True,
)
def _parabolas(cut, lp, phi_s, w, L, pasos, tipo='II'):
    return plot_parabolas_superpuestas(cut, lp, phi_s, w, L, pasos=int(pasos), tipo=tipo)


_ITS_SINGLE_OVERRIDES = {
    "puntos": {"value": 100},
    "N_pares": {"max_value": 50000, "value": 10000, "step": 500},
}


@register_plot(
    name="Mapa de phase matching: cálculo vs. simulación",
    escala_color=True,
    figsize=(11, 3.8),
    params=["cut", "lp", "ls", "phi_s", "w", "L",
            "puntos", "N_pares", "theta_min_deg", "theta_max_deg"],
    description=(
        'A la izquierda, el mapa analítico de intensidad en (θs, θi). A la '
        'derecha, el histograma 2D de los pares sorteados de ese mismo mapa '
        'por Inverse Transform Sampling (ITS). Si el muestreo es correcto, '
        'los dos paneles se parecen.'),
    category="Estocástico",
    param_overrides=_ITS_SINGLE_OVERRIDES,
    soporta_tipo=True,
)
def _its_2d(cut, lp, ls, phi_s, w, L, puntos, N_pares,
            theta_min_deg, theta_max_deg, tipo='II'):
    return plot_its_pares_2d_single(cut, lp, ls, phi_s, w, L,
                                     puntos=int(puntos), N_pares=int(N_pares),
                                     theta_min_deg=theta_min_deg,
                                     theta_max_deg=theta_max_deg, tipo=tipo)


@register_plot(
    name="Distribución angular de los pares",
    figsize=(7, 3.2),
    params=["cut", "lp", "ls", "phi_s", "w", "L",
            "puntos", "N_pares", "theta_min_deg", "theta_max_deg"],
    description=(
        'Histograma de θ_signal (hacia la derecha) y −θ_idler (hacia la '
        'izquierda) de los pares sorteados por ITS.'),
    category="Estocástico",
    param_overrides=_ITS_SINGLE_OVERRIDES,
    soporta_tipo=True,
)
def _its_1d(cut, lp, ls, phi_s, w, L, puntos, N_pares,
            theta_min_deg, theta_max_deg, tipo='II'):
    return plot_its_angular_1d(cut, lp, ls, phi_s, w, L,
                                puntos=int(puntos), N_pares=int(N_pares),
                                theta_min_deg=theta_min_deg,
                                theta_max_deg=theta_max_deg, tipo=tipo)


@register_plot(
    name="Fotones simulados sobre el detector",
    escala_color=True,
    figsize=(5.2, 4.2),
    params=["cut", "lp", "ls", "phi_s", "w", "L",
            "puntos", "N_pares", "theta_min_deg", "theta_max_deg"],
    description=(
        'Cada punto es un fotón, ubicado en (θx, θy) y coloreado por las '
        'cuentas del píxel del que salió. El signal sale en φs y su idler en '
        'φs + 180°.'),
    category="Estocástico",
    param_overrides=_ITS_SINGLE_OVERRIDES,
    soporta_tipo=True,
)
def _its_cart(cut, lp, ls, phi_s, w, L, puntos, N_pares,
              theta_min_deg, theta_max_deg, tipo='II'):
    return plot_its_fotones_cartesianas(cut, lp, ls, phi_s, w, L,
                                         puntos=int(puntos), N_pares=int(N_pares),
                                         theta_min_deg=theta_min_deg,
                                         theta_max_deg=theta_max_deg, tipo=tipo)


@register_plot(
    name="Distribución angular por longitud de onda",
    figsize=(8, 3.5),
    params=["cut", "lp", "phi_s", "w", "L",
            "ls_min", "ls_max", "N_lambdas",
            "puntos", "N_pares", "theta_min_deg", "theta_max_deg"],
    description=(
        'El histograma angular acumulado sobre un barrido en la longitud de '
        'onda del signal, con un color por λ. Se ve cómo se abre el cono al '
        'alejarse de la degeneración.'),
    category="Estocástico",
    param_overrides={
        "ls_min": {"value": 790.0},
        "ls_max": {"value": 830.0},
        "N_lambdas": {"value": 10},
        "puntos": {"value": 100},
        "N_pares": {"max_value": 20000, "value": 1000, "step": 100},
    },
    soporta_tipo=True,
)
def _its_espectral(cut, lp, phi_s, w, L, ls_min, ls_max, N_lambdas,
                    puntos, N_pares, theta_min_deg, theta_max_deg, tipo='II'):
    progress_placeholder = st.empty()

    def _cb(idx, total, msg):
        progress_placeholder.progress(idx / total, text=msg)

    fig = plot_its_barrido_espectral(
        cut, lp, phi_s, w, L,
        ls_min=ls_min, ls_max=ls_max, N_lambdas=int(N_lambdas),
        puntos=int(puntos), N_pares=int(N_pares),
        theta_min_deg=theta_min_deg, theta_max_deg=theta_max_deg,
        progress_callback=_cb, tipo=tipo,
    )
    progress_placeholder.empty()
    return fig
