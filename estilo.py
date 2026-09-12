"""Estilo unificado de las figuras de la app.

Fuente única de verdad: `aplicar_estilo()` (rcParams) más las constantes de
color, colormap y tamaño de figura. Se importa igual desde cualquier carpeta
porque la raíz del repo está en `sys.path`::

    from estilo import aplicar_estilo, SIGNAL_COLOR, FIGSIZE_TESIS_1COL
    aplicar_estilo()

En scripts con celdas `#%%` conviene llamarlo también al inicio de cada celda
que grafica: es idempotente y así re-correr una celda suelta no depende del
orden de ejecución.

Rojo para el signal y azul para el idler en todo el catálogo.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


SIGNAL_COLOR = '#d62728'
IDLER_COLOR = '#1f77b4'
PUMP_COLOR = 'black'

ORDINARIO_COLOR = '#2ca02c'
EXTRAORDINARIO_COLOR = '#9467bd'

VIOLETA_OSCURO = '#4b2e83'
VIOLETA_CLARO = '#a78bda'

TURQUESA_OSCURO = '#008080'
TURQUESA_CLARO = '#8fd0d0'


CMAP_DENSIDAD = 'magma'
CMAP_ESPECTRAL = 'cividis'
CMAP_ESPECTRAL_GRIS = LinearSegmentedColormap.from_list(
    'Greys_clip', plt.get_cmap('Greys')(np.linspace(0.30, 0.90, 256)))


def _extremo_a_blanco(base, frac, n=256):
    """`base` con su extremo claro llevado a BLANCO puro.

    Mezcla hacia el blanco sólo los primeros `frac` del rango. La fracción es un
    compromiso: magma_r arranca en amarillo pálido y un panel entero de ese
    amarillo pesa demasiado, pero mezclando de más se destiñen los valores bajos,
    que en un mapa de phase matching son justo los lóbulos laterales que hay que
    ver. Con 0.03 el cero sale blanco y los lóbulos conservan el color.
    """
    colores = plt.get_cmap(base)(np.linspace(0, 1, n))
    k = max(int(frac * n), 1)
    peso = np.linspace(1.0, 0.0, k)[:, None]
    colores[:k, :3] = peso + (1 - peso) * colores[:k, :3]
    return LinearSegmentedColormap.from_list(base + '_blanco', colores)


CMAP_DENSIDAD_CLARO = _extremo_a_blanco('magma_r', 0.03)
CMAP_DENSIDAD_PUNTOS = LinearSegmentedColormap.from_list(
    'magma_r_clip', plt.get_cmap('magma_r')(np.linspace(0.20, 1.00, 256)))


ANCHO_TEXTO = 6.30
ESCALA_TESIS = 0.70


def figsize_tesis(ancho, alto, frac=1.0):
    """(ancho, alto) llevados al tamaño de letra canónico del capítulo.

    Se le pasan las proporciones que uno quiere (los mismos numeros de siempre) y
    devuelve esa MISMA forma al ancho que corresponde para incluirla a `frac` veces
    el ancho del texto. Escala ancho y alto por igual, así que el layout no cambia:
    lo único que se mueve es el tamaño relativo del texto, que es el punto.
    """
    ancho_final = frac * ANCHO_TEXTO / ESCALA_TESIS
    return (ancho_final, alto * ancho_final / ancho)


FIGSIZE_TESIS_1COL = (6.0, 4.0)
FIGSIZE_TESIS_2COL = (12.0, 4.5)
FIGSIZE_CUADRADO = (6.0, 6.0)
FIGSIZE_POSTER = (8.0, 6.0)


__all__ = [
    'aplicar_estilo',
    'SIGNAL_COLOR', 'IDLER_COLOR', 'PUMP_COLOR',
    'ORDINARIO_COLOR', 'EXTRAORDINARIO_COLOR',
    'VIOLETA_OSCURO', 'VIOLETA_CLARO',
    'TURQUESA_OSCURO', 'TURQUESA_CLARO',
    'CMAP_DENSIDAD', 'CMAP_ESPECTRAL', 'CMAP_ESPECTRAL_GRIS',
    'CMAP_DENSIDAD_CLARO', 'CMAP_DENSIDAD_PUNTOS',
    'ANCHO_TEXTO', 'ESCALA_TESIS', 'figsize_tesis',
    'FIGSIZE_TESIS_1COL', 'FIGSIZE_TESIS_2COL',
    'FIGSIZE_CUADRADO', 'FIGSIZE_POSTER',
]


def aplicar_estilo():
    """Configura matplotlib con el estilo común. Idempotente.

    Decisión: sans-serif tanto para la tesis como para el póster, aunque el
    cuerpo de la tesis sea Computer Modern serif. Para matchear el cuerpo,
    cambiar `font.family` a 'serif' y `mathtext.fontset` a 'cm'.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',

        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',
        'axes.axisbelow': True,

        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 2.4,
        'ytick.major.size': 2.4,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.minor.size': 1.4,
        'ytick.minor.size': 1.4,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,

        'figure.dpi': 110,
        'figure.facecolor': 'white',
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',

        'grid.linestyle': '-',
        'grid.color': '#b0b0b0',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.5,

        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',
        'legend.fancybox': False,
    })
