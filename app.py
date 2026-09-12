import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from registro import PLOT_REGISTRY
from fisica import onset_colineal, residuo_phase_matching
from estilo import (SIGNAL_COLOR, IDLER_COLOR,
                    ORDINARIO_COLOR, EXTRAORDINARIO_COLOR)

st.set_page_config(
    page_title="SPDC BBO - Phase Matching",
    page_icon=str(Path(__file__).resolve().parent / "icono.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* OJO: el botón para reabrir el sidebar (stExpandSidebarButton) vive DENTRO
       de stToolbar, no suelto en el header. Ocultar el toolbar entero con
       display:none se lo lleva puesto, y si el sidebar quedaba colapsado no
       había forma de reabrirlo: se veía un gráfico suelto sin ningún control.
       Por eso se ocultan sólo las tres piezas de chrome, nunca el contenedor. */
    [data-testid="stHeader"] { background: transparent; height: 2.5rem; }
    [data-testid="stToolbarActions"],
    [data-testid="stMainMenu"],
    [data-testid="stAppDeployButton"] { display: none !important; }
    /* El botón de reabrir: fijo arriba a la izquierda, con fondo y borde propios
       para que se vea sí o sí contra el blanco de la página. */
    [data-testid="stExpandSidebarButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 0.4rem;
        left: 0.4rem;
        z-index: 1000;
        background: #ffffff !important;
        border: 1px solid #c8c8c8 !important;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
    }
    /* Sidebar ancho: los grupos de parametros entran de a dos por fila en vez
       de apilarse en una columna larguisima. Se fija min-width y no width, para
       que se pueda seguir arrastrando el borde y agrandarlo (el ancho lo pone
       streamlit inline; con width:!important el arrastre deja de funcionar). */
    [data-testid="stSidebar"] { min-width: 33rem !important; }
    [data-testid="stSidebarContent"] { width: 100% !important; }
    /* Con dos columnas los expanders quedan angostos: recorto el padding para
       que el slider tenga ancho util. */
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        padding-left: 0.4rem; padding-right: 0.4rem;
    }
    .block-container { padding-top: 0.2rem; padding-bottom: 0rem; }
    /* Tope de tamaño de la figura. OJO: el testid es stFullScreenFrame; el
       stStyledFullScreenFrame que había antes NO existe en streamlit 1.56, así
       que el tope no se aplicaba nunca y los plots cuadrados salían enormes. */
    [data-testid="stFullScreenFrame"] { display: flex; justify-content: center; }
    [data-testid="stFullScreenFrame"] img,
    [data-testid="stImage"] img {
        max-height: 62vh;
        max-width: 100%;
        width: auto !important;
        height: auto;
        object-fit: contain;
    }
    /* Titulo compacto y subtitulo inline */
    .app-title   { font-size: 1.35rem; margin: 0.1rem 0 0.1rem; font-weight: 600; }
    .app-subtitle { color: #888; font-size: 0.85rem; margin: 0 0 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-title">SPDC en BBO — Simulador de phase matching</div>'
    '<div class="app-subtitle">BBO · Spontaneous Parametric Down-Conversion · tipo I y tipo II</div>',
    unsafe_allow_html=True,
)

PARAM_POR_TIPO = {
    "I": {
        "cut": {"min_value": 28.5, "max_value": 34.0, "value": 30.0, "step": 0.05},
    },
    "II": {},
}

PARAM_DEFS = {
    "cut": {
        "label": "Ángulo de corte",
        "min_value": 38.0, "max_value": 48.0, "value": 43.0, "step": 0.1,
        "to_si": 1.0, "group": "Cristal", "format": "%.1f°",
    },
    "L": {
        "label": "Largo del cristal",
        "min_value": 0.5, "max_value": 10.0, "value": 2.0, "step": 0.1,
        "to_si": 1e-3, "group": "Cristal", "format": "%.1f mm",
    },
    "lp": {
        "label": "Longitud de onda del pump",
        "min_value": 350.0, "max_value": 450.0, "value": 405.0, "step": 1.0,
        "to_si": 1e-9, "group": "Bombeo", "format": "%.0f nm",
    },
    "w": {
        "label": "Cintura del pump",
        "min_value": 10.0, "max_value": 500.0, "value": 90.0, "step": 5.0,
        "to_si": 1e-6, "group": "Bombeo", "format": "%.0f μm",
    },
    "ls": {
        "label": "Longitud de onda del signal",
        "min_value": 700.0, "max_value": 900.0, "value": 810.0, "step": 1.0,
        "to_si": 1e-9, "group": "Signal / Idler", "format": "%.0f nm",
    },
    "phi_s": {
        "label": "Ángulo azimutal φs",
        "min_value": 0.0, "max_value": 360.0, "value": 0.0, "step": 1.0,
        "to_si": np.pi / 180, "group": "Signal / Idler", "format": "%.0f°",
    },
    "f_lente": {
        "label": "Distancia focal de la lente",
        "min_value": 5.0, "max_value": 50.0, "value": 15.0, "step": 0.5,
        "to_si": 1e-2, "group": "Cámara", "format": "%.1f cm",
    },
    "d_camara": {
        "label": "Distancia cristal–cámara",
        "min_value": 1.0, "max_value": 100.0, "value": 15.0, "step": 0.5,
        "to_si": 1e-2, "group": "Cámara", "format": "%.1f cm",
    },
    "pixel_size": {
        "label": "Tamaño de píxel",
        "min_value": 5.0, "max_value": 100.0, "value": 15.0, "step": 1.0,
        "to_si": 1e-6, "group": "Cámara", "format": "%.0f μm",
    },
    "ls_min": {
        "label": "Signal mínimo (barrido)",
        "min_value": 700.0, "max_value": 900.0, "value": 810.0, "step": 1.0,
        "to_si": 1e-9, "group": "Barrido espectral", "format": "%.0f nm",
    },
    "ls_max": {
        "label": "Signal máximo (barrido)",
        "min_value": 700.0, "max_value": 900.0, "value": 820.0, "step": 1.0,
        "to_si": 1e-9, "group": "Barrido espectral", "format": "%.0f nm",
    },
    "N_lambdas": {
        "label": "N de longitudes de onda",
        "min_value": 1, "max_value": 30, "value": 5, "step": 1,
        "to_si": 1, "group": "Barrido espectral", "format": "%d",
    },
    "puntos": {
        "label": "Resolución de la grilla 2D",
        "min_value": 20, "max_value": 400, "value": 200, "step": 10,
        "to_si": 1, "group": "Muestreo ITS", "format": "%d",
    },
    "N_pares": {
        "label": "Pares ITS",
        "min_value": 50, "max_value": 50000, "value": 10000, "step": 500,
        "to_si": 1, "group": "Muestreo ITS", "format": "%d",
    },
    "theta_min_deg": {
        "label": "θ mínimo de la grilla",
        "min_value": 0.0, "max_value": 10.0, "value": 0.0, "step": 0.1,
        "to_si": 1.0, "group": "Muestreo ITS", "format": "%.1f°",
    },
    "theta_max_deg": {
        "label": "θ máximo de la grilla",
        "min_value": 0.5, "max_value": 15.0, "value": 2.0, "step": 0.1,
        "to_si": 1.0, "group": "Muestreo ITS", "format": "%.1f°",
    },
    "pasos": {
        "label": "Pasos de cálculo",
        "min_value": 10, "max_value": 500, "value": 100, "step": 10,
        "to_si": 1, "group": "Cómputo", "format": "%d",
    },
}

GROUP_ORDER = [
    "Cristal", "Bombeo", "Signal / Idler", "Cámara",
    "Barrido espectral", "Muestreo ITS", "Cómputo",
]

DPI_APP = 100


def _extract_and_clear_main_title(fig):
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        text = suptitle.get_text()
        suptitle.set_text("")
        return text

    axes_con_titulo = [ax for ax in fig.get_axes() if ax.get_title()]
    if len(axes_con_titulo) == 1:
        ax = axes_con_titulo[0]
        text = ax.get_title()
        ax.set_title("")
        return text

    return None


MAX_ENTRADAS_LEYENDA = 8


def _leyenda_afuera(fig):
    total = sum(len([e for e in ax.get_legend_handles_labels()[1]
                     if e and not e.startswith('_')])
                for ax in fig.get_axes())
    if total > MAX_ENTRADAS_LEYENDA:
        return []

    entradas, vistas = [], set()
    for ax in fig.get_axes():
        handles, labels = ax.get_legend_handles_labels()
        for h, etiqueta in zip(handles, labels):
            if not etiqueta or etiqueta.startswith("_") or etiqueta in vistas:
                continue
            vistas.add(etiqueta)
            try:
                color = mcolors.to_hex(h.get_color())
            except Exception:
                color = "#444444"
            marcador = getattr(h, "get_marker", lambda: "None")()
            estilo = getattr(h, "get_linestyle", lambda: "-")()
            if marcador not in (None, "None", "", " ") and estilo in ("None", "", " "):
                tipo = "punto"
            elif estilo in ("--", ":", "-."):
                tipo = "guion"
            else:
                tipo = "linea"
            entradas.append((color, tipo, etiqueta))
        leyenda = ax.get_legend()
        if leyenda is not None:
            leyenda.remove()
    return entradas


_TEX = [
    (r'\\theta', 'θ'), (r'\theta', 'θ'),
    (r'\\phi', 'φ'), (r'\phi', 'φ'),
    (r'\\lambda', 'λ'), (r'\lambda', 'λ'),
    (r'\Delta', 'Δ'), (r'\ ', ' '), ('_{signal}', '_s'), ('_{idler}', '_i'),
]
_SUB = {'s': 'ₛ', 'i': 'ᵢ', 'x': 'ₓ', 'y': 'ᵧ', 'p': 'ₚ', '0': '₀',
        '1': '₁', '2': '₂', '+': '₊', '-': '₋'}


def _tex_a_texto(etiqueta):
    t = etiqueta
    for viejo, nuevo in _TEX:
        t = t.replace(viejo, nuevo)
    t = t.replace('$', '')
    fuera, i = [], 0
    while i < len(t):
        if t[i] == '_' and i + 1 < len(t):
            if t[i + 1] == '{':
                fin = t.find('}', i)
                cuerpo = t[i + 2:fin] if fin > 0 else t[i + 2:]
                i = (fin + 1) if fin > 0 else len(t)
            else:
                cuerpo = t[i + 1]
                i += 2
            fuera.append(''.join(_SUB.get(c, c) for c in cuerpo))
        else:
            fuera.append(t[i]); i += 1
    return ''.join(fuera).strip()


_SWATCH = {
    "linea": '<span style="display:inline-block;width:20px;height:0;'
             'border-top:3px solid {c};vertical-align:middle;margin-right:6px"></span>',
    "guion": '<span style="display:inline-block;width:20px;height:0;'
             'border-top:3px dashed {c};vertical-align:middle;margin-right:6px"></span>',
    "punto": '<span style="display:inline-block;width:20px;text-align:center;'
             'color:{c};font-weight:700;vertical-align:middle;margin-right:6px">✚</span>',
}


def _mostrar_leyenda(entradas):
    if not entradas:
        return
    chips = "".join(
        '<span style="display:inline-flex;align-items:center;margin:0 14px 4px 0;'
        'font-size:0.9rem;white-space:nowrap">'
        + _SWATCH[tipo].format(c=color) + _tex_a_texto(etiqueta) + "</span>"
        for color, tipo, etiqueta in entradas
    )
    st.markdown(
        '<div style="margin:0.1rem 0 0.4rem 0;line-height:1.9">' + chips + "</div>",
        unsafe_allow_html=True,
    )


def _aplicar_grilla(fig, prender):
    for ax in fig.get_axes():
        if ax.get_label() == '<colorbar>':
            continue
        ax.grid(prender)


_GAMMAS = {"lineal": 1.0, "medio": 0.5, "alto": 0.3}


def _aplicar_realce(fig, modo):
    gamma = _GAMMAS.get(modo, 1.0)
    for ax in fig.get_axes():
        if ax.get_label() == '<colorbar>':
            continue
        for artista in list(ax.images) + list(ax.collections):
            if artista.get_array() is None:
                continue
            vmin, vmax = artista.get_clim()
            if vmin is None or vmax is None or vmin < 0:
                continue
            artista.set_norm(mcolors.Normalize(vmin, vmax) if gamma == 1.0
                             else mcolors.PowerNorm(gamma, vmin, vmax))


def _unificar_escala(fig_a, fig_b):
    ejes_a = [ax for ax in fig_a.get_axes() if ax.get_label() != '<colorbar>']
    ejes_b = [ax for ax in fig_b.get_axes() if ax.get_label() != '<colorbar>']
    if len(ejes_a) != len(ejes_b):
        return
    for ax_a, ax_b in zip(ejes_a, ejes_b):
        for get_lim, set_lim in (('get_xlim', 'set_xlim'), ('get_ylim', 'set_ylim')):
            la, lb = getattr(ax_a, get_lim)(), getattr(ax_b, get_lim)()
            union = (min(la[0], lb[0]), max(la[1], lb[1]))
            getattr(ax_a, set_lim)(union)
            getattr(ax_b, set_lim)(union)


def _texto_parametros(raw):
    partes = []
    for nombre, valor in raw.items():
        d = PARAM_DEFS.get(nombre, {})
        etiqueta = d.get('label', nombre)
        formato = d.get('format')
        try:
            texto = (formato % valor) if formato else f'{valor:g}'
        except (TypeError, ValueError):
            texto = f'{valor:g}'
        partes.append(f'{etiqueta} {texto}')
    return ' · '.join(partes)


_POR_COLOR = {
    SIGNAL_COLOR.lower(): 'signal',
    IDLER_COLOR.lower(): 'idler',
    ORDINARIO_COLOR.lower(): 'indice ordinario',
    EXTRAORDINARIO_COLOR.lower(): 'indice extraordinario',
    '#000000': 'pump',
}


def _color_de(artista):
    for metodo in ('get_color', 'get_facecolor', 'get_edgecolor'):
        try:
            valor = getattr(artista, metodo)()
        except Exception:
            continue
        try:
            return mcolors.to_hex(valor).lower()
        except (ValueError, TypeError):
            valor = np.asarray(valor)
            if valor.ndim == 2 and len(valor):
                valor = valor[0]
            try:
                return mcolors.to_hex(valor).lower()
            except Exception:
                continue
    return None


def _datos_de_figura(fig):
    filas = []
    ejes = [ax for ax in fig.get_axes() if ax.get_label() != '<colorbar>']

    for i, ax in enumerate(ejes):
        prefijo = f'panel{i + 1}_' if len(ejes) > 1 else ''
        n = 0
        handles, labels = ax.get_legend_handles_labels()
        por_id = {id(h): l for h, l in zip(handles, labels)}

        def nombrar(artistas, bruta, n):
            if not isinstance(artistas, (list, tuple)):
                artistas = [artistas]
            etiqueta = next((por_id[id(a)] for a in artistas if id(a) in por_id), None)
            if not etiqueta and bruta and not bruta.startswith('_'):
                etiqueta = bruta
            if not etiqueta:
                etiqueta = next((_POR_COLOR.get(_color_de(a)) for a in artistas
                                 if _POR_COLOR.get(_color_de(a))), None)
            return prefijo + (_tex_a_texto(etiqueta) if etiqueta else f'serie{n}')

        for linea in ax.get_lines():
            xs = np.atleast_1d(linea.get_xdata())
            ys = np.atleast_1d(linea.get_ydata())
            if not len(xs):
                continue
            n += 1
            nombre = nombrar(linea, linea.get_label(), n)
            for x, y in zip(xs, ys):
                if np.isfinite(x) and np.isfinite(y):
                    filas.append((nombre, float(x), float(y)))

        for col in ax.collections:
            try:
                puntos = np.asarray(col.get_offsets())
            except Exception:
                continue
            if not len(puntos):
                continue
            n += 1
            nombre = nombrar(col, col.get_label(), n)
            for x, y in puntos:
                if np.isfinite(x) and np.isfinite(y):
                    filas.append((nombre, float(x), float(y)))

        barras_vistas = set()
        for cont in getattr(ax, 'containers', []):
            parches = list(getattr(cont, 'patches', []))
            if not parches:
                continue
            n += 1
            nombre = nombrar([cont] + parches[:1],
                             getattr(cont, 'get_label', lambda: '')(), n)
            for parche in parches:
                barras_vistas.add(id(parche))
                try:
                    x = parche.get_x() + parche.get_width() / 2
                    y = parche.get_height()
                except AttributeError:
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    filas.append((nombre, float(x), float(y)))

        for parche in ax.patches:
            if id(parche) in barras_vistas:
                continue
            if hasattr(parche, 'get_height') and hasattr(parche, 'get_x'):
                n += 1
                nombre = nombrar(parche, parche.get_label(), n)
                x = parche.get_x() + parche.get_width() / 2
                y = parche.get_height()
                if np.isfinite(x) and np.isfinite(y):
                    filas.append((nombre, float(x), float(y)))
            elif hasattr(parche, 'get_xy'):
                vertices = np.asarray(parche.get_xy())
                if vertices.ndim != 2 or not len(vertices):
                    continue
                n += 1
                nombre = nombrar(parche, parche.get_label(), n)
                for x, y in vertices:
                    if np.isfinite(x) and np.isfinite(y):
                        filas.append((nombre, float(x), float(y)))

        for imagen in ax.images:
            datos = np.asarray(imagen.get_array())
            if datos.ndim != 2:
                continue
            n += 1
            nombre = nombrar(imagen, ax.get_title() or imagen.get_label(), n)
            x0, x1, y0, y1 = imagen.get_extent()
            ny, nx = datos.shape
            xc = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
            yc = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
            for iy in range(ny):
                for ix in range(nx):
                    v = datos[iy, ix]
                    if np.isfinite(v):
                        filas.append((nombre, float(xc[ix]), float(yc[iy]), float(v)))

    filas.sort(key=lambda f: (f[0], f[1], f[2]))
    return filas


def _csv_de_figura(fig, titulo, texto_params, ejes):
    filas = _datos_de_figura(fig)
    if not filas:
        return None
    salida = io.StringIO()
    salida.write(f'# {titulo}\n')
    salida.write(f'# parametros: {texto_params}\n')
    salida.write(f'# ejes: x = {ejes[0]} ; y = {ejes[1]}\n')
    hay_valor = any(len(f) == 4 for f in filas)
    escritor = csv.writer(salida)
    escritor.writerow(['serie', 'x', 'y'] + (['valor'] if hay_valor else []))
    for fila in filas:
        serie, x, y = fila[0], fila[1], fila[2]
        celdas = [serie, repr(x), repr(y)]
        if hay_valor:
            celdas.append(repr(fila[3]) if len(fila) == 4 else '')
        escritor.writerow(celdas)
    return salida.getvalue()


def _nombres_de_ejes(fig):
    for ax in fig.get_axes():
        if ax.get_label() != '<colorbar>':
            return (_tex_a_texto(ax.get_xlabel()) or 'x',
                    _tex_a_texto(ax.get_ylabel()) or 'y')
    return ('x', 'y')


def _figura_a_bytes(fig, formato):
    buf = io.BytesIO()
    fig.savefig(buf, format=formato, dpi=200, bbox_inches='tight')
    return buf.getvalue()


def _coincide(valor, esperado):
    if isinstance(esperado, (list, tuple)):
        return valor in esperado
    return valor == esperado


_categorias = {}
for _name, _info in PLOT_REGISTRY.items():
    _cat = _info.get("category", "Determinístico")
    _categorias.setdefault(_cat, []).append(_name)

categoria = st.sidebar.radio("Tipo de análisis", list(_categorias.keys()))
st.sidebar.divider()
selected_plot = st.sidebar.selectbox("Gráfico", _categorias[categoria])
plot_info = PLOT_REGISTRY[selected_plot]

if plot_info.get("soporta_tipo"):
    tipo_cristal = st.sidebar.radio(
        "Tipo de cristal", ["II", "I"],
        format_func=lambda t: f"Tipo {t}",
        help="Tipo II: signal y idler con polarizaciones ortogonales, dos anillos "
             "corridos. Tipo I: los dos ordinarios, un unico anillo circular.",
    )
else:
    tipo_cristal = "II"

opciones_elegidas = {}
opciones_visibles = {}
for _clave, _spec in plot_info.get("opciones", {}).items():
    _cond = _spec.get("solo_si")
    if _cond and not _coincide(opciones_elegidas.get(_cond[0]), _cond[1]):
        opciones_elegidas[_clave] = _spec["valores"][0]
        continue
    opciones_elegidas[_clave] = opciones_visibles[_clave] = st.sidebar.radio(
        _spec["label"], _spec["valores"], help=_spec.get("help"),
        key=f"opcion_{selected_plot}_{_clave}")

params_activos = list(plot_info["params"])
for _p, _cond in plot_info.get("params_opcionales", {}).items():
    if all(_coincide(opciones_elegidas.get(k), v) for k, v in _cond.items()):
        params_activos.append(_p)

with st.sidebar.form("params"):
    st.subheader("Parámetros")
    overrides = plot_info.get("param_overrides", {})
    raw_params = {}

    params_por_grupo = {}
    for p in params_activos:
        grupo = PARAM_DEFS[p].get("group", "Otros")
        params_por_grupo.setdefault(grupo, []).append(p)

    grupos = [g for g in GROUP_ORDER if g in params_por_grupo]
    for i in range(0, len(grupos), 2):
        for columna, grupo in zip(st.columns(2), grupos[i:i + 2]):
            with columna, st.expander(grupo, expanded=True):
                for p in params_por_grupo[grupo]:
                    ov = overrides.get(p, {})
                    if ov and set(ov) <= {"I", "II"}:
                        ov = ov.get(tipo_cristal, {})
                    d = {**PARAM_DEFS[p],
                         **PARAM_POR_TIPO.get(tipo_cristal, {}).get(p, {}),
                         **ov}
                    raw_params[p] = st.slider(
                        d["label"],
                        min_value=d["min_value"],
                        max_value=d["max_value"],
                        value=d["value"],
                        step=d["step"],
                        format=d.get("format"),
                    )

    submitted = st.form_submit_button("Calcular")

with st.sidebar.expander("Presentación", expanded=False):
    mostrar_grilla = st.checkbox("Grilla", value=True)
    comparar = st.checkbox("Mostrar el gráfico anterior abajo", value=False,
                           help="Deja la corrida previa debajo, con los MISMOS "
                                "límites de ejes, para poder compararlas.")
    if plot_info.get("escala_color"):
        realce = st.radio(
            "Escala de color", list(_GAMMAS), index=1, horizontal=True,
            help="Comprime la escala (no los datos) para que se vean los lóbulos "
                 "laterales, que están por debajo del 1% del máximo. En «lineal» "
                 "sólo se ve el lóbulo principal.")
    else:
        realce = "lineal"

if True:
    si_params = {}
    for p, val in raw_params.items():
        si_params[p] = val * PARAM_DEFS[p]["to_si"]

    lp_si = si_params.get("lp")
    ls_si = si_params.get("ls")
    ls_min_si = si_params.get("ls_min")
    ls_max_si = si_params.get("ls_max")
    theta_min_si = si_params.get("theta_min_deg")
    theta_max_si = si_params.get("theta_max_deg")

    error_msg = None
    if lp_si is not None and ls_si is not None and ls_si <= lp_si:
        error_msg = (
            "La longitud de onda del signal debe ser mayor que la del pump "
            f"({raw_params['lp']:.0f} nm) para conservar energia."
        )
    elif ls_min_si is not None and lp_si is not None and ls_min_si <= lp_si:
        error_msg = (
            f"El signal mínimo ({raw_params['ls_min']:.0f} nm) debe ser mayor "
            f"que el pump ({raw_params['lp']:.0f} nm)."
        )
    elif (ls_min_si is not None and ls_max_si is not None
            and ls_max_si < ls_min_si):
        error_msg = "El signal máximo del barrido debe ser mayor o igual al mínimo."
    elif (theta_min_si is not None and theta_max_si is not None
            and theta_max_si <= theta_min_si):
        error_msg = "El θ máximo debe ser mayor que el θ mínimo."

    if error_msg is None and plot_info.get("soporta_tipo") and lp_si is not None:
        cut_val = si_params.get("cut")
        if cut_val is not None:
            ls_chequeo = ls_si if ls_si is not None else 2 * lp_si
            residuo = residuo_phase_matching(
                cut_val, lp_si, ls_chequeo, si_params.get("phi_s", 0.0),
                si_params["w"], si_params["L"], tipo_cristal)
            if residuo > 1e-6:
                onset = onset_colineal(lp_si, tipo_cristal)
                donde = (f" El onset colineal del tipo {tipo_cristal} a "
                         f"{raw_params['lp']:.0f} nm está en {onset:.2f}°; "
                         "por debajo el cono no existe." if onset else "")
                error_msg = (
                    f"No hay phase matching a {cut_val:.2f}° con estos "
                    f"parámetros (tipo {tipo_cristal}).{donde}"
                )

    if error_msg is not None:
        st.error(error_msg)
    else:
        try:
            if plot_info.get("soporta_tipo"):
                si_params["tipo"] = tipo_cristal
            si_params.update(opciones_elegidas)

            firma = (selected_plot, repr(sorted(si_params.items())))

            actual = st.session_state.get("corrida_actual")
            if actual is None or actual["firma"] != firma:
                with st.spinner("Calculando..."):
                    fig = plot_info["func"](**si_params)
                    titulo = _extract_and_clear_main_title(fig)
                    leyenda = _leyenda_afuera(fig)
                    figsize = plot_info.get("figsize")
                    if figsize:
                        fig.set_size_inches(*figsize)
                    fig.tight_layout()
                texto_params = ' · '.join(list(opciones_visibles.values())
                                          + [_texto_parametros(raw_params)])
                nueva = {"firma": firma, "fig": fig, "titulo": titulo,
                         "leyenda": leyenda, "params": texto_params,
                         "plot": selected_plot}
                previa = st.session_state.get("corrida_previa")
                if previa is not None:
                    plt.close(previa["fig"])
                if actual is not None:
                    st.session_state["corrida_previa"] = actual
                st.session_state["corrida_actual"] = nueva
                actual = nueva

            previa = st.session_state.get("corrida_previa")
            comparando = (comparar and previa is not None
                          and previa["plot"] == actual["plot"])

            _aplicar_grilla(actual["fig"], mostrar_grilla)
            _aplicar_realce(actual["fig"], realce)
            if comparando:
                _aplicar_grilla(previa["fig"], mostrar_grilla)
                _aplicar_realce(previa["fig"], realce)
                _unificar_escala(actual["fig"], previa["fig"])

            if actual["titulo"]:
                st.subheader(actual["titulo"])
            st.pyplot(actual["fig"], width="content", dpi=DPI_APP)
            _mostrar_leyenda(actual["leyenda"])
            st.caption(actual["params"])

            if comparando:
                st.divider()
                st.caption("**Corrida anterior** — mismos límites de ejes")
                st.pyplot(previa["fig"], width="content", dpi=DPI_APP)
                st.caption(previa["params"])

            ejes = _nombres_de_ejes(actual["fig"])
            csv_datos = _csv_de_figura(actual["fig"], actual["titulo"] or selected_plot,
                                       actual["params"], ejes)
            base = "".join(c if c.isalnum() else "_" for c in selected_plot)[:50]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "⬇ Datos (CSV)", csv_datos or "",
                    file_name=f"{base}.csv", mime="text/csv",
                    disabled=csv_datos is None,
                    help="Los datos que se ven en el gráfico, en CSV.",
                    width="stretch")
            with col2:
                st.download_button(
                    "⬇ Figura (PDF)", _figura_a_bytes(actual["fig"], "pdf"),
                    file_name=f"{base}.pdf", mime="application/pdf",
                    help="Vectorial: no pierde calidad al ampliar ni al imprimir.",
                    width="stretch")
            with col3:
                st.download_button(
                    "⬇ Figura (PNG)", _figura_a_bytes(actual["fig"], "png"),
                    file_name=f"{base}.png", mime="image/png", width="stretch")

            with st.expander("ℹ️ Sobre este gráfico", expanded=False):
                st.caption(plot_info["description"])
        except Exception as e:
            st.error(f"No se pudo generar «{selected_plot}»: {e}")
            st.exception(e)
