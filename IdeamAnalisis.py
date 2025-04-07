import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, kstest
import plotly.express as px
from dataclasses import dataclass
import re
import folium
from io import BytesIO
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# Variable para definir el percentil de filtrado
PERCENTILE_THRESHOLD = 90  # Modifica este valor según sea necesario

# Funciones necesarias
@st.cache_data
def load_and_clean_data(file):
    chunks = pd.read_csv(file, delimiter="|", names=["Fecha", "Velocidad"], skiprows=1, chunksize=100000)
    df_chunks = []
    for chunk in chunks:
        chunk["Fecha"] = pd.to_datetime(chunk["Fecha"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        chunk["Velocidad"] = chunk["Velocidad"].astype(float).abs()
        clean_chunk = chunk.dropna(subset=["Fecha", "Velocidad"]).query("Velocidad < 50")
        df_chunks.append(clean_chunk)
    df = pd.concat(df_chunks, ignore_index=True)
    return df

def extraer_datos(nombre_archivo):
    patrones = [
        r'UBICACION\((.*?)\).*?LATITUD\(([-\d\.]+)\)LONGITUD\(([-\d\.]+)\)',
        r'_UBICACION_(.*?)_.*?_LATITUD_([-\d\.]+)_LONGITUD_([-\d\.]+)_',
        r'UBICACION\s*([\w\s\-\[\]]+).*?LATITUD_([-\d\.]+)_LONGITUD_([-\d\.]+)',
        r'_UBICACION_([\w\s\-\[\]]+)_LATITUD_([-\d\.]+)_LONGITUD_([-\d\.]+)'
    ]
    for patron in patrones:
        coincidencia = re.search(patron, nombre_archivo)
        if coincidencia:
            ubicacion, latitud, longitud = coincidencia.groups()
            return ubicacion.strip(), latitud.strip(), longitud.strip()
    return None, None, None

@dataclass
class WeibullParams:
    k: float
    c: float

def estimate_weibull_ml(data):
    params = weibull_min.fit(data, floc=0)
    return WeibullParams(params[0], params[2])

def crear_mapa(latitud, longitud, ubicacion):
    mapa = folium.Map(location=[latitud, longitud], zoom_start=12)
    folium.Marker(location=[latitud, longitud], popup=ubicacion, tooltip=ubicacion).add_to(mapa)
    mapa_io = BytesIO()
    mapa.save(mapa_io, close_file=False)
    mapa_io.seek(0)
    return mapa_io

st.title("Análisis de Velocidades del Viento")

# Ruta a la carpeta dentro del repositorio
carpeta_datos = "DatosIdeamProcesados"

# Buscar todos los archivos .DATA en esa carpeta
archivos_data = [f for f in os.listdir(carpeta_datos) if f.lower().endswith(".data")]

ubicaciones = []
for i, nombre_archivo in enumerate(archivos_data, start=1):
    ruta_archivo = os.path.join(carpeta_datos, nombre_archivo)
    ubicacion, latitud, longitud = extraer_datos(nombre_archivo)
    if ubicacion:
        ubicaciones.append((f"Archivo {i} - {ubicacion}", latitud, longitud, ruta_archivo))

if ubicaciones:
    ubicacion_seleccionada = st.selectbox(
        "Selecciona la ubicación para el análisis",
        [u[0] for u in ubicaciones]
    )

    # Obtén los datos del archivo seleccionado
    seleccion = next(u for u in ubicaciones if u[0] == ubicacion_seleccionada)
    ruta_archivo_seleccionado = seleccion[3]
    latitud = float(seleccion[1])
    longitud = float(seleccion[2])
    ubicacion = seleccion[0]

    # Carga y limpia el archivo
    df = load_and_clean_data(ruta_archivo_seleccionado)

    if df.empty:
        st.error("Error: No hay datos válidos después de la limpieza.")
    else:
        stats = df["Velocidad"].describe()

        # Extrae municipio y departamento desde el nombre del archivo
        selected_file_name = os.path.basename(ruta_archivo_seleccionado)
        municipio, departamento = None, None
        patrones_municipio_departamento = [
            r'MUNICIPIO\((.*?)\).*?DEPARTAMENTO\((.*?)\)',
            r'_MUNICIPIO_(.*?)_.*?_DEPARTAMENTO_(.*?)_',
        ]
        for patron in patrones_municipio_departamento:
            coincidencia = re.search(patron, selected_file_name)
            if coincidencia:
                municipio, departamento = coincidencia.groups()
                break



        if municipio and departamento:
            st.markdown(f"**Municipio:** {municipio}")
            st.markdown(f"**Departamento:** {departamento}")
        else:
            st.warning("No se pudo extraer el municipio y el departamento.")
        # Redondear y formatear estadísticas
        # Redondear manualmente count como entero y el resto con dos decimales
        stats_rounded = stats.copy()
        stats_rounded["count"] = int(stats_rounded["count"])
        for col in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            stats_rounded[col] = round(stats_rounded[col], 2)
      
        stats_with_units = stats_rounded.rename({
            "mean": "mean (m/s)",
            "std": "std (m/s)",
            "min": "min (m/s)",
            "25%": "25% (m/s)",
            "50%": "50% (m/s)",
            "75%": "75% (m/s)",
            "max": "max (m/s)"
        })
        stats_df = pd.DataFrame(stats_with_units).reset_index()
        stats_df.columns = ["Estadistica", "Valor"]

        st.table(stats_df.style.set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]},
             {'selector': 'td', 'props': [('text-align', 'center')]}]
        ).set_properties(**{'width': 'auto'}).set_caption("Estadisticas Descriptivas Velocidad del Viento - Cuantitativa Continua"))

        st.markdown("##### Distribución de la Velocidad del Viento")
        df["Fecha2"] = df["Fecha"].dt.date
        df_daily = df.groupby("Fecha2")["Velocidad"].mean().reset_index()
        fig5 = px.scatter(df_daily, x="Fecha2", y="Velocidad", title="Velocidad del Viento Promedio Diario", labels={"Fecha2": "Fecha", "Velocidad": "Velocidad del Viento (m/s)"})
        st.plotly_chart(fig5)

        # Diagramas de caja (sin filtrado por percentil)
        df["Año"] = df["Fecha"].dt.year
        df["Grupo 5 años"] = (df["Año"] // 5) * 5
        fig4 = px.box(df, x="Grupo 5 años", y="Velocidad", title="Diagramas de Caja cada 5 años", labels={"Grupo 5 años": "Periodo (Años)", "Velocidad": "Velocidad del Viento (m/s)"})
        st.plotly_chart(fig4)

        # Filtrado por percentil
        cutoff = np.percentile(df["Velocidad"], PERCENTILE_THRESHOLD)
        df_weibull = df[df["Velocidad"] <= cutoff]

        weibull_ml = estimate_weibull_ml(df_weibull["Velocidad"].to_numpy())

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col3:
            # CDF empírica
            sorted_data = np.sort(df_weibull["Velocidad"])
            empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            # CDF Weibull
            weibull_cdf = weibull_min.cdf(sorted_data, weibull_ml.k, scale=weibull_ml.c)

            # Gráfica
            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(x=sorted_data, y=empirical_cdf, mode="lines", name="CDF Empírica"))
            fig_cdf.add_trace(go.Scatter(x=sorted_data, y=weibull_cdf, mode="lines", name="CDF Weibull", line=dict(dash="dash")))
            fig_cdf.update_layout(title="Comparación de CDF Empírica vs Weibull",
                xaxis_title="Velocidad (m/s)",
                yaxis_title="Probabilidad acumulada",
                template="plotly_white")
            st.plotly_chart(fig_cdf)




        with col4:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=df_weibull["Velocidad"], nbinsx=10, histnorm="probability density", name="Datos", opacity=0.8, marker=dict(line=dict(width=1, color="black"))))
            x = np.linspace(df_weibull["Velocidad"].min(), df_weibull["Velocidad"].max(), 100)
            fig2.add_trace(go.Scatter(x=x, y=weibull_min.pdf(x, weibull_ml.k, scale=weibull_ml.c), mode="lines", name="Weibull ML", line=dict(dash="dash")))
            fig2.add_annotation(x=df_weibull["Velocidad"].max() * 0.7, y=0.8, text=f" k = {weibull_ml.k:.4f} <br> c = {weibull_ml.c:.4f}", showarrow=False, font=dict(size=12, color="black"), bgcolor="white", bordercolor="black", borderwidth=1, borderpad=4, opacity=0.8)
            fig2.update_layout(title="Distribución Weibull", xaxis_title="Velocidad (m/s)", yaxis_title="Densidad", barmode="overlay", template="plotly_white")
            st.plotly_chart(fig2)

        with col1:
            st.markdown("<p style='font-family:Arial, sans-serif; font-size:15px; font-weight:bold; margin-top:30px;'>Ubicación Geográfica</p>", unsafe_allow_html=True)
            selected_location = next(u for u in ubicaciones if u[0] == ubicacion_seleccionada)
            ubicacion = selected_location[0]
            latitud = float(selected_location[1])
            longitud = float(selected_location[2])
            if ubicacion and latitud and longitud:
                mapa_io = crear_mapa(latitud, longitud, ubicacion)
                st.components.v1.html(mapa_io.getvalue().decode(), width=500, height=400)

        with col2:
            df_filtered = df[df["Velocidad"] <= np.percentile(df["Velocidad"], 97)]
            df_filtered["Año"] = df_filtered["Fecha"].dt.year
            df_filtered["Grupo 5 años"] = (df_filtered["Año"] // 5) * 5
            fig4 = px.box(df_filtered, x="Grupo 5 años", y="Velocidad", title="Diagramas de Caja cada 5 años (Percentil 97)", labels={"Grupo 5 años": "Periodo (Años)", "Velocidad": "Velocidad del Viento (m/s)"})
            st.plotly_chart(fig4)

                # Evaluación de la hipótesis nula
        st.write("**Hipótesis Nula (H₀):** Los datos siguen una distribución Weibull.")
        max_diff = np.max(np.abs(empirical_cdf - weibull_cdf))
        st.write(f"**Máxima diferencia (D):** {max_diff:.4f}")
        if max_diff < 0.09:  # Umbral arbitrario para aceptar/rechazar H₀
          st.success("No se rechaza H₀: Los datos siguen una distribución Weibull.")
        else:
          st.error("Se rechaza H₀: Los datos no siguen una distribución Weibull.")
        # Nueva sección 2x3 para mostrar fórmula y resultados de potencia
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("#### Fórmula de Potencia por unidad de área (Weibull)")
            st.latex(r"P = \frac{1}{2} \rho c^3 \Gamma\left(1 + \frac{3}{k}\right)")

        with col6:
            st.markdown("#### Parámetros Weibull y Potencia estimada")
            # Corrección de la densidad del aire por altitud
            from math import exp
            def calcular_densidad_altura(h):
                rho0 = 1.225  # kg/m³ a nivel del mar
                T0 = 288.15  # K
                g = 9.80665  # m/s²
                M = 0.0289644  # kg/mol
                R = 8.3144598  # J/(mol·K)
                return rho0 * (1 - 0.0065 * h / T0) ** ((g * M / (R * 0.0065)) - 1)

            import requests
            def obtener_altura(lat, lon):
                try:
                    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        return data['results'][0]['elevation']
                except:
                    pass
                return 0  # valor por defecto si falla

            altura = obtener_altura(latitud, longitud)
            rho = calcular_densidad_altura(altura)
            from math import gamma as gamma_func
            gamma = gamma_func(1 + 3 / weibull_ml.k)
            potencia = 0.5 * rho * (weibull_ml.c ** 3) * gamma

            st.latex(r"k = {:.2f}".format(weibull_ml.k))
            st.latex(r"c = {:.2f}".format(weibull_ml.c))
            st.latex(r"\Gamma(1 + 3/k) = {:.2f}".format(gamma))
            st.latex(r"\rho = {:.2f} \ kg/m^3".format(rho))
            st.latex(r"P = {:.2f} \ W/m^2".format(potencia))
