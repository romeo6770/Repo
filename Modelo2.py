'''
Dashboard de Análisis de Tráfico (HTML + Navegador)
Este script carga datos desde SQLite, genera gráficas con Matplotlib y Seaborn,
y construye un dashboard responsivo en HTML/CSS que se abre en el navegador predeterminado.

Motivo de cambio (macOS):
- Para evitar errores de compatibilidad con Tcl/Tk en macOS.
- No requiere tkinter ni tkhtmlview.

Dependencias (instalar con pip si faltan):
    pip install pandas matplotlib seaborn scikit-learn
'''
import sqlite3
import webbrowser
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Intentar importar scikit-learn para el modelo predictivo
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
except ImportError:
    raise ImportError(
        "scikit-learn no está instalado.\n" \
        "Ejecuta: pip install scikit-learn"
    )

# Configuración inicial de Seaborn
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.8)
FIGSIZE = (4, 3)  # Tamaño de cada figura
IMG_DIR = Path.cwd() / 'images'
IMG_DIR.mkdir(exist_ok=True)

# Función para convertir columnas a tipo numérico
def convert_numeric(df, cols):
    """
    Convierte las columnas de `cols` a numérico (float),
    estableciendo NaN en valores no convertibles.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Cargar tablas desde SQLite
def load_data(db_path='traffic_analysis.db'):
    """
    Conecta a la base SQLite y devuelve DataFrames para:
    audiences, demographics, engagement, pages,
    reports, tech_details, tech_overview, user_acquisition.
    """
    conn = sqlite3.connect(db_path)
    tables = ['audiences','demographics','engagement','pages',
              'reports','tech_details','tech_overview','user_acquisition']
    data = {tbl: pd.read_sql_query(f"SELECT * FROM {tbl}", conn) for tbl in tables}
    conn.close()
    return data

# Guardar figura y devolver (título, archivo)
def save_plot(fig, name, title):
    filepath = IMG_DIR / f"{name}.png"
    fig.savefig(filepath, bbox_inches='tight', dpi=600)
    plt.close(fig)
    return title, filepath.name

# 1. Usuarios Totales vs Nuevos
def plot_audiences_bar(aud_df):
    if aud_df.empty: return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    df = aud_df.copy()
    df = convert_numeric(df, ["Total users","New users"])
    melt = df.melt(id_vars=["Audience name"],
                   value_vars=["Total users","New users"],
                   var_name="Tipo",value_name="Usuarios")
    sns.barplot(x="Audience name", y="Usuarios", hue="Tipo", data=melt, ax=ax)
    ax.set_title("Usuarios Totales vs Nuevos", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    return save_plot(fig,'audiences_bar',"Usuarios Totales vs Nuevos")

# 2. Usuarios Activos por País
def plot_active_by_country(dem_df):
    if dem_df.empty or 'Country' not in dem_df.columns: return None
    df = convert_numeric(dem_df, ["Active users"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x="Country", y="Active users", data=df, ax=ax)
    ax.set_title("Usuarios Activos por País", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    return save_plot(fig,'active_by_country',"Usuarios Activos por País")

# 3. Tendencia de Engagement
def plot_engagement_trend(eng_df):
    if eng_df.empty or 'Nth day' not in eng_df.columns: return None
    df = convert_numeric(eng_df,["Nth day","Average engagement time per active user"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(x="Nth day", y="Average engagement time per active user",
                 data=df, marker='o', ax=ax)
    ax.set_title("Tendencia de Engagement", fontsize=10)
    return save_plot(fig,'engagement_trend',"Tendencia de Engagement")

# 4. Activos por Plataforma
def plot_platform_active(tech_ov):
    if tech_ov.empty or 'Platform' not in tech_ov.columns: return None
    df = convert_numeric(tech_ov,["Active users"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x="Platform", y="Active users", data=df, ax=ax)
    ax.set_title("Activos por Plataforma", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    return save_plot(fig,'platform_active',"Activos por Plataforma")

# 5. Canales de Adquisición (Pie)
def plot_acquisition_pie(ua_df):
    key='First user primary channel group (Default Channel Group)'
    if ua_df.empty or key not in ua_df.columns: return None
    df=ua_df.groupby(key)["Total users"].sum().reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.pie(df['Total users'], labels=df[key], autopct='%1.1f%%', startangle=140,
           textprops={'fontsize':6})
    ax.set_title("Canales de Adquisición", fontsize=10)
    return save_plot(fig,'acquisition_pie',"Canales de Adquisición")

# 6. Activos por Dispositivo
def plot_device_category(tech_ov):
    df=convert_numeric(tech_ov,[c for c in tech_ov.columns if 'Active users' in c])
    col = next((c for c in tech_ov.columns if 'device' in c.lower()),None)
    if df.empty or not col: return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x=col,y="Active users",data=df,ax=ax)
    ax.set_title("Activos por Dispositivo", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    return save_plot(fig,'device_category',"Activos por Dispositivo")

# 7. Activos por Navegador
def plot_browser_active(tech_det):
    if tech_det.empty or 'Browser' not in tech_det.columns: return None
    df=convert_numeric(tech_det,["Active users"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x="Browser",y="Active users",data=df,ax=ax)
    ax.set_title("Activos por Navegador", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    return save_plot(fig,'browser_active',"Activos por Navegador")

# 8. Ratio Engagement por País
def plot_engagement_ratio(dem_df):
    if dem_df.empty or 'Engaged sessions' not in dem_df.columns: return None
    df=dem_df.copy()
    df=convert_numeric(df,["Active users","Engaged sessions"])
    df['Ratio']=df['Engaged sessions']/df['Active users'].replace(0,pd.NA)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x='Country',y='Ratio',data=df,ax=ax)
    ax.set_title("Ratio Engagement por País", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    return save_plot(fig,'engagement_ratio',"Ratio Engagement por País")

# 9. Análisis Predictivo con Datos Suficientes (Demographics)
def plot_predictive_demographics(dem_df):
    """
    Usa RandomForest para predecir 'Active users' según:
    ['New users','Engaged sessions','Engagement rate',
     'Engaged sessions per active user','Average engagement time per active user','Event count'].
    """
    cols=['New users','Engaged sessions','Engagement rate',
          'Engaged sessions per active user','Average engagement time per active user','Event count']
    df=convert_numeric(dem_df,cols+['Active users'])
    df=df.dropna(subset=cols+['Active users'])
    if len(df)<3:
        print("[Predictivo] No hay suficientes datos en demographics.")
        return None
    X=df[cols]
    y=df['Active users']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    print(f"[Predictivo Demographics] MSE: {mse:.2f}")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pd.DataFrame({'Real':y_test.values,'Predicho':y_pred}).plot(ax=ax,marker='o')
    ax.set_title(f"Predictivo Demographics (MSE {mse:.1f})",fontsize=10)
    ax.set_xlabel("Muestra",fontsize=8)
    ax.set_ylabel("Active users",fontsize=8)
    fig.tight_layout()
    return save_plot(fig,'predictive_demographics',f"Predictivo Demographics (MSE {mse:.1f})")

# Bloque principal
def main():
    dfs=load_data()
    aud_df=dfs.get('audiences',pd.DataFrame())
    dem_df=dfs.get('demographics',pd.DataFrame())
    eng_df=dfs.get('engagement',pd.DataFrame())
    tech_ov=dfs.get('tech_overview',pd.DataFrame())
    ua_df=dfs.get('user_acquisition',pd.DataFrame())
    tech_det=dfs.get('tech_details',pd.DataFrame())

    # Generar gráficas estáticas
    static_plots=[
        plot_audiences_bar(aud_df),
        plot_active_by_country(dem_df),
        plot_engagement_trend(eng_df),
        plot_platform_active(tech_ov),
        plot_acquisition_pie(ua_df),
        plot_device_category(tech_ov),
        plot_browser_active(tech_det),
        plot_engagement_ratio(dem_df)
    ]
    static_plots=[p for p in static_plots if p]

    # Generar modelo predictivo con demographics
    pred=plot_predictive_demographics(dem_df)
    if pred:
        static_plots.append(pred)

    # Construir HTML responsivo
    html=[
        '<!DOCTYPE html>','<html><head>','<meta charset="utf-8">',
        '<style>','body{font-family:Arial;margin:20px;}',
        '.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;}',
        '.card{box-shadow:0 2px 5px rgba(0,0,0,0.1);border-radius:8px;}',
        '.card img{width:100%;}',
        '.card h3{margin:0;padding:10px;background:#f0f0f0;}',
        '.footer{margin-top:30px;padding-top:10px;border-top:1px solid #ddd;font-size:12px;color:#666;}',
        '</style>','</head><body>',
        '<h1>Dashboard de Análisis de Tráfico</h1>','<div class="grid">'
    ]
    for title,fname in static_plots:
        html+=[f'<div class="card"><h3>{title}</h3>',f'<img src="images/{fname}"/>','</div>']
    html+=['</div>',
           '<div class="footer">Desarrollado por Romeo de la Garza #2935144</div>',
           '</body></html>']

    dashboard=Path.cwd()/'dashboard.html'
    dashboard.write_text("\n".join(html),encoding='utf-8')
    print(f"Dashboard generado en: {dashboard}")
    webbrowser.open_new_tab(dashboard.as_uri())

if __name__=='__main__':
    main()
