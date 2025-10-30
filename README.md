# Massive BI · Enterprise Demo v3
# - Mejora visual y navegación
# - NPS con velocímetro + objetivo + opción "Modo Demo" (escenario positivo)
# - Churn con gauge y alerta por umbral
# - Comparativa Objetivos vs Real + Forecast
# - Explicación ejecutiva por módulo
# - Mantiene datasets en /data (sin alterarlos)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import sqrt

# ML / Stats
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_HW = True
except Exception:
    HAS_HW = False

# Viz
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Massive BI · Enterprise v3", layout="wide", initial_sidebar_state="expanded")

# ---------------- THEME ----------------
st.markdown("""
<style>
:root { --brand:#0D6EFD; --ok:#16a34a; --warn:#f59e0b; --bad:#dc2626; --purple:#6a1b9a; --indigo:#3949AB; --teal:#00838F; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #eef1f5; padding:14px 20px; border-radius:8px; }
div[data-testid="stMetric"] > div[data-testid="stMetricLabel"] { font-weight:500; }
div[data-testid="stMetric"] > div[data-testid="stMetricValue"] { font-size:2.1rem; }
div[data-testid="stMetric"] svg { display:none; }
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; flex-direction:column; gap:4px; }
.stTabs [data-baseweb="tab"] > div { font-weight:600; }
.stTabs [data-baseweb="tab--selected"] > div { color:var(--brand); }
span[data-baseweb="tag"] { background-color: var(--brand) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------- HELPERS ----------------
@st.cache_data
def load_data(allow_output_mutation=True):
    DATA_PATH = "data/"
    try:
        customers = pd.read_csv(f"{DATA_PATH}customers.csv", parse_dates=["acquisition_date"])
        products = pd.read_csv(f"{DATA_PATH}products.csv")
        nps = pd.read_csv(f"{DATA_PATH}nps_surveys.csv", parse_dates=["date"])
        tx = pd.read_csv(f"{DATA_PATH}transactions.csv", parse_dates=["date"])
        web = pd.read_csv(f"{DATA_PATH}web_events.csv", parse_dates=["date"])
        campaigns = pd.read_csv(f"{DATA_PATH}campaigns.csv", parse_dates=["start_date", "end_date"])
    except FileNotFoundError:
        st.error(f"Error: No se encontraron los archivos CSV en la carpeta '{DATA_PATH}'. Asegúrate de que la estructura del proyecto es correcta.")
        return None, None, None, None, None, None

    # Merge para enriquecer transacciones
    df = tx.merge(products, on="product_id").merge(customers, on="customer_id")
    return df, customers, products, nps, web, campaigns

def get_kpi_color(val, target, higher_is_better=True):
    if higher_is_better:
        if val >= target: return "ok"
        if val >= target * 0.9: return "warn"
        return "bad"
    else:
        if val <= target: return "ok"
        if val <= target * 1.1: return "warn"
        return "bad"

# Load all data
df, customers, products, nps, web, campaigns = load_data()

if df is not None:
    # ---------------- SIDEBAR / FILTERS ----------------
    with st.sidebar:
        st.title("Massive BI")
        st.subheader("Enterprise Demo v3")
        st.info("Panel de Inteligencia de Negocio con modelos de Machine Learning integrados.")
        
        cities = ["Todas"] + sorted(df["city"].unique().tolist())
        selected_city = st.selectbox("📍 Ciudad", cities)
        
        min_date, max_date = df["date"].min(), df["date"].max()
        sd, ed = st.date_input("🗓️ Periodo", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        st.subheader("Configuración de Modelos")
        demo_mode = st.toggle("Activar Modo Demo", value=False, help="Suma un ajuste al NPS para mostrar un escenario positivo. No altera los datos originales.")
        demo_lift = st.slider("Ajuste NPS (en Modo Demo)", -10, 20, 10) if demo_mode else 0

        st.markdown("---")
        st.write("👨‍💻 **Desarrollado por:** [Tu Nombre/Empresa]")


    # ---------------- FILTERED DATA ----------------
    query_filters = "(date >= @sd) & (date <= @ed)"
    if selected_city != "Todas":
        query_filters += f" & (city == '{selected_city}')"

    df_f = df.query(query_filters)
    nps_f = nps.query("(date >= @sd) & (date <= @ed)")
    if selected_city != "Todas":
        nps_f = nps_f.query(f"location.str.contains('{selected_city}')")


    # ---------------- MAIN CONTENT ----------------
    st.markdown("### 📊 **Top Acciones Recomendadas**")
    st.markdown("Basado en el análisis de datos, estas son las acciones sugeridas con mayor impacto potencial (ROI).")
    
    # Placeholder for actions
    actions = []

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 **Visión General**", "🔮 **Forecast (vs Objetivo)**", "⭐ **NPS & Voz del Cliente**", "⚠️ **Churn Prediction**", "🧠 **Cohortes & CLV**"])

    with tab1:
        st.subheader("KPIs Principales del Negocio")
        st.markdown(f"Análisis para **{selected_city}** desde **{sd.strftime('%d-%b-%Y')}** hasta **{ed.strftime('%d-%b-%Y')}**.")
        
        # KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_revenue = df_f['revenue_usd'].sum()
        total_tx = len(df_f)
        unique_customers = df_f['customer_id'].nunique()
        arpu = total_revenue / unique_customers if unique_customers else 0
        
        kpi1.metric("Ingresos (USD)", f"${total_revenue:,.0f}", delta="vs periodo anterior")
        kpi2.metric("Transacciones", f"{total_tx:,}")
        kpi3.metric("ARPU (Ingreso Promedio/Usuario)", f"${arpu:,.2f}")
        kpi4.metric("Clientes Únicos", f"{unique_customers:,}")
        
        st.markdown("---")
        
        c1, c2 = st.columns((1,1))
        with c1:
            st.subheader("Ventas por Categoría de Producto")
            cat_sales = df_f.groupby("category")["revenue_usd"].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(cat_sales, x="category", y="revenue_usd", title="Ingresos Totales por Categoría", text_auto=".2s", color_discrete_sequence=[px.colors.qualitative.Plotly[0]])
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Top 10 Productos más Vendidos")
            prod_sales = df_f.groupby("product_name")["revenue_usd"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(prod_sales, y="product_name", x="revenue_usd", orientation='h', title="Top 10 Productos por Ingresos", text_auto=".2s", color_discrete_sequence=[px.colors.qualitative.Plotly[1]])
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)


    with tab2:
        st.subheader("Proyección de Ingresos vs. Objetivos")
        st.markdown("Este módulo utiliza modelos de series de tiempo para predecir los ingresos futuros y compararlos con las metas establecidas.")
        
        # Data prep
        sales_monthly = df.set_index("date").resample("MS")["revenue_usd"].sum().reset_index()
        sales_monthly["date_str"] = sales_monthly["date"].dt.strftime("%b %Y")

        # Config
        forecast_steps = st.slider("Meses a proyectar", 1, 12, 6)
        growth_target = st.number_input("Objetivo de Crecimiento Mensual (%)", value=5.0, step=0.5) / 100

        # Model Training
        if HAS_HW and len(sales_monthly) > 12:
            model = ExponentialSmoothing(sales_monthly["revenue_usd"], seasonal="add", seasonal_periods=12).fit()
            forecast = model.forecast(forecast_steps)
            
            future_dates = pd.date_range(start=sales_monthly["date"].max() + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
            forecast_df = pd.DataFrame({"date": future_dates, "revenue_usd": forecast})
            forecast_df["date_str"] = forecast_df["date"].dt.strftime("%b %Y")
            
            # Combine
            full_df = pd.concat([sales_monthly, forecast_df])
            full_df["type"] = ["Real"]*len(sales_monthly) + ["Proyectado"]*len(forecast_df)
            
            # Target line
            last_real_revenue = sales_monthly['revenue_usd'].iloc[-1]
            target_values = [last_real_revenue * (1 + growth_target)**i for i in range(forecast_steps + 1)]
            target_dates = pd.date_range(start=sales_monthly["date"].max(), periods=forecast_steps+1, freq='MS')
            target_df = pd.DataFrame({'date': target_dates, 'target': target_values})

            # Plot
            fig = px.bar(full_df, x="date_str", y="revenue_usd", color="type", title=f"Proyección de Ingresos a {forecast_steps} Meses",
                         labels={"revenue_usd": "Ingresos (USD)", "date_str": "Mes"},
                         color_discrete_map={"Real": "var(--brand)", "Proyectado": "var(--indigo)"})
            fig.add_trace(go.Scatter(x=target_df["date"].dt.strftime("%b %Y"), y=target_df["target"], mode='lines+markers', name='Objetivo', line=dict(color='var(--ok)', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast Action
            if forecast.mean() < target_df['target'].mean() * 0.9:
                 actions.append({"prioridad":2,"acción":"Revisar estrategia comercial/marketing","dominio":"Ventas","razón":"Proyección por debajo del objetivo","ROI_esperado":"Alineación al target","dueño":"Gerencia","cuando":"Próximo comité"})

        else:
            st.warning("No hay suficientes datos (>12 meses) o falta la librería `statsmodels` para realizar una proyección con estacionalidad.")

    with tab3:
        st.subheader("Net Promoter Score (NPS) y Análisis de Comentarios")
        st.markdown("El NPS mide la lealtad del cliente. Un puntaje alto indica una base de clientes sólida y promotora de la marca.")

        nps_target = 45 # Target NPS
        
        # Calculate NPS
        def classify_nps(score):
            if score >= 9: return "Promotor"
            if score >= 7: return "Pasivo"
            return "Detractor"
        nps_f["type"] = nps_f["score"].apply(classify_nps)
        
        if not nps_f.empty:
            promoters = len(nps_f[nps_f["type"] == "Promotor"])
            detractors = len(nps_f[nps_f["type"] == "Detractor"])
            total_responses = len(nps_f)
            
            nps_global = ((promoters - detractors) / total_responses) * 100 if total_responses > 0 else 0
            nps_global += demo_lift

            c1, c2 = st.columns((1,2))
            with c1:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = nps_global,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "NPS Global", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': f'var(--{get_kpi_color(nps_global, nps_target)})'},
                        'steps' : [
                            {'range': [-100, 0], 'color': 'rgba(220, 38, 38, 0.2)'},
                            {'range': [0, 40], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [40, 100], 'color': 'rgba(22, 163, 74, 0.2)'}],
                        'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': nps_target}
                    }))
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                # NPS over time
                nps_monthly = nps_f.set_index("date").groupby([pd.Grouper(freq="MS"), "type"]).size().unstack(fill_value=0)
                nps_monthly["total"] = nps_monthly.sum(axis=1)
                nps_monthly["nps"] = ((nps_monthly["Promotor"] - nps_monthly["Detractor"]) / nps_monthly["total"]) * 100
                nps_monthly["nps"] += demo_lift

                fig = px.line(nps_monthly, y="nps", title="Evolución Mensual del NPS", markers=True)
                fig.add_hline(y=nps_target, line_dash="dash", line_color="red", annotation_text="Objetivo")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Desglose y Voz del Cliente")
            c1,c2 = st.columns(2)
            with c1:
                # Breakdown by category
                nps_cat = nps_f.groupby('category').apply(lambda x: ((len(x[x['type']=='Promotor']) - len(x[x['type']=='Detractor'])) / len(x)) * 100).sort_values()
                st.dataframe(nps_cat.reset_index().rename(columns={0: "NPS"}), use_container_width=True)

            with c2:
                # Word Cloud from comments
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                text = " ".join(comment for comment in nps_f.comment.astype(str) if comment != 'nan')
                if text:
                    wordcloud = WordCloud(background_color="white", colormap="viridis").generate(text)
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("No hay comentarios para generar la nube de palabras.")
        else:
            st.warning("No hay datos de NPS para el periodo o ciudad seleccionada.")
        
        # NPS Action
        if 'nps_global' in locals() and nps_global < 20:
            actions.append({"prioridad":1,"acción":"Plan de choque para Detractores","dominio":"CX","razón":f"NPS {nps_global:,.0f}% < umbral","ROI_esperado":"+10 pts NPS","dueño":"CX","cuando":"Inmediato"})
        elif 'nps_global' in locals() and nps_global < nps_target and not demo_mode:
            actions.append({"prioridad":2,"acción":"Entrenamiento de atención y optimización de turnos","dominio":"CX","razón":f"NPS {nps_global:,.0f}% por debajo del objetivo","ROI_esperado":"+5 pts NPS","dueño":"RRHH/CX","cuando":"Esta semana"})
        else:
            actions.append({"prioridad":3,"acción":"Programa de referidos / reputación","dominio":"Marketing","razón":"NPS saludable","ROI_esperado":"+5–8% captación","dueño":"Marketing","cuando":"Próximos 15 días"})


    with tab4:
        st.subheader("Predicción de Fuga de Clientes (Churn)")
        st.markdown("Utilizamos un modelo de Machine Learning (RFM + Regresión Logística) para identificar clientes con alta probabilidad de abandonar nuestro servicio. Esto permite tomar acciones de retención proactivas.")

        churn_threshold = st.slider("Umbral de Probabilidad de Churn", 0.0, 1.0, 0.65)
        
        # RFM Model
        today = df['date'].max() + timedelta(days=1)
        rfm = df.groupby('customer_id').agg({
            'date': lambda date: (today - date.max()).days,
            'tx_id': 'count',
            'revenue_usd': 'sum'
        }).rename(columns={'date': 'recency', 'tx_id': 'frequency', 'revenue_usd': 'monetary'})

        if not rfm.empty:
            # Simple Churn definition: no purchase in last 90 days
            rfm['churn'] = (rfm['recency'] > 90).astype(int)
            
            # ML Model
            X = rfm[['recency', 'frequency', 'monetary']]
            y = rfm['churn']
            
            pipeline = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(class_weight='balanced', random_state=42))])
            pipeline.fit(X, y)
            
            rfm['churn_proba'] = pipeline.predict_proba(X)[:, 1]
            rfm['churn_score'] = rfm['churn_proba'] * 100 # as percentage

            c1, c2 = st.columns((1,2))
            with c1:
                churn_rate = y.mean() * 100
                high_prop = (rfm["churn_score"] >= churn_threshold*100).mean()*100

                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = high_prop,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {"text": "% Clientes en Riesgo"},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': f'var(--{get_kpi_color(high_prop, 15, False)})'},
                             'steps' : [{'range': [0, 15], 'color': 'lightgray'}, {'range': [15, 30], 'color': 'gray'}]}
                ))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.histogram(rfm, x='churn_score', title='Distribución de Probabilidad de Churn', nbins=50,
                                   color_discrete_sequence=['var(--indigo)'])
                fig.add_vline(x=churn_threshold*100, line_dash="dash", line_color="red", annotation_text="Umbral")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Clientes con Mayor Riesgo de Fuga")
            high_risk_customers = rfm[rfm['churn_proba'] > churn_threshold].sort_values('churn_proba', ascending=False)
            st.dataframe(high_risk_customers.head(10))

            # Churn Action
            if 'high_prop' in locals() and high_prop > 30:
                actions.append({"prioridad":1,"acción":"Campaña de retención a clientes en riesgo alto","dominio":"CRM","razón":">30% clientes sobre umbral de churn","ROI_esperado":"-10–20% churn","dueño":"CRM","cuando":"Inmediato"})

        else:
            st.warning("No hay suficientes datos para el modelo de Churn.")


    with tab5:
        st.subheader("Análisis de Cohortes y Valor de Vida del Cliente (CLV)")
        st.markdown("El análisis de cohortes agrupa a los clientes por fecha de adquisición para entender su comportamiento y retención a lo largo del tiempo. El CLV estima el valor total que un cliente aportará al negocio.")

        # Cohort Analysis
        customers['cohort'] = customers['acquisition_date'].dt.to_period('M')
        df_cohort = df.merge(customers[['customer_id', 'cohort']], on='customer_id')
        df_cohort['tx_month'] = df_cohort['date'].dt.to_period('M')
        
        def get_cohort_index(df, event_month_col='tx_month', cohort_month_col='cohort'):
            year_diff = df[event_month_col].dt.year - df[cohort_month_col].dt.year
            month_diff = df[event_month_col].dt.month - df[cohort_month_col].dt.month
            return year_diff * 12 + month_diff

        df_cohort['cohort_index'] = get_cohort_index(df_cohort)
        
        cohort_data = df_cohort.groupby(['cohort', 'cohort_index'])['customer_id'].nunique().reset_index()
        cohort_counts = cohort_data.pivot_table(index='cohort', columns='cohort_index', values='customer_id')
        
        cohort_sizes = cohort_counts.iloc[:, 0]
        retention_matrix = cohort_counts.divide(cohort_sizes, axis=0)
        
        st.subheader("Matriz de Retención de Clientes (%)")
        fig = go.Figure(data=go.Heatmap(
                   z=retention_matrix.mul(100).round(1),
                   x=retention_matrix.columns,
                   y=retention_matrix.index.strftime('%b %Y'),
                   hoverongaps = False,
                   colorscale='Blues'))
        fig.update_layout(title='Retención por Cohorte de Adquisición',
                          xaxis_title='Meses desde Adquisición',
                          yaxis_title='Cohorte')
        st.plotly_chart(fig, use_container_width=True)

        # CLV Calculation (simplified)
        st.subheader("Estimación del Customer Lifetime Value (CLV)")
        clv_days = st.slider("Periodo de cálculo para CLV (días)", 30, 365, 90)
        clv_df = df[df['date'] > df['date'].max() - timedelta(days=clv_days)]

        if not clv_df.empty:
            avg_order_value = clv_df.groupby('tx_id')['revenue_usd'].sum().mean()
            purchase_freq = clv_df.groupby('customer_id')['tx_id'].count().mean()
            
            # Churn rate for CLV
            churn_rate_clv = rfm['churn'].mean() if 'rfm' in locals() else 0.1 # default if no churn model
            if churn_rate_clv == 0: churn_rate_clv = 0.05 # Avoid division by zero
            
            customer_lifetime = 1 / churn_rate_clv
            clv = avg_order_value * purchase_freq * customer_lifetime

            c1,c2,c3 = st.columns(3)
            c1.metric("Valor de Compra Promedio", f"${avg_order_value:,.2f}")
            c2.metric("Frecuencia de Compra Promedio", f"{purchase_freq:,.2f}")
            c3.metric("CLV Estimado", f"${clv:,.0f}")
        else:
            st.warning("No hay datos en el periodo seleccionado para calcular el CLV.")

    # Show Actions DataFrame
    if actions:
        dfA = pd.DataFrame(actions).sort_values("prioridad")
        st.dataframe(dfA, use_container_width=True)
    else:
        st.success("Todos los indicadores se encuentran dentro de los rangos esperados. ¡Buen trabajo!")