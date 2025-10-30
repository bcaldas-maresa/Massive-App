
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
div[data-testid="stMetric"] { background:#fff; border:1px solid #eef1f5; padding:14px 16px; border-radius:14px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; }
.badge-ok { background: rgba(22,163,74,0.1); color: var(--ok); border:1px solid rgba(22,163,74,0.25); }
.badge-warn { background: rgba(245,158,11,0.1); color: var(--warn); border:1px solid rgba(245,158,11,0.25); }
.badge-bad { background: rgba(220,38,38,0.08); color: var(--bad); border:1px solid rgba(220,38,38,0.25); }
.card { border:1px solid #eef1f5; border-radius:16px; padding:16px; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
h2, h3 { color: #0D1829; }
</style>
""", unsafe_allow_html=True)

st.markdown("## Massive BI · Enterprise v3")
st.caption("Incluye: Forecast con objetivos, Cohortes, NPTB, Campañas & What-if, **NPS con Velocímetro y Objetivo**, **Churn Prediction**, y **Top Acciones**. Ahora con más colores, visuales intuitivos y un 'Modo Demo' opcional para preventa.")

# ---------------- DATA LOAD ----------------
@st.cache_data
def load_data():
    customers = pd.read_csv("data/customers.csv", parse_dates=["acquisition_date"])
    products  = pd.read_csv("data/products.csv")
    campaigns = pd.read_csv("data/campaigns.csv", parse_dates=["start_date","end_date"])
    tx        = pd.read_csv("data/transactions.csv", parse_dates=["date"])
    events    = pd.read_csv("data/web_events.csv", parse_dates=["date"])
    try:
        nps = pd.read_csv("data/nps_surveys.csv", parse_dates=["date"])
    except Exception:
        nps = pd.DataFrame(columns=["survey_id","date","customer_id","location","score","comment","category"])
    return customers, products, campaigns, tx, events, nps

customers, products, campaigns, tx, events, nps = load_data()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🔎 Filtros")
    city_sel    = st.multiselect("Ciudad", sorted(customers["city"].unique().tolist()))
    segment_sel = st.multiselect("Segmento", sorted(customers["segment"].unique().tolist()))
    channel_sel = st.multiselect("Canal adquisición", sorted(customers["acquisition_channel"].unique().tolist()))
    date_min, date_max = tx["date"].min(), tx["date"].max()
    date_range = st.date_input("Rango de fechas (ventas)", [date_min, date_max], min_value=date_min, max_value=date_max)
    st.divider()
    st.subheader("🎯 Objetivos / Presupuesto")
    sales_growth = st.slider("Crecimiento objetivo vs. histórico (%)", -20, 50, 10, 1)
    nps_target = st.slider("Objetivo NPS (%)", -100, 100, 60, 1)
    st.caption("El objetivo se usa como línea de referencia en NPS mensual.")
    st.divider()
    st.subheader("⚙️ Forecast")
    model_name = st.selectbox("Modelo", ["Media Móvil", "Tendencia Lineal", "Holt-Winters (si disponible)"])
    horizon    = st.slider("Horizonte (semanas)", 4, 26, 12)
    window_ma  = st.slider("Ventana Media Móvil", 2, 12, 4)
    st.divider()
    st.subheader("🚨 Umbrales de alerta")
    thr_nps_bad, thr_nps_good  = st.slider("NPS (malo/bueno)", -100, 100, (-10, 60))
    thr_uplift_low, thr_uplift_high = st.slider("Uplift campañas (bajo/alto) %", 0, 40, (5, 15))
    thr_retention_m3 = st.slider("Retención cohorte mes 3 (mínimo %)", 0, 100, 20)
    churn_threshold = st.slider("Umbral de Churn para alerta (%)", 0, 100, 35)
    st.divider()
    st.subheader("🧪 Modo Demo (opcional)")
    demo_mode = st.toggle("Activar escenario positivo para demos (no altera datos)", value=False)
    demo_nps_uplift = st.slider("Ajuste NPS DEMO (puntos)", 0, 40, 15) if demo_mode else 0

def apply_filters(df_tx, df_customers):
    df = df_tx.merge(df_customers[["customer_id","city","segment","acquisition_channel","acquisition_date"]], on="customer_id", how="left")
    if city_sel:    df = df[df["city"].isin(city_sel)]
    if segment_sel: df = df[df["segment"].isin(segment_sel)]
    if channel_sel: df = df[df["acquisition_channel"].isin(channel_sel)]
    if len(date_range)==2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["date"]>=start) & (df["date"]<=end)]
    return df

tx_f = apply_filters(tx, customers)

# ------------- HELPERS -------------
def monthly_targets_from_history(ts_daily, growth_pct):
    tsm = ts_daily.resample("M").sum()
    if len(tsm) == 0: return pd.Series(dtype=float)
    by_month = tsm.groupby(tsm.index.month).mean()
    future = tsm.index
    targets = pd.Series(index=future, dtype=float)
    for idx in future:
        targets.loc[idx] = by_month.loc[idx.month] * (1 + growth_pct/100.0)
    targets.name = "Objetivo"
    return targets

def nps_monthly(df):
    m = df.copy()
    m["score"] = pd.to_numeric(m["score"], errors="coerce")
    m["month"] = m["date"].values.astype('datetime64[M]')
    agg = m.groupby("month")["score"].apply(lambda s: ((s>=9).mean() - (s<=6).mean())*100)
    agg.index = pd.to_datetime(agg.index)
    agg.name = "NPS"
    return agg

# ------------- KPIs -------------
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Ingresos (USD)", f"{tx_f['revenue_usd'].sum():,.0f}")
with k2: st.metric("Transacciones", f"{len(tx_f):,}")
with k3:
    arpu = tx_f.groupby("customer_id")["revenue_usd"].sum().mean() if len(tx_f)>0 else 0
    st.metric("ARPU", f"{arpu:,.2f}")
with k4: st.metric("Clientes únicos", f"{tx_f['customer_id'].nunique():,}")

st.divider()

tab_overview, tab_forecast, tab_cohorts, tab_nptb, tab_campaigns, tab_nps, tab_churn, tab_actions = st.tabs([
    "📈 Visión General",
    "🔮 Forecast (vs Objetivo)",
    "👥 Cohortes",
    "🧠 NPTB",
    "📣 Campañas & What-if",
    "⭐ NPS & Voz del Cliente",
    "⚠️ Churn Prediction",
    "🏁 Top Acciones"
])

# -------- OVERVIEW --------
with tab_overview:
    left, right = st.columns([2,1])
    with left:
        ts = tx_f.set_index("date").resample("W")["revenue_usd"].sum().rename("Ventas")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="Ventas semanales", line=dict(color="#1565C0")))
        fig.update_layout(height=380, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Semana", yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True, key="ov_sales_v3")
    with right:
        top_p = (tx_f.merge(products, on="product_id")
                 .groupby(["product_name"])["revenue_usd"].sum()
                 .sort_values(ascending=False).head(10).reset_index())
        fig2 = px.bar(top_p, x="revenue_usd", y="product_name", orientation="h", labels={"revenue_usd":"USD","product_name":"Producto"},
                      color_discrete_sequence=["#3949AB"])
        fig2.update_layout(height=380, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig2, use_container_width=True, key="ov_top_v3")
    st.markdown("""<div class='card'><b>Explicación:</b> vista gerencial para entender tendencia y mix de productos.
    <br><b>Decisión:</b> priorizar surtido, pricing y enfoque comercial.</div>""", unsafe_allow_html=True)

# -------- FORECAST VS OBJETIVO --------
with tab_forecast:
    st.subheader("Forecast de Ventas vs. Objetivo")
    tsd = tx_f.set_index("date")["revenue_usd"].astype(float).resample("D").sum()
    tsm = tsd.resample("M").sum().rename("Real")
    targets = monthly_targets_from_history(tsd, sales_growth)
    tsw = tsd.resample("W").sum().rename("y")
    if len(tsw) < 20:
        st.warning("Se necesitan al menos 20 semanas para un forecast robusto.")
    else:
        def backtest(series, model, h=4, window_ma=4):
            n = len(series); split = int(n*0.7)
            train, test = series.iloc[:split], series.iloc[split:]
            preds, idxs = [], []
            hist = train.copy()
            for t in range(0, len(test), h):
                if model == "Media Móvil":
                    fc = hist.rolling(window_ma, min_periods=1).mean().iloc[-1]
                    pred = [fc]*min(h, len(test)-t)
                elif model == "Tendencia Lineal":
                    x = np.arange(len(hist)).reshape(-1,1)
                    lr = LinearRegression().fit(x, hist.values)
                    start = len(hist); steps = min(h, len(test)-t)
                    pred = lr.predict(np.arange(start, start+steps).reshape(-1,1)).tolist()
                elif model == "Holt-Winters" and HAS_HW:
                    hw = ExponentialSmoothing(hist, trend='add', seasonal=None).fit()
                    pred = hw.forecast(min(h, len(test)-t)).tolist()
                else:
                    fc = hist.rolling(window_ma, min_periods=1).mean().iloc[-1]
                    pred = [fc]*min(h, len(test)-t)
                preds.extend(pred); idxs.extend(test.index[t:t+len(pred)])
                hist = series.loc[:idxs[-1]]
            pred_series = pd.Series(preds, index=idxs, name="pred")
            y_true = series.loc[pred_series.index]
            rmse = sqrt(mean_squared_error(y_true, pred_series))
            mape = mean_absolute_percentage_error(y_true, pred_series)
            return pred_series, rmse, mape

        model_internal = "Media Móvil"
        if model_name.startswith("Tendencia"): model_internal = "Tendencia Lineal"
        elif model_name.startswith("Holt") and HAS_HW: model_internal = "Holt-Winters"

        pred_bt, rmse, mape = backtest(tsw, model_internal, h=4, window_ma=window_ma)

        if model_internal == "Media Móvil":
            last_fc = tsw.rolling(window_ma, min_periods=1).mean().iloc[-1]; forecast_vals = [last_fc]*horizon
        elif model_internal == "Tendencia Lineal":
            x = np.arange(len(tsw)).reshape(-1,1); lr = LinearRegression().fit(x, tsw.values)
            forecast_vals = lr.predict(np.arange(len(tsw), len(tsw)+horizon).reshape(-1,1)).tolist()
        elif model_internal == "Holt-Winters" and HAS_HW:
            hw = ExponentialSmoothing(tsw, trend='add', seasonal=None).fit(); forecast_vals = hw.forecast(horizon).tolist()
        else:
            last_fc = tsw.rolling(window_ma, min_periods=1).mean().iloc[-1]; forecast_vals = [last_fc]*horizon

        future_idx = pd.date_range(tsw.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W")
        fcst = pd.Series(forecast_vals, index=future_idx, name="Forecast")
        fcst_daily = fcst.resample("D").interpolate()
        fcst_month = fcst_daily.resample("M").sum().rename("Forecast_M")

        comp = pd.concat([tsm, targets, fcst_month], axis=1)
        comp = comp.dropna(how="all")
        comp["Cumplimiento %"] = (comp["Real"] / comp["Objetivo"]) * 100.0

        figf = go.Figure()
        figf.add_trace(go.Bar(x=comp.index, y=comp["Real"], name="Real", marker_color="#3949AB"))
        figf.add_trace(go.Bar(x=comp.index, y=comp["Objetivo"], name="Objetivo", marker_color="#00838F"))
        figf.add_trace(go.Scatter(x=comp.index, y=comp["Forecast_M"], name="Forecast", mode="lines+markers", yaxis="y2", line=dict(color="#6A1B9A")))
        figf.update_layout(height=460, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20),
                           xaxis_title="Mes", yaxis_title="USD",
                           yaxis2=dict(title="USD (Forecast)", overlaying="y", side="right", showgrid=False))
        st.plotly_chart(figf, use_container_width=True, key="fc_vs_target_v3")

        today = pd.to_datetime(datetime.today().date())
        this_month = pd.to_datetime(today.replace(day=1))
        if this_month in comp.index:
            real_m = comp.loc[this_month, "Real"]
            obj_m  = comp.loc[this_month, "Objetivo"]
            if pd.notna(real_m) and pd.notna(obj_m) and obj_m>0:
                cumplimiento = 100*real_m/obj_m
                if cumplimiento < 90:
                    st.markdown("<span class='badge badge-bad'>🔴 Cumplimiento mensual &lt; 90%</span>", unsafe_allow_html=True)
                elif cumplimiento >= 100:
                    st.markdown("<span class='badge badge-ok'>🟢 Objetivo mensual alcanzado</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge badge-warn'>🟡 En seguimiento (90–100%)</span>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE (backtest)", f"{pred_bt.sub(tsw.loc[pred_bt.index]).pow(2).mean()**0.5:,.0f}")
        c2.metric("MAPE (backtest)", f"{mape*100:,.1f}%")
        c3.metric("Horizonte", f"{horizon} semanas")

        st.markdown("""<div class='card'><b>Explicación:</b> comparamos Real vs Objetivo mensual y proyectamos el cierre con el forecast.
        <br><b>Decisión:</b> ajustar inversión, inventario o staffing según brechas.</div>""", unsafe_allow_html=True)

# -------- COHORTS --------
with tab_cohorts:
    st.subheader("Cohortes de Adquisición")
    cust = customers.copy(); cust["cohort"] = cust["acquisition_date"].dt.to_period("M").astype(str)
    tx_coh = tx_f.copy(); tx_coh["year_month"] = tx_coh["date"].dt.to_period("M").astype(str)
    m = tx_coh.merge(cust[["customer_id","cohort"]], on="customer_id", how="left")

    if len(m)>0:
        first_month = pd.Period(m["year_month"].min(), "M")
        def month_index(p): return (pd.Period(p, "M") - pd.Period(first_month, "M")).n
        m["cohort_index"] = m["year_month"].apply(lambda x: month_index(x))
        active = m.groupby(["cohort","cohort_index"])["customer_id"].nunique().reset_index()
        base = active[active["cohort_index"]==0][["cohort","customer_id"]].rename(columns={"customer_id":"cohort_size"})
        ret = active.merge(base, on="cohort", how="left")
        ret["retention"] = (ret["customer_id"] / ret["cohort_size"]).fillna(0.0)
        coh_table = ret.pivot_table(index="cohort", columns="cohort_index", values="retention", fill_value=0.0)

        m3 = coh_table[3] if 3 in coh_table.columns else pd.Series(dtype=float)
        low_ret = m3[m3*100 < thr_retention_m3]
        if len(low_ret)>0:
            st.markdown(f"<span class='badge badge-bad'>🔴 Retención mes 3 &lt; {thr_retention_m3}%: {', '.join(low_ret.index.tolist())}</span>", unsafe_allow_html=True)

        figc = px.imshow(coh_table, color_continuous_scale="Blues", aspect="auto",
                         labels=dict(color="Retención"), title="Heatmap Retención por Cohorte (mes 0 = adquisición)")
        figc.update_layout(height=460, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(figc, use_container_width=True, key="coh_heat_v3")

        st.markdown("""<div class='card'><b>Explicación:</b> medimos persistencia por cohorte para detectar canales/meses más saludables.
        <br><b>Decisión:</b> reasignar presupuesto y diseñar onboarding/recordatorios.</div>""", unsafe_allow_html=True)
    else:
        st.info("No hay transacciones en el rango seleccionado.")

# -------- NPTB --------
with tab_nptb:
    st.subheader("Propensión / NPTB")
    basket = (tx_f.groupby(["customer_id","product_id"])["quantity"].sum()>0).astype(int).reset_index()
    if len(basket)==0:
        st.info("Seleccione un rango con ventas para ver recomendaciones.")
    else:
        user_item = basket.pivot_table(index="customer_id", columns="product_id", values="quantity", fill_value=0)
        item_item = user_item.T.dot(user_item); np.fill_diagonal(item_item.values, 0)
        scores = user_item.dot(item_item)
        scores_norm = 100 * (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        cust_id = st.selectbox("Cliente", sorted(user_item.index.tolist()))
        row = scores_norm.loc[cust_id].sort_values(ascending=False).head(5)
        recs = (row.reset_index().merge(products, on="product_id", how="left").rename(columns={0:"score"}))
        recs["score"] = row.values

        figr = px.bar(recs, x="score", y="product_name", orientation="h", labels={"score":"Afinidad (0-100)","product_name":"Producto"},
                      color_discrete_sequence=["#16a34a"])
        figr.update_layout(height=380, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(figr, use_container_width=True, key="nptb_bar_v3")

        high = recs[recs["score"]>=80]
        if len(high)>0:
            st.markdown("<span class='badge badge-ok'>🟢 Alta probabilidad en: " + ", ".join(high["product_name"].tolist()) + "</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-warn'>🟡 Afinidad moderada; enriquecer datos/campañas.</span>", unsafe_allow_html=True)

        st.markdown("""<div class='card'><b>Explicación:</b> recomienda siguientes productos a partir de co-compra.
        <br><b>Valor:</b> aumenta ticket y frecuencia con ofertas relevantes.</div>""", unsafe_allow_html=True)

# -------- CAMPAIGNS --------
with tab_campaigns:
    st.subheader("Campañas & What-if")
    tx_c = tx.copy(); tx_c["campaign_day"] = 0
    for _, r in campaigns.iterrows():
        mask = (tx_c["date"] >= r["start_date"]) & (tx_c["date"] <= r["end_date"])
        tx_c.loc[mask, "campaign_day"] = 1
    daily = (tx_c.groupby([tx_c["date"].dt.date, "campaign_day"])["revenue_usd"].sum()
             .reset_index().rename(columns={"date":"day"}))
    if len(daily)>0:
        avg_with = daily[daily["campaign_day"]==1]["revenue_usd"].mean()
        avg_without = daily[daily["campaign_day"]==0]["revenue_usd"].mean()
        uplift_obs = (avg_with - avg_without) / (avg_without + 1e-6)

        c1,c2,c3 = st.columns(3)
        c1.metric("Promedio con campaña", f"{avg_with:,.0f}")
        c2.metric("Promedio sin campaña", f"{avg_without:,.0f}")
        c3.metric("Uplift observado", f"{uplift_obs*100:,.1f}%")

        if uplift_obs*100 < thr_uplift_low:
            st.markdown(f"<span class='badge badge-bad'>🔴 Uplift &lt; {thr_uplift_low}%</span>", unsafe_allow_html=True)
        elif uplift_obs*100 > thr_uplift_high:
            st.markdown(f"<span class='badge badge-ok'>🟢 Uplift &gt; {thr_uplift_high}%</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-warn'>🟡 Uplift moderado</span>", unsafe_allow_html=True)

        pivot = daily.pivot_table(index="day", columns="campaign_day", values="revenue_usd").rename(columns={0:"Sin campaña",1:"Con campaña"}).reset_index()
        figd = go.Figure()
        figd.add_trace(go.Scatter(x=pivot["day"], y=pivot["Sin campaña"], mode="lines", name="Sin campaña", line=dict(color="#dc2626")))
        figd.add_trace(go.Scatter(x=pivot["day"], y=pivot["Con campaña"], mode="lines", name="Con campaña", line=dict(color="#16a34a")))
        figd.update_layout(height=380, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Día", yaxis_title="USD")
        st.plotly_chart(figd, use_container_width=True, key="camp_lines_v3")

        st.caption("Tip: esta vista puede conectarse a MMM para optimizar el mix por canal.")
    else:
        st.info("No hay datos para impacto de campañas.")

# -------- NPS --------
with tab_nps:
    st.subheader("NPS & Voz del Cliente")
    if nps.empty:
        st.info("No hay datos de NPS.")
    else:
        nps_df = nps.copy()
        nps_df["score"] = pd.to_numeric(nps_df["score"], errors="coerce")

        # NPS Global (con opción demo)
        promoters = (nps_df["score"]>=9).mean(); detractors=(nps_df["score"]<=6).mean()
        nps_global = (promoters - detractors)*100
        nps_demo = min(100, nps_global + demo_nps_uplift) if demo_mode else nps_global

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=nps_demo,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "#0D6EFD"},
                'steps': [
                    {'range': [-100, 0],  'color': 'rgba(220,38,38,0.2)'},
                    {'range': [0, 60],   'color': 'rgba(245,158,11,0.2)'},
                    {'range': [60, 100], 'color': 'rgba(22,163,74,0.2)'},
                ]
            },
            title={"text": "NPS Global (%)" + (" — DEMO" if demo_mode else "")}
        ))
        gauge.update_layout(height=260, margin=dict(l=20,r=20,t=30,b=0), template="plotly_white")
        st.plotly_chart(gauge, use_container_width=True, key="nps_gauge_v3")

        # Mensual (septiembre completo si hay datos; si demo ON, aplicar uplift)
        nps_df["date"] = pd.to_datetime(nps_df["date"])
        nps_m = nps_df.copy()
        nps_m["month"] = nps_m["date"].values.astype('datetime64[M]')
        nps_m = nps_m.groupby("month")["score"].apply(lambda s: ((s>=9).mean() - (s<=6).mean())*100).rename("NPS")
        nps_m.index = pd.to_datetime(nps_m.index)
        if demo_mode:
            nps_m = (nps_m + demo_nps_uplift).clip(upper=100)

        target_series = pd.Series(nps_target, index=nps_m.index).rename("Objetivo NPS")
        comp = pd.concat([nps_m.rename("NPS"), target_series], axis=1)

        fig_nps_m = go.Figure()
        fig_nps_m.add_trace(go.Scatter(x=comp.index, y=comp["NPS"], mode="lines+markers", name="NPS mensual", line=dict(color="#1565C0")))
        fig_nps_m.add_trace(go.Scatter(x=comp.index, y=comp["Objetivo NPS"], mode="lines", name="Objetivo NPS", line=dict(dash="dash", color="#16a34a")))
        fig_nps_m.update_layout(height=360, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Mes", yaxis_title="NPS (%)")
        st.plotly_chart(fig_nps_m, use_container_width=True, key="nps_month_line_v3")

        # Categorías
        cat_agg = nps_df.groupby("category").agg(encuestas=("survey_id","count"), score_prom=("score","mean")).reset_index().sort_values("encuestas", ascending=False)
        fig_cat = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cat.add_trace(go.Bar(x=cat_agg["category"], y=cat_agg["encuestas"], name="Encuestas", marker_color="#3949AB"), secondary_y=False)
        fig_cat.add_trace(go.Scatter(x=cat_agg["category"], y=cat_agg["score_prom"], mode="lines+markers", name="Score Promedio", line=dict(color="#00838F")), secondary_y=True)
        fig_cat.update_yaxes(title_text="Encuestas", secondary_y=False)
        fig_cat.update_yaxes(title_text="Score", secondary_y=True, range=[0,10])
        fig_cat.update_layout(height=380, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20), title_text="Volumen vs. Score por Categoría")
        st.plotly_chart(fig_cat, use_container_width=True, key="nps_cat_v3")

        # Alertas
        if nps_global < thr_nps_bad and not demo_mode:
            st.markdown(f"<span class='badge badge-bad'>🔴 NPS {nps_global:,.0f}% por debajo del umbral</span>", unsafe_allow_html=True)
        elif comp["NPS"].iloc[-1] < nps_target and not demo_mode:
            st.markdown("<span class='badge badge-warn'>🟡 NPS por debajo del objetivo</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-ok'>🟢 NPS saludable o en objetivo</span>" + (" <b>(DEMO)</b>" if demo_mode else ""), unsafe_allow_html=True)

        st.markdown("""<div class='card'><b>Explicación:</b> NPS mide lealtad (%Promotores - %Detractores). 
        Agregamos vista mensual y velocímetro con objetivo; el <i>Modo Demo</i> permite ilustrar un escenario positivo <b>sin alterar datos</b>.</div>""", unsafe_allow_html=True)

# -------- CHURN --------
with tab_churn:
    st.subheader("Churn Prediction (Abandono)")
    st.caption("Modelo supervisado (demo) que estima la probabilidad de que un cliente abandone. Se construyen features RFM + web.")
    N_INACTIVE = 120
    max_date = tx_f["date"].max() if len(tx_f)>0 else tx["date"].max()
    last_tx = tx_f.groupby("customer_id")["date"].max().reset_index().rename(columns={"date":"last_tx"})
    freq = tx_f.groupby("customer_id")["tx_id"].count().reset_index().rename(columns={"tx_id":"tx_count"})
    money = tx_f.groupby("customer_id")["revenue_usd"].sum().reset_index().rename(columns={"revenue_usd":"monetary"})
    rfm = customers[["customer_id","acquisition_date"]].merge(last_tx, on="customer_id", how="left").merge(freq, on="customer_id", how="left").merge(money, on="customer_id", how="left")
    rfm["last_tx"] = rfm["last_tx"].fillna(pd.Timestamp(max_date) - pd.Timedelta(days=N_INACTIVE+1))
    rfm["tx_count"] = rfm["tx_count"].fillna(0)
    rfm["monetary"] = rfm["monetary"].fillna(0.0)
    rfm["recency_days"] = (pd.Timestamp(max_date) - rfm["last_tx"]).dt.days
    rfm["tenure_days"]  = (pd.Timestamp(max_date) - rfm["acquisition_date"]).dt.days
    web = events.groupby("customer_id")["date"].count().reset_index().rename(columns={"date":"web_sessions"})
    rfm = rfm.merge(web, on="customer_id", how="left").fillna({"web_sessions":0})
    rfm["churned"] = (rfm["recency_days"] > N_INACTIVE).astype(int)

    feats = ["recency_days","tx_count","monetary","tenure_days","web_sessions"]
    X = rfm[feats].values
    y = rfm["churned"].values
    if len(np.unique(y)) < 2:
        st.info("No hay variación suficiente para entrenar el modelo de churn en el filtro actual.")
    else:
        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
        pipe.fit(X, y)
        rfm["churn_score"] = pipe.predict_proba(X)[:,1] * 100

        mean_churn = rfm["churn_score"].mean()
        g_churn = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_churn,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#dc2626"},
                'steps': [
                    {'range': [0, 25],  'color': 'rgba(22,163,74,0.2)'},
                    {'range': [25, 50], 'color': 'rgba(245,158,11,0.2)'},
                    {'range': [50, 100],'color': 'rgba(220,38,38,0.2)'},
                ]
            },
            title={"text": "Churn promedio (%)"}
        ))
        g_churn.update_layout(height=260, template="plotly_white", margin=dict(l=20,r=20,t=30,b=0))
        st.plotly_chart(g_churn, use_container_width=True, key="churn_gauge_v3")

        hist = px.histogram(rfm, x="churn_score", nbins=30, title="Distribución de score de churn", color_discrete_sequence=["#6A1B9A"])
        hist.update_layout(height=360, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Score (%)")
        st.plotly_chart(hist, use_container_width=True, key="churn_hist_v3")

        top_at_risk = rfm.sort_values("churn_score", ascending=False).head(20)
        st.dataframe(top_at_risk[["customer_id","churn_score","recency_days","tx_count","monetary","web_sessions"]])

        high_risk = (rfm["churn_score"] >= churn_threshold).mean()*100
        if high_risk > 30:
            st.markdown("<span class='badge badge-bad'>🔴 Alta proporción de clientes en riesgo</span>", unsafe_allow_html=True)
        elif high_risk > 10:
            st.markdown("<span class='badge badge-warn'>🟡 Riesgo moderado</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-ok'>🟢 Riesgo controlado</span>", unsafe_allow_html=True)

        st.markdown("""<div class='card'><b>Qué es:</b> modelo que predice probabilidad de abandono combinando recencia, frecuencia, valor, antigüedad y actividad web.
        <br><b>Qué se espera:</b> un score 0–100% por cliente.
        <br><b>Cómo ayuda:</b> activar campañas de retención personalizadas y medir recuperación.</div>""", unsafe_allow_html=True)

# -------- ACTIONS --------
with tab_actions:
    st.subheader("Top Acciones")
    actions = []

    tsd = tx_f.set_index("date")["revenue_usd"].astype(float).resample("D").sum()
    tsm = tsd.resample("M").sum().rename("Real")
    targets = monthly_targets_from_history(tsd, sales_growth)
    today = pd.to_datetime(datetime.today().date())
    this_month = today.replace(day=1)
    if this_month in tsm.index and this_month in targets.index:
        real_m = tsm.loc[this_month]; obj_m = targets.loc[this_month]
        if pd.notna(real_m) and pd.notna(obj_m) and obj_m>0:
            cumplimiento = 100*real_m/obj_m
            if cumplimiento < 90:
                actions.append({"prioridad":1,"acción":"Acelerar campañas y paquetes de valor","dominio":"Ventas/Marketing","razón":"Cumplimiento mensual <90%","ROI_esperado":"+8–15%","dueño":"Marketing","cuando":"Inmediato"})
            elif cumplimiento < 100:
                actions.append({"prioridad":2,"acción":"Ajustar mix de medios hacia canales de mayor retorno","dominio":"Marketing","razón":"Cumplimiento 90–100%","ROI_esperado":"+3–6%","dueño":"Mkt Performance","cuando":"Esta semana"})
            else:
                actions.append({"prioridad":3,"acción":"Asegurar stock y capacidad por demanda alta","dominio":"Operaciones","razón":"Objetivo superado","ROI_esperado":"Evitar quiebre","dueño":"Operaciones","cuando":"Próximos 7 días"})

    if not nps.empty:
        nps_df = nps.copy(); nps_df["score"] = pd.to_numeric(nps_df["score"], errors="coerce")
        promoters = (nps_df["score"]>=9).mean(); detractors=(nps_df["score"]<=6).mean()
        nps_global = (promoters - detractors)*100
        if nps_global < thr_nps_bad and not demo_mode:
            actions.append({"prioridad":1,"acción":"Plan de choque CX (Trato/Tiempo/Limpieza)","dominio":"CX/Operaciones","razón":f"NPS {nps_global:,.0f}% < umbral","ROI_esperado":"+10 pts NPS","dueño":"CX","cuando":"Inmediato"})
        elif nps_global < nps_target and not demo_mode:
            actions.append({"prioridad":2,"acción":"Entrenamiento de atención y optimización de turnos","dominio":"CX","razón":f"NPS {nps_global:,.0f}% por debajo del objetivo","ROI_esperado":"+5 pts NPS","dueño":"RRHH/CX","cuando":"Esta semana"})
        else:
            actions.append({"prioridad":3,"acción":"Programa de referidos / reputación","dominio":"Marketing","razón":"NPS saludable","ROI_esperado":"+5–8% captación","dueño":"Marketing","cuando":"Próximos 15 días"})

    # Churn
    try:
        if 'rfm' in locals():
            high_prop = (rfm["churn_score"] >= churn_threshold).mean()*100
            if high_prop > 30:
                actions.append({"prioridad":1,"acción":"Campaña de retención a clientes en riesgo alto","dominio":"CRM","razón":">30% clientes sobre umbral de churn","ROI_esperado":"-10–20% churn","dueño":"CRM","cuando":"Inmediato"})
    except Exception:
        pass

    if actions:
        dfA = pd.DataFrame(actions).sort_values("prioridad")
        st.dataframe(dfA, use_container_width=True)
    else:
        st.success("✅ Sin alertas críticas. Mantener plan y monitoreo.")

    st.markdown("""<div class='card'><b>Uso:</b> prioriza de arriba hacia abajo, asigna responsables y tiempos.
    <br><b>Valor:</b> traduce analítica en decisiones con ROI esperado.</div>""", unsafe_allow_html=True)
