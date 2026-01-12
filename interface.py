import streamlit as st
import pandas as pd
import numpy as np
import openai
import google.generativeai as genai
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bem‑estar mental - Demonstração AASE", layout="wide")

# -------------------------
# Carregar modelo de deployment
# -------------------------
@st.cache_resource
def load_artifact(path: str = "wellness_model.joblib"):
    return load(path)

try:
    artifact = load_artifact()
    MODEL = artifact["model"]
    FEATURES = artifact["feature_names"]
    # Fallback if class_names missing
    CLASS_NAMES = artifact.get("class_names", ["Alto", "Baixo", "Médio"])
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()


# -------------------------
# Helpers
# -------------------------
def gerar_relatorio_classico(valores, classe):
    """
    Fallback: Gera relatório baseado em regras simples caso a IA falhe.
    """
    erros = []
    sugestoes = []
    pontos_fortes = []

    screen = valores["screen_time_hours"]
    stress = valores["stress_level_0_10"]
    sleep_h = valores["sleep_hours"]
    sleep_q = valores["sleep_quality_1_5"]
    exer = valores["exercise_minutes_per_week"]
    prod = valores["productivity_0_100"]

    if screen > 8:
        erros.append("Tempo de ecrã diário acima de 8 horas.")
        sugestoes.append("Reduzir o tempo de ecrã (especialmente fora do trabalho).")
    elif screen < 4:
        pontos_fortes.append("Tempo de ecrã moderado.")

    if stress >= 7:
        erros.append("Nível de stress muito elevado.")
        sugestoes.append("Introduzir momentos diários de relaxamento.")
    elif stress <= 3:
        pontos_fortes.append("Bom controlo do stress.")

    if sleep_h < 7 or sleep_q <= 2:
        erros.append("Sono insuficiente ou de fraca qualidade.")
        sugestoes.append("Garantir 7–9 horas de sono.")
    else:
        pontos_fortes.append("Rotina de sono razoável.")

    if exer < 150:
        erros.append("Atividade física abaixo de 150 min/semana.")
        sugestoes.append("Aumentar gradualmente o exercício.")
    else:
        pontos_fortes.append("Boa prática de atividade física.")

    relatorio = f"""
    ### Relatório (Modo Clássico)
    **Classificação:** {classe}
    
    **Pontos Fortes:**
    {''.join([f'- {p}  \n' for p in pontos_fortes]) if pontos_fortes else '- (Nenhum destaque registado)'}
    
    **Atenção:**
    {''.join([f'- {e}  \n' for e in erros]) if erros else '- (Nenhum alerta registado)'}
    
    **Sugestões:**
    {''.join([f'- {s}  \n' for s in sugestoes]) if sugestoes else '- (Manter a rotina atual)'}
    """
    return relatorio

def gerar_relatorio_ia(valores, classe, api_key_val):
    if not api_key_val or api_key_val == "...":
        return gerar_relatorio_classico(valores, classe)
    
    try:
        # Configurar Google Gemini
        genai.configure(api_key=api_key_val)
        
        # Tentar usar o alias 'gemini-flash-latest' que aponta para a versão estável mais recente
        model = genai.GenerativeModel('gemini-flash-latest')
        
        prompt = f"""
        Atue como um especialista em saúde mental e bem-estar digital.
        O sistema classificou o bem-estar deste utilizador como: **{classe}**.
        
        Dados:
        - Tempo de ecrã: {valores['screen_time_hours']}h
        - Stress (0-10): {valores['stress_level_0_10']}
        - Sono: {valores['sleep_hours']}h (Qualidade: {valores['sleep_quality_1_5']}/5)
        - Exercício: {valores['exercise_minutes_per_week']} min/sem
        - Produtividade: {valores['productivity_0_100']}
        
        Escreva um relatório curto, direto e motivador. Fale diretamente com o utilizador.
        IMPORTANTE: Escreva estritamente em Português de Portugal (pt-PT). Use termos como "ecrã" em vez de "tela", "equipa" em vez de "time", etc.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"⚠️ **Erro na IA (Gemini):** {e}\n\n" + gerar_relatorio_classico(valores, classe)


# -------------------------
# PÁGINA 1: Previsão Individual
# -------------------------
def run_forecast_page():
    st.title("Previsão de bem‑estar mental")
    st.caption("Interface de deployment do modelo treinado no Projeto 1 (AASE).")

    st.header("Previsão para um utilizador")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            screen_time = st.number_input("Horas de ecrã por dia", 0.0, 24.0, 6.0, 0.5)
            stress = st.slider("Nível de stress (0–10)", 0, 10, 5)
        with c2:
            sleep_hours = st.number_input("Horas de sono", 0.0, 24.0, 7.0, 0.5)
            sleep_quality = st.slider("Qualidade do sono (1–5)", 1, 5, 3)
        with c3:
            exercise = st.number_input("Minutos de exercício por semana", 0.0, 2000.0, 150.0, 10.0)
            productivity = st.number_input("Produtividade (0–100)", 0.0, 100.0, 70.0, 1.0)

        submit = st.form_submit_button("Prever classe de bem‑estar")

    if submit:
        input_row = {
            "screen_time_hours": screen_time,
            "stress_level_0_10": stress,
            "sleep_hours": sleep_hours,
            "sleep_quality_1_5": sleep_quality,
            "exercise_minutes_per_week": exercise,
            "productivity_0_100": productivity,
        }
        X = pd.DataFrame([input_row])[FEATURES]

        st.subheader("Input do modelo")
        st.write(X)

        try:
            pred = MODEL.predict(X)
            pred_class = pred[0]
            st.success(f"Classe prevista de bem‑estar mental: **{pred_class}**")

            # Gerar relatório com IA
            with st.spinner('A gerar o relatório personalizado com IA...'):
                relatorio = gerar_relatorio_ia(input_row, pred_class, GOOGLE_API_KEY)
            
            st.markdown("### 🤖 Relatório Personalizado com Parceria com o (Gemini)")
            st.markdown(relatorio)

        except Exception as e:
            st.error(f"Falha na previsão: {e}")

    # Section Batch
    st.header("Previsão em lote a partir de CSV")
    batch_file = st.file_uploader("Upload do CSV", type=["csv"])

    if batch_file is not None:
        try:
            df_batch = pd.read_csv(batch_file)
            missing = [c for c in FEATURES if c not in df_batch.columns]
            if missing:
                st.error(f"Faltam colunas: {missing}")
            else:
                if st.button("Prever em lote"):
                    preds = MODEL.predict(df_batch[FEATURES])
                    df_out = df_batch.copy()
                    df_out["mental_wellness_class_pred"] = preds
                    st.dataframe(df_out.head())
                    st.download_button("Download", df_out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Erro no CSV: {e}")


# -------------------------
# PÁGINA 2: Avaliação dos Modelos (Dashboard)
# -------------------------
def run_evaluation_page():
    st.title("Avaliação dos Modelos (Dashboard)")
    
    # Carregar Dataset
    try:
        df = pd.read_csv("ScreenTime vs MentalWellness.csv")
    except Exception:
        st.error("Ficheiro 'ScreenTime vs MentalWellness.csv' não encontrado no diretório.")
        return

    target_reg = 'mental_wellness_index_0_100'
    if target_reg not in df.columns:
        st.error(f"Coluna de índice '{target_reg}' em falta.")
        return

    # Verificar features
    missing_feats = [c for c in FEATURES if c not in df.columns]
    if missing_feats:
        st.error(f"Features em falta no dataset: {missing_feats}")
        return

    # 1. Preparação (Gerar Ground Truth e Predições on-the-fly se necessário)
    st.info("A calcular métricas e gerar visualizações a partir dos dados carregados...")
    
    # Gerar Classe Real (Ground Truth) baseada em quantis 33/66
    lower = df[target_reg].quantile(0.33)
    upper = df[target_reg].quantile(0.66)
    
    def get_class_label(x):
        if x <= lower: return 'Baixo'
        elif x <= upper: return 'Médio'
        else: return 'Alto'
        
    df['Mental Wellness Class (Real)'] = df[target_reg].apply(get_class_label)
    
    X = df[FEATURES]
    y_reg = df[target_reg]
    y_class = df['Mental Wellness Class (Real)']
    
    # ---------------------------------------------------
    # 2. Bloco de Regressão
    # ---------------------------------------------------
    st.markdown("## 2. Bloco de Regressão")
    st.caption("Nota: Modelo de regressão treinado em tempo real para demonstração.")
    
    @st.cache_resource
    def train_regressor(X_data, y_data):
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        rf.fit(X_data, y_data)
        return rf
    
    reg_model = train_regressor(X, y_reg)
    df['pred_wellness_index'] = reg_model.predict(X)
    
    # KPIs
    mae = mean_absolute_error(y_reg, df['pred_wellness_index'])
    rmse = np.sqrt(mean_squared_error(y_reg, df['pred_wellness_index']))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    
    # Scatter Plot
    fig_scatter = px.scatter(
        df, x=target_reg, y='pred_wellness_index',
        color='Mental Wellness Class (Real)',
        title="Real vs Previsto (Regressão)",
        labels={target_reg: "Índice Real (0-100)", 'pred_wellness_index': "Índice Previsto"}
    )
    # Linha diagonal de referência
    fig_scatter.add_shape(type="line", line=dict(dash='dash', color='gray'),
        x0=0, y0=0, x1=100, y1=100)
    
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------------------------------------------------
    # 3. Bloco de Classificação
    # ---------------------------------------------------
    st.markdown("## 3. Bloco de Classificação")
    
    # Predição usando o modelo carregado (Global artifact)
    df['pred_wellness_class'] = MODEL.predict(X)
    
    # KPI Accuracy
    acc = accuracy_score(y_class, df['pred_wellness_class'])
    st.metric("Accuracy (Percentagem de Acerto)", f"{acc:.2%}")
    
    # Matriz de Confusão
    labels = sorted(y_class.unique())
    cm = confusion_matrix(y_class, df['pred_wellness_class'], labels=labels)
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("Matriz de Confusão")
        fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                           color_continuous_scale='Blues',
                           labels=dict(x="Previsto", y="Real", color="Contagem"))
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_viz2:
        st.subheader("Erros por Classe")
        # Coluna de Resultado
        df['Resultado Classificação'] = np.where(df['Mental Wellness Class (Real)'] == df['pred_wellness_class'], 'Correto', 'Incorreto')
        
        # Gráfico de Barras Empilhadas
        fig_bar = px.histogram(df, x='pred_wellness_class', color='Resultado Classificação',
                               barmode='group', # ou 'stack'
                               category_orders={'pred_wellness_class': labels},
                               color_discrete_map={'Correto': '#2ca02c', 'Incorreto': '#d62728'},
                               title="Distribuição de Acertos/Erros")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------------------------------
    # 4. Bloco de Clustering
    # ---------------------------------------------------
    st.markdown("## 4. Bloco de Clustering")
    
    @st.cache_resource
    def run_clustering(X_data):
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        return km.fit_predict(X_data)
        
    df['cluster_label'] = run_clustering(X)
    df['cluster_label'] = df['cluster_label'].astype(str) # Para visualização categórica
    
    c_clust1, c_clust2 = st.columns(2)
    
    with c_clust1:
        st.subheader("Distribuição dos Clusters")
        fig_clust_count = px.histogram(df, x='cluster_label', title="Contagem por Cluster")
        st.plotly_chart(fig_clust_count, use_container_width=True)
        
    with c_clust2:
        st.subheader("Perfil Médio por Cluster")
        # Agregar colunas numéricas
        cols_to_avg = ['pred_wellness_index', 'stress_level_0_10', 'sleep_hours', 'screen_time_hours']
        avg_profile = df.groupby('cluster_label')[cols_to_avg].mean().reset_index()
        
        # Melt para gráfico agrupado
        avg_melt = avg_profile.melt(id_vars='cluster_label', var_name='Métrica', value_name='Valor Médio')
        
        fig_profile = px.bar(avg_melt, x='cluster_label', y='Valor Médio', color='Métrica', barmode='group')
        st.plotly_chart(fig_profile, use_container_width=True)


# -------------------------
# PÁGINA 3: Dashboards Power BI
# -------------------------
def run_powerbi_page():
    st.title("Dashboards Power BI")

    
    # Campo de texto para colar o URL
    # Podes até deixar um valor default se já tiveres o link
    powerbi_url = st.text_input("Cola o Link de Incorporação aqui:", "")
    
    if powerbi_url:
        # Tenta converter link direto (browser) para link de embed
        if "app.powerbi.com" in powerbi_url and "reportEmbed" not in powerbi_url:
            import re
            # Padrão para extrair o report ID da URL: .../reports/<UUID>/...
            match = re.search(r'reports/([a-f0-9-]{36})', powerbi_url)
            if match:
                report_id = match.group(1)
                powerbi_url = f"https://app.powerbi.com/reportEmbed?reportId={report_id}&autoAuth=true"
                st.caption("ℹ️ Link convertido automaticamente para formato de incorporação.")
                
        st.markdown(f'<iframe title="AASE Dashboard" width="100%" height="800" src="{powerbi_url}" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)
    else:
        st.info("À espera do link do Power BI...")


# -------------------------
# Navegação Principal
# -------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=50) # Ícone generico
st.sidebar.title("Navegação AASE")

# API Key Fixa (Substitua "..." pela sua chave real da Google)
GOOGLE_API_KEY = "AIzaSyCqIo6PihbSF6Zt7HQPK-0xvXKxYCmDw6c" 

page = st.sidebar.radio("Selecione a página:", 
    ["Previsão Individual", "Avaliação dos Modelos", "Dashboards Power BI"]
)

if page == "Previsão Individual":
    run_forecast_page()
elif page == "Avaliação dos Modelos":
    run_evaluation_page()
elif page == "Dashboards Power BI":
    run_powerbi_page()

