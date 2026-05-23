import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Carregar modelo, scaler e encoder
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Título
st.title('🏛️ Previsão de Visitantes — Museus de Los Angeles')
st.markdown('Selecione o museu e o período para ver a previsão de visitantes.')

# Inputs
museu = st.selectbox('Museu', le.classes_)
mes = st.slider('Mês', 1, 12, 6)
ano = st.slider('Ano', 2014, 2025, 2023)

# Preparar entrada
trimestre = (mes - 1) // 3 + 1
verao = 1 if mes in [6, 7, 8] else 0
museu_enc = le.transform([museu])[0]

entrada = pd.DataFrame([[mes, ano, trimestre, verao, museu_enc]],
                       columns=['Mes', 'Ano', 'Trimestre', 'Verao', 'Museu_enc'])

entrada_scaled = scaler.transform(entrada)

# Previsão
if st.button('Prever'):
    previsao = model.predict(entrada_scaled)[0]
    
    st.subheader('Resultado')
    st.metric('Visitantes previstos', f'{int(previsao):,}')
    
    # Nível de demanda
    if previsao > 10000:
        st.error(' Demanda ALTA — reforce a equipe')
    elif previsao > 5000:
        st.warning(' Demanda MÉDIA — operação normal')
    else:
        st.success(' Demanda BAIXA — pode reduzir equipe')
    
   # SHAP
    st.subheader('Por que o modelo previu isso?')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(entrada_scaled)
    
    feature_names = ['Mes', 'Ano', 'Trimestre', 'Verao', 'Museu_enc']
    importancias = pd.Series(shap_values[0], index=feature_names).sort_values()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    cores = ['#e74c3c' if v < 0 else '#2ecc71' for v in importancias]
    ax.barh(importancias.index, importancias.values, color=cores)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Impacto na previsão (SHAP value)')
    ax.set_title('Fatores que influenciaram a previsão')
    st.pyplot(fig)
