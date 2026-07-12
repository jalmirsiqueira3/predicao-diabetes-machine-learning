from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PREPROCESSED_INFO_PATH = ROOT / "models" / "preprocessed" / "diabetes_preprocessed.joblib"
METRICS_PATH = ROOT / "models" / "trained" / "model_metrics.json"

st.set_page_config(page_title="Classificador de Diabetes", page_icon="🩺", layout="wide")

st.title("🩺 Sistema Inteligente de Classificação de Diabetes")
st.write("Insira os sinais clínicos e receba uma previsão automática para apoiar a análise do caso.")

if not PREPROCESSED_INFO_PATH.exists():
    st.error("Os arquivos de pré-processamento não foram encontrados. Execute primeiro: python src/preprocess_diabetes.py")
    st.stop()

if not METRICS_PATH.exists():
    st.error("Os modelos treinados não foram encontrados. Execute primeiro: python src/train_models.py")
    st.stop()

try:
    preprocessed_info = joblib.load(PREPROCESSED_INFO_PATH)
except Exception as exc:
    st.error(f"Não foi possível carregar o arquivo de pré-processamento: {exc}")
    st.stop()

feature_names = preprocessed_info.get("feature_names")
scaler = preprocessed_info.get("scaler")

if feature_names is None or scaler is None:
    st.error("O arquivo de pré-processamento está incompleto. Execute novamente o script de pré-processamento.")
    st.stop()

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    model_metrics = json.load(f)

model_options = {
    "Árvore de Decisão (depth = 3)": ROOT / "models" / "trained" / "decision_tree_depth_3.joblib",
    "Árvore de Decisão (depth = 5)": ROOT / "models" / "trained" / "decision_tree_depth_5.joblib",
    "Árvore de Decisão (depth = 7)": ROOT / "models" / "trained" / "decision_tree_depth_7.joblib",
    "Naive Bayes Gaussiano": ROOT / "models" / "trained" / "naive_bayes_gaussian.joblib",
}

st.sidebar.header("Configuração")
model_name = st.sidebar.selectbox("Escolha o modelo", list(model_options.keys()))
selected_model_path = model_options[model_name]

if not selected_model_path.exists():
    st.error(f"O modelo selecionado não foi encontrado: {selected_model_path}")
    st.stop()

model = joblib.load(selected_model_path)
model_metric = next((item for item in model_metrics if item["model"] == selected_model_path.stem), None)

st.sidebar.markdown("### Informações do modelo")
if model_metric:
    st.sidebar.write(f"Acurácia: {model_metric['accuracy']:.4f}")
    st.sidebar.write(f"Precisão: {model_metric['precision']:.4f}")
    st.sidebar.write(f"Recall: {model_metric['recall']:.4f}")
    st.sidebar.write(f"F1: {model_metric['f1']:.4f}")
else:
    st.sidebar.write("Métricas não disponíveis para este modelo.")

st.sidebar.write("Pré-processamento: padronização com StandardScaler")

st.subheader("Dados do paciente")
pregnancies = st.number_input("Número de gestações", min_value=0, value=3)
glucose = st.number_input("Glicose", min_value=0, value=121)
blood_pressure = st.number_input("Pressão arterial", min_value=0, value=72)
skin_thickness = st.number_input("Espessura da pele", min_value=0, value=29)
insulin = st.number_input("Insulina", min_value=0, value=140)
bmi = st.number_input("IMC", min_value=0.0, value=32.0, step=0.1)
pedigree = st.number_input("Função de pedigree", min_value=0.0, value=0.47, step=0.01)
age = st.number_input("Idade", min_value=0, value=33)

if st.button("Fazer previsão"):
    input_df = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]],
        columns=feature_names,
    )

    input_scaled = scaler.transform(input_df)
    prediction = int(model.predict(input_scaled)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        probability_positive = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
    else:
        probability_positive = None

    st.markdown("---")
    st.subheader("Resultado da previsão")

    if prediction == 1:
        st.error("⚠️ Previsão: o paciente pode ter diabetes.")
    else:
        st.success("✅ Previsão: o paciente provavelmente não tem diabetes.")

    if probability_positive is not None:
        st.write(f"Probabilidade estimada de diabetes: {probability_positive:.2%}")

    st.write("### Dados informados")
    st.write(input_df)
