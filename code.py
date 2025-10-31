import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Red Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "acidity_input" not in st.session_state or "alcohol_input" not in st.session_state:
    
    st.session_state.acidity_input = 7.0
    st.session_state.alcohol_input = 10.0

with st.sidebar:
    st.header("Controls ⚙️")
    st.caption("Choose inputs and explore the model.")


st.title("Red Wine Quality Prediction 🍷")
st.caption("Predicting 0–10 quality from acidity and alcohol using a manual linear regression model.")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/winequality-red.csv"
    return pd.read_csv(url)

with st.spinner("Loading dataset..."):
    df = load_data()

with st.container():
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Dataset preview")
        st.dataframe(df.head(), use_container_width=True)
    with c2:
        st.subheader("Summary (selected columns)")
        st.table(df.describe()[["fixed acidity", "alcohol", "quality"]].round(2))

with st.expander("What features are used?"):
    st.write("- Acidity (Fixed Acidity)")
    st.write("- Alcohol")
    st.write("- Target: Quality (0–10)")

# Select features and target
X_cols = ["fixed acidity", "alcohol"]
df_small = df[X_cols + ["quality"]].copy()
df_small.rename(columns={"fixed acidity": "acidity"}, inplace=True)

# Manual Linear Regression (Normal Equation)
X_only = df_small[["acidity", "alcohol"]].values
ones = np.ones(len(X_only))
X = np.c_[ones, X_only]

y = df_small["quality"].values.reshape(-1, 1)

XT_X = X.T @ X
if np.linalg.det(XT_X) == 0:
    XT_X = XT_X + 1e-8 * np.eye(XT_X.shape[0])

theta = np.linalg.inv(XT_X) @ (X.T @ y)
b0, b1, b2 = float(theta[0]), float(theta[1]), float(theta[2])

# Predictions on dataset
df_small["predicted_quality"] = (X @ theta).flatten()

# Evaluation (Accuracy via rounding)
actual = df_small["quality"].values
pred = df_small["predicted_quality"].values

y_true = actual.astype(int)
y_pred_round = np.rint(pred).astype(int)

# Fixed project range 0–10
min_q, max_q = 0, 10

y_pred_labels = np.clip(y_pred_round, min_q, max_q)
accuracy = (y_pred_labels == y_true).mean()

k1, k2, k3 = st.columns(3)
k1.metric("Accuracy", f"{accuracy*100:.2f}%")
k2.metric("Samples", f"{len(df_small):,}")
k3.metric("Quality range", f"{min_q} – {max_q}")

st.divider()

# Bigger title for equation, and bold equation in a green box (one line)
st.markdown("## Regression equation")
equation_text = f"**Predicted Quality = {b0:.3f} + ({b1:.3f} × Acidity) + ({b2:.3f} × Alcohol)**"
st.success(equation_text)

# Plot: Predicted vs Actual (compact size)
st.markdown("## Predicted vs Actual")

slope, intercept = np.polyfit(actual, pred, 1)
xy_min = float(min(actual.min(), pred.min()))
xy_max = float(max(actual.max(), pred.max()))
line_x = np.linspace(xy_min, xy_max, 200)
best_fit_y = intercept + slope * line_x

fig2, ax2 = plt.subplots(figsize=(5.5, 3.6))
ax2.plot([xy_min, xy_max], [xy_min, xy_max], color='red', linestyle='--', linewidth=2, label='Perfect fit (y = x)')
ax2.plot(line_x, best_fit_y, color='blue', linestyle='-', linewidth=3, label='Best-fit line')
ax2.set_xlabel("Actual Quality")
ax2.set_ylabel("Predicted Quality")
ax2.set_title("Predicted vs Actual")
ax2.legend(loc="best")
st.pyplot(fig2, clear_figure=True)
plt.close(fig2)

st.divider()

with st.sidebar:
    st.subheader("Try your own inputs")
    acidity_input = st.slider(
        "Acidity (Fixed Acidity)",
        float(df_small["acidity"].min()),
        float(df_small["acidity"].max()),
        float(st.session_state.acidity_input)
    )
    st.session_state.acidity_input = acidity_input

    alcohol_input = st.slider(
        "Alcohol (%)",
        float(df_small["alcohol"].min()),
        float(df_small["alcohol"].max()),
        float(st.session_state.alcohol_input)
    )
    st.session_state.alcohol_input = alcohol_input

predicted_value = b0 + b1 * st.session_state.acidity_input + b2 * st.session_state.alcohol_input

st.subheader("Your prediction")
st.success(f"### 🍷 Predicted Quality: **{predicted_value:.2f} / 10**")

st.caption("Developed by ASHISH KUMAR(AI-DS 1st yr) & PUSHKAR(AI-DS 1st yr) | Regression based miniproject")
