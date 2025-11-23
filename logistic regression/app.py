import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.title("Income Classification App (Logistic Regression)")
st.write("Prediction task: Determine whether a person's income is over $50,000 a year.")

@st.cache_data
def load_data():
    
    df = pd.read_csv("adult_converted.csv")
    return df
@st.cache_data
def load_data():
    df = pd.read_csv("adult_converted.csv")

    
    df.columns = df.columns.str.replace('\xa0', ' ', regex=False)  
    df.columns = df.columns.str.strip()                            
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)   

    
    df.rename(columns={'Income (label / target)': 'income'}, inplace=True)

    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ File 'adult_converted.csv' not found. Put it in the same folder as app.py.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Shape of data:", df.shape)



st.subheader("Data Preprocessing")


categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
st.write("Categorical columns:", categorical_cols)


encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

st.write("After encoding, preview:")
st.dataframe(df.head())

st.subheader("Train-Test Split")


if "income" not in df.columns:
    st.error("❌ Column 'income' not found in dataset. Check your CSV column names.")
    st.stop()

X = df.drop("income", axis=1) 
y = df["income"]              

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("Training samples:", X_train.shape[0])
st.write("Testing samples:", X_test.shape[0])


st.subheader("Model Training")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.subheader("Model Evaluation")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Accuracy: {accuracy:.2f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))


st.subheader("Try Your Own Prediction")

st.write("Enter feature values to predict whether income is > 50K:")

user_input = {}
for col in X.columns:
    
    default_val = float(df[col].mean())
    val = st.number_input(f"{col}", value=default_val)
    user_input[col] = val

if st.button("Predict Income"):
    user_df = pd.DataFrame([user_input])
    pred = model.predict(user_df)[0]
    result = "Income > 50K" if pred == 1 else "Income ≤ 50K"
    st.write("### 🔮 Prediction:", result)
