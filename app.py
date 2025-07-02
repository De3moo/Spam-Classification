import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Prepare the model once
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['text'] = df['text'].str.lower()

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate once
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Streamlit UI
st.title("📧 Spam Classifier")

st.write(f"**Model Accuracy:** {accuracy:.4f}")

# Display classification report as a table
st.subheader("📊 Classification Report")
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Display Confusion Matrix
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(fig)

# Message input
st.subheader("✉️ Test Your Own Message")
input_text = st.text_area("Enter a message here to classify it as spam or ham:")

if st.button("Classify"):
    input_vec = vectorizer.transform([input_text.lower()])
    prediction = model.predict(input_vec)[0]
    st.write(f"**Prediction:** {prediction.upper()}")
