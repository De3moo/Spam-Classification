from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

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

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()

# Ensure static folder exists and save plot
if not os.path.exists('static'):
    os.makedirs('static')
plt.savefig('static/confusion_matrix.png')
plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_text = ""
    if request.method == 'POST':
        input_text = request.form['message'].lower()
        input_vec = vectorizer.transform([input_text])
        prediction = model.predict(input_vec)[0]

    return render_template('results.html',
                           accuracy=accuracy,
                           report=report,
                           prediction=prediction,
                           input_text=input_text)

if __name__ == '__main__':
    app.run(debug=False)
