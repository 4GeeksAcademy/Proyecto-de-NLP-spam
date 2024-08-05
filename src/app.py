from utils import db_connect
engine = db_connect()

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import regex as re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import download

# Descargar stopwords y wordnet
download('stopwords')
download('wordnet')

# Paso 1: Cargar el conjunto de datos
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv")

# Paso 2: Preprocesar los enlaces
total_data["is_spam"] = total_data["is_spam"].apply(lambda x: 1 if x else 0).astype(int)
total_data = total_data.drop_duplicates().reset_index(drop=True)

# Función de preprocesamiento
def preprocess_text(text):
    text = re.sub(r'[^a-z ]', " ", text.lower())
    text = re.sub(r'\s+', " ", text)
    return text.split()

total_data["url"] = total_data["url"].apply(preprocess_text)

# Lemmatización y eliminación de stopwords
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")

def lemmatize_text(words):
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 3]
    return tokens

total_data["url"] = total_data["url"].apply(lemmatize_text)

# Generar WordCloud para visualización
wordcloud = WordCloud(width=800, height=800, background_color="black", max_words=1000, min_font_size=20, random_state=42)\
    .generate(str(total_data["url"]))

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Vectorización TF-IDF
tokens_list = total_data["url"].apply(lambda tokens: " ".join(tokens))
vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=5)
X = vectorizer.fit_transform(tokens_list).toarray()
y = total_data["is_spam"]

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 3: Construir y entrenar la SVM
model = SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Paso 4: Optimizar el modelo
hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [1, 2, 3, 4, 5],
    "gamma": ["scale", "auto"]
}

grid = GridSearchCV(SVC(), hyperparams, scoring="accuracy", cv=5)
grid.fit(X_train, y_train)

print(f"Best hyperparameters: {grid.best_params_}")

# Evaluación del mejor modelo
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Best Model Accuracy: {accuracy_score(y_test, y_pred_best)}")
print(classification_report(y_test, y_pred_best))

# Paso 5: Guardar el modelo
import joblib
joblib.dump(best_model, 'best_svm_model.pkl')