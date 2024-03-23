# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Leximi i datasetit "SMS Spam Collection" 
dataset_path = 'SMS_Spam_Collection.csv'
df = pd.read_csv(dataset_path, encoding='latin-1')

# Shfaqim informacion bazik mbi datasetin
print(df.describe())
print(df.head())

# Dataseti mund të përmbajë kolona shtesë që nuk janë të nevojshme, prandaj i largojmë ato
# Supozojmë se kolonat e vlefshme janë 'v1' për etiketën dhe 'v2' për mesazhin
df = df[['v1', 'v2']]
df.columns = ['Label', 'Message']

# Ndajmë datasetin në features (X) dhe target (y)
X = df['Message']
y = df['Label']

# Ndajmë të dhënat në setin e trajnimit dhe testimit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Përdorimi i TF-IDF Vectorizer për të kthyer tekstet në një matricë të peshuar të termave
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Trajnojmë modelin përdorur Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Parashikojmë dhe vlerësojmë modelin
predictions = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:", confusion_matrix(y_test, predictions))
print("Classification Report:", classification_report(y_test, predictions))
