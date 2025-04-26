import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Definir o caminho do arquivo de dados
file_path = os.path.join(os.getcwd(), "data", "SMSSpamCollection")

# Carregar os dados
dados = pd.read_csv(file_path, sep="\t", header=None, names=["label", "message"])

# Visualização inicial dos dados
print("Visualização inicial dos dados:")
print(dados.head())

# Análise exploratória
print("\nResumo estatístico dos dados:")
print(dados.describe())

print("\nInformação sobre os dados:")
print(dados.info())

# Distribuição das classes
print("\nDistribuição das classes:")
print(dados['label'].value_counts())

sns.countplot(x='label', data=dados, palette="viridis")
plt.show()

# Pré-processamento dos dados
dados['label'] = dados['label'].map({'ham': 0, 'spam': 1})

# Dividir os dados em treino e teste
X = dados['message']
y = dados['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline com CountVectorizer e SVM
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  # Vetorização de texto
    ('svm', SVC())  # Classificador SVM
])

# Parâmetros para otimização com GridSearch
parameters = {
    'vectorizer__max_features': [1000, 5000],  # Número máximo de características do CountVectorizer
    'svm__C': [0.1, 1, 10],  # Parâmetros de regularização para o SVM
    'svm__kernel': ['linear', 'rbf']  # Tipos de kernel para o SVM
}

# GridSearchCV com o pipeline
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
print("\nMelhores parâmetros encontrados:")
print(grid_search.best_params_)

# Avaliação do modelo otimizado
best_svm_model = grid_search.best_estimator_
y_pred_best_svm = best_svm_model.predict(X_test)

# Avaliar performance do modelo otimizado
print("\nAcurácia do melhor modelo: ", accuracy_score(y_test, y_pred_best_svm))
print("Precisão do melhor modelo: ", precision_score(y_test, y_pred_best_svm))
print("Recall do melhor modelo: ", recall_score(y_test, y_pred_best_svm))
print("F1-Score do melhor modelo: ", f1_score(y_test, y_pred_best_svm))

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred_best_svm)
print("\nMatriz de Confusão:")
print(conf_matrix)

# Garantir que estamos alinhando apenas os dados de teste com y_test e y_pred
# Criando o DataFrame para os dados de teste
dados_teste = dados.iloc[y_test.index].copy()  # Copia as linhas de teste de dados

# Adicionando as colunas 'real' e 'predito' aos dados de teste
dados_teste['real'] = y_test
dados_teste['predito'] = y_pred_best_svm

# Identificando os falsos positivos (classificados como spam, mas na verdade são ham)
falsos_positivos = dados_teste[(dados_teste['real'] == 0) & (dados_teste['predito'] == 1)].reset_index(drop=True)

# Identificando os falsos negativos (classificados como ham, mas na verdade são spam)
falsos_negativos = dados_teste[(dados_teste['real'] == 1) & (dados_teste['predito'] == 0)].reset_index(drop=True)

# Exibindo os resultados
print("\nFalsos Positivos:")
print(falsos_positivos[['message', 'real', 'predito']])

print("\nFalsos Negativos:")
print(falsos_negativos[['message', 'real', 'predito']])

# Avaliação cruzada
from sklearn.model_selection import cross_val_score
scores = cross_val_score(best_svm_model, X_train, y_train, cv=5)
print("\nAcurácia média da validação cruzada: ", scores.mean())
