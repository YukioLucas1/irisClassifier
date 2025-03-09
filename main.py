import numpy as np
from numpy.random.mtrand import hypergeometric
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Definição do caminho do arquivo contendo o conjunto de dados
total_path = 'iris.data'
column = ["sepallength", "sepalwidth", "petallength", "petalwidth", "Type"]

# Carregando os dados do arquivo CSV para um DataFrame
data = pd.read_csv(total_path, sep=',', names=column)

# Exibindo a contagem de cada classe no conjunto de dados
print(data["Type"].value_counts())

# Inicializando o LabelEncoder para transformar classes categóricas em numéricas
le = LabelEncoder()
data['Type_encoded'] = le.fit_transform(data['Type'])

# Gerando gráficos para visualizar a relação entre os atributos e a classe da flor
for column in ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']:
  plt.figure()
  sns.regplot(x=data['Type_encoded'], y=data[column])
  plt.ylabel(column)
  plt.xlabel('Type Encoded')
  plt.xticks(ticks=[0, 1, 2], labels=le.classes_)
plt.show()

# Definição das variáveis preditoras (X) e da variável alvo (Y)
y = data.Type
X = data.drop('Type', axis=1)

# Divisão dos dados em conjuntos de treino e teste
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando um pipeline com normalização e RandomForestClassifier
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier(random_state=42))

# Definição dos hiperparâmetros para otimização
hyperparameters = {'randomforestclassifier__max_depth': [None, 5, 3, 1]}

# Utilizando GridSearchCV para encontrar os melhores hiperparâmetros
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_Train, y_Train)

# Realizando previsões no conjunto de teste
pred = clf.predict(X_Test)

# Criando um DataFrame para comparar valores reais e previstos
df_preds = pd.DataFrame({"Actual": y_Test.squeeze(), "Predicted": pred.squeeze()})
print(df_preds)

# Avaliação do modelo usando métricas de desempenho
le = LabelEncoder()
print(classification_report(y_Test, pred))
y_enc = le.fit_transform(y_Test)
pred_enc = le.fit_transform(pred)
print(le.classes_)
print("Score:", r2_score(y_enc, pred_enc))
print("mse:", mean_absolute_error(y_enc, pred_enc))