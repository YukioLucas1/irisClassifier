<H1>Iris Classifier</H1>

Este projeto utiliza Python para implementar um modelo de Inteligência Artificial que classifica flores da base de dados Iris. O objetivo é treinar um modelo de aprendizado supervisionado para alcançar um Score = 1.0 e um Erro Quadrático Médio = 0.0.
<h2>🔗 Link do Projeto</h2>
o projeto pode ser visto, também, em:
https://replit.com/@YukioLucas1/irisClassifier?v=1

<h2>📌 bibliotecas utilizadas</h2>

    Pandas
    NumPy
    Scikit-learn
    Seaborn & Matplotlib

O código foi desenvolvido e testado utilizando o ambiente Replit, permitindo fácil execução e compartilhamento do projeto.

<h2>📂 Estrutura do projeto</h2>

 ├── .gitignore  
 ├── .replit               # Configuração do Replit  
 ├── iris.data             # Base de dados  
 ├── main.py               # Código principal  
 ├── pyproject.toml        # Dependências do projeto  
 ├── README.md             # Documento explicativo  

<h2>📊 Base de Dados</h2>

    Nome: Iris
    Quantidade de elementos: 150 amostras
    Quantidade de atributos e respectivos nomes:
        Sepal Length (Comprimento da Sépala)
        Sepal Width (Largura da Sépala)
        Petal Length (Comprimento da Pétala)
        Petal Width (Largura da Pétala)
    Descrição da Classe (Y): Tipo da flor (Setosa, Versicolor ou Virginica)
    Distribuição de elementos em cada classe:
        Setosa: 50 amostras
        Versicolor: 50 amostras
        Virginica: 50 amostras

<h2>🔍 Modelo Utilizado</h2>

O modelo utilizado foi um Random Forest Classifier, que foi escolhido por sua robustez e capacidade de lidar bem com pequenos conjuntos de dados sem sobreajuste.

A base foi dividida em 80% para treinamento e 20% para teste, garantindo generalização do modelo.
📈 Análise da Correlação

Gráficos foram gerados:

![image](https://github.com/user-attachments/assets/ec1a6675-aafd-4a89-9f2f-9f7e832f5866)

![image](https://github.com/user-attachments/assets/c608cdaa-cfdf-4045-8e41-72ac1ed134c0)

![image](https://github.com/user-attachments/assets/c38f795e-6cb6-45b4-935e-a3d760931ad1)

![image](https://github.com/user-attachments/assets/17325e9d-a705-4ee1-b317-794497f731be)






<h2>📉 Resultados </h2>

✅ Score: 1.0
✅ Erro Quadrático Médio: 0.0

![image](https://github.com/user-attachments/assets/b577d8c4-222f-440a-8c63-1a73e2297f99)


Isso significa que o modelo conseguiu classificar todas as amostras corretamente, atingindo o objetivo da atividade.


