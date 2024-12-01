import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt

# Pré-processamento

# Carregar dados
dados = pd.read_csv('db/vendas.csv')

# Remover valores ausentes
dados = dados.dropna()

# Conversão de formatos inconsistentes
dados['Data_Compra'] = pd.to_datetime(dados['Data_Compra'])


# Transformação

# Aplicar Label Encoding na categoria do produto, mantendo a coluna original para o gráfico 
encoder = LabelEncoder() 
dados['Categoria_Produto_Cod'] = encoder.fit_transform(dados['Categoria_Produto'])


# Classificação

# Definir as variáveis independentes (X) e alvo (y)
X = dados[['Quantidade', 'Categoria_Produto_Cod']]
y = dados['Cliente_VIP'].apply(lambda x: 1 if x == 'Sim' else 0)  # Convertendo para binário

# Treinamento do modelo
modelo = DecisionTreeClassifier()
modelo.fit(X, y)

# Exemplo de predição
predicao = modelo.predict([[2, 1]])  # Exemplo com quantidade 2 e categoria codificada como 1
print("Predição do modelo:", predicao)


# Clusterização

# Aplicação do algoritmo k-means
kmeans = KMeans(n_clusters=3, random_state=42)
dados['Cluster'] = kmeans.fit_predict(X)

# Visualização dos clusters
print(dados[['ID_Cliente', 'Cluster']])


# Associação

# Preparar os dados para análise de associação
dados_crosstab = pd.crosstab(dados['ID_Cliente'], dados['Categoria_Produto_Cod']).astype(bool)

# Encontrar itens frequentes usando fpgrowth
frequent_itemsets = fpgrowth(dados_crosstab, min_support=0.2, use_colnames=True)
print(frequent_itemsets)


# Interpretação/Avaliação

# Agrupar os dados por Categoria_Produto e somar o Valor_Total
receita_por_categoria = dados.groupby('Categoria_Produto')['Valor_Total'].sum()

# Gerar o gráfico de barras
plt.figure(figsize=(8, 6))
receita_por_categoria.plot(kind='bar', color=['blue', 'green', 'orange', 'red'], alpha=0.7)
plt.title('Receita Total por Categoria de Produto')
plt.xlabel('Categoria de Produto')
plt.ylabel('Receita Total (R$)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('db/graph.png')
plt.show()
