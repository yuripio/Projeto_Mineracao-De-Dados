import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from sklearn.tree import plot_tree

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# Modifica de , para . (formato utilizado na biblioteca)
df = pd.read_csv('DataSetNormalizado.csv', delimiter=';', encoding='ISO-8859-1', decimal=',')


numeric_cols = ['causa_acidente', 'fase_dia', 'condicao_metereologica',
               'tracado_via', 'tipo_veiculo', 'severidade', 'idade_normalizada']

features = numeric_cols
target = 'tipo_acidente'

# Cria categorias para os tipos de acidente (Colisão (1), Atropelamento (2), ...)
le = LabelEncoder()
df['tipo_acidente_encoded'] = le.fit_transform(df[target])

# Separar dados em features (X) e target (y)
X = df[features]
y = df['tipo_acidente_encoded']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Previsões com os testes
y_pred = rf.predict(X_test)

print("\nAcuracia:", accuracy_score(y_test, y_pred))
print("\nRelatorio de Classificacao:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

feature_importances = pd.Series(rf.feature_importances_, index=features)
feature_importances.sort_values(ascending=True, inplace=True)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh')
plt.title('Importância das Features no Modelo Random Forest')
plt.xlabel('Importância Relativa')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Matriz de Confusão - Tipos de Acidente')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Visualização de Árvores Individuais
plt.figure(figsize=(20, 10))
for i, tree_in_forest in enumerate(rf.estimators_[:1]):  # Mostra a 1 árvore, se quiser ver mais árvores mude os valores embaixo támbem
    plt.subplot(1, 1, i+1)                               # plt.subplot((árvore inicial), (arvore final), i+1)
    plot_tree(tree_in_forest,
              feature_names=features,
              class_names=le.classes_,
              filled=True,
              rounded=True,
              proportion=True,
              max_depth=3)  # Limita a profundidade
    plt.title(f'Árvore {i+1} do Random Forest')
plt.tight_layout()
plt.show()