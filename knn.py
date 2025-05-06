import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import UndefinedMetricWarning
import warnings

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

# Ajustar o melhor valor de K com validação cruzada
param_grid = {'n_neighbors': list(range(1, 21))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Melhor modelo KNN
knn = grid.best_estimator_
y_pred = knn.predict(X_test)

print(f"\nMelhor K: {grid.best_params_['n_neighbors']}")
print("\nAcuracia:", accuracy_score(y_test, y_pred))
print("\nRelatorio de Classificacao:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Matriz de Confusão
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Matriz de Confusão - Tipos de Acidente (KNN)')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()