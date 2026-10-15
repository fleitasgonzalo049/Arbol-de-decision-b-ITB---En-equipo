import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Carga de datos (ajusta los nombres exactos de los  archivos en data/)
df_ent = pd.read_csv('../data/TablaPrediccionAbandono-Entrenamiento(PrediccionAbandonoTabla).csv', sep=';', encoding='latin1')
df_final = pd.read_csv('../data/TablaPrediccionAbandono-DatosFinal(PrediccionAbandono-DatosFinal).csv', sep=';', encoding='latin1')

# Limpieza de comas decimales
for col in ['PromedioPrimerCuatrimestre', 'AsistenciaPromedio(%)']:
    df_ent[col] = df_ent[col].astype(str).str.replace(',', '.').astype(float)
    df_final[col] = df_final[col].astype(str).str.replace(',', '.').astype(float)

# Preparación y codificación
df_ent['origen'] = 'train'
df_final['origen'] = 'test'
df_final['EstadoFinal'] = 'Desconocido'
combinado = pd.concat([df_ent, df_final], ignore_index=True)

combinado_encoded = pd.get_dummies(combinado, columns=['genero', 'carrera', 'trabaja/NoTrabaja', 'ActividadesExtracurriculares(Estudio)'], drop_first=True)

train_proc = combinado_encoded[combinado_encoded['origen'] == 'train'].drop(columns=['origen'])
test_proc = combinado_encoded[combinado_encoded['origen'] == 'test'].drop(columns=['origen', 'EstadoFinal'])

X = train_proc.drop(columns=['EstadoFinal'])
y = train_proc['EstadoFinal']

# Entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar árbol en imagen
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=['Abandonó', 'Continúa'], filled=True, rounded=True)
plt.savefig('../outputs/arbol_decision.png', dpi=300, bbox_inches='tight')
plt.close()
