import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Simulación del conjunto de datos (Dataset)
np.random.seed(42)
n_muestras = 1000

data = {
    'Antiguedad_Meses': np.random.randint(1, 60, size=n_muestras),
    'Interacciones_Plataforma': np.random.randint(5, 100, size=n_muestras),
    'Gasto_Promedio': np.random.uniform(20.0, 500.0, size=n_muestras),
    'Soporte_Tecnico_Usado': np.random.choice([0, 1], size=n_muestras, p=[0.7, 0.3])
}

df = pd.DataFrame(data)
# Variable objetivo (0: Sin riesgo de abandono/baja, 1: Con riesgo/éxito de conversión)
df['Objetivo'] = np.where((df['Interacciones_Plataforma'] > 50) & (df['Gasto_Promedio'] > 200), 1, 0)

# 2. Preparación de datos (Separación de variables predictoras X y etiqueta y)
X = df[['Antiguedad_Meses', 'Interacciones_Plataforma', 'Gasto_Promedio', 'Soporte_Tecnico_Usado']]
y = df['Objetivo']

# División en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Implementación y Entrenamiento del Modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train_scaled, y_train)

# 4. Evaluación de Resultados
y_pred = modelo.predict(X_test_scaled)

print("Exactitud del Modelo (Accuracy):", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))