# -*- coding: utf-8 -*-
"""Copia de ProyectyoIA-Con datos reales.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ol_4w-wkKYekGMdD_jmrZiG4pQu5zhI8
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from google.colab import files
import pandas as pd
import io
import ipywidgets as widgets
from IPython.display import display, clear_output

output = widgets.Output()

upload_button = widgets.Button(
    description='📤 Subir archivo CSV',
    layout=widgets.Layout(width='200px', height='40px'),
    button_style='primary'
)

def on_upload_clicked(b):
    clear_output()
    uploaded = files.upload()
    for nombre_archivo in uploaded.keys():
        df = pd.read_csv(io.BytesIO(uploaded[nombre_archivo]))
        print(f"✅ Archivo cargado: {nombre_archivo}")
        print(f"📊 Dimensiones del DataFrame: {df.shape}")
        print(df.head())
        globals()['df'] = df  # hace df accesible globalmente

upload_button.on_click(on_upload_clicked)

display(upload_button, output)

np.random.seed(42)
num_samples = 100

datos = {
    "horas_sueno": np.random.normal(6.5, 1.0, num_samples).clip(3, 10),
    "tareas_tarde": np.random.poisson(2, num_samples).clip(0, 10),
    "horas_pantalla": np.random.normal(7.5, 1.5, num_samples).clip(2, 14),
    "descansos_dia": np.random.randint(0, 6, num_samples),
    "estres_auto": np.random.randint(1, 6, num_samples),
    "actividades_extras": np.random.randint(0, 4, num_samples),
    "interaccion_social": np.random.randint(1, 6, num_samples),
    "cambios_animo": np.random.randint(1, 6, num_samples)
}

df = pd.DataFrame(datos)

def clasificar_burnout(row):
    score = (
        (6 - row["horas_sueno"]) * 1.2 +
        row["tareas_tarde"] * 0.7 +
        (row["horas_pantalla"] - 6) * 0.8 +
        (5 - row["descansos_dia"]) * 0.6 +
        row["estres_auto"] * 1.5 +
        (5 - row["interaccion_social"]) * 0.5 +
        row["cambios_animo"] * 1.0
    )
    if score < 6:
        return 0
    elif score < 10:
        return 1
    elif score < 14:
        return 2
    else:
        return 3

df["nivel_burnout"] = df.apply(clasificar_burnout, axis=1)
df.head()

X = df.drop("nivel_burnout", axis=1).values
y = df["nivel_burnout"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_cat = to_categorical(y, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dense(12, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=200, batch_size=8, validation_split=0.2,
                    callbacks=[early_stop], verbose=0)


loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión del modelo: {acc:.2f}")

# Evaluación detallada del modelo
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print("Matriz de confusión:")
print(confusion_matrix(y_true, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred))

df_nuevos = df.copy()  # o usa df directamente

# Drop the target variable column from the new data before scaling
X_nuevos = scaler.transform(df_nuevos.drop("nivel_burnout", axis=1))
predicciones = model.predict(X_nuevos)
niveles_predichos = np.argmax(predicciones, axis=1)

niveles = ["Sin riesgo", "Riesgo leve", "Riesgo moderado", "Riesgo alto"]
print("\nResultados de predicción:")
for i, nivel in enumerate(niveles_predichos):
    print(f"Estudiante {i+1}: {niveles[nivel]}")

import matplotlib.pyplot as plt
import collections

# Contar cuántos estudiantes hay por nivel
conteo_niveles = collections.Counter(niveles_predichos)

# Definir etiquetas legibles
etiquetas = ["Sin riesgo", "Riesgo leve", "Riesgo moderado", "Riesgo alto"]
valores = [conteo_niveles[i] for i in range(4)]

# Crear gráfico de barras
plt.figure(figsize=(8, 5))
plt.bar(etiquetas, valores, color='lightblue', edgecolor='black')
plt.title("Distribución de niveles de burnout")
plt.xlabel("Nivel de burnout")
plt.ylabel("Número de estudiantes")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()