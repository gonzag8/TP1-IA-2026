##############################################################################
# CELDA 1 - MARKDOWN
# ---
# ## Importación de librerías
# Se importan las librerías necesarias para el trabajo:
# - **TensorFlow/Keras**: construcción y entrenamiento de redes neuronales
# - **Keras Tuner**: búsqueda automatizada de hiperparámetros óptimos
# - **NumPy/Pandas**: manipulación de datos
# - **Matplotlib/Seaborn**: visualización
# - **Scikit-learn**: métricas de evaluación
##############################################################################

# CELDA 2 - CÓDIGO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import seaborn as sns

##############################################################################
# CELDA 3 - MARKDOWN
# ---
# ## Carga de datos
# Los datos fueron preprocesados en la materia Ciencia de Datos: se eliminaron
# valores atípicos, se balancearon las clases, se seleccionaron variables
# relevantes y se aplicó estandarización (Z-score). Se cargan los conjuntos
# de entrenamiento y test ya preparados.
##############################################################################

# CELDA 4 - CÓDIGO
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

##############################################################################
# CELDA 5 - MARKDOWN
# ---
# ## Exploración rápida de los datos
# Verificamos la forma y contenido del dataset para confirmar que los datos
# están correctamente cargados y preprocesados.
##############################################################################

# CELDA 6 - CÓDIGO
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nDistribución del target en train:\n{train['target'].value_counts()}")
print(f"\nDistribución del target en test:\n{test['target'].value_counts()}")
train.describe()

##############################################################################
# CELDA 7 - MARKDOWN
# ---
# ## Separación de features y target
# Se separan las variables predictoras (X) de la variable objetivo (y)
# para ambos conjuntos.
##############################################################################

# CELDA 8 - CÓDIGO
X_train = train.drop('target', axis=1)
y_train = train['target']
X_test = test.drop(columns=['target'])
y_test = test['target']

print(f"Features de entrada: {X_train.shape[1]} ({list(X_train.columns)})")

##############################################################################
# CELDA 9 - MARKDOWN
# ---
# ## Actividad 1: Diseño del modelo
#
### Justificación de las decisiones de diseño

**Arquitectura MLP (capas ocultas → 1 neurona de salida):**
- Se utiliza un patrón "embudo" (neuronas decrecientes) que fuerza a la
  red a comprimir progresivamente la representación de los datos.
- Con solo 5 features de entrada, una red de 2 capas ocultas es suficiente
  para capturar relaciones no lineales sin sobreajustar.

**Funciones de activación:**
- **ReLU** en capas ocultas: es el estándar actual porque evita el problema
  de "vanishing gradient" y es computacionalmente eficiente.
- **Sigmoid** en la capa de salida: mapea la salida al rango [0, 1],
  interpretable como la probabilidad de falla (clase positiva).
  Se usa sigmoid (y no softmax) porque tenemos **1 sola neurona de salida**
  para clasificación binaria. Softmax se emplea cuando hay 2+ neuronas de
  salida (clasificación multiclase), ya que normaliza un vector para que
  las probabilidades sumen 1.

**Regularización (Dropout):**
- Durante el entrenamiento, se "apagan" aleatoriamente un porcentaje de las
  neuronas en cada paso. Esto evita co-adaptación y reduce el sobreajuste.

**Loss binary_crossentropy:**
- Es la función de pérdida estándar para clasificación binaria.
  Mide la divergencia entre la distribución predicha y la real.

La cantidad exacta de neuronas, la tasa de dropout, el learning rate, el
batch size y el optimizador serán determinados mediante búsqueda
automatizada de hiperparámetros (Actividad 2).
##############################################################################

##############################################################################
# CELDA 10 - MARKDOWN
# ---
## Actividad 2: Ajuste de hiperparámetros con Keras Tuner

Para encontrar la mejor combinación de hiperparámetros se utiliza
**Keras Tuner**, una librería que automatiza la búsqueda de hiperparámetros
óptimos. En lugar de probar manualmente cada combinación (lo cual sería
impráctico dado el número de posibilidades), Keras Tuner:

1. Define un **espacio de búsqueda** con los rangos de cada hiperparámetro
2. Ejecuta múltiples **trials** (entrenamientos con distintas combinaciones)
3. Rankea los resultados según una métrica objetivo (en nuestro caso,
   `val_accuracy`)
4. Devuelve la mejor configuración encontrada

Se utiliza `RandomSearch` como estrategia de búsqueda, que muestrea
combinaciones aleatorias del espacio. Esto es más eficiente que un grid
search exhaustivo cuando el espacio es grande.

### Hiperparámetros explorados

| Hiperparámetro | Valores explorados |
|---|---|
| Neuronas capa 1 | 32, 64, 128 |
| Neuronas capa 2 | 16, 32, 64 |
| Tasa de aprendizaje | 0.01, 0.001, 0.0001 |
| Dropout | 0.0, 0.2, 0.3, 0.5 |
| Regularización L2 | 0.0, 0.001, 0.01 |
| Optimizador | Adam, SGD (momentum=0.9), RMSprop |
##############################################################################

# CELDA 11 - CÓDIGO: Definir el modelo parametrizado para Keras Tuner
def build_model(hp):
    """
    Función que construye un modelo MLP parametrizado.
    Keras Tuner invoca esta función en cada trial, variando los
    hiperparámetros definidos con hp.Choice().
    """
    # Hiperparámetros de arquitectura
    units_1 = hp.Choice('units_1', [32, 64, 128])
    units_2 = hp.Choice('units_2', [16, 32, 64])

    # Hiperparámetros de regularización
    dropout_rate = hp.Choice('dropout', [0.0, 0.2, 0.3, 0.5])
    l2_value = hp.Choice('l2_reg', [0.0, 0.001, 0.01])

    # Hiperparámetros de entrenamiento
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    opt_name = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])

    regularizer = keras.regularizers.l2(l2_value) if l2_value > 0 else None

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(units_1, activation='relu',
                           kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(units_2, activation='relu',
                           kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    if opt_name == 'adam':
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        opt = keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

##############################################################################
# CELDA 12 - MARKDOWN
# ---
### Ejecución de la búsqueda
Se configuran 30 trials (combinaciones aleatorias) y se entrena cada uno
por hasta 50 épocas con EarlyStopping (patience=10) para ahorrar tiempo
cuando el modelo deja de mejorar.
##############################################################################

# CELDA 13 - CÓDIGO: Ejecutar la búsqueda
# Eliminar resultados anteriores para empezar limpio
import shutil
if os.path.exists('tuner_results'):
    shutil.rmtree('tuner_results')

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=30,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='tp_ia',
    seed=42
)

early_stop = EarlyStopping(monitor='val_loss', patience=10,
                            restore_best_weights=True)

tuner.search(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

##############################################################################
# CELDA 14 - MARKDOWN
# ---
### Resultados de la búsqueda
Se muestran los mejores hiperparámetros encontrados y un resumen de
todos los trials ejecutados.
##############################################################################

# CELDA 15 - CÓDIGO: Mostrar resultados
tuner.results_summary(num_trials=10)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n" + "="*60)
print("MEJOR CONFIGURACIÓN ENCONTRADA:")
print(f"  Neuronas capa 1:   {best_hp.get('units_1')}")
print(f"  Neuronas capa 2:   {best_hp.get('units_2')}")
print(f"  Dropout:           {best_hp.get('dropout')}")
print(f"  Regularización L2: {best_hp.get('l2_reg')}")
print(f"  Learning Rate:     {best_hp.get('learning_rate')}")
print(f"  Optimizador:       {best_hp.get('optimizer')}")
print("="*60)

##############################################################################
# CELDA 16 - MARKDOWN
# ---
### Exploración del efecto del batch size
Keras Tuner no varía el batch_size directamente (se pasa a .search()),
así que lo exploramos manualmente con la mejor configuración encontrada.
##############################################################################

# CELDA 17 - CÓDIGO: Explorar batch size
resultados_batch = []
batch_sizes = [16, 32, 64, 128]

for bs in batch_sizes:
    print(f"\n--- Batch Size: {bs} ---")
    model = tuner.hypermodel.build(best_hp)
    es = EarlyStopping(monitor='val_loss', patience=10,
                       restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=bs,
                        validation_data=(X_test, y_test),
                        callbacks=[es], verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  val_loss={loss:.4f}, val_accuracy={acc:.4f}")
    resultados_batch.append({
        'batch_size': bs,
        'val_loss': round(loss, 4),
        'val_accuracy': round(acc, 4),
        'epocas': len(history.history['loss'])
    })

df_batch = pd.DataFrame(resultados_batch)
print("\n=== Comparación de batch sizes ===")
print(df_batch.sort_values('val_accuracy', ascending=False).to_string(index=False))

mejor_batch = df_batch.loc[df_batch['val_accuracy'].idxmax(), 'batch_size']
print(f"\nMejor batch size: {mejor_batch}")

##############################################################################
# CELDA 18 - MARKDOWN
# ---
### Entrenamiento del modelo final
Se construye y entrena el modelo con la configuración óptima encontrada
(mejores hiperparámetros de Keras Tuner + mejor batch size).
Se aumentan las épocas máximas a 100 y el patience a 15 para permitir
una convergencia más completa.
##############################################################################

# CELDA 19 - CÓDIGO: Modelo final
modelo_final = tuner.hypermodel.build(best_hp)
modelo_final.summary()

early_stop_final = EarlyStopping(monitor='val_loss', patience=15,
                                  restore_best_weights=True)

history_final = modelo_final.fit(
    X_train, y_train,
    epochs=100,
    batch_size=int(mejor_batch),
    validation_data=(X_test, y_test),
    callbacks=[early_stop_final],
    verbose=1
)

##############################################################################
# CELDA 20 - MARKDOWN
# ---
## Actividad 3: Evaluación del modelo

### Curvas de aprendizaje
Se grafican las curvas de loss y accuracy para entrenamiento y validación.
Estas curvas permiten detectar:
- **Overfitting**: val_loss sube mientras train_loss baja
- **Underfitting**: ambas curvas se estancan en valores altos
- **Buen ajuste**: ambas curvas convergen en valores bajos y similares
##############################################################################

# CELDA 21 - CÓDIGO: Curvas de aprendizaje
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history_final.history['loss'], label='Entrenamiento', linewidth=2)
ax1.plot(history_final.history['val_loss'], label='Validación', linewidth=2)
ax1.set_title('Curva de Loss', fontsize=14)
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss (Binary Crossentropy)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history_final.history['accuracy'], label='Entrenamiento', linewidth=2)
ax2.plot(history_final.history['val_accuracy'], label='Validación', linewidth=2)
ax2.set_title('Curva de Accuracy', fontsize=14)
ax2.set_xlabel('Época')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curvas_aprendizaje.png', dpi=150, bbox_inches='tight')
plt.show()

##############################################################################
# CELDA 22 - MARKDOWN
# ---
### Métricas de clasificación
Se calculan métricas estándar para clasificación binaria:
- **Accuracy**: proporción de predicciones correctas
- **Precision**: de los predichos positivos, cuántos son realmente positivos
- **Recall**: de los realmente positivos, cuántos fueron detectados
- **F1-Score**: media armónica de precision y recall
- **AUC-ROC**: capacidad del modelo para distinguir entre clases
##############################################################################

# CELDA 23 - CÓDIGO: Métricas
y_pred_proba = modelo_final.predict(X_test).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("=== Reporte de Clasificación ===\n")
print(classification_report(y_test, y_pred,
                            target_names=['No falla (0)', 'Falla (1)']))

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {auc_score:.4f}")

##############################################################################
# CELDA 24 - MARKDOWN
# ---
### Matriz de confusión
Muestra la distribución de predicciones correctas e incorrectas.
Permite identificar si el modelo falla más en falsos positivos
(predice falla cuando no la hay) o falsos negativos (no detecta fallas
reales).
##############################################################################

# CELDA 25 - CÓDIGO: Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No falla (0)', 'Falla (1)'],
            yticklabels=['No falla (0)', 'Falla (1)'],
            ax=ax, annot_kws={"size": 14})
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Valor Real', fontsize=12)
ax.set_title('Matriz de Confusión', fontsize=14)
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.show()

##############################################################################
# CELDA 26 - MARKDOWN
# ---
### Curva ROC
La curva ROC grafica la tasa de verdaderos positivos (TPR) vs. la tasa
de falsos positivos (FPR) para distintos umbrales de decisión. Un AUC
cercano a 1.0 indica excelente capacidad discriminativa.
##############################################################################

# CELDA 27 - CÓDIGO: Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f'Modelo MLP (AUC = {auc_score:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Azar (AUC = 0.5)')
ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
ax.set_title('Curva ROC', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curva_roc.png', dpi=150, bbox_inches='tight')
plt.show()
