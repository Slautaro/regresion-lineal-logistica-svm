# Regresión lineal, regresión logística y SVM

TP3 de la materia Aprendizaje Automático. Compara tres enfoques sobre un dataset de rendimiento estudiantil: regresión lineal múltiple para predecir la calificación, y regresión logística vs. SVM (con distintos kernels y valores de C) para predecir si el estudiante aprueba.

## Contenido

- `tp3.py`: script principal, entrena y evalúa los tres modelos, y compara SVM contra regresión logística por F1-score.
- `algoritmos.py`: implementación de la regresión lineal múltiple (desde cero) y wrapper de entrenamiento de SVM.
- `herramientas.py`: funciones de preprocesamiento (lectura de CSV, nulos, duplicados, codificación binaria, train/test split) y de evaluación (R², matriz de confusión, accuracy, F1-score).
- `Estudiantes.csv`: dataset de rendimiento estudiantil usado para entrenar y evaluar los modelos.
- `TP3.pdf` / `TP3.pptx`: consigna y presentación del trabajo práctico.

## Resultado

- Regresión lineal múltiple: **R² = 0.60** sobre el conjunto de prueba para predecir la calificación.
- Regresión logística: **83% accuracy, F1 = 0.67** para predecir si el estudiante aprueba.
- Mejor SVM (kernel lineal, C=10): **85% accuracy, F1 = 0.71**, superando a la regresión logística.

## Cómo correrlo

```bash
pip install numpy pandas scikit-learn
python tp3.py
```
