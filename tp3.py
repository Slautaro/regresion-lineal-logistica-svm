import herramientas
import algoritmos as alg
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from itertools import product


D = herramientas.leer_csv("Estudiantes.csv") 

D = D.rename(columns={
    "Horas de estudio por semana": "Horas_estudio",
    "Porcentaje de asistencia": "Asistencia",
    "Calificaciones previas": "Calif_previa",
    "Nivel de educación de los padres": "Educ_padres",
    "Acceso a internet": "Internet",
    "Extracurricular_Activities": "Extracurricular",
    "Calificación": "Calificacion",
    "Condición": "Condicion",
})

print("Filas totales:", len(D))
print("Nulos por columna:\n", herramientas.contar_nulos(D))

D = herramientas.eliminar_duplicados(D)
print("Filas luego de eliminar duplicados:", len(D))

print(D[["Horas_estudio", "Asistencia", "Calif_previa", "Calificacion"]].describe())

D["Condicion_bin"] = herramientas.codificar_binaria(D["Condicion"], "Aprobado")

df_train, df_test = herramientas.separar_train_test(D, tam_test=0.2, semilla=42)
print("Train:", len(df_train), "Test:", len(df_test))

columnas_x = ["Horas_estudio", "Calif_previa", "Asistencia"]

X_train = df_train[columnas_x].values
y_train = df_train["Calificacion"].values
X_test = df_test[columnas_x].values
y_test = df_test["Calificacion"].values

beta = alg.regresion_lineal_multiple(X_train, y_train)
y_pred_reg = alg.predecir_regresion_lineal(beta, X_test)
r2 = herramientas.coeficiente_determinacion(y_test, y_pred_reg)

print("\nRegresión lineal múltiple")
print(f"Calificacion = {beta[0]:.4f} + {beta[1]:.4f}*Horas_estudio + {beta[2]:.4f}*Calif_previa + {beta[3]:.4f}*Asistencia")
print("R^2 (conjunto de prueba):", r2)

y_train_bin = df_train["Condicion_bin"].values
y_test_bin = df_test["Condicion_bin"].values

modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_train, y_train_bin)

beta0 = modelo_logistico.intercept_[0]
beta_log = modelo_logistico.coef_[0]

print("\nRegresión logística")
print(f"p(x) = exp({beta0:.4f} + {beta_log[0]:.4f}*Horas_estudio + {beta_log[1]:.4f}*Calif_previa + {beta_log[2]:.4f}*Asistencia)"
      " / (1 + exp(...))")

y_pred_log = modelo_logistico.predict(X_test)

resultado_log = herramientas.evaluar_clasificacion(y_test_bin, y_pred_log)
print("Matriz de confusión:", resultado_log["matriz_confusion"])
print("Accuracy:", resultado_log["accuracy"])
print("F1-score:", resultado_log["f1_score"])

nuevo_alumno = [[25, 68, 58]]

calificacion_predicha = alg.predecir_regresion_lineal(beta, nuevo_alumno)[0]
condicion_predicha = modelo_logistico.predict(nuevo_alumno)[0]
prob_aprobar = modelo_logistico.predict_proba(nuevo_alumno)[0][1]

print("\nPredicción para el estudiante nuevo (25 hs, 68 pts previos, 58% asistencia)")
print(f"Calificación estimada (regresión lineal): {calificacion_predicha:.2f}")
print(f"Condición estimada (regresión logística): {'Aprobado' if condicion_predicha == 1 else 'No aprobado'}"
      f" (probabilidad de aprobar: {prob_aprobar:.2%})")


kernels = ["linear", "rbf", "poly"]
valores_C = [0.1, 1, 10]

resultados_svm = []

print("\nSVM: comparación de kernels y valores de C")
for kernel, C in product(kernels, valores_C):
    modelo_svm = alg.entrenar_svm(X_train, y_train_bin, kernel=kernel, C=C)
    y_pred_svm = modelo_svm.predict(X_test)
    resultado = herramientas.evaluar_clasificacion(y_test_bin, y_pred_svm)

    resultados_svm.append({"kernel": kernel, "C": C, **resultado})
    print(f"kernel={kernel:6s} C={C:<4} -> "
          f"accuracy={resultado['accuracy']:.3f}  f1={resultado['f1_score']:.3f}  "
          f"matriz={resultado['matriz_confusion']}")

mejor_svm = max(resultados_svm, key=lambda r: r["f1_score"])

print("\nComparación final (conjunto de prueba)")
print(f"Regresión logística -> accuracy={resultado_log['accuracy']:.3f}  f1={resultado_log['f1_score']:.3f}")
print(f"Mejor SVM (kernel={mejor_svm['kernel']}, C={mejor_svm['C']}) -> "
      f"accuracy={mejor_svm['accuracy']:.3f}  f1={mejor_svm['f1_score']:.3f}")

if mejor_svm["f1_score"] > resultado_log["f1_score"]:
    print(f"Se recomienda el SVM con kernel={mejor_svm['kernel']} y C={mejor_svm['C']}, "
          f"porque logra mejor F1-score ({mejor_svm['f1_score']:.3f} contra {resultado_log['f1_score']:.3f} "
          "de la regresión logística) sobre el conjunto de prueba.")
else:
    print(f"Se recomienda la regresión logística, porque logra un F1-score igual o mejor "
          f"({resultado_log['f1_score']:.3f} contra {mejor_svm['f1_score']:.3f} del mejor SVM) "
          "y además da una probabilidad interpretable de aprobación, algo que el SVM no ofrece directamente.")

