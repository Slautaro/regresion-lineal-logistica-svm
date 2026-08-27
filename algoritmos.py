import herramientas
import numpy as np
import pandas as pd
from sklearn.svm import SVC

def find_s(df) -> list:
    X = herramientas.obtener_X(df)
    y = herramientas.obtener_y(df)

    h = [None] * len(X[0])
    for fila in range(len(X)):
        if y[fila] == True: 
            if h[0] is None:
                h = X[fila]  
            else:
                for atributo in range(len(h)):
                    if h[atributo] != X[fila][atributo]:
                        h[atributo] = "?"

    return h

def regresion_lineal_multiple(X, y) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    unos = np.ones((X.shape[0], 1))
    X_ext = np.hstack([unos, X])

    beta = np.linalg.inv(X_ext.T @ X_ext) @ X_ext.T @ y
    return beta

def predecir_regresion_lineal(beta, X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    unos = np.ones((X.shape[0], 1))
    X_ext = np.hstack([unos, X])
    return X_ext @ beta

def entrenar_svm(X_train, y_train, kernel="rbf", C=1.0) -> SVC:
    modelo = SVC(kernel=kernel, C=C)
    modelo.fit(X_train, y_train)
    return modelo

def predecir_y(h, X_test) -> list:
    y_pred = []

    for fila in range(len(X_test)):
        prediccion = True
        for atributo in range(len(h)): 
            if h[atributo] != "?" and h[atributo] != X_test[fila][atributo]:
                prediccion = False
                break
        y_pred.append(prediccion)

    return y_pred
def evaluar_casos_base(df, atributos):
    columna_clase = df.columns[-1]
    clases_unicas = df[columna_clase].unique()
    if len(clases_unicas) == 1:
        return clases_unicas[0]
    if len(atributos) == 0:
        return df[columna_clase].mode()[0]
    return None  

def buscar_mejor_atributo(df, atributos, dominio_global):
    frecuencias_actuales = herramientas.frecuencia_atributos_dado_clase(df, dominio_global)
    mejor_attr = None
    mejor_ganancia = -1.0
    for attr in atributos:
        ganancia = herramientas.calcular_ganancia_id3(df, attr, frecuencias_actuales)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_attr = attr
            
    return mejor_attr, mejor_ganancia

def crear_arbol_ID3(df, atributos, dominio_global):
    if df.empty:
        return "Dataset Vacío"
    resultado_base = evaluar_casos_base(df, atributos)
    if resultado_base is not None:
        return resultado_base
    mejor_attr, mejor_ganancia = buscar_mejor_atributo(df, atributos, dominio_global)
    if mejor_ganancia <= 0:
        return df[df.columns[-1]].mode()[0]
    arbol = {mejor_attr: {}}
    atributos_restantes = [a for a in atributos if a != mejor_attr]
    for valor in dominio_global[mejor_attr]:
        sub_df = df[df[mejor_attr] == valor]
        if sub_df.empty:
            arbol[mejor_attr][valor] = df[df.columns[-1]].mode()[0]
        else:
            arbol[mejor_attr][valor] = crear_arbol_ID3(sub_df, atributos_restantes, dominio_global)
            
    return arbol

def predecir_dataset(df, arbol, columna_clase):
    predicciones_texto = df.apply(lambda fila: predecir_fila(arbol, fila), axis=1)
    y_true = (df[columna_clase] == "OTORGADO").tolist()
    y_pred = (predicciones_texto == "OTORGADO").tolist()
    return y_true, y_pred

def evaluar_modelo_id3(df_test, arbol):
    columna_clase = df_test.columns[-1]
    y_true, y_pred = predecir_dataset(df_test, arbol, columna_clase)
    mc = herramientas.matriz_confusion(y_true, y_pred)
    acc = herramientas.accuracy(y_true, y_pred)
    f1 = herramientas.f1_score(y_true, y_pred)
    return mc, f1, acc

def predecir_fila(arbol, fila):
    if not isinstance(arbol, dict):
        return arbol
    atributo = list(arbol.keys())[0]
    valor_persona = fila[atributo]
    sub_arbol = arbol[atributo].get(valor_persona)
    if sub_arbol is None:
        return "RECHAZADO"
    return predecir_fila(sub_arbol, fila)


def predecir_dataset(df, arbol, columna_clase):
    predicciones_texto = df.apply(lambda fila: predecir_fila(arbol, fila), axis=1)
    
    y_true = (df[columna_clase] == "OTORGADO").tolist()
    y_pred = (predicciones_texto == "OTORGADO").tolist()
    
    return y_true, y_pred

def evaluar_modelo_id3(df_test, arbol):
    columna_clase = df_test.columns[-1]
    
    y_true, y_pred = predecir_dataset(df_test, arbol, columna_clase)
    
    mc = herramientas.matriz_confusion(y_true, y_pred)
    acc = herramientas.accuracy(y_true, y_pred)
    f1 = herramientas.f1_score(y_true, y_pred)
    
    return mc, f1, acc


def construir_bosque_rf(df_train, atributos, dominio_global, n_arboles, max_features):
    bosque = []
    for _ in range(n_arboles):
        df_muestra = df_train.sample(n=len(df_train), replace=True).reset_index(drop=True)
        atributos_sorteados = list(np.random.choice(atributos, size=max_features, replace=False))
        arbol = crear_arbol_ID3(df_muestra, atributos_sorteados, dominio_global)
        bosque.append(arbol)
    return bosque

def obtener_predicciones_bosque(df_test, bosque):
    predicciones = []
    for _, fila in df_test.iterrows():
        votos = [predecir_fila(arbol, fila) for arbol in bosque]
        predicciones.append(pd.Series(votos).mode()[0])
    return predicciones

def evaluar_random_forest(df_train, df_test, atributos, dominio_global, n_arboles=30):
    columna_clase = df_train.columns[-1]
    max_features = 4  
    if max_features > len(atributos): 
        max_features = len(atributos)
    bosque = construir_bosque_rf(df_train, atributos, dominio_global, n_arboles, max_features)
    predicciones = obtener_predicciones_bosque(df_test, bosque)
    y_true = (df_test[columna_clase] == "OTORGADO").tolist()
    y_pred = (pd.Series(predicciones) == "OTORGADO").tolist() 
    mc = herramientas.matriz_confusion(y_true, y_pred)
    acc = herramientas.accuracy(y_true, y_pred)
    f1 = herramientas.f1_score(y_true, y_pred)
    return mc, f1, acc