import numpy as np
import pandas as pd

def leer_csv(ruta, separador=",", encoding="latin1") -> pd.DataFrame:
    df = pd.read_csv(ruta, sep=separador, encoding=encoding)
    return df

def seleccionar_filas(df, columna, condicion) -> pd.DataFrame:
    if callable(condicion):
        return df[df[columna].apply(condicion)]
    else:
        return df[df[columna] == condicion]

def seleccionar_columnas(df, columnas) -> pd.DataFrame:
    return df[columnas]

def columnas_categoricas(df) -> list:
    columnas_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return columnas_cat

def dominio_columna(df, columna) -> list:
    if isinstance(columna, str):
        serie = df[columna]
    else:
        serie = columna
    if pd.api.types.is_categorical_dtype(serie):
        return serie.cat.categories.tolist()

    return serie.dropna().unique().tolist()

def dominio_categorico(df, columna=None) -> dict:
    if columna is not None:
        return {columna: dominio_columna(df, columna)}

    columnas_cat = columnas_categoricas(df)
    return {col: dominio_columna(df, col) for col in columnas_cat}

def frecuencia_atributos(df, columna) -> dict:
    return df[columna].value_counts().to_dict()

def frecuencia_atributos_dado_clase(df, A) -> dict:
    columna_clase = df.columns[-1]  
    resultado_global = {}
    columnas_atributos = df.columns[:-1]  
    for col in columnas_atributos:
        resultado_columna = {}
        for valor in A[col]:
            sub_df = df[df[col] == valor]
            conteos = sub_df[columna_clase].value_counts().to_dict()
            resultado_columna[valor] = conteos
        resultado_global[col] = resultado_columna
    return resultado_global

def obtener_X(df) -> list:
    X = []
    for i in range(len(df)):
        fila = df.values[i]
        X.append(fila[:-1]) 
    return X

def obtener_y(df) -> list:
    y = []
    for i in range(len(df)):
        fila = df.values[i]
        y.append(fila[-1]) 
    return y

def separar_train_test(df, tam_test=0.2, semilla=None) -> list[pd.DataFrame, pd.DataFrame]:
    if semilla is not None:
        np.random.seed(semilla)

    indices = np.random.permutation(len(df))
    corte = int(len(df) * tam_test)

    indices_test = indices[:corte]
    indices_train = indices[corte:]

    df_train = df.iloc[indices_train].reset_index(drop=True)
    df_test = df.iloc[indices_test].reset_index(drop=True)

    return [df_train, df_test]

def matriz_confusion(y_true, y_pred)-> dict:
    TP = FP = TN = FN = 0

    for true, pred in zip(y_true, y_pred):
        if true == True and pred == True:
            TP += 1
        elif true == False and pred == True:
            FP += 1
        elif true == True and pred == False:
            FN += 1
        elif true == False and pred == False:
            TN += 1

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

def accuracy(y_true, y_pred)-> float:
    mc = matriz_confusion(y_true, y_pred)
    total = mc['TP'] + mc['FP'] + mc['TN'] + mc['FN']
    return (mc['TP'] + mc['TN']) / total if total > 0 else 0

def recall(y_true, y_pred)-> float:
    mc = matriz_confusion(y_true, y_pred)
    return mc['TP'] / (mc['TP'] + mc['FN']) if (mc['TP'] + mc['FN']) > 0 else 0

def especificidad(y_true, y_pred)-> float:
    mc = matriz_confusion(y_true, y_pred)
    return mc['TN'] / (mc['TN'] + mc['FP']) if (mc['TN'] + mc['FP']) > 0 else 0

def precision(y_true, y_pred)-> float   :
    mc = matriz_confusion(y_true, y_pred)
    return mc['TP'] / (mc['TP'] + mc['FP']) if (mc['TP'] + mc['FP']) > 0 else 0

def f1_score(y_true, y_pred)-> float:
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def tasa_verdaderos_positivos(y_true, y_pred)-> float:
    return recall(y_true, y_pred)

def tasa_falsos_positivos(y_true, y_pred)-> float:
    mc = matriz_confusion(y_true, y_pred)
    return mc['FP'] / (mc['FP'] + mc['TN']) 

def calcular_shannon(conteos) -> float:
    total = sum(conteos)
    if total == 0:
        return 0.0
    
    entropia = 0.0
    for c in conteos:
        if c > 0:
            p = c / total
            entropia -= p * np.log2(p)
    return entropia

def coeficiente_determinacion(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

def evaluar_clasificacion(y_true, y_pred) -> dict:
    return {
        "matriz_confusion": matriz_confusion(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

def contar_nulos(df) -> pd.Series:
    return df.isnull().sum()

def eliminar_duplicados(df) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def codificar_binaria(serie, valor_positivo) -> pd.Series:
    return (serie == valor_positivo).astype(int)

def obtener_entropia_total(df) -> float:
    columna_clase = df.columns[-1]
    totales_clase = df[columna_clase].value_counts().to_dict()
    return calcular_shannon(totales_clase.values())

def obtener_entropia_condicional(df, columna_attr, frecuencias_globales) -> float:
    total_general = len(df)
    entropia_condicional = 0.0
    
    frecuencias_del_atributo = frecuencias_globales[columna_attr]
    
    for valor, conteos_clases in frecuencias_del_atributo.items():
        total_valor = sum(conteos_clases.values())
        entropia_subgrupo = calcular_shannon(conteos_clases.values())
        entropia_condicional += (total_valor / total_general) * entropia_subgrupo
        
    return entropia_condicional

def calcular_ganancia_id3(df, columna_attr, frecuencias_globales) -> float:
    en_total = obtener_entropia_total(df)
    en_condicional = obtener_entropia_condicional(df, columna_attr, frecuencias_globales)
    return en_total - en_condicional

def obtener_muestra_bootstrap(df) -> pd.DataFrame:
    return df.sample(n=len(df), replace=True).reset_index(drop=True)
