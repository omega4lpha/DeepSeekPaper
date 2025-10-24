"""
===============================================================================
Proyecto:    Clasificación vía LLM sobre datos tabulares (modo 100% consola)
Archivo:     app.py
Autor:       (tu nombre)
Descripción:
    Ejecuta un pipeline reproducible que:
      1) Carga y valida dos archivos CSV: 'clases' y 'datos' (separador ';').
      2) Prepara una muestra reproducible (150 primeras + 150 últimas filas)
         y separa entrenamiento/prueba (60/40).
      3) Genera un prompt con ejemplos etiquetados y casos a clasificar.
      4) Invoca un LLM (DeepSeek vía cliente OpenAI-compatible) a distintas
         temperaturas y parsea las predicciones devueltas.
      5) Calcula métricas de evaluación (accuracy, kappa, AUC opcional),
         matriz de confusión, etiquetas y tiempos por fase.
      6) Persiste TODOS los artefactos en un CSV acumulativo (una fila por corrida),
         serializando objetos complejos en JSON (reportes, matrices, tablas, etc.).

Requisitos:
    - Python 3.9+
    - Paquetes: pandas, scikit-learn, openai
    - Variable de entorno: DEEPSEEK_API_KEY (clave para el endpoint DeepSeek)

Notas:
    - Los CSV de entrada deben estar separados por ';' y sin cabecera. La primera
      columna de ambos archivos se descarta como supuesto ID externo.
    - Por defecto, el script buscará 'clases' y 'datos' con o sin extensión '.csv'.
    - El archivo de salida es 'Resuldatos.csv' y se ANEXA en corridas sucesivas.
    - Las columnas JSON incluyen: report_json, conf_matrix_json, labels_json,
      resultados_json, tiempos_json y ancho_banda_json.
    - Si Excel/PQ está en configuración regional con coma decimal, se recomienda
      usar separador ';' al importar el CSV para que los decimales 0.2/0.5/0.8
      no se trunquen a 2/5/8.

Ejemplos de uso (PowerShell / CMD en Windows):
    # Tres corridas por cada temperatura por defecto (0.2, 0.5, 0.8)
    py app.py --clases clases --datos datos --out Resuldatos.csv

    # Temperaturas explícitas y 3 repeticiones por temperatura
    py app.py --clases clases --datos datos --out Resuldatos.csv --temps 0.2 0.5 0.8 --reps 3

Ejemplos de uso (bash en Linux/macOS):
    export DEEPSEEK_API_KEY="sk-..."
    python3 app.py --clases clases --datos datos --out Resuldatos.csv
    python3 app.py --clases clases --datos datos --out Resuldatos.csv --temps 0.2 0.5 0.8 --reps 3
===============================================================================
"""

import os
import time
import uuid
import json
import argparse
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, cohen_kappa_score
)
from openai import OpenAI


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def resolve_path(p: str) -> str:
    """
    Resuelve la ruta de archivo. Si no existe exactamente como se entregó,
    intenta con sufijo '.csv'. Lanza FileNotFoundError si no existe.

    Parámetros
    ----------
    p : str
        Ruta o nombre base del archivo (con o sin '.csv').

    Retorna
    -------
    str
        Ruta existente al archivo.

    Excepciones
    -----------
    FileNotFoundError
        Si no se localiza el archivo ni con ni sin sufijo '.csv'.
    """
    if os.path.isfile(p):
        return p
    if not p.lower().endswith(".csv") and os.path.isfile(p + ".csv"):
        return p + ".csv"
    raise FileNotFoundError(f"No se encontró el archivo: {p}")


def require_api_key(env_name: str = "DEEPSEEK_API_KEY") -> str:
    """
    Obtiene la API key requerida desde variable de entorno.

    Parámetros
    ----------
    env_name : str
        Nombre de la variable de entorno a consultar.

    Retorna
    -------
    str
        Valor de la API key.

    Excepciones
    -----------
    ValueError
        Si la variable de entorno no está definida.
    """
    key = os.getenv(env_name)
    if not key:
        raise ValueError(f"Variable de entorno {env_name} no está definida.")
    return key


# -----------------------------------------------------------------------------
# Cliente LLM
# -----------------------------------------------------------------------------

def get_client() -> OpenAI:
    """
    Construye el cliente OpenAI configurado para DeepSeek (endpoint compatible).

    Retorna
    -------
    OpenAI
        Cliente inicializado con base_url y api_key.
    """
    api_key = require_api_key("DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# -----------------------------------------------------------------------------
# Procesamiento de datos
# -----------------------------------------------------------------------------

def procesar_archivos(clases_path: str, datos_path: str) -> pd.DataFrame:
    """
    Lee, valida y consolida los archivos de clases y datos.

    Flujo:
      1) Carga ambos CSV separados por ';' y sin cabecera.
      2) Verifica que cada archivo tenga al menos 2 columnas.
      3) Descarta la primera columna (ID/índice externo) en ambos.
      4) Verifica que el número de filas coincida entre 'clases' y 'datos'.
      5) Estandariza los nombres de columnas de atributos a d1..dn.
      6) Concatena 'clase' + atributos y genera una muestra reproducible:
         150 primeras + 150 últimas filas (si existen), barajada con semilla fija.

    Parámetros
    ----------
    clases_path : str
        Ruta al CSV con las clases por fila.
    datos_path : str
        Ruta al CSV con los atributos por fila.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columna 'clase' seguida por d1..dn.
    """
    df_clases = pd.read_csv(clases_path, header=None, sep=";")
    df_datos = pd.read_csv(datos_path, header=None, sep=";")

    if df_clases.shape[1] < 2:
        raise ValueError("El archivo de clases debe tener al menos 2 columnas.")
    if df_datos.shape[1] < 2:
        raise ValueError("El archivo de datos debe tener al menos 2 columnas.")

    # Se descarta la primera columna (id/índice externo).
    df_clases = df_clases.iloc[:, 1:]
    df_datos = df_datos.iloc[:, 1:]

    if df_clases.shape[0] != df_datos.shape[0]:
        raise ValueError("Los archivos no tienen la misma cantidad de filas.")

    # Estandariza nombres de columnas de datos.
    df_datos.columns = [f'd{i+1}' for i in range(df_datos.shape[1])]
    df_final = pd.concat([df_clases, df_datos], axis=1)
    df_final.columns = ['clase'] + df_datos.columns.tolist()

    # Muestra reproducible: 150 primeras + 150 últimas; si hay menos, usa lo disponible.
    head = df_final.head(150)
    tail = df_final.tail(150)
    df_sample = pd.concat([head, tail]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_sample


# -----------------------------------------------------------------------------
# Prompting y parsing
# -----------------------------------------------------------------------------

def generate_prompt(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> str:
    """
    Construye el prompt con ejemplos etiquetados y casos a clasificar.

    Diseño:
      - Lista ejemplos de entrenamiento como "idx: f1, f2, ... -> clase".
      - Solicita clasificar el set de prueba con el formato "idx: clase".
      - Ese formato facilita un parseo determinista posterior.

    Parámetros
    ----------
    X_train : pd.DataFrame
        Atributos de entrenamiento.
    X_test : pd.DataFrame
        Atributos a clasificar.
    y_train : pd.Series
        Etiquetas correspondientes al entrenamiento.

    Retorna
    -------
    str
        Prompt listo para enviar al LLM.
    """
    prompt = "Estas son muestras del conjunto de entrenamiento con sus clases:\n"
    for i, row in X_train.iterrows():
        features = ", ".join([str(round(val, 2)) for val in row.values])
        prompt += f"{i}: {features} -> {y_train[i]}\n"

    prompt += "\nClasifica las siguientes muestras (solo pon 1 o 0 según similitud):\n"
    for i, row in X_test.iterrows():
        features = ", ".join([str(round(val, 2)) for val in row.values])
        prompt += f"{i}: {features}\n"

    prompt += "\nResponde en el formato:\n0: clase\n1: clase\n..."
    return prompt


def call_openai(prompt: str, temperature: float, client: OpenAI) -> str:
    """
    Invoca el modelo con la temperatura indicada y devuelve el texto de la primera elección.

    Parámetros
    ----------
    prompt : str
        Instrucción generada.
    temperature : float
        Temperatura de muestreo del LLM.
    client : OpenAI
        Cliente ya inicializado.

    Retorna
    -------
    str
        Respuesta textual del LLM (única mejor elección).
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Eres un clasificador de datos tabulares."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


def parse_predictions(response: str) -> Dict[int, str]:
    """
    Parsea líneas con el patrón 'indice: etiqueta' a un diccionario índice->etiqueta.

    Comportamiento:
      - Ignora líneas sin ':' o con índices no enteros.
      - Normaliza la etiqueta (minúsculas y sin espacios exteriores).

    Parámetros
    ----------
    response : str
        Respuesta del LLM en texto plano.

    Retorna
    -------
    Dict[int, str]
        Mapa de índice a etiqueta predicha.
    """
    preds: Dict[int, str] = {}
    for line in response.strip().splitlines():
        if ':' in line:
            key, val = line.split(':', 1)
            try:
                idx = int(key.strip())
                preds[idx] = val.strip().lower()
            except ValueError:
                continue
    return preds


# -----------------------------------------------------------------------------
# Pipeline de evaluación
# -----------------------------------------------------------------------------

def evaluar_llm_sobre_archivos(
    clases_path: str,
    datos_path: str,
    temperature: float,
    client: OpenAI
) -> Dict:
    """
    Ejecuta el pipeline completo para una temperatura dada y devuelve
    un diccionario con todos los artefactos requeridos para persistencia.

    Retorna
    -------
    Dict
        Estructura serializable con: métricas, matriz de confusión, etiquetas,
        tiempos por fase, ancho de banda estimado, prompt y respuesta, tamaños
        de entrenamiento/prueba y conteos de bytes enviados/recibidos.
    """
    start_total = time.time()

    # Fase 1: Procesamiento
    start_pre = time.time()
    df_final = procesar_archivos(clases_path, datos_path)

    # Split reproducible 60/40
    X = df_final.iloc[:, 1:]
    y = df_final.iloc[:, 0]
    train_df = df_final.sample(frac=0.6, random_state=42).reset_index(drop=True)
    test_df = df_final.drop(train_df.index).reset_index(drop=True)
    X_train = train_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]
    y_test = test_df.iloc[:, 0]
    end_pre = time.time()

    # Fase 2: Prompt
    start_prompt = time.time()
    prompt = generate_prompt(X_train, X_test, y_train)
    approx_tokens = len(prompt.split())
    input_bytes = len(prompt.encode('utf-8'))
    end_prompt = time.time()

    # Fase 3: LLM
    start_llm = time.time()
    response = call_openai(prompt, temperature=temperature, client=client)
    output_bytes = len(response.encode('utf-8'))
    end_llm = time.time()

    ancho_banda = {
        "enviados_kb": round(input_bytes / 1024, 2),
        "recibidos_kb": round(output_bytes / 1024, 2),
        "total_kb": round((input_bytes + output_bytes) / 1024, 2)
    }

    # Fase 4: Evaluación
    start_eval = time.time()
    preds_dict = parse_predictions(response)
    pred_labels: List[str] = [preds_dict.get(i, 'no_detectado') for i in range(len(X_test))]
    y_test_list: List[str] = [str(label).strip().lower() for label in y_test]

    report = classification_report(y_test_list, pred_labels, output_dict=True)
    conf_matrix = confusion_matrix(y_test_list, pred_labels)
    labels = sorted(list(set(y_test_list + pred_labels)))
    accuracy = accuracy_score(y_test_list, pred_labels)
    kappa = cohen_kappa_score(y_test_list, pred_labels)

    # AUC binario opcional: mapeo de clases a {0,1}. Si falla (multiclase/no mapeable), se omite.
    try:
        y_true_bin = [1 if val in ['1', 1, 'true'] else 0 for val in y_test_list]
        y_pred_bin = [1 if val in ['1', 1, 'true'] else 0 for val in pred_labels]
        auc: Optional[float] = roc_auc_score(y_true_bin, y_pred_bin)
    except Exception:
        auc = None

    # Resultados por muestra (índice, clase real, predicción)
    df_results = pd.DataFrame({
        "indice": range(len(X_test)),
        "clase_real": y_test_list,
        "prediccion_llm": pred_labels
    })
    end_eval = time.time()

    # Consolidación de tiempos
    end_total = time.time()
    tiempos = {
        "procesamiento_s": end_pre - start_pre,
        "generacion_prompt_s": end_prompt - start_prompt,
        "llamada_llm_s": end_llm - start_llm,
        "evaluacion_s": end_eval - start_eval,
        "total_s": end_total - start_total
    }

    # Empaquetado: listo para persistir en CSV (una fila por corrida)
    payload = {
        "run_id": str(uuid.uuid4()),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "temperature": temperature,
        "approx_tokens": approx_tokens,
        "accuracy": accuracy,
        "kappa": kappa,
        "auc": auc if auc is not None else "",
        "labels_json": json.dumps(labels, ensure_ascii=False),
        "conf_matrix_json": json.dumps(conf_matrix.tolist(), ensure_ascii=False),
        "report_json": json.dumps(report, ensure_ascii=False),
        "resultados_json": df_results.to_json(orient="records", force_ascii=False),
        "prompt": prompt,
        "response": response,
        "tiempos_json": json.dumps(tiempos, ensure_ascii=False),
        "ancho_banda_json": json.dumps(ancho_banda, ensure_ascii=False),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "bytes_enviados": input_bytes,
        "bytes_recibidos": output_bytes,
        "bytes_totales": input_bytes + output_bytes
    }
    return payload


# -----------------------------------------------------------------------------
# Persistencia en CSV acumulativo
# -----------------------------------------------------------------------------

def append_dict_to_csv(row: Dict, out_csv: str) -> None:
    """
    Anexa una fila (dict) a un CSV acumulativo. Crea el archivo con cabeceras
    si no existe. Por defecto utiliza coma como delimitador.

    Sugerencia:
      Si Excel/PQ está en configuración regional con coma decimal, conviene
      importar este CSV declarando el delimitador como ';' o, alternativamente,
      modificar este método para usar sep=';' y escribir en un archivo distinto.
    """
    df = pd.DataFrame([row])
    file_exists = os.path.isfile(out_csv)
    df.to_csv(out_csv, mode='a', header=not file_exists, index=False, encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    """
    Punto de entrada CLI. Lanza N repeticiones por cada temperatura indicada,
    ejecuta el pipeline y anexa los resultados al CSV acumulativo.

    Ejemplos:
        py app.py --clases clases --datos datos --out Resuldatos.csv
        py app.py --clases clases --datos datos --out Resuldatos.csv --temps 0.2 0.5 0.8 --reps 3
    """
    parser = argparse.ArgumentParser(
        description="Automatiza corridas LLM y guarda resultados en un CSV acumulativo."
    )
    parser.add_argument(
        "--clases", default="clases",
        help="Ruta al CSV de clases (con o sin .csv). Por defecto: 'clases'"
    )
    parser.add_argument(
        "--datos", default="datos",
        help="Ruta al CSV de datos (con o sin .csv). Por defecto: 'datos'"
    )
    parser.add_argument(
        "--out", default="Resuldatos.csv",
        help="CSV acumulativo de salida. Por defecto: Resuldatos.csv"
    )
    parser.add_argument(
        "--temps", nargs="*", type=float, default=[0.2, 0.5, 0.8],
        help="Lista de temperaturas. Por defecto: 0.2 0.5 0.8"
    )
    parser.add_argument(
        "--reps", type=int, default=3,
        help="Repeticiones por temperatura. Por defecto: 3"
    )
    args = parser.parse_args()

    # Resolución robusta de rutas (acepta nombres con o sin '.csv').
    clases_path = resolve_path(args.clases)
    datos_path = resolve_path(args.datos)

    # Cliente LLM
    client = get_client()

    # Ejecución de corridas y persistencia
    total_runs = 0
    for t in args.temps:
        for r in range(args.reps):
            payload = evaluar_llm_sobre_archivos(
                clases_path, datos_path, temperature=t, client=client
            )
            append_dict_to_csv(payload, args.out)
            total_runs += 1
            print(f"[OK] Temp={t} rep={r+1}/{args.reps} -> run_id={payload['run_id']} guardado en {args.out}")

    print(f"Completado: {total_runs} corridas. Resultados acumulados en {args.out}")


if __name__ == "__main__":
    main()
