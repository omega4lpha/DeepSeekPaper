"""
===============================================================================
Proyecto:    Clasificación vía LLM sobre datos tabulares (modo 100% consola)
Archivo:     app.py
Autor:       Boris Herrera Flores
Descripción:
    Ejecuta un pipeline reproducible que:
      1) Carga y valida dos archivos CSV: 'clases' y 'datos' (separador ';').
      2) Prepara una muestra reproducible (150 primeras + 150 últimas filas)
         y separa entrenamiento/prueba (60/40).
      3) Genera un prompt con ejemplos etiquetados y casos a clasificar.
      4) Invoca un LLM (DeepSeek vía cliente OpenAI-compatible) a distintas
         temperaturas y parsea las predicciones devueltas.
      5) Calcula métricas de evaluación (accuracy, kappa, AUC, curva ROC),
         matriz de confusión, etiquetas y tiempos por fase.
      6) Persiste TODOS los artefactos en un CSV acumulativo (una fila por corrida),
         serializando objetos complejos en JSON (reportes, matrices, tablas, etc.).

Requisitos:
    - Python 3.9+
    - Paquetes: pandas, scikit-learn, openai
    - Variable de entorno: DEEPSEEK_API_KEY

Notas:
    - Los CSV de entrada deben estar separados por ';' y sin cabecera. La primera
      columna de ambos archivos se descarta como supuesto ID externo.
    - Por defecto, el script buscará 'clases' y 'datos' con o sin extensión '.csv'.
    - El archivo de salida es 'Resuldatos.csv' y se ANEXA en corridas sucesivas.
    - Columnas JSON: report_json, conf_matrix_json, labels_json,
      resultados_json, tiempos_json, ancho_banda_json, roc_curve_json.
    - resultados_json ahora incluye prob_llm (probabilidad prevista de clase=1).
===============================================================================
"""

import os
import time
import uuid
import json
import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    cohen_kappa_score
)
from openai import OpenAI


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def resolve_path(p: str) -> str:
    """
    Resuelve la ruta de archivo. Si no existe como se entregó
    intenta con sufijo '.csv'. Lanza FileNotFoundError si no existe.
    """
    if os.path.isfile(p):
        return p
    if not p.lower().endswith(".csv") and os.path.isfile(p + ".csv"):
        return p + ".csv"
    raise FileNotFoundError(f"No se encontró el archivo: {p}")


def require_api_key(env_name: str = "DEEPSEEK_API_KEY") -> str:
    """
    Obtiene la API key requerida desde variable de entorno.
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

    Nuevo formato de salida requerido al LLM:
        idx: clase_predicha prob=ppp
    donde:
        - clase_predicha es 0 o 1
        - ppp es la probabilidad (entre 0 y 1) de que la clase verdadera sea 1.
    """
    prompt = (
        "Eres un clasificador binario. Devuelves para cada fila dos cosas:\n"
        "1) La clase predicha (0 o 1).\n"
        "2) Tu probabilidad estimada de que la fila pertenezca a la clase 1,\n"
        "   en el rango [0,1] con hasta 3 decimales.\n\n"
        "Formato EXACTO por línea:\n"
        "<indice>: <clase_predicha> prob=<probabilidad_de_clase_1>\n\n"
        "Ejemplo:\n"
        "12: 1 prob=0.87\n"
        "13: 0 prob=0.12\n\n"
        "No expliques nada más. Sólo entrega una línea por índice solicitado.\n\n"
        "Ahora ve el set de entrenamiento etiquetado:\n"
    )

    for i, row in X_train.iterrows():
        features = ", ".join([str(round(val, 2)) for val in row.values])
        prompt += f"{i}: {features} -> {y_train[i]}\n"

    prompt += "\nClasifica las siguientes muestras. Recuerda el formato indicado:\n"
    for i, row in X_test.iterrows():
        features = ", ".join([str(round(val, 2)) for val in row.values])
        prompt += f"{i}: {features}\n"

    return prompt


def call_openai(prompt: str, temperature: float, client: OpenAI) -> str:
    """
    Invoca el modelo con la temperatura indicada y devuelve el texto de la primera elección.
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Eres un clasificador de datos tabulares binario estricto."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


def parse_predictions(response: str) -> Dict[int, Tuple[str, Optional[float]]]:
    """
    Parsea líneas con el patrón:
        indice: etiqueta prob=probabilidad

    Devuelve dict:
        idx -> (etiqueta_predicha:str, prob_positiva:float)

    Si no encuentra probabilidad válida asigna None.
    Si no encuentra etiqueta válida asigna 'no_detectado'.

    Ejemplo de línea válida:
        7: 1 prob=0.82
    """
    preds: Dict[int, Tuple[str, Optional[float]]] = {}
    for raw_line in response.strip().splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue

        # split por "idx:" (primera aparición)
        left, right = line.split(":", 1)
        try:
            idx = int(left.strip())
        except ValueError:
            continue

        etiqueta_pred = "no_detectado"
        prob_val: Optional[float] = None

        # ejemplo right = "1 prob=0.82"
        parts = right.strip().split()
        # buscar primero número 0/1 como clase
        if len(parts) > 0:
            cand = parts[0].strip().lower()
            if cand in ["0", "1"]:
                etiqueta_pred = cand

        # buscar token tipo prob=0.82
        for p in parts[1:]:
            if p.lower().startswith("prob="):
                try:
                    prob_val = float(p.split("=", 1)[1])
                except ValueError:
                    prob_val = None
                break

        preds[idx] = (etiqueta_pred, prob_val)

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

    # Construir listas alineadas
    pred_labels: List[str] = []
    prob_list: List[Optional[float]] = []
    for i in range(len(X_test)):
        etiqueta_i, prob_i = preds_dict.get(i, ("no_detectado", None))
        pred_labels.append(etiqueta_i)
        prob_list.append(prob_i)

    y_test_list: List[str] = [str(label).strip().lower() for label in y_test]

    # Métricas clásicas basadas en etiquetas duras
    report = classification_report(y_test_list, pred_labels, output_dict=True)
    conf_matrix = confusion_matrix(y_test_list, pred_labels)
    labels = sorted(list(set(y_test_list + pred_labels)))
    accuracy = accuracy_score(y_test_list, pred_labels)
    kappa = cohen_kappa_score(y_test_list, pred_labels)

    # Métricas probabilísticas
    # AUC y curva ROC requieren:
    #   y_true_bin: 0/1 reales
    #   y_score: probabilidad estimada de clase=1
    # Se intentará sólo si las clases parecen binarias.
    auc: Optional[float] = None
    roc_points: Optional[Dict[str, List[float]]] = None
    try:
        y_true_bin = [1 if val in ['1', 1, 'true'] else 0 for val in y_test_list]

        # Si el modelo no entregó probabilidad se hace fallback:
        #   prob=1.0 si predijo clase=1
        #   prob=0.0 si predijo clase=0
        #   prob=None -> usar 0.5
        score_list: List[float] = []
        for hard_label, p in zip(pred_labels, prob_list):
            if p is not None:
                score_list.append(p)
            else:
                if hard_label in ['1', 'true', 1]:
                    score_list.append(1.0)
                elif hard_label in ['0', 'false', 0]:
                    score_list.append(0.0)
                else:
                    score_list.append(0.5)

        auc = roc_auc_score(y_true_bin, score_list)

        fpr, tpr, thr = roc_curve(y_true_bin, score_list)
        roc_points = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in thr]
        }
    except Exception:
        auc = None
        roc_points = None

    # Resultados por muestra
    df_results = pd.DataFrame({
        "indice": range(len(X_test)),
        "clase_real": y_test_list,
        "prediccion_llm": pred_labels,
        "prob_llm": prob_list
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
        # Guardamos también la curva ROC completa para análisis posterior
        "roc_curve_json": json.dumps(roc_points, ensure_ascii=False) if roc_points is not None else "",
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

    Si Excel/PQ está en configuración regional con coma decimal
    conviene importar este CSV declarando el delimitador como ';'.
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
