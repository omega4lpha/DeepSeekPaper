from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
import os
import time
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, cohen_kappa_score
)
from openai import OpenAI

# -----------------------------------------------------------------------------
# Configuración de la aplicación Flask y del cliente LLM
# -----------------------------------------------------------------------------

app = Flask(__name__)
# Nota: para despliegues reales, el secreto debe provenir de una variable de entorno.
app.secret_key = "secret-key"

# Directorio de carga de archivos; se crea si no existe.
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cliente OpenAI configurado para utilizar DeepSeek vía endpoint compatible.
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# -----------------------------------------------------------------------------
# Funciones de procesamiento de datos y soporte LLM
# -----------------------------------------------------------------------------

def procesar_archivos(clases_path: str, datos_path: str) -> pd.DataFrame:
    """
    Lee, valida y consolida los archivos de clases y datos.

    Flujo:
    1) Carga ambos CSV separados por ';' sin cabecera.
    2) Verifica dimensión mínima de columnas (>=2).
    3) Descarta la primera columna de cada archivo (p. ej., índice o id externo).
    4) Valida consistencia de filas entre ambos archivos.
    5) Estandariza nombres de columnas de datos como d1..dn y concatena la clase.
    6) Construye una muestra balanceada simple tomando 150 primeras y 150 últimas
       filas (si existen), barajándolas con semilla fija para reproducibilidad.

    Parámetros
    ----------
    clases_path : str
        Ruta al archivo CSV con las clases por fila.
    datos_path : str
        Ruta al archivo CSV con los atributos por fila.

    Retorna
    -------
    pd.DataFrame
        DataFrame con la primera columna 'clase' y el resto atributos d1..dn.
        Devuelve una muestra barajada de hasta 300 filas.
    """
    df_clases = pd.read_csv(clases_path, header=None, sep=";")
    df_datos = pd.read_csv(datos_path, header=None, sep=";")

    if df_clases.shape[1] < 2:
        raise ValueError("El archivo de clases debe tener al menos 2 columnas.")
    if df_datos.shape[1] < 2:
        raise ValueError("El archivo de datos debe tener al menos 2 columnas.")

    # Se elimina la primera columna en ambos (típicamente un índice externo).
    df_clases = df_clases.iloc[:, 1:]
    df_datos = df_datos.iloc[:, 1:]

    if df_clases.shape[0] != df_datos.shape[0]:
        raise ValueError("Los archivos no tienen la misma cantidad de filas.")

    # Estandarización de nombres de columnas de datos.
    df_datos.columns = [f'd{i+1}' for i in range(df_datos.shape[1])]
    df_final = pd.concat([df_clases, df_datos], axis=1)
    df_final.columns = ['clase'] + df_datos.columns.tolist()

    # Muestra reproducible: 150 primeras + 150 últimas, luego shuffle.
    # Si el dataset es menor, concatena lo disponible sin fallar.
    head = df_final.head(150)
    tail = df_final.tail(150)
    df_sample = pd.concat([head, tail]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df_sample


def generate_prompt(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> str:
    """
    Genera un prompt instructivo para un LLM a partir de ejemplos etiquetados
    (entrenamiento) y ejemplos sin etiqueta (prueba).

    Diseño:
    - Presenta pares (características -> clase) del conjunto de entrenamiento.
    - Solicita la clasificación de las muestras de prueba con salida estricta
      índice: etiqueta, para facilitar el parseo determinista.

    Parámetros
    ----------
    X_train : pd.DataFrame
        Atributos de entrenamiento.
    X_test : pd.DataFrame
        Atributos de evaluación/prueba.
    y_train : pd.Series
        Etiquetas correspondientes a X_train.

    Retorna
    -------
    str
        Prompt para ser enviado al LLM.
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


def call_openai(prompt: str) -> str:
    """
    Envía el prompt al modelo LLM y devuelve el contenido textual de la primera
    elección de respuesta.

    Parámetros
    ----------
    prompt : str
        Instrucción generada para el LLM.

    Retorna
    -------
    str
        Respuesta en texto plano del LLM.
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Eres un clasificador de datos tabulares."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )
    return response.choices[0].message.content


def parse_predictions(response: str) -> Dict[int, str]:
    """
    Analiza la respuesta del LLM con el formato 'indice: etiqueta' por línea,
    devolviendo un diccionario índice->etiqueta.

    Comportamiento:
    - Ignora líneas sin carácter ':' o con índices no enteros.
    - Normaliza la etiqueta a minúsculas y elimina espacios.

    Parámetros
    ----------
    response : str
        Respuesta en texto plano emitida por el LLM.

    Retorna
    -------
    Dict[int, str]
        Mapa de índice de muestra a etiqueta predicha (como cadena).
    """
    preds: Dict[int, str] = {}
    for line in response.strip().splitlines():
        if ':' in line:
            key, val = line.split(':', 1)
            try:
                idx = int(key.strip())
                preds[idx] = val.strip().lower()
            except ValueError:
                # Se omiten encabezados, comentarios u otras líneas no parseables.
                continue
    return preds

# -----------------------------------------------------------------------------
# Rutas de la aplicación
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    """
    Página inicial: entrega el formulario para cargar archivos y ejecutar el flujo.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Maneja la carga de archivos de clase y datos, los guarda en disco de forma
    segura y persiste las rutas en la sesión para su uso posterior.

    Seguridad:
    - secure_filename evita rutas maliciosas e inserción de caracteres inválidos.
    """
    class_file = request.files['class_file']
    data_file = request.files['data_file']

    class_filename = secure_filename(class_file.filename)
    data_filename = secure_filename(data_file.filename)

    class_path = os.path.join(app.config['UPLOAD_FOLDER'], class_filename)
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

    class_file.save(class_path)
    data_file.save(data_path)

    session['class_path'] = class_path
    session['data_path'] = data_path

    return render_template('preview.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Ejecuta el pipeline completo:
    - Procesamiento y muestreo de datos.
    - Construcción de prompt.
    - Llamada al LLM.
    - Parseo de predicciones y evaluación con métricas clásicas de clasificación.
    - Medición de tiempos por fase y estimación de volumen de datos intercambiado.

    Devuelve el render de 'results.html' con:
    - Prompt emitido y respuesta del LLM (para trazabilidad).
    - Tokens aproximados y bytes enviados/recibidos.
    - Métricas: classification_report, matriz de confusión, accuracy, kappa, AUC (si aplica).
    - Tiempos por fase y totales.
    - Tabla índice/clase real/predicción.
    """
    start_total = time.time()

    class_path = session['class_path']
    data_path = session['data_path']

    # Fase 1: Procesamiento de datos
    start_pre = time.time()
    df_final = procesar_archivos(class_path, data_path)

    # Separación de variables y partición simple entrenamiento/prueba (60/40).
    X = df_final.iloc[:, 1:]
    y = df_final.iloc[:, 0]
    train_df = df_final.sample(frac=0.6, random_state=42).reset_index(drop=True)
    test_df = df_final.drop(train_df.index).reset_index(drop=True)

    X_train = train_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]
    y_test = test_df.iloc[:, 0]
    end_pre = time.time()

    # Fase 2: Generación del prompt
    start_prompt = time.time()
    prompt = generate_prompt(X_train, X_test, y_train)
    approx_tokens = len(prompt.split())
    # Estimación de bytes transmitidos para diagnósticos de consumo.
    input_bytes = len(prompt.encode('utf-8'))
    end_prompt = time.time()

    # Fase 3: Llamada al LLM
    start_llm = time.time()
    response = call_openai(prompt)
    output_bytes = len(response.encode('utf-8'))
    end_llm = time.time()

    # Estimación de ancho de banda (KB) intercambiado con el LLM.
    ancho_banda = {
        "Enviados (KB)": round(input_bytes / 1024, 2),
        "Recibidos (KB)": round(output_bytes / 1024, 2),
        "Total (KB)": round((input_bytes + output_bytes) / 1024, 2)
    }

    # Fase 4: Evaluación cuantitativa
    start_eval = time.time()
    preds_dict = parse_predictions(response)

    # Se construye la lista de predicciones para cada índice de X_test.
    # Si el LLM no devuelve una línea para un índice, se marca como 'no_detectado'.
    pred_labels: List[str] = [preds_dict.get(i, 'no_detectado') for i in range(len(X_test))]

    # Normalización de clases verdaderas a cadenas en minúscula para métrica coherente.
    y_test_list: List[str] = [str(label).strip().lower() for label in y_test]

    # Reporte de clasificación estructurado (dict) para facilitar renderizado en plantilla.
    report = classification_report(y_test_list, pred_labels, output_dict=True)

    # Matriz de confusión. Los labels se infieren de la unión de verdaderos y predichos.
    conf_matrix = confusion_matrix(y_test_list, pred_labels)
    labels = sorted(list(set(y_test_list + pred_labels)))

    # Métricas escalares frecuentes.
    accuracy = accuracy_score(y_test_list, pred_labels)
    kappa = cohen_kappa_score(y_test_list, pred_labels)

    # Cálculo opcional de AUC para problemas binarios (se mapea a 0/1).
    try:
        y_true_bin = [1 if val in ['1', 1, 'true'] else 0 for val in y_test_list]
        y_pred_bin = [1 if val in ['1', 1, 'true'] else 0 for val in pred_labels]
        auc = roc_auc_score(y_true_bin, y_pred_bin)
    except Exception:
        # Si el mapeo no es válido (p. ej., clases multiclase), no se reporta AUC.
        auc = "No disponible"

    # Tabla de resultados por muestra para inspección manual.
    df_results = pd.DataFrame({
        "Índice": range(len(X_test)),
        "Clase real": y_test_list,
        "Predicción LLM": pred_labels
    })
    end_eval = time.time()

    # Tiempos de ejecución por fase y total.
    end_total = time.time()
    tiempos = {
        "Procesamiento": end_pre - start_pre,
        "Generación de prompt": end_prompt - start_prompt,
        "Llamada al LLM": end_llm - start_llm,
        "Evaluación": end_eval - start_eval,
        "Total": end_total - start_total
    }

    return render_template(
        'results.html',
        prompt=prompt,
        response=response,
        approx_tokens=approx_tokens,
        results=df_results.values,
        metrics=report,
        conf_matrix=conf_matrix,
        labels=labels,
        accuracy=accuracy,
        auc=auc,
        kappa=kappa,
        tiempos=tiempos,
        ancho_banda=ancho_banda
    )


if __name__ == '__main__':
    # Modo debug solo para desarrollo local. En producción usar un servidor WSGI/ASGI.
    app.run(debug=True, port=8000)
