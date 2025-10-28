# Clasificación vía LLM sobre datos tabulares (CLI)

Pipeline reproducible que:
1) Carga y valida dos archivos CSV (`clases` y `datos`, separador `;`).
2) Construye una muestra reproducible (150 primeras + 150 últimas filas) y separa entrenamiento/prueba (60/40).
3) Genera un prompt con ejemplos etiquetados y casos a clasificar.
4) Invoca un LLM (DeepSeek vía cliente OpenAI-compatible) a distintas temperaturas.
5) Parsea predicciones, incluyendo clase y probabilidad asociada, y calcula métricas (accuracy, kappa, AUC, curva ROC), matriz de confusión y tiempos por fase.
6) Persiste todos los artefactos de cada corrida en un CSV acumulativo (una fila por corrida), serializando objetos complejos en JSON.

---

## Requisitos

- Python 3.9+
- Dependencias:
  - `pandas`
  - `scikit-learn`
  - `openai`

Instalación de dependencias:

```bash
pip install pandas scikit-learn openai
```

Variable de entorno requerida:

- `DEEPSEEK_API_KEY` (clave para el endpoint DeepSeek compatible con el SDK de OpenAI)

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="sk-..."
# Windows PowerShell / CMD
$env:DEEPSEEK_API_KEY="sk-..."
```

---

## Estructura de entrada

- **`clases.csv`**  
  CSV separado por `;`, sin cabecera. La **primera columna** se descarta (ID/índice externo).  
  Desde la **segunda columna** en adelante: una sola columna con la **clase** por fila.

- **`datos.csv`**  
  CSV separado por `;`, sin cabecera. La **primera columna** se descarta (ID/índice externo).  
  Desde la **segunda columna** en adelante: atributos numéricos (`d1..dn`) por fila.

Ambos archivos deben tener igual número de filas.  
El script acepta nombres con o sin la extensión `.csv` (ej.: `--clases clases` o `--clases clases.csv`).

---

## Uso rápido

Ejecuta tres corridas por cada temperatura por defecto (`0.2`, `0.5`, `0.8`) y anexa resultados a `Resuldatos.csv`:

```powershell
# Windows
py app.py --clases clases --datos datos --out Resuldatos.csv
```

```bash
# Linux/macOS
python3 app.py --clases clases --datos datos --out Resuldatos.csv
```

Ejecuta con temperaturas y repeticiones explícitas:

```powershell
py app.py --clases clases --datos datos --out Resuldatos.csv --temps 0.2 0.5 0.8 --reps 3
```

Parámetros CLI disponibles:

```text
--clases   Ruta al CSV de clases (con o sin .csv). Por defecto: 'clases'
--datos    Ruta al CSV de datos (con o sin .csv). Por defecto: 'datos'
--out      CSV acumulativo de salida. Por defecto: 'Resuldatos.csv'
--temps    Lista de temperaturas. Por defecto: 0.2 0.5 0.8
--reps     Repeticiones por temperatura. Por defecto: 3
```

El archivo `Resuldatos.csv` se crea si no existe y se va anexando en corridas sucesivas.  
Cada corrida agrega un `run_id` (UUID) y `timestamp` UTC únicos.

---

## Salida: columnas y formatos

Cada fila del CSV de salida representa una corrida. Incluye, entre otras, las columnas:

- `run_id` (UUID de la corrida)  
- `timestamp` (UTC ISO 8601)  
- `temperature` (temperatura del LLM)  
- `approx_tokens` (estimación por división de palabras del prompt)  
- `accuracy`, `kappa`, `auc`  
- `n_train`, `n_test`  
- `bytes_enviados`, `bytes_recibidos`, `bytes_totales`  
- `prompt`, `response` (texto plano del modelo)  
- JSON serializados:
  - `report_json` (salida de `classification_report(output_dict=True)`)  
  - `conf_matrix_json` (matriz de confusión `list[list[int]]`)  
  - `labels_json` (lista de etiquetas ordenadas)  
  - `resultados_json` (lista de dicts con {indice, clase_real, prediccion_llm, prob_llm})  
  - `tiempos_json` (duraciones por fase en segundos)  
  - `ancho_banda_json` (KB enviados/recibidos/total)  
  - `roc_curve_json` (estructura con FPR, TPR y thresholds de la curva ROC)

Cuando se visualiza en Excel o Power Query con configuración regional que usa coma decimal, se debe importar el CSV declarando el delimitador como `;` para evitar que los decimales se interpreten como enteros.

---

## Arquitectura del pipeline

1. **Carga y validación** (`procesar_archivos`)  
   Lee `clases.csv` y `datos.csv` (sep `;`, sin cabecera), descarta la primera columna y verifica que tengan igual número de filas.  
   Estandariza columnas de atributos (`d1..dn`) y concatena con la columna `clase`.  
   Genera muestra reproducible: 150 primeras + 150 últimas filas barajadas con semilla `42`.

2. **Partición 60/40**  
   Separa entrenamiento y prueba de forma reproducible con semilla fija.

3. **Prompting** (`generate_prompt`)  
   Genera un prompt con formato estricto que exige devolver, por cada muestra de prueba, la clase (`0` o `1`) y la probabilidad de clase positiva (`prob=valor`).  
   Ejemplo:  
   ```
   12: 1 prob=0.87
   13: 0 prob=0.12
   ```

4. **Inferencia LLM** (`call_openai`)  
   Usa cliente OpenAI compatible (`model="deepseek-chat"`) con la temperatura indicada.  
   Devuelve el texto de la primera respuesta del modelo.

5. **Parseo** (`parse_predictions`)  
   Extrae el índice, la clase predicha y la probabilidad.  
   Si no hay probabilidad, se asigna `None`.  
   Si no hay clase, se asigna `no_detectado`.

6. **Evaluación** (`evaluar_llm_sobre_archivos`)  
   Calcula métricas `accuracy`, `kappa`, `auc` y genera la curva ROC.  
   Si el modelo entrega probabilidades válidas, se usan para el AUC; si no, se infieren según la clase.  
   Guarda reportes, matriz de confusión, etiquetas, tiempos y anchos de banda estimados.  
   Incluye en el JSON de resultados la probabilidad por muestra (`prob_llm`) y la curva ROC (`roc_curve_json`).

7. **Persistencia** (`append_dict_to_csv`)  
   Anexa una fila con todos los artefactos al CSV acumulativo de salida.  
   Crea cabecera si no existe.

---

## Buenas prácticas

- Configura correctamente la variable de entorno `DEEPSEEK_API_KEY`.  
- Usa separador `;` al importar el CSV si tu sistema usa coma decimal.  
- Conserva la semilla `42` para reproducibilidad.  
- No incluyas datos sensibles en `prompt` ni en `response` si compartes resultados.  
- Cada corrida produce datos independientes que pueden auditarse por `run_id`.

---

## Ejemplo de ejecución

```powershell
py app.py --clases clases --datos datos --out Resuldatos.csv
```

Ejemplo con temperatura única y una repetición:

```bash
python3 app.py --clases clases --datos datos --out Resuldatos.csv --temps 0.5 --reps 1
```

---

## Licencia

MIT

