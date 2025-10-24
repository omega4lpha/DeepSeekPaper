# Clasificación vía LLM sobre datos tabulares (CLI)

Pipeline reproducible que:
1) Carga y valida dos archivos CSV (`clases` y `datos`, separador `;`).
2) Construye una muestra reproducible (150 primeras + 150 últimas filas) y separa entrenamiento/prueba (60/40).
3) Genera un prompt con ejemplos etiquetados y casos a clasificar.
4) Invoca un LLM (DeepSeek vía cliente OpenAI-compatible) a distintas temperaturas.
5) Parsea predicciones y calcula métricas (accuracy, kappa, AUC opcional), matriz de confusión y tiempos por fase.
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

> Ambos archivos deben tener **igual número de filas**.  
> El script acepta nombres **con o sin** la extensión `.csv` (ej.: `--clases clases` o `--clases clases.csv`).

---

## Uso rápido

Ejecuta **3 corridas por cada temperatura** por defecto (`0.2`, `0.5`, `0.8`) y **anexa** resultados a `Resuldatos.csv`:

```powershell
# Windows
py app.py --clases clases --datos datos --out Resuldatos.csv
```

```bash
# Linux/macOS
python3 app.py --clases clases --datos datos --out Resuldatos.csv
```

Especifica temperaturas y repeticiones:

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

> El archivo `Resuldatos.csv` se crea si no existe y **se va anexando** en corridas sucesivas.  
> Cada corrida agrega un `run_id` (UUID) y `timestamp` UTC únicos.

---

## Salida: columnas y formatos

Cada fila del CSV de salida representa **una corrida**. Incluye, entre otras, las columnas:

- `run_id` (UUID de la corrida)  
- `timestamp` (UTC ISO 8601)  
- `temperature` (temperatura del LLM)  
- `approx_tokens` (aproximación simple por división de palabras del prompt)  
- `accuracy`, `kappa`, `auc` (si aplica)  
- `n_train`, `n_test`  
- `bytes_enviados`, `bytes_recibidos`, `bytes_totales`  
- `prompt`, `response` (texto plano)  
- JSON serializados:
  - `report_json` (salida de `classification_report(output_dict=True)`)  
  - `conf_matrix_json` (matriz de confusión `list[list[int]]`)  
  - `labels_json` (lista de etiquetas ordenadas)  
  - `resultados_json` (lista de dicts con {indice, clase_real, prediccion_llm})  
  - `tiempos_json` (duraciones por fase en segundos)  
  - `ancho_banda_json` (KB enviados/recibidos/total)

> Sugerencia para Excel/Power Query en es-CL/es-ES: si ves que `0.2/0.5/0.8` aparecen como `2/5/8`, importa el CSV con **delimitador `;`** o cambia el tipo usando **“Usar configuración regional…” → Inglés (Estados Unidos)** para la columna `temperature`.

---

## Arquitectura del pipeline

1. **Carga y validación** (`procesar_archivos`)  
   - Lee `clases.csv` y `datos.csv` (sep `;`, sin cabecera), descarta la primera columna y verifica misma cantidad de filas.  
   - Estandariza columnas de atributos (`d1..dn`) y concatena con la columna `clase`.  
   - Genera muestra reproducible: `head(150) + tail(150)` barajadas (semilla `42`).

2. **Partición 60/40**  
   - `train_df = sample(frac=0.6, random_state=42)`  
   - `test_df = resto`

3. **Prompting** (`generate_prompt`)  
   - Lista ejemplos de entrenamiento en formato: `idx: f1, f2, ... -> clase`.  
   - Pide clasificar el set de prueba en formato estricto: `idx: clase`.

4. **Inferencia LLM** (`call_openai`)  
   - Cliente OpenAI compatible, `model="deepseek-chat"`, temperatura variable.  
   - Devuelve la respuesta textual de la primera elección.

5. **Parseo** (`parse_predictions`)  
   - Extrae pares `idx: etiqueta` (minúsculas, recorta espacios).  
   - Si falta un índice, su predicción se marca como `no_detectado`.

6. **Evaluación** (`evaluar_llm_sobre_archivos`)  
   - `classification_report`, `confusion_matrix`, `accuracy`, `cohen_kappa_score`.  
   - AUC binario opcional (mapea clases a `{0,1}` si corresponde).  
   - Mide tiempos por fase y bytes intercambiados (estimados).

7. **Persistencia** (`append_dict_to_csv`)  
   - Anexa una fila por corrida a `Resuldatos.csv` (crea cabecera si no existe).

---

## Buenas prácticas y notas

- **API key**: no la hardcodees. Usa `DEEPSEEK_API_KEY`.  
- **Reproducibilidad**: la semilla fija (`random_state=42`) asegura el mismo split y shuffle.  
- **Privacidad**: `prompt` y `response` pueden contener datos sensibles. Versiona el CSV con criterio.  
- **Regiones decimales**: si tu sistema usa coma `,` como decimal, considera exportar otro CSV con `sep=';'` o importar explícitamente como `;` en Excel/PQ.  
- **Extensibilidad**: para más temperaturas o repeticiones, ajusta `--temps` y `--reps`.

---

## Ejecución de ejemplo

Tres repeticiones por cada temperatura por defecto (`0.2`, `0.5`, `0.8`):

```powershell
py app.py --clases clases --datos datos --out Resuldatos.csv
```

Corridas con temperaturas explícitas y 3 repeticiones por temperatura:

```powershell
py app.py --clases clases --datos datos --out Resuldatos.csv --temps 0.2 0.5 0.8 --reps 3
```

---

## Licencia

MIT (ajústala según tus necesidades).
