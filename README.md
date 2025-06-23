# TFG-RAG-Evaluation
Este repositorio contiene un sistema para **evaluar automáticamente la calidad de cada uno de los componentes de un sistema RAG**, usando modelos de lenguaje como evaluadores.

## Configuración 
### Intalación de dependencias

Comenzaremos instalando todas las dependencias necesarias ejecutando el siguiente comando en la raíz del proyecto:
```python
pip install -r requirements.txt

```
Esto instalará bibliotecas como ragas, langchain, unstructured, entre otras, que son fundamentales para el funcionamiento del sistema.

### Crear token en RAGAS

Accede a https://app.ragas.io/ y crea una cuenta (si aún no la tienes). Una vez dentro crearemos un token. Para ello accederemos en la parte superior derecha y seleccionaremos el boton **app token** tal y como se muestra en la siguiente imagen 
![image](https://github.com/user-attachments/assets/e1915c5c-4b07-4483-8869-3f8f90cbedc4)
Genera y copia tu token.


### Insertar token Ragas App en el código
Una vez creado, copiaremos el token y accederemos a los ficheros *create_synthetic_dataset.py* y sustituiremos el fragmento *your_token_here* por tu token real .

```python

df = testset.to_pandas()
os.environ["RAGAS_APP_TOKEN"] = "your_token_here"  # Reemplaza con tu token real
testset.upload()
```

De igual forma accederemos a los ficheros *main.py* y*evaluation.py* y realizaremos el mismo procedimiento.

```python
result = evaluate(dataset=eval_ds, metrics=metrics, llm=evaluator_llm)
os.environ['RAGAS_APP_TOKEN'] = 'your_token_here'
result.upload()
```
### Insertar token Openrouter en el código
En el caso de usar un modelo alojado en la plataforma Openrouter será necesario el uso de un token. Para introducirlo en el código será necesario acceder al fichero *main.py* e insertarlo en el lugar marcado como #YOUR API KEY HERE


```python
 generator_llm = LangchainLLMWrapper(ChatOpenAI(model=gen_llm,
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            base_url="https://openrouter.ai/api/v1",
            api_key = # YOUR API KEY HERE
        ))
````

## Ejecución
Una vez configurado todo, puedes ejecutar el sistema desde la terminal. Asegúrate de estar ubicado en el directorio raíz del proyecto y ejecuta:

```python
python main.py
```
Este script coordina las distintas etapas del pipeline: carga del documento, ejecución del RAG, creación del dataset y evaluación de las respuestas generadas.

Al ejecutarlo, aparecerá un menú interactivo con tres opciones principales.
1. Generar y subir dataset
2. Convertir JSON revisado a CSV
3. Evaluar dataset final

**¿Que hace cada ópción?**
1. **Genenar y subir dataset**. Esta opción crea un conjunto de datos sintético a partir de un pdf. El lresultado será subido a Ragas App para su revisión. En dicha opción se le solicitará al usuario el nombre del pdf, la plataforma que se quiere usar (Openrouter o Ollama) y los modelos (LLMs) que se desean usar como generador del dataset. Para el nombre del modelo se deberá proporcionar el identificador del mismo. Este se puede encontrar en la página de la herramienta ( Openrouter u Ollama) que se desea utilizar. 

2. **Convertir JSON revisado a CSV**. Esta opción es útil si se ha decidido hacer una revisión del dataset para validar o rechazar las entradas. Permite convertir el JSON exportado de Ragas App a un formato aceptable para la evaluación. En dicha opción se le solicitará al usuario la ruta del JSON y el nombre del CSV final que se desee.
   
4. **Evaluar dataset**. Esta opción permite la evaluación de un dataset con distintos modelos y métricas. Permite saber como rinde el sistema (RAG) que estamos queriendo evaluar. En dicha opción se le solicitará al usuario la ruta del CSV que contiene el dataset y el pdf que subimos al RAG.
Como resultado se imprimirá por pantalla el resultado global de las métricas agrupadas en función del componente del RAG que se está evaluando (retrieval, generador) y del tipo de métrica (basadas en LLM, no basadas en LLMs y de lenguaje claro)

## Resultados
Todos los resultados estarán disponibles directamente en tu cuenta de RAGAS, dentro del panel de Dashboard. Allí podrás analizar las métricas obtenidas para cada entrada del dataset


## Estructura del proyecto
| Archivo                       | Descripción                                                                                                                                   |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                     | Script principal e interfaz por consola. Muestra un menú para generar datasets sintéticos, convertir JSON revisados a CSV y evaluar datasets. |
| `data_ingestion.py`           | Carga y limpia el documento fuente (PDF), eliminando texto irrelevante. Divide el texto en fragmentos y los convierte en vectores para búsqueda.       |
| `create_synthetic_dataset.py` | Genera conjuntos de datos sintéticos a partir del documento cargado usando LLMs. Incluye la creación de grafos de conocimiento y testsets con preguntas/respuestas simuladas.     |
| `rag.py`                      | Implementa la arquitectura RAG: define cómo recuperar los fragmentos más relevantes y cómo el modelo genera respuestas a partir de ellos.            |
| `evaluation.py`         | Evalúa la calidad de las respuestas del sistema usando métricas automáticas (LLM y no LLM) con la librería RAGAS. Imprime resultados y los sube a la plataforma. Sube los resultados para su análisis en la plataforma.  |
| `utils.py`                 | Funciones auxiliares para procesar bloques de texto, analizar contextos de referencia y limpiar datos.  |
| `requirements.txt`            | Lista de todas las dependencias necesarias para ejecutar el proyecto correctamente.                                                           |


