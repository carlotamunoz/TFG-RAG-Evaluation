# TFG-RAG-Evaluation
Este repositorio contiene un sistema para **evaluar automáticamente la calidad de respuestas generadas por sistemas RAG**, usando modelos de lenguaje como evaluadores.

## Configuración 
### Intalación de dependencias

Para comenzar lo primero que deberemos hacer será instalar las dependencias ejecutando el siguiente comando
```python
pip install -r requirements.txt

```
### Crear token en RAGAS

A continuación deberemos crearnos una cuenta en la aplicación de ragas https://app.ragas.io/
y crearemos un token. Para ello accederemos en la parte superior derecha y seleccionaremos el boton **app token** tal y como se muestra en la siguiente imagen 
![image](https://github.com/user-attachments/assets/e1915c5c-4b07-4483-8869-3f8f90cbedc4)

### Insertar token en el código
Una vez creado, copiaremos el token y accederemos al fichero *create_synthetic_dataset.py* y sustituiremos el fragmento *your_token_here* por el nuevo.

```python

df = testset.to_pandas()
os.environ["RAGAS_APP_TOKEN"] = "your_token_here"  # Reemplaza con tu token real
testset.upload()
```

Por otro lado, accederemos al fichero *evaluate_dataset.py* y realizaremos el mismo procedimiento

```python
result = evaluate(dataset=eval_ds, metrics=metrics, llm=evaluator_llm)
os.environ['RAGAS_APP_TOKEN'] = 'your_token_here'
result.upload()
```
## Ejecución

En una terminal, tras haber accedido al directorio en el que hemos clonado el repositorio ejecutaremos 

```python
python main.py
```

## Resultados
Una vez ejecutados los scripts de creación y evaluación, los resultados estarán disponibles en tu cuenta de Ragas, dentro del panel de datasets.


## Estructura del proyecto
| Archivo                       | Descripción                                                                                                                                         |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                     | Script principal que orquesta el flujo del sistema. Aquí se define la ruta del documento a cargar y se coordinan las distintas etapas del pipeline. |
| `data_ingestion.py`           | Encargado de la limpieza y carga de los documentos. Prepara los datos para ser utilizados por el RAG.                                               |
| `create_synthetic_dataset.py` | Genera un conjunto de datos sintéticos con consultas y respuestas, que serán evaluados posteriormente.                                              |
| `rag.py`                      | Define la arquitectura del sistema RAG que se desea evaluar, incluyendo recuperación y generación.                                                  |
| `evaluate_dataset.py`         | Realiza la evaluación del dataset generado, aplicando métricas como `Faithfulness`, `FactualCorrectness` y `LLMContextRecall`.                      |
| `requirements.txt`            | Contiene todas las dependencias necesarias para ejecutar el proyecto en un entorno Python.                                                          |



