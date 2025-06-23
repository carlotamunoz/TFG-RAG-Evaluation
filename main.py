from data_ingestion import load_documents
from rag import RAG
from create_synthetic_dataset import generar_dataset_sintetico, crear_testset
from evaluation import evaluar_por_metricas
from langchain_ollama import OllamaEmbeddings
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from utils import crear_bloques_por_paginas, parse_reference_contexts, contexto_no_vacio
import ast
from ragas.testset import Testset
import time
import json
import pandas as pd

def generar_y_subir_dataset(pdf_path, plataforma, gen_llm, gen_embed = "nomic-embed-text", preguntas_por_bloque=25):
    '''
    1. Carga el PDF y fragmenta en bloques.
    2. Genera datasets sintéticos y los junta.
    3. Sube a RAGAS para revisión humana.
    4. El usuario revisa y descarga el JSON.
    '''
    # --- Carga y fragmenta documento ---
    rag = RAG()
    rag.docs, rag.vectorstore, elements_utiles = load_documents(pdf_path, embeddings=rag.embeddings)

    # Detecta páginas y crea bloques
    paginas_detectadas = set()
    for el in elements_utiles:
        if hasattr(el.metadata, "page_number") and el.metadata.page_number is not None:
            paginas_detectadas.add(el.metadata.page_number)
    total_paginas = len(paginas_detectadas)
    print(f"📑 Total de páginas detectadas: {total_paginas}")

    MAX_PAGINAS_BLOQUE = 75
    if total_paginas > MAX_PAGINAS_BLOQUE:
        bloques = crear_bloques_por_paginas(elements_utiles, max_paginas_por_bloque=75)
    else:
        bloques = [[el.text.strip() for el in elements_utiles]]

    # --- Genera datasets por bloque ---

    # INICIO DEL TEMPORIZADOR
    start_time = time.time()
    if plataforma == "openrouter":

        generator_llm = LangchainLLMWrapper(ChatOpenAI(model=gen_llm,
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            base_url="https://openrouter.ai/api/v1",
            api_key = # YOUR API KEY HERE
        ))
        generator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model = gen_embed))

    elif plataforma == "ollama":
        generator_llm = LangchainLLMWrapper(ChatOllama(model="llama3", base_url="https://ollama.gsi.upm.es/", format = 'json'))
        generator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=gen_embed))


    csvs = []
    for idx, bloque in enumerate(bloques):
        try:
            bloque_texto = "\n".join(bloque)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            docs_bloque = splitter.create_documents([bloque_texto])
            nombre_kg = f"knowledge_graph_block_{idx+1}.json"
            nombre_csv = f"dataset_block_{idx+1}.csv"
            generar_dataset_sintetico(docs=docs_bloque, generator_llm=generator_llm, generator_embeddings=generator_embeddings, output_path=nombre_kg)
            crear_testset(graph=nombre_kg, output_path=nombre_csv, generator_llm=generator_llm, generator_embeddings=generator_embeddings, testset_size=preguntas_por_bloque)
            csvs.append(nombre_csv)
        except Exception as e:
            print(f"⚠️ Error en el bloque {idx+1}: {e}")
            continue

    # --- Une datasets y sube a RAGAS ---
    dfs = [pd.read_csv(csv) for csv in csvs]
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_csv("dataset_completo.csv", index=False)


    # FIN DEL TEMPORIZADOR
    end_time = time.time()
    print(f"⏱️ Tiempo total de ejecución de la generación de datasets: {end_time - start_time:.2f} segundos")
    
    df_final = pd.read_csv("dataset_completo.csv")
    
    df_final["reference_contexts"] = df_final["reference_contexts"].apply(parse_reference_contexts)
    df_filtrado = df_final[df_final["reference_contexts"].apply(contexto_no_vacio)].reset_index(drop=True)

    print(f"Filas antes: {len(df_final)} | Filas después de limpiar: {len(df_filtrado)}")
    if len(df_filtrado) == 0:
        raise ValueError("❌ El dataframe está vacío después de limpiar. Revisa los datos.")

    df_filtrado.to_csv("testset_adaptado.csv", index=False)

    testset = Testset.from_pandas(df_filtrado)
    os.environ["RAGAS_APP_TOKEN"] = # YOUR RAGAS APP TOKEN HERE
    testset.upload()
    print("🚀 Dataset subido correctamente a RAGAS")



def json_revisado_a_csv(json_path = "testset.json", csv_out = None):
    """Convierte el JSON exportado desde RAGAS en CSV sólo con aprobados."""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for sample in data:
        if sample.get("approval_status") == "approved":
            eval_sample = sample["eval_sample"]
            records.append({
                "user_input": eval_sample["user_input"],
                "reference_contexts": eval_sample["reference_contexts"],
                "reference": eval_sample["reference"],
                "synthesizer_name": sample.get("synthesizer_name", "unknown")
            })
    df = pd.DataFrame(records)
    print(f"✅ cvs creado correctamente SOLO con aprobados.")

    df["reference_contexts"] = df["reference_contexts"].apply(parse_reference_contexts)
    df_filtrado = df[df["reference_contexts"].apply(contexto_no_vacio)].reset_index(drop=True)

    print(f"Filas antes: {len(df)} | Filas después de limpiar: {len(df_filtrado)}")
    if len(df_filtrado) == 0:
        raise ValueError("❌ El dataframe está vacío después de limpiar. Revisa los datos.")

    df_filtrado.to_csv(csv_out, index=False)

    testset = Testset.from_pandas(df_filtrado)
    os.environ["RAGAS_APP_TOKEN"] = # YOUR RAGAS APP TOKEN HERE
    testset.upload()
    print("🚀 Dataset subido correctamente a RAGAS")




def evaluar_dataset_final(csv_path, pdf_path):
    """Evalúa el CSV validado con los modelos y métricas elegidas."""
    rag = RAG()
    rag.docs, rag.vectorstore, elements_utiles = load_documents(pdf_path, embeddings=rag.embeddings)
    df = pd.read_csv(csv_path)
    df["reference_contexts"] = df["reference_contexts"].apply(parse_reference_contexts)

    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5", base_url="https://ollama.gsi.upm.es/"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    resultados = evaluar_por_metricas(
        df=df,
        rag=rag,
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings)
    # ...imprime resultados...
    
    print("Evaluación del RETRIEVER con métricas LLM-based:")
    print(resultados["llm_based_retrieval"])
    print("Evaluación del RETRIEVER con métricas NON-LLM-based:")
    print(resultados["nonllm_based_retrieval"])
    print("Evaluación del GENERADOR con métricas LLM-based:")
    print(resultados["llm_based_generator"])
    print("Evaluación del GENERADOR con métricas NON-LLM-based:")
    print(resultados["nonllm_based_generator"])
    print("Evaluación de texto claro (Plain Language):")
    print(resultados["plain_language"])





if __name__ == "__main__":
    print("¡Hola!, ¿Que quieres hacer hoy?:")
    print("1. Generar y subir dataset")
    print("2. Convertir JSON revisado a CSV")
    print("3. Evaluar dataset final")
    opcion = input("Opción: ")

    if opcion == "1":
        pdf = input("Nombre del PDF (ejemplo: book3.pdf): ")
        # Aquí pide el modelo y embedding que quieres usar
        plataforma = input("Elige la plataforma (openrouter/ollama): ").strip().lower()
        gen_llm = input("Modelo LLM para generación: ")
        gen_embed = input("Modelo LLM para embedding: ")
        generar_y_subir_dataset(pdf, plataforma, gen_llm, gen_embed)
    elif opcion == "2":
       
        json_path = input("Ruta al JSON revisado exportado de RAGAS (ej: testset.json): ")
        csv_out = input("Nombre para el CSV final (ej: testset_adaptado.csv): ")
        json_revisado_a_csv(json_path, csv_out)

    elif opcion == "3":
        csv_path = input("Ruta al CSV validado: ")
        pdf = input("Nombre del PDF original: ")

        evaluar_dataset_final(csv_path, pdf)
    else:
        print("Opción no reconocida.")
