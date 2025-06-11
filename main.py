from data_ingestion_epub import load_documents
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
from utils import crear_bloques_por_paginas
import ast
from ragas.testset import Testset


os.environ["OPENAI_API_KEY"] = "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"

def main():
    rag = RAG()
    rag.docs, rag.vectorstore, elements_utiles = load_documents("book3.pdf", embeddings=rag.embeddings)


    
    # CREACIÓN DE BLOQUES DE PÁGINAS
    # Detecta si tiene páginas (PDF) o no (EPUB)
    paginas_detectadas = set()
    for el in elements_utiles:
        if hasattr(el.metadata, "page_number") and el.metadata.page_number is not None:
            paginas_detectadas.add(el.metadata.page_number)
    total_paginas = len(paginas_detectadas)
    print(f"📑 Total de páginas detectadas: {total_paginas}")

    MAX_PAGINAS_BLOQUE = 75
    if total_paginas > MAX_PAGINAS_BLOQUE:
        bloques = crear_bloques_por_paginas(elements_utiles, max_paginas_por_bloque=MAX_PAGINAS_BLOQUE)
    else:
        bloques = [[el.text.strip() for el in elements_utiles]]





    # GENERACIÓN DE DATASETS SINTÉTICOS Y CREACIÓN DE TESTSET PARA CADA BLOQUE
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-2025-04-14",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        base_url="https://openrouter.ai/api/v1",
        api_key="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"))
    generator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    csvs = [
        "dataset_block_1.csv",
        "dataset_block_2.csv",
        "dataset_block_3.csv",
        "dataset_block_4.csv",
        "dataset_block_5.csv",
        "dataset_block_6.csv",
        "dataset_block_7.csv",
        "dataset_block_8.csv",
        "dataset_block_9.csv",
    ]
    
    
    for idx, bloque in enumerate(bloques):
        try:
            bloque_texto = "\n".join(bloque)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            docs_bloque = splitter.create_documents([bloque_texto])
            nombre_kg = f"knowledge_graph_block_{idx+1}.json"
            nombre_csv = f"dataset_block_{idx+1}.csv"
            generar_dataset_sintetico(docs=docs_bloque, generator_llm=generator_llm, generator_embeddings=generator_embeddings, output_path=nombre_kg)
            crear_testset(graph=nombre_kg, output_path=nombre_csv, generator_llm=generator_llm, generator_embeddings=generator_embeddings)
            csvs.append(nombre_csv)
        except Exception as e:
            print(f"⚠️ Error en el bloque {idx+1}: {e}")
            continue

    # Unir todos los CSVs generados en un único dataset
    dfs = [pd.read_csv(csv) for csv in csvs]
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_csv("dataset_completo.csv", index=False)
    





    
    #TRATAMIENTO TRAS EJECUTAR jsontocsv.py (HUMAN IN THE LOOP)
    df = pd.read_csv("testset_adaptado.csv")
    def parse_reference_contexts(val):
        if isinstance(val, list):
            return val
        if pd.isna(val) or str(val).strip() == "":
            return []
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    res = ast.literal_eval(val)
                    if isinstance(res, list):
                        return res
                    else:
                        return [res]
                except Exception as e:
                    print(f"[WARN] Error al parsear reference_contexts: {val} | Error: {e}")
                    return [val]
            else:
                return [val]
        return [val]

    df["reference_contexts"] = df["reference_contexts"].apply(parse_reference_contexts)
  
    # FUNCIÓN para saber si una lista es vacía o contiene solo strings vacíos
    def contexto_no_vacio(lista):
        if not lista:
            return False
        if all((x is None) or (isinstance(x, str) and x.strip() == '') for x in lista):
            return False
        return True

    df_filtrado = df[df["reference_contexts"].apply(contexto_no_vacio)].reset_index(drop=True)

    print(f"Filas antes: {len(df)} | Filas después de limpiar: {len(df_filtrado)}")
    if len(df_filtrado) == 0:
        raise ValueError("❌ El dataframe está vacío después de limpiar. Revisa los datos.")

    df_filtrado.to_csv("testset_adaptado.csv", index=False)

    testset = Testset.from_pandas(df_filtrado)
    os.environ["RAGAS_APP_TOKEN"] = 'apt.4054-53fd2731274f-4395-87c1-2cd16721-ebca3'
    testset.upload()
    print("🚀 Dataset subido correctamente a RAGAS")
        

    




    #EVALUCIÓN Y RESULTADOS
        
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5", base_url="https://ollama.gsi.upm.es/"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    resultados = evaluar_por_metricas(
        df= df_filtrado,
        rag=rag,
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings)
    
    
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
    main()
