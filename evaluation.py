import pandas as pd
from datasets import Dataset
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import RubricsScore
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

from ragas.metrics import SemanticSimilarity
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
import time
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, ResponseRelevancy
import time
import json
from ragas import RunConfig


def evaluar_por_metricas(df, rag, evaluator_llm, evaluator_embeddings) -> dict:

    def get_response_text(response):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed["answer"]
            return response
        except Exception:
            return response

    
    result_list = []

    for index, row in df.iterrows():
        try:
            user_input = row["user_input"]
            reference = row["reference"]
            reference_contexts = row["reference_contexts"]
            relevant_docs = rag.get_most_relevant_docs(user_input)
            response = rag.generate_answer(user_input, relevant_docs)
            respuesta = get_response_text(response)
            retrieved_contexts = [doc.page_content for doc in relevant_docs]
            # No añadas la muestra si retrieved_contexts es vacío
            if not retrieved_contexts or all((x is None) or (isinstance(x, str) and x.strip() == '') for x in retrieved_contexts):
                print(f"Fila {index} ignorada: contexto vacío.")
                continue
            sample = {
                "user_input": user_input,
                "retrieved_contexts": retrieved_contexts,
                "response": respuesta,
                "reference": reference,
                "reference_contexts": reference_contexts
            }
            result_list.append(sample)
        except Exception as e:
            print(f"Error en fila {index}: {e}")
            continue
        time.sleep(0.75)

    # Si no hay filas válidas, lanza error
    if not result_list:
        raise ValueError("❌ No hay muestras válidas tras el parseo y filtrado. Corrige tu dataset o código.")

    dataset = Dataset.from_dict({
        "user_input": [sample["user_input"] for sample in result_list],
        "retrieved_contexts": [sample["retrieved_contexts"] for sample in result_list],
        "response": [sample["response"] for sample in result_list],
        "reference": [sample["reference"] for sample in result_list]
        ,"reference_contexts": [sample["reference_contexts"] for sample in result_list]
    })

    

    run_config = RunConfig(
        timeout=120,          # Tiempo máximo de espera por evaluación (en segundos)
        max_retries=3,        # Número máximo de reintentos en caso de fallo
        max_workers=1         # Número de tareas concurrentes; ajusta según tu entorno
    )
    
    eval_ds = EvaluationDataset.from_list(dataset)


    metrics_retrieval_llm = [context_precision, context_recall]
    metrics_retrieval_nonllm = [NonLLMContextPrecisionWithReference(), NonLLMContextRecall() ]
    metrics_generation_llm = [answer_relevancy,faithfulness, FactualCorrectness(llm = evaluator_llm)]
    metrics_generation_nonllm = [SemanticSimilarity()]
    
    rubrics = {
    "score0_description": "The response is confusing, uses jargon or complex words, and is hard to read or understand. The message is unclear and the structure does not help the reader.",
    "score1_description": "The response is somewhat understandable but includes some complex sentences, unnecessary words, or technical terms. It requires effort to understand and could confuse the reader.",
    "score2_description": "The response is mostly clear and uses simple language but could be shorter or better organized. The main idea is understandable, but some sentences or words make it harder to read.",
    "score3_description": "The response is clear, concise, and logically organized. It uses everyday language and the main ideas are easy to understand on first reading. There may be small areas for improvement.",
    "score4_description": "The response is very clear, direct, and easy to understand for the target audience. It uses plain language, short sentences, and a simple structure. The reader can quickly understand and act on the information.",
    }
    plain_language_metrics = [RubricsScore(rubrics=rubrics, llm=evaluator_llm)]




    resultados = {}
    os.environ['RAGAS_APP_TOKEN'] = 'apt.4054-53fd2731274f-4395-87c1-2cd16721-ebca3'
    # Ejecuta la evaluación
    try:
        
        result_llm_retrieval = evaluate(eval_ds, llm = evaluator_llm, embeddings= evaluator_embeddings, metrics=metrics_retrieval_llm, run_config=run_config)
        resultados["llm_based_retrieval"] = result_llm_retrieval.to_pandas()
        result_llm_retrieval.upload()
        print("Evaluación de métricas LLM-based para el RETRIEVAL:", result_llm_retrieval)

        
        result_llm_generator = evaluate(eval_ds, llm = evaluator_llm, embeddings= evaluator_embeddings, metrics=metrics_retrieval_nonllm, run_config=run_config)
        resultados["nonllm_based_retrieval"] = result_llm_generator.to_pandas()
        result_llm_generator.upload()
        print("Evaluación de métricas LLM-based para el GENERADOR:", result_llm_generator)

        result_nonllm_retrieval = evaluate(eval_ds, llm = evaluator_llm, embeddings= evaluator_embeddings, metrics=metrics_generation_llm, run_config=run_config)
        resultados["llm_based_generator"] = result_nonllm_retrieval.to_pandas()
        result_nonllm_retrieval.upload()
        print("Evaluación de métricas non-LLM-based para el RETRIEVAL:", result_nonllm_retrieval)

        
        result_nonllm_generator = evaluate(eval_ds, llm = evaluator_llm, embeddings= evaluator_embeddings, metrics=metrics_generation_nonllm, run_config=run_config)            
        resultados["nonllm_based_generator"] = result_nonllm_generator.to_pandas()
        result_nonllm_generator.upload()
        print("Evaluación de métricas non-LLM-based para el GENERADOR:", result_nonllm_generator)

        
        result_plain_language = evaluate(eval_ds, llm=evaluator_llm, embeddings=None, metrics=plain_language_metrics, run_config=run_config)
        resultados["plain_language"] = result_plain_language.to_pandas()
        result_plain_language.upload()
        print("Evaluación de texto plano:", result_plain_language/5)

        
    

    except Exception as e:
        print(f"Error durante la evaluación: {e}")


    return resultados
