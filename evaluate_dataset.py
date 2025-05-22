import pandas as pd
from datasets import Dataset
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os


def evaluar_dataset_llm(testset, rag, evaluator_llm) -> dict:
    """Evalúa un dataset cvs y retorna resultados."""
    df = pd.read_csv(testset)    
    dataset = Dataset.from_dict(df)
    result_list = []

    for index, row in df.iterrows():
        user_input = row["user_input"]
        reference = row["reference"]
        
        # Recuperar los documentos relevantes a partir de la consulta del usuario
        relevant_docs = rag.get_most_relevant_docs(user_input)
        # Generar la respuesta usando los documentos relevantes
        response = rag.generate_answer(user_input, relevant_docs)
        
        # Extraer el contenido de los documentos relevantes
        retrieved_contexts = [doc.page_content for doc in relevant_docs]
        
        sample = {
            "user_input": user_input,
            "retrieved_contexts": retrieved_contexts,
            "response": response,
            "reference": reference
        }
        
        result_list.append(sample)

    # Convertir la lista de diccionarios en un Dataset de Hugging Face
    dataset = Dataset.from_dict({
        "user_input": [sample["user_input"] for sample in result_list],
        "retrieved_contexts": [sample["retrieved_contexts"] for sample in result_list],
        "response": [sample["response"] for sample in result_list],
        "reference": [sample["reference"] for sample in result_list]
    })

    eval_ds = EvaluationDataset.from_list(dataset)
    
    metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
    result = evaluate(dataset=eval_ds, metrics=metrics, llm=evaluator_llm)
    os.environ['RAGAS_APP_TOKEN'] = 'your_token_here'
    result.upload()

    return result