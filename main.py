from data_ingestion import load_documents
from rag import RAG
from create_synthetic_dataset import generar_dataset_sintetico, crear_testset
from evaluate_dataset import evaluar_dataset_llm
from langchain_ollama import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.testset.graph import KnowledgeGraph

def main():
    # 1. Cargar documentos y vectorizar
    rag = RAG()
    rag.docs, rag.vectorstore = load_documents("book2.pdf", embeddings=rag.embeddings)

    generator_llm = LangchainLLMWrapper(ChatOllama(model="mistral", base_url="https://ollama.gsi.upm.es/"))
    generator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

                                                
    # 2. Generar dataset sintético
    
    generar_dataset_sintetico(docs = rag.docs, generator_llm=generator_llm, generator_embeddings=generator_embeddings)
    crear_testset(generator_llm=generator_llm, generator_embeddings=generator_embeddings)
    # 3. Evaluar dataset
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5", base_url="https://ollama.gsi.upm.es/"))
    #evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    resultado = evaluar_dataset_llm(testset = "dataset.csv", rag=rag, evaluator_llm=evaluator_llm)
    print("🔍 Resultados de evaluación:\n", resultado)

if __name__ == "__main__":
    main()