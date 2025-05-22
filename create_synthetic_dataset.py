import os
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset import TestsetGenerator
from ragas.testset.transforms import default_transforms, apply_transforms
from langchain.prompts import ChatPromptTemplate



def generar_dataset_sintetico( docs, generator_llm=None, generator_embeddings=None):
    """Genera y guarda un dataset sintético basado en RAG."""
    # 1. Construir KnowledgeGraph
    kg = KnowledgeGraph()
    print("KnowledgeGraph inicial:", kg)  # Debería mostrar: KnowledgeGraph(nodes: 0, relationships: 0)

    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata
                }
            )
        )
    print("KnowledgeGraph tras agregar documentos:", kg)  # Ahora debería mostrar varios nodos (por ejemplo, 10)

    #2. extraccion
    headlines_extractor_prompt = ChatPromptTemplate.from_template(
    """
You are a headlines extractor for document nodes. Extract a concise headline from the given page_content.
Return your answer as a JSON object with the key "headlines". 
For example, if the content is:
"Joint Consultation Command & Control Information Exchange Data Model (JC3IEDM) facilitates interoperability..."
Then your output should be:
{"headlines": "Interoperability in JC3IEDM"}
If no clear headline is found, return {"headlines": ""}.
"""
)


    # Usamos el mismo LLM y modelo de embeddings que para la generación del testset
    transformer_llm = generator_llm
    embedding_model = generator_embeddings

    # Obtener las transformaciones por defecto
    trans = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)

    # Inyectar el prompt personalizado para la transformación de headlines (si existe)
    for transform in trans:
        if hasattr(transform, "name") and transform.name == "headlines_extractor":
            transform.prompt = headlines_extractor_prompt


    apply_transforms(kg, trans)

    # 5. Guardar y recargar el KnowledgeGraph (opcional pero útil para verificar el enriquecimiento)
    kg.save("knowledge_graph_deploy_prueba.json")

def crear_testset(graph = "knowledge_graph_deploy_prueba.json" , output_path: str = "dataset.csv", testset_size: int = 120, generator_llm=None, generator_embeddings=None):
    """Genera un testset sintético y lo guarda en un archivo JSON."""
    # 1. Cargar KnowledgeGraph desde el archivo JSON
    loaded_kg = KnowledgeGraph.load(graph)

    # 3. Generar testset
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=loaded_kg
    )
    query_distribution = default_query_distribution(generator_llm)
    print("Distribución original de queries:", query_distribution)

    # Filtrar la distribución para eliminar el sintetizador multi_hop_specific_query_synthesizer
    filtered_query_distribution = [
        (synth, prob) for synth, prob in query_distribution 
        if synth.name != "multi_hop_specific_query_synthesizer"
    ]

    print("Distribución filtrada de queries:", filtered_query_distribution)
    
    testset = generator.generate(testset_size=testset_size, query_distribution=filtered_query_distribution)
    df = testset.to_pandas()
    
    os.environ["RAGAS_APP_TOKEN"] = "apt.4054-53fd2731274f-4395-87c1-2cd16721-ebca3"
    testset.upload()


    # 4. Guardar JSON
    testset = df.to_csv(output_path)
    return testset