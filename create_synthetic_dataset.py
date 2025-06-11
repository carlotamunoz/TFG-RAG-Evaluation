import os
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.transforms import apply_transforms, Parallel, Transforms
from langchain.prompts import ChatPromptTemplate
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor

from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
import typing as t

from ragas.testset.graph import NodeType
from ragas.testset.transforms.extractors import (
    EmbeddingExtractor,
    HeadlinesExtractor,
    SummaryExtractor,
)

from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.utils import num_tokens_from_string
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from langchain_core.documents import Document as LCDocument
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from langchain.prompts import ChatPromptTemplate



def default_transforms(
    documents: t.List[LCDocument],
    llm: BaseRagasLLM,
    embedding_model: BaseRagasEmbeddings,
) -> Transforms:
    """
    Aplica todas las transformaciones posibles, ajustando cada una a los nodos/documentos adecuados.
    """

    def filter_doc_with_num_tokens(node, min_num_tokens=250):
        return (
            node.type == NodeType.DOCUMENT
            and num_tokens_from_string(node.properties["page_content"]) > min_num_tokens
        )

    def filter_docs(node):
        return node.type == NodeType.DOCUMENT

    def filter_chunks(node):
        return node.type == NodeType.CHUNK

    # Headline extractor y splitter solo para docs largos
    headline_extractor = HeadlinesExtractor(
        llm=llm, filter_nodes=lambda node: filter_doc_with_num_tokens(node, 250)
    )
    headline_splitter = HeadlineSplitter(min_tokens=500)

    # Summary extractor para docs >100 tokens
    summary_extractor = SummaryExtractor(
        llm=llm, filter_nodes=lambda node: filter_doc_with_num_tokens(node, 50)
    )

    # NER y themes: para chunks y docs (ajustar filtro según tu pipeline)
    ner_extractor = NERExtractor(
        llm=llm, filter_nodes=lambda node: filter_chunks(node) or filter_docs(node)
    )
    theme_extractor = ThemesExtractor(
        llm=llm, filter_nodes=lambda node: filter_chunks(node) or filter_docs(node)
    )

    # Embeddings y similitud: para docs medianos y largos
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=lambda node: filter_doc_with_num_tokens(node, 50)
    )
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.7,
        filter_nodes=lambda node: filter_doc_with_num_tokens(node, 50)
    )

    ner_overlap_sim = OverlapScoreBuilder(
        threshold=0.01, filter_nodes=lambda node: filter_chunks(node) or filter_docs(node)
    )

    node_filter = CustomNodeFilter(
        llm=llm, filter_nodes=lambda node: filter_chunks(node) or filter_docs(node)
    )

    transforms = [
        headline_extractor,
        headline_splitter,
        summary_extractor,
        node_filter,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]

    return transforms


def generar_dataset_sintetico(docs, generator_llm=None, generator_embeddings=None, output_path= None):
    """Genera un KnowledgeGraph robusto aplicando transformaciones obligatorias."""

    kg = KnowledgeGraph()
    print("📊 KnowledgeGraph inicial:", kg)

    for doc in docs:
        kg.nodes.append(Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata if doc.metadata else {}
            }
        ))

    print("✅ KnowledgeGraph tras agregar documentos:", kg)

    # Prompt opcional para headlines
    headlines_extractor_prompt = ChatPromptTemplate.from_template("""
    You are a headlines extractor for document nodes. Extract a concise headline from the given page_content.
    Return your answer as a JSON object with the key "headlines".
    If no clear headline is found, return {"headlines": ""}.
    """)

    # Paso 1: aplicar transformaciones por defecto
    print("⚙️ Aplicando default_transforms...")
    transforms = default_transforms(
        documents=docs,
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )
    apply_transforms(kg, transforms)

    # Paso 2: verificar transformadores aplicados y forzar los que faltan
    transform_names = {getattr(t, "name", type(t).__name__).lower() for t in transforms}
    forced_transforms = []

    if "headlineextractor" not in transform_names:
        print("🔁 Forzando HeadlineExtractor...")
        headline_extractor = HeadlinesExtractor(llm=generator_llm)
        headline_extractor.prompt = headlines_extractor_prompt
        forced_transforms.append(headline_extractor)

    if "headlinesplitter" not in transform_names:
        print("🔁 Forzando HeadlineSplitter...")
        forced_transforms.append(HeadlineSplitter())

    if "nerextractor" not in transform_names:
        print("🔁 Forzando NERExtractor...")
        forced_transforms.append(NERExtractor(llm=generator_llm))

    if "themesextractor" not in transform_names:
        print("🔁 Forzando ThemesExtractor...")
        forced_transforms.append(ThemesExtractor(llm=generator_llm))

    if forced_transforms:
        apply_transforms(kg, forced_transforms)

    # Paso 3: verificación final
    enriched = 0
    for node in kg.nodes:
        entities = node.properties.get("entities")
        themes = node.properties.get("themes")
        if entities or themes:
            enriched += 1
            print("🔍 Fragmento:", node.properties["page_content"][:80].replace("\n", " "))
            print("📌 Entities:", entities)
            print("📌 Themes:", themes)

    print(f"✅ Nodos enriquecidos: {enriched}/{len(kg.nodes)}")

    kg.save(output_path)
    print("💾 KnowledgeGraph guardado como", output_path)

def crear_testset(
    graph: str = "",
    output_path: str = "",
    testset_size: int = 20,
    generator_llm=None,
    generator_embeddings=None,
):
    """Genera un testset sintético y lo guarda en un archivo CSV."""
    from ragas.testset.graph import Relationship

    loaded_kg = KnowledgeGraph.load(graph)
    print(f"✅ Nodos cargados: {len(loaded_kg.nodes)}")
    print(f"🔗 Relaciones cargadas: {len(loaded_kg.relationships)}")

    # Verificar nodos útiles
    valid_nodes = sum(
        1 for node in loaded_kg.nodes if node.properties.get("entities") or node.properties.get("themes")
    )
    print(f"🧠 Nodos con entidades o temas: {valid_nodes}/{len(loaded_kg.nodes)}")

    # Si no hay relaciones, crear una falsa para pruebas
    if len(loaded_kg.relationships) == 0 and len(loaded_kg.nodes) > 1:
        print("⚠️ No hay relaciones, creando una dummy...")
        loaded_kg.relationships.append(
        Relationship(
        source=loaded_kg.nodes[0],
        target=loaded_kg.nodes[1],
        type="related"
            )
        )


    # Crear el generador de testset
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=loaded_kg,
    )

    # Distribución de queries
    query_distribution = default_query_distribution(generator_llm)
    filtered_query_distribution = [
        (synth, prob) for synth, prob in query_distribution
        if synth.name != "multi_hop_specific_query_synthesizer"
    ]

    if not filtered_query_distribution:
        print("⚠️ Distribución vacía tras filtrar. Usando la original.")
        filtered_query_distribution = query_distribution

    print("🎯 Generando testset...")
    testset = generator.generate(
        testset_size=testset_size,
        query_distribution=filtered_query_distribution,
    )

    df = testset.to_pandas()
    print("📈 Shape del dataset generado:", df.shape)

    if df.empty:
        print("❌ El dataset está vacío. Revisa los nodos, relaciones o transformaciones.")
    else:
        df.to_csv(output_path, index=False)
        print(f"💾 Dataset guardado en {output_path}")

    os.environ["RAGAS_APP_TOKEN"] = "apt.4c07-9615c0f47003-1f3a-9f3e-b1242b14-f5f68"
    testset.upload()
    print("🚀 Dataset subido a RAGAS")

    return df
