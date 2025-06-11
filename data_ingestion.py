from unstructured.partition.auto import partition
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def is_useful(el) -> bool:
    text = el.text.strip().lower()
    bad_keywords = [
        "table of contents", "contents", "editor:", "series editor",
        "synthesis lectures", "morgan & claypool", "isbn", "issn",
        "publisher", "graeme hirst", "copyright", "this book", "volume"
    ]
    if not text or len(text) < 30:
        return False
    if hasattr(el, 'category') and el.category in {"Title", "Header", "UncategorizedText"}:
        return False
    if any(bad_kw in text for bad_kw in bad_keywords):
        return False
    return True

def load_documents(file_path: str, embeddings, chunk_size: int = 500):
    print(f"📥 Cargando y limpiando: {file_path}")
    elements = partition(filename=file_path)
    print(f"📊 Elementos totales: {len(elements)}")
    elements_utiles = [el for el in elements if is_useful(el)]
    print(f"✅ Elementos útiles: {len(elements_utiles)}")

    # Texto completo para el vectorstore global
    texts = [el.text.strip() for el in elements_utiles]
    full_text = "\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    docs = splitter.create_documents([full_text])
    print(f"📦 Chunks generados: {len(docs)}")

    #persist_dir = "chroma_db"  # O cualquier ruta temporal
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    
    
    return docs, vectorstore, elements_utiles  # Ojo: aquí devuelvo elements_utiles, no solo los textos
