from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
from collections import Counter
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def is_useful(el: Element) -> bool:
    """Filtra elementos irrelevantes como encabezados o editoriales."""
    text = el.text.strip().lower()
    bad_keywords = [
        "table of contents", "contents", "editor:", "series editor",
        "synthesis lectures", "morgan & claypool", "isbn", "issn",
        "publisher", "graeme hirst", "copyright", "this book", "volume"
    ]
    if not text or len(text) < 30:
        return False
    if el.category in {"Title", "Header", "UncategorizedText"}:
        return False
    if any(bad_kw in text for bad_kw in bad_keywords):
        return False
    return True


def group_elements_by_page_and_block(elements: list[Element], min_block_len: int = 200) -> list[str]:
    grouped_texts = []
    current_page = -1
    buffer = []

    def flush_buffer():
        if buffer:
            block = " ".join(buffer).strip()
            if len(block) >= min_block_len:
                grouped_texts.append(block)
            buffer.clear()

    for el in elements:
        if not el.text or not el.text.strip():
            continue
        page_num = getattr(el.metadata, "page_number", None)
        if page_num is not None and page_num != current_page:
            flush_buffer()
            current_page = page_num
        buffer.append(el.text.strip())

    flush_buffer()
    return grouped_texts


def load_documents(file_path: str, embeddings, chunk_size: int = 500) -> tuple[list[Document], Chroma]:
    """
    Carga un PDF, filtra contenido irrelevante y crea un vectorstore listo para RAG.
    """
    print(f"📥 Cargando y limpiando PDF: {file_path}")
    elements = partition_pdf(filename=file_path, strategy="fast", extract_images_in_pdf=False)
    print(f"📊 Elementos totales: {len(elements)} - Categorías:", Counter([el.category for el in elements]))

    filtered = [el for el in elements if is_useful(el)]
    print(f"✅ Elementos útiles: {len(filtered)}")

    blocks = group_elements_by_page_and_block(filtered)
    print(f"📦 Bloques agrupados: {len(blocks)}")

    docs = [Document(page_content=blk) for blk in blocks]

    # Chunk + vectorstore
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    return docs, vectorstore
