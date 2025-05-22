from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate


class RAG:
    RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}
"""

    def __init__(self, model="mistral", base_url="https://ollama.gsi.upm.es/"):
        self.model = ChatOllama(model=model, base_url=base_url)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = None
        self.docs = None
        self.prompt_template = ChatPromptTemplate.from_template(self.RAG_TEMPLATE)

    def get_most_relevant_docs(self, query):
        if not self.vectorstore:
            raise ValueError("No se han cargado documentos. Ejecuta load_documents primero.")
        retriever = self.vectorstore.as_retriever()
        return retriever.invoke(query)

    def generate_answer(self, query, relevant_docs):
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        prompt = self.prompt_template.format(context=context, question=query)
        messages = [
            ("system", "You are an assistant that answers questions based solely on the provided context."),
            ("human", prompt)
        ]
        ai_msg = self.model.invoke(messages)
        return ai_msg.content
