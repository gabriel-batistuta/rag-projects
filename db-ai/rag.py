# rag_ollama_pg.py
"""
Exemplo RAG: Ollama (LLM) + Ollama embeddings + LangChain + Postgres (pgvector)
Supondo: tabela `documentos(id int, title text, content text)` na sua BD de aplicação.
"""

import os
from typing import List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain.schema import Document

# LangChain + Ollama imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings

# ---------- CONFIG (troque conforme seu ambiente) ----------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/meubd")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")  # padrão Ollama
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")  # exemplo: nomic-embed-text
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")  # modelo de geração
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "rag_documents")
EMBED_BATCH_SIZE = 32

# ---------- DB (aplicação) - exemplo SQLAlchemy ----------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def fetch_documents_from_app_db(limit=1000) -> List[Document]:
    """
    Extrai textos (ex: conteúdos de artigos) do seu BD de aplicação e retorna lista de Documents do LangChain.
    Ajuste a query conforme seu esquema real.
    """
    sql = "SELECT id, title, content FROM documentos WHERE content IS NOT NULL"
    docs: List[Document] = []
    with engine.connect() as conn:
        res = conn.execute(text(sql))
        for row in res:
            doc_id = row["id"]
            title = row["title"] or ""
            content = row["content"] or ""
            metadata = {"id": doc_id, "title": title}
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

# ---------- Embeddings (Ollama) ----------
def get_ollama_embeddings() -> Embeddings:
    """
    Instancia o wrapper de embeddings do Ollama (via langchain-ollama).
    """
    emb = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    return emb

# ---------- Indexar no Postgres (pgvector) via LangChain PGVector ----------
def create_or_connect_vectorstore(embeddings: Embeddings):
    """
    Conecta/Cria coleção pgvector (langchain_postgres.PGVector).
    """
    # conexão usada internamente por PGVector; o construtor pede "client" (sqlalchemy URL) e embeddings
    vectorstore = PGVector.from_documents(
        documents=[],  # docs vazios aqui; usaremos add_documents depois
        embeddings=embeddings,
        client=DATABASE_URL,
        collection_name=PGVECTOR_TABLE,  # nome da collection/tabela usada
    )
    return vectorstore

def ingest_documents_to_vectorstore(vectorstore: PGVector, docs: List[Document], embeddings: Embeddings):
    """
    Gera embeddings (em batch) e insere no vectorstore.
    """
    # a forma simples: vectorstore.add_documents(docs, embedding=embeddings) -- mas varia por versões.
    # Utilizamos add_documents se disponível:
    try:
        vectorstore.add_documents(docs)
    except Exception as e:
        # fallback manual: gerar embeddings e chamar add_texts
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        embs = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i+EMBED_BATCH_SIZE]
            embs_batch = embeddings.embed_documents(batch)
            embs.extend(embs_batch)
        vectorstore.add_texts(texts=texts, metadatas=metadatas, embeddings=embs)

# ---------- Build RAG chain ----------
def build_rag_chain():
    # LLM (Chat) via langchain-ollama
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

    # embeddings
    embeddings = get_ollama_embeddings()

    # connect vectorstore
    vectorstore = create_or_connect_vectorstore(embeddings)

    # retriever
    retriever = vectorstore.as_retriever(search_type="cosine", search_kwargs={"k": 4})

    # RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "map_reduce" ou "refine" também são opções para contextos maiores
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain, embeddings, vectorstore

# ---------- Exemplo de uso ----------
if __name__ == "__main__":
    print("1) Buscando documentos da base da aplicação...")
    docs = fetch_documents_from_app_db()
    print(f"-> encontrados {len(docs)} docs (amostra)...")

    print("2) Conectando embeddings Ollama e vectorstore pgvector...")
    embeddings = get_ollama_embeddings()
    vs = create_or_connect_vectorstore(embeddings)

    if len(docs) > 0:
        print("3) Ingestando documentos (embedding + inserção em pgvector)...")
        ingest_documents_to_vectorstore(vs, docs, embeddings)
        print("-> ingest concluída.")

    print("4) Construindo RAG chain (retriever + LLM)...")
    qa_chain, _, _ = build_rag_chain()

    # exemplo de pergunta
    pergunta = "Qual é o procedimento para renovação de matrícula?"
    print(f"Pergunta: {pergunta}")
    resp = qa_chain({"query": pergunta})
    answer = resp["result"]
    sources = resp.get("source_documents", [])
    print("Resposta:\n", answer)
    print("\nFontes (IDs / titles):")
    for s in sources:
        print("-", s.metadata.get("id"), s.metadata.get("title"))
