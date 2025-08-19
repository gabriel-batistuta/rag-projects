"""
setup_test_db_and_data.py

Cria extensão pgvector (se necessário), cria tabela `documentos`
e insere muitos registros de teste para alimentar o pipeline RAG.
"""

import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, Table, Column, Integer, Text, String, MetaData, TIMESTAMP
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.sql import select

# ---------- CONFIG ----------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gabriel:0000@localhost:5432/documentos-teste")
BATCH_SIZE = 100

# ---------- Engine ----------
engine = create_engine(DATABASE_URL, future=True)

def ensure_pgvector_extension(conn):
    """
    Cria extensão 'vector' (pgvector) se não existir.
    Observação: o nome da extensão é normalmente 'vector' (pgvector).
    """
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
        print("[OK] extensão 'vector' verificada/criada.")
    except ProgrammingError as e:
        # Em alguns ambientes (usuário sem permissão) pode falhar:
        print("[ERRO] não foi possível criar/garantir extensão 'vector':", e)
        raise

def create_documentos_table(conn):
    """
    Cria a tabela de documentos de teste se não existir.
    esquema: id, title, content, status, updated_at
    """
    meta = MetaData()
    documentos = Table(
        "documentos", meta,
        Column("id", Integer, primary_key=True),
        Column("title", String, nullable=False),
        Column("content", Text, nullable=False),
        Column("status", String, nullable=False, default="published"),
        Column("updated_at", TIMESTAMP, nullable=False),
    )
    meta.create_all(conn.engine)
    print("[OK] tabela 'documentos' verificada/criada.")
    return documentos

def generate_sample_texts() -> list:
    """
    Gera uma lista de dicionários com dados de teste.
    Cada item tem 'title', 'content', 'status', 'updated_at'.
    """
    now = datetime.utcnow()
    samples = [
        {
            "title": "Procedimento para renovação de matrícula",
            "content": (
                "Para renovar sua matrícula é necessário acessar o portal do aluno, "
                "verificar se não há pendências financeiras e emitir a guia de pagamento. "
                "Se a guia não for paga até a data limite, a matrícula será cancelada automaticamente."
            ),
            "status": "published",
            "updated_at": now - timedelta(days=10)
        },
        {
            "title": "Como pagar com Pix",
            "content": (
                "Para pagar via Pix, abra o aplicativo do seu banco, selecione 'Pix', escolha 'Pagar com QR code' "
                "ou 'Chave Pix' e insira o valor. Confirme os dados do recebedor e confirme o pagamento."
            ),
            "status": "published",
            "updated_at": now - timedelta(days=9)
        },
        {
            "title": "Política de reembolso",
            "content": (
                "Pedidos de reembolso serão analisados em até 30 dias úteis. Reembolsos por problemas técnicos "
                "exigem evidências (prints, logs) e podem ter retenção para verificação de fraude."
            ),
            "status": "published",
            "updated_at": now - timedelta(days=30)
        },
        {
            "title": "FAQ: como alterar senha",
            "content": (
                "Vá em 'Configurações' > 'Minha conta' > 'Alterar senha'. A nova senha deve conter pelo menos 8 caracteres, "
                "uma letra maiúscula e um número."
            ),
            "status": "published",
            "updated_at": now - timedelta(days=2)
        },
        {
            "title": "Guia de integração do sistema",
            "content": (
                "Para integrar via API, gere a chave em 'Desenvolvedores', use o endpoint /api/v1/events e envie JSON usando "
                "o header Authorization: Bearer <token>. O payload deve seguir o schema descrito na documentação."
            ),
            "status": "published",
            "updated_at": now - timedelta(days=40)
        },
        {
            "title": "Termos de uso - resumo",
            "content": (
                "Ao usar a plataforma você concorda em não publicar conteúdo ilegal. Conteúdos que violem direitos autorais serão removidos."
            ),
            "status": "archived",
            "updated_at": now - timedelta(days=400)
        },
        {
            "title": "Procedimentos de segurança física",
            "content": (
                "A entrada em áreas restritas exige crachá ativo e registro na portaria. Visitantes devem ser acompanhados por um funcionário."
            ),
            "status": "published",
            "updated_at": now - timedelta(days=7)
        },
        {
            "title": "Modelo de contrato de serviço",
            "content": (
                "Contrato de prestação de serviços: entre as partes, fica acordado que a contratada prestará os serviços descritos, "
                "com prazos e condições de pagamento especificados no Anexo A."
            ),
            "status": "draft",
            "updated_at": now - timedelta(days=1)
        },
        # -- gerar variações pequenas para popular --
    ]

    # adiciona outros registros gerados automaticamente para ter volume
    for i in range(10):
        samples.append({
            "title": f"Documento de processo #{i+1}",
            "content": (
                f"Este é o documento de processo número {i+1}. Contém instruções operacionais, "
                "checklists e contatos de suporte. Use este texto para testar buscas semânticas."
            ),
            "status": "published" if i % 2 == 0 else "draft",
            "updated_at": now - timedelta(days=(i % 15))
        })

    return samples

def insert_samples(conn, documentos_table, samples):
    """
    Insere os samples em lotes para evitar problemas de memória.
    """
    inserted = 0
    with conn.begin() as trans:
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i+BATCH_SIZE]
            conn.execute(documentos_table.insert(), batch)
            inserted += len(batch)
    print(f"[OK] inseridos {inserted} registros de teste na tabela 'documentos'.")

def verify_some_rows(conn, documentos_table, limit=5):
    """
    Exibe algumas linhas para verificar.
    """
    sel = select(documentos_table).limit(limit)
    res = conn.execute(sel).all()
    print("[AMOSTRA]")
    for r in res:
        print(f" - id={r.id} title={r.title[:60]} updated_at={r.updated_at} status={r.status}")

if __name__ == "__main__":
    print("Conectando a:", DATABASE_URL)
    with engine.connect() as conn:
        ensure_pgvector_extension(conn)
        documentos = create_documentos_table(conn)
        samples = generate_sample_texts()
        insert_samples(conn, documentos, samples)
        verify_some_rows(conn, documentos, limit=8)
    print("Tudo pronto! Agora você tem uma tabela 'documentos' populada para testes.")
