# Run:
```bash
streamlit run app.py
```

# Result

PDF com o número de referência:

![Before](assets/pdf-example.png)

Resposta do bot ao usar RAG:

![After](assets/rag-example.png)

# AI
```bash
CREATE DATABASE "documentos-teste";
# exemplo para Postgres 17 (substitua pela sua versão)
sudo apt update
sudo apt install postgresql-17-pgvector

CREATE EXTENSION IF NOT EXISTS vector;

python3 db-ai/setup.py
python3 db-ai/rag.py
```