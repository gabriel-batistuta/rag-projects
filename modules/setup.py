"""
RAG integration for your RequestAPI using Ollama (Llama3.2) + Ollama embeddings + LangChain + Chroma

Prerequisitos (execute antes):
 1) Tenha o Ollama rodando localmente: https://ollama.com/docs/installation
 2) Baixe os modelos:
    ollama pull llama3
    ollama pull mxbai-embed-large    # ou outro modelo de embeddings disponível
 3) Instale as dependências Python (recomendado criar virtualenv):
    pip install requests python-dotenv langchain langchain-ollama langchain-community chromadb ollama

Como funciona:
 - Pega os funcionários via RequestAPI.pegar_funcionarios()
 - Converte cada funcionário em um documento de texto (metadata normalizada)
 - Gera embeddings com OllamaEmbeddings
 - Salva no Chroma local (persistência opcional)
 - Cria um RetrievalQA pipeline com ChatOllama (modelo de geração: llama3 / llama3.2)
 - Permite perguntas em que o sistema recupera trechos relevantes e gera resposta

Notas importantes:
 - Não há como garantir 100% que não haverá erro em todos os ambientes (dependências, Ollama rodando, modelos puxados).
 - O código tenta capturar erros comuns e fornece mensagens orientadoras.
 - Ajuste nomes dos modelos (embed_model / llm_model) caso você use outros modelos.

Uso (exemplos):
  python rag_with_ollama_langchain.py --build  # busca funcionários, gera embeddings e persiste no Chroma
  python rag_with_ollama_langchain.py --ask "Qual o email do João?"

"""

import os
import sys
import json
import argparse
from typing import List, Dict

# ----- Inclui sua classe RequestAPI (a mesma que você enviou) -----
from abc import ABC, abstractmethod
import requests
import dotenv

class API(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url

class RequestAPI():
    def __init__(self, base_url:str="http://localhost:3000"):
        self.base_url = base_url
        # tenta carregar token pelo .env ou realizando login
        self.__token = dotenv.get_key(".env", "ADMIN_TOKEN") or self.login_admin()
        print("Credenciais carregadas com sucesso...")

    def __login_super(self):
        response = requests.post(f"{self.base_url}/login", json={
            "usuario": {
                "email": "titan@titan.com", 
                "password": "senha123"
            }
        })
        print(response.status_code, response.json())
        credentials = response.json()
        token = credentials.get("token")
        if token:
            dotenv.set_key(".env", "SUPER_TOKEN", token)
        return token
     
    def login_admin(self):
        response = requests.post(f"{self.base_url}/login", json={
            "usuario": {
                "email": "admin@abc-loja.com", 
                "password": "123456"
            }
        })
        print(response.status_code, response.json())
        credentials = response.json()
        token = credentials.get("token")
        if token:
            dotenv.set_key(".env", "SUPER_TOKEN", token)
        return token

    def pegar_funcionarios(self):
        url = f"{self.base_url}/api/v1/funcionarios"
        headers = {
            "Authorization": f"Bearer {self.__token}"
        }

        response = requests.get(url, headers=headers)
        print(response.status_code)
        try:
            return response.json()
        except Exception:
            print("Erro ao decodificar JSON da resposta de funcionarios")
            return []

# ----- Fim RequestAPI -----

# ----- Imports de LangChain / Ollama / Chroma -----
try:
    # wrapper oficial/mais comum para Ollama + LangChain
    from langchain_ollama import OllamaEmbeddings, ChatOllama
except Exception:
    # fallback para pacotes alternativos do ecossistema
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_ollama import ChatOllama
    except Exception as e:
        print("Erro ao importar classes do Ollama/LangChain. Instale 'langchain-ollama' e 'langchain-community'.\nDetalhe:", e)
        raise

try:
    from langchain_community.vectorstores import Chroma
    from langchain.docstore.document import Document
    from langchain.chains import RetrievalQA
except Exception as e:
    print("Erro ao importar LangChain Vectorstores/Chains. Instale 'langchain' e 'chromadb'.\nDetalhe:", e)
    raise


class RAGManager:
    def __init__(self,
                 request_api: RequestAPI,
                 embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large"),
                 llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.1"),
                 ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                 chroma_dir: str = "./chroma_db"):
        self.request_api = request_api
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url
        self.chroma_dir = chroma_dir

        # init clients
        try:
            self.embeddings = OllamaEmbeddings(model=self.embed_model, base_url=self.ollama_base_url)
        except Exception as e:
            print("Falha ao inicializar OllamaEmbeddings:", e)
            raise

        try:
            self.llm = ChatOllama(model=self.llm_model, base_url=self.ollama_base_url)
        except Exception as e:
            print("Falha ao inicializar ChatOllama (LLM):", e)
            raise

        self.vectorstore = None
        self.qa_chain = None

    def _employee_to_text(self, emp: Dict) -> str:
        # Normaliza campos mais comuns e concatena tudo em um blob de texto para indexação
        lines = []
        for key, val in emp.items():
            # ignora valores falsy longos
            if val is None:
                continue
            # se for dicionário, transforma em json curto
            if isinstance(val, dict):
                try:
                    val = json.dumps(val, ensure_ascii=False)
                except Exception:
                    val = str(val)
            lines.append(f"{key}: {val}")
        return "\n".join(lines)

    def _clean_metadata_for_chroma(self, metadata: dict) -> dict:
        """
        Garante que todos os valores da metadata sejam str/int/float/bool/None.
        - valores complexos (list/dict/obj) são convertidos para JSON str.
        """
        cleaned = {}
        for k, v in metadata.items():
            # aceita apenas primitivos
            if isinstance(v, (str, int, float, bool)) or v is None:
                cleaned[k] = v
            else:
                # tenta serializar; se falhar, usa str()
                try:
                    cleaned[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    cleaned[k] = str(v)
        return cleaned

    def build_index_from_api(self, persist: bool = True, overwrite: bool = False):
        print("Buscando funcionarios da API...")
        employees = self.request_api.pegar_funcionarios()
        if not employees:
            print("Nenhum funcionário retornado da API. Abortando construção do índice.")
            return

        docs: List[Document] = []
        for i, emp in enumerate(employees):
            content = self._employee_to_text(emp)
            # metadata simples: id e email (exemplo) + raw convertido para string JSON
            meta = {
                "employee_id": emp.get("id", f"idx_{i}"),
                # inclua campos simples que você queira consultar diretamente:
                "nome": emp.get("nome"),
                "email": emp.get("email"),
                "cargo": emp.get("cargo"),
            }
            # Se quiser guardar o objeto inteiro, converta para JSON string (aceito por Chroma)
            meta["raw_json"] = json.dumps(emp, ensure_ascii=False)

            # garante que todos os campos sejam primitivos (por segurança)
            clean_meta = self._clean_metadata_for_chroma(meta)

            docs.append(Document(page_content=content, metadata=clean_meta))


        # Se já existe e overwrite=True, apaga o diretório
        if overwrite and os.path.isdir(self.chroma_dir):
            import shutil
            shutil.rmtree(self.chroma_dir)

        print(f"Criando embeddings com o modelo {self.embed_model} e populando Chroma na pasta {self.chroma_dir}...")
        try:
            self.vectorstore = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory=self.chroma_dir)
            if persist:
                try:
                    self.vectorstore.persist()
                except Exception:
                    # nem todas as versões do vectorstore têm persist
                    pass
        except Exception as e:
            print("Erro ao criar/ popular o vetorstore:", e)
            raise

        # cria o retriever e o pipeline RAG
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        try:
            self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever)
        except Exception as e:
            print("Erro ao criar RetrievalQA:", e)
            raise

        print("Índice construído com sucesso.")

    def load_index(self):
        # tenta carregar um Chroma persistido
        if not os.path.isdir(self.chroma_dir):
            raise FileNotFoundError(f"Diretório Chroma não encontrado: {self.chroma_dir}. Rode build primeiro.")
        print("Carregando Chroma persistido...")
        self.vectorstore = Chroma(persist_directory=self.chroma_dir, embedding_function=self.embeddings)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever)
        print("Carregado e pipeline criado.")

    def ask(self, question: str) -> str:
        if not self.qa_chain:
            raise RuntimeError("Chain RAG não inicializada. Rode build_index_from_api() ou load_index() antes.")
        print(f"Pergunta: {question}")
        try:
            answer = self.qa_chain.run(question)
            return answer
        except Exception as e:
            print("Erro ao executar QA:", e)
            raise


# ----- CLI simples para usar o módulo -----
def main():
    parser = argparse.ArgumentParser(description="RAG com Ollama + LangChain para funcionários")
    parser.add_argument("--build", action="store_true", help="Buscar funcionários e construir índice")
    parser.add_argument("--ask", type=str, help="Pergunta a fazer após construir/carregar índice")
    parser.add_argument("--base_url", type=str, default="http://localhost:3000", help="URL base da API da sua aplicação")
    parser.add_argument("--chroma_dir", type=str, default="./chroma_db", help="Pasta onde Chroma persiste dados")
    parser.add_argument("--embed_model", type=str, default=os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large"))
    parser.add_argument("--llm_model", type=str, default=os.getenv("OLLAMA_LLM_MODEL", "llama3.1"))
    args = parser.parse_args()

    api = RequestAPI(base_url=args.base_url)
    rag = RAGManager(request_api=api, embed_model=args.embed_model, llm_model=args.llm_model, chroma_dir=args.chroma_dir)

    try:
        if args.build:
            rag.build_index_from_api(persist=True, overwrite=False)

        # se já existe índice, carrega
        if not rag.qa_chain:
            try:
                rag.load_index()
            except FileNotFoundError:
                print("Índice não encontrado. Rode --build primeiro.")
                sys.exit(1)

        if args.ask:
            answer = rag.ask(args.ask)
            print("\nResposta da IA:\n", answer)
    except Exception as e:
        print("Erro executando fluxo RAG:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
