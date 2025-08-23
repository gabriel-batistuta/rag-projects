from abc import ABC, abstractmethod

import requests
# from fastapi import FastAPI, HTTPException
# from uvicorn import run
import dotenv

class API(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url

# class AI_API():
    # pass
    
class RequestAPI():
    def __init__(self, base_url:str="http://localhost:3000"):
        self.base_url = base_url
        # self.email = 
        self.__token = self.login_admin()
        if not self.__token:
            self.__token = dotenv.get_key(".env", "ADMIN_TOKEN")
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
        token = credentials["token"]
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
        token = credentials["token"]
        dotenv.set_key(".env", "SUPER_TOKEN", token)
        return token

    def pegar_funcionarios(self):
        url = f"{self.base_url}/api/v1/funcionarios"
        headers = {
            "Authorization": f"Bearer {self.__token}"
        }

        response = requests.get(url, headers=headers)
        print(response.status_code, response.json())
        return response.json()

    def criar_loja(self):
        # adicionar credenciais de usuário da loja ao .env logo depois de criar a loja com sucesso 

        """
        "informacao_loja": {
            "nome_da_loja": "Minhadsfdfs fffLoja Exemplo",
            "nome_dono": "Fulano de Tal",
            "forma_de_pagamento": "cartão",
            "endereco": "Rua Exemplo, 123",
            "cidade": "São Paulo",
            "estado": "SP",
            "cnpj": "12345678142227901234",
            "telefone": "113999992224999",
            "email": "contato@minh48loja.com",
            "plano_contratado": "premium",
            "data_vencimento_plano": "2025-12-31",
                "ativo": True,
                "usuario": {
                    "password": "123456",
                    "password_confirmation": "123456"
                }
            }
        """

        url = f"{self.base_url}/api/v1/informacoes_lojas"

        headers = {
            "Authorization": f"Bearer {self.__token_super}"
        }

        response = requests.post(url, headers=headers, json={
            "informacao_loja": {
            "nome_da_loja": "Minhadsfdfs fffLoja Exemplo",
            "nome_dono": "Fulano de Tal",
            "forma_de_pagamento": "cartão",
            "endereco": "Rua Exemplo, 123",
            "cidade": "São Paulo",
            "estado": "SP",
            "cnpj": "12345678142227901234",
            "telefone": "113999992224999",
            "email": "contato@minh48loja.com",
            "plano_contratado": "premium",
            "data_vencimento_plano": "2025-12-31",
                "ativo": True,
                "usuario": {
                    "password": "123456",
                    "password_confirmation": "123456"
                }
            }
        })

        print(response.status_code, response.json())

    def get_todos_usuarios(self):
        url = f"{self.base_url}/usuarios"
        headers = {
            "Authorization": f"Bearer {self.__token_super}"
        }

        response = requests.get(url, headers=headers)
        r = response.json()
        for usuario in r:
            print("Usuário: ")
            print(usuario)
            print("----------")  
    
    

if __name__ == "__main__":
    r = RequestAPI()
    r.pegar_funcionarios()