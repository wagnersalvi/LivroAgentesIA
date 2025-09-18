# meu_primeiro_agente.py

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate # Para prompts mais estruturados
from langchain.chains import LLMChain # Para encadear o LLM

# 1. Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Verifica se as chaves de API estão configuradas
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Erro: A chave OPENAI_API_KEY não está configurada no arquivo .env")
    exit()

# 2. Inicializa o Modelo de Linguagem (LLM)
# Escolhemos o modelo gpt-3.5-turbo por ser rápido e eficiente para este exemplo
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

# Conceito de Temperatura:
# A 'temperature' controla a aleatoriedade da saída do LLM.
# Um valor de 0.0 tornará a saída muito determinística e previsível (ótimo para tarefas de fatos).
# Um valor mais alto (até 1.0) tornará a saída mais criativa e variada (bom para geração de texto).
# Para este agente, 0.7 oferece um bom equilíbrio.

# Exemplo de uma requisição básica para testar a conexão com o LLM
if __name__ == "__main__":
    print("Conexão com o LLM estabelecida. Testando uma pergunta simples:")
    try:
        response = llm.invoke("Qual é a capital da França?")
        print(f"Resposta do LLM: {response.content}")

        # Exemplo com um prompt mais elaborado usando PromptTemplate e LLMChain
        prompt = PromptTemplate.from_template("Responda à pergunta: {question}")
        chain = LLMChain(llm=llm, prompt=prompt)
        response_chain = chain.invoke({"question": "Qual é a capital da Alemanha?"})
        print(f"Resposta do LLM (via Chain): {response_chain['text']}")

    except Exception as e:
        print(f"Erro ao chamar o LLM: {e}")
        print("Verifique sua chave de API da OpenAI e sua conexão com a internet.")
