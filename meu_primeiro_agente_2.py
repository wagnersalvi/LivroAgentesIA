# meu_primeiro_agente.py (continuação)

# ... (código anterior) ...
from langchain_community.tools import SerpAPIWrapper # Importa a ferramenta de busca da SerpAPI
from langchain.agents import AgentExecutor, create_react_agent # Importa os componentes do agente
from langchain.tools import Tool # Para definir ferramentas personalizadas

# ... (código anterior de carregamento de .env e verificação de chaves) ...

serpapi_api_key = os.getenv("SERPAPI_API_KEY")
if not serpapi_api_key:
    print("Erro: A chave SERPAPI_API_KEY não está configurada no arquivo .env")
    exit()

# 3. Inicializa o Modelo de Linguagem (LLM) - Já feito anteriormente

# 4. Cria a Ferramenta de Busca na Web (Tool)
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

# Define a ferramenta de forma que o agente LangChain possa usá-la
# É crucial que o 'name' e 'description' sejam claros, pois o LLM os usa para decidir quando e como usar a ferramenta.
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Útil para buscar informações gerais na internet, sobre pessoas, lugares, eventos, definições e fatos atuais."
    )
]

if __name__ == "__main__":
    # ... (código de teste de conexão com LLM) ...
    
    print("\nFerramenta 'Google Search' configurada.")
    # Exemplo de como a ferramenta de busca funciona isoladamente
    # try:
    #     search_query = "Quem é o atual presidente do Brasil?"
    #     print(f"Testando a ferramenta de busca com a query: '{search_query}'")
    #     search_result = search.run(search_query)
    #     print(f"Resultado da busca: {search_result[:200]}...") # Limita a saída para não sobrecarregar
    # except Exception as e:
    #     print(f"Erro ao usar a ferramenta SerpAPI: {e}")
    #     print("Verifique sua chave de API da SerpAPI e sua conexão com a internet.")
