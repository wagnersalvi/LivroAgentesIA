# agente_social_media.py

from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import SerpAPIWrapper # Para a ferramenta de busca

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Verifica e obtém as chaves de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENAI_API_KEY:
    print("Erro: A chave OPENAI_API_KEY não está configurada no arquivo .env")
    exit()
if not SERPAPI_API_KEY:
    print("Erro: A chave SERPAPI_API_KEY não está configurada no arquivo .env. A ferramenta de busca não funcionará.")
    exit()

# Inicializa o LLM para todos os agentes (pode ser configurado individualmente)
# Usando gpt-4 para melhor desempenho no raciocínio complexo de multiagentes.
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# --- Ferramentas Compartilhadas (ou específicas, se preferir) ---
# Nossa ferramenta de busca será usada por vários agentes.
search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

# 1. Agente de Pesquisa
researcher = Agent(
    role='Agente de Pesquisa',
    goal='Coletar informações relevantes e tendências sobre o tópico fornecido.',
    backstory='Você é um analista de mercado experiente, especializado em encontrar dados, fatos e tendências online de forma eficiente e precisa.',
    verbose=True, # Para ver o raciocínio detalhado deste agente
    allow_delegation=False, # Este agente não delega, ele executa a pesquisa
    llm=llm,
    tools=[search_tool] # Este agente usa a ferramenta de busca
)

# 2. Agente de Conteúdo
content_creator = Agent(
    role='Agente de Conteúdo',
    goal='Gerar ideias criativas e um rascunho de conteúdo persuasivo para redes sociais.',
    backstory='Você é um estrategista de conteúdo criativo, com um olhar aguçado para engajamento e storytelling. Você transforma informações em narrativas.',
    verbose=True,
    allow_delegation=True, # Pode delegar a outros se necessário, embora não aconteça neste fluxo
    llm=llm
)

# 3. Agente de Redator Criativo
creative_writer = Agent(
    role='Agente de Redator Criativo',
    goal='Refinar o rascunho do conteúdo em um post de rede social polido, envolvente e otimizado.',
    backstory='Você é um copywriter de alto nível, com maestria na arte de criar textos curtos e impactantes para mídias sociais, que geram cliques e conversões.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)
