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
# ... (definição dos Agentes) ...

# 1. Tarefa do Agente de Pesquisa
research_task = Task(
    description=(
        "Pesquise as últimas tendências e informações sobre o lançamento do 'Novo Recurso X' em inteligência artificial."
        "Foque em casos de uso práticos, benefícios para o usuário e diferenciais competitivos."
        "Colete links relevantes, citações e estatísticas para apoiar o conteúdo."
    ),
    expected_output=(
        "Um relatório conciso e detalhado com os principais pontos de pesquisa, incluindo: "
        "1. Principais casos de uso do Novo Recurso X. "
        "2. Benefícios claros para os usuários. "
        "3. Como se diferencia da concorrência (se relevante). "
        "4. 3-5 links para artigos ou notícias relevantes. "
        "5. Citações ou estatísticas que podem ser usadas no post."
    ),
    agent=researcher # Esta tarefa será atribuída ao 'researcher'
)

# 2. Tarefa do Agente de Conteúdo
content_task = Task(
    description=(
        "Com base no relatório de pesquisa do 'Agente de Pesquisa',"
        "gere 3 ideias de conteúdo criativas para um post de rede social sobre o 'Novo Recurso X'."
        "Para cada ideia, forneça:"
        "1. Um título impactante. "
        "2. O público-alvo principal. "
        "3. Os 3 principais pontos a serem comunicados. "
        "4. Um call-to-action (CTA) claro."
    ),
    expected_output=(
        "Uma lista de 3 ideias de conteúdo formatadas. Exemplo:"
        "---"
        "Ideia 1: [Título]"
        "Público: [Público]"
        "Pontos: [Ponto 1, Ponto 2, Ponto 3]"
        "CTA: [Call-to-Action]"
        "---"
        "Ideia 2: ..."
    ),
    agent=content_creator # Esta tarefa será atribuída ao 'content_creator'
)

# 3. Tarefa do Agente de Redator Criativo
# Esta tarefa receberá a saída do Agente de Conteúdo como entrada
write_task = Task(
    description=(
        "Baseado nas ideias de conteúdo geradas pelo 'Agente de Conteúdo',"
        "escolha a melhor ideia e desenvolva um post completo para rede social (ex: LinkedIn/X - Twitter) sobre o 'Novo Recurso X'."
        "O post deve ser envolvente, usar emojis relevantes, ter hashtags apropriadas e incorporar um Call-to-Action eficaz."
        "Garanta que o tom seja profissional, mas acessível, e que o post tenha no máximo 280 caracteres (para X) ou seja conciso para LinkedIn."
    ),
    expected_output=(
        "O post final para rede social, pronto para publicação, incluindo texto, emojis e hashtags."
        "O post deve ser otimizado para engajamento."
    ),
    agent=creative_writer # Esta tarefa será atribuída ao 'creative_writer'
)

# Monta a equipe (Crew)
social_media_crew = Crew(
    agents=[researcher, content_creator, creative_writer], # Todos os agentes da equipe
    tasks=[research_task, content_task, write_task],       # Todas as tarefas a serem executadas
    process=Process.sequential,                            # Ordem sequencial
    verbose=True                                           # Para ver o fluxo de trabalho detalhado
)

# Inicia o processo da equipe
if __name__ == "__main__":
    print("Iniciando a equipe de criação de conteúdo para redes sociais...")
    result = social_media_crew.kickoff() # 'kickoff()' inicia o processo!
    print("\n--- Resultado Final da Equipe ---")
    print(result)
