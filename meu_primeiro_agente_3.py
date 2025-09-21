# meu_primeiro_agente.py (final)

# --- 1. Importações Necessárias ---
from dotenv import load_dotenv
import os
import datetime # Adicionado para uso com datas no exemplo

# Importações do LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import SerpAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

# Importa as funções do módulo de ferramentas que acabamos de criar
from tools_module import (
    send_email_function,
    create_calendar_event_function,
    check_calendar_availability_function,
    post_slack_message_function,
    query_knowledge_base_function # Nova função importada!
)

# --- 2. Preparando o Terreno: Configuração do Ambiente e Chaves de API ---
load_dotenv() # Carrega as variáveis de ambiente do arquivo .env

# Verifica e obtém as chaves de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENAI_API_KEY:
    print("Erro: A chave OPENAI_API_KEY não está configurada no arquivo .env")
    exit()
if not SERPAPI_API_KEY:
    print("Aviso: SERPAPI_API_KEY não configurada. A ferramenta 'Google Search' não funcionará.")


print("Chaves de API carregadas com sucesso!")

# --- 3. A Primeira Peça: Conectando-se ao Cérebro (o LLM) ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# --- 4. Dando Olhos e Mãos ao Agente: Criando Ferramentas (Tools) ---
# Lista de ferramentas que o agente poderá usar

tools = [
    # Ferramenta de Busca na Web (do Capítulo 5)
    Tool(
        name="Google Search",
        func=SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY).run if SERPAPI_API_KEY else lambda x: "SerpAPI Key não configurada, busca na web indisponível.",
        description="Útil para buscar informações gerais na internet, sobre pessoas, lugares, eventos, definições e fatos atuais."
    ),
    # Ferramenta para Envio de E-mail (do Capítulo 6)
    Tool(
        name="Send Email",
        func=send_email_function,
        description="Útil para enviar e-mails. Parâmetros: recipient_email (str), subject (str), body (str)."
    ),
    # Ferramenta para Criar Evento de Calendário (do Capítulo 6)
    Tool(
        name="Create Calendar Event",
        func=create_calendar_event_function,
        description="Útil para agendar um evento no calendário. Parâmetros: title (str), start_time (str YYYY-MM-DD HH:MM), end_time (str YYYY-MM-DD HH:MM), attendees (str, e-mails separados por vírgula), description (str, opcional)."
    ),
    # Ferramenta para Verificar Disponibilidade no Calendário (do Capítulo 6)
    Tool(
        name="Check Calendar Availability",
        func=check_calendar_availability_function,
        description="Útil para verificar a disponibilidade de participantes para um evento. Parâmetros: start_time (str YYYY-MM-DD HH:MM), end_time (str YYYY-MM-DD HH:MM), attendees (str, e-mails separados por vírgula)."
    ),
    # Ferramenta para Postar no Slack (do Capítulo 6)
    Tool(
        name="Post Slack Message",
        func=post_slack_message_function,
        description="Útil para enviar mensagens para canais do Slack. Parâmetros: channel (str, nome do canal sem #), message (str)."
    ),
    # NOVA Ferramenta para Consultar a Base de Conhecimento (deste capítulo!)
    Tool(
        name="Query Internal Knowledge Base",
        func=query_knowledge_base_function,
        description="Útil para consultar informações internas da empresa, como políticas, FAQs ou documentos. Use para perguntas sobre regras, procedimentos ou informações específicas da organização."
    )
]

# --- 5. Montando o Agente: O Coração do Nosso Primeiro Sistema Autônomo ---

# Prompt do Agente (com placeholder para chat_history)
# O prompt foi ligeiramente ajustado para incluir a nova ferramenta.
prompt_template_with_memory = PromptTemplate.from_template("""
Você é um agente de IA útil e atencioso.
Seu objetivo é responder perguntas da melhor forma possível, utilizando as ferramentas disponíveis e executando ações quando apropriado.
Você tem acesso às seguintes ferramentas:

{tools}

Aqui está o histórico da sua conversa com o usuário:
{chat_history}

Para responder a uma pergunta ou executar uma ação, siga este processo:
1. Pense no que você precisa fazer.
2. Se precisar de uma ferramenta, use 'Action:' e 'Action Input:'.
3. Se tiver a resposta final, use 'Final Answer:'.
4. Sempre forneça a 'Final Answer' ao usuário depois de completar uma tarefa, mesmo que a tarefa seja uma ação.

Formato do seu raciocínio e ações:
Question: a pergunta/requisição de entrada
Thought: você deve sempre pensar no que fazer
Action: a ação a ser executada, deve ser uma das {tool_names}
Action Input: a entrada para a ação (NÃO inclua as aspas duplas, apenas o valor)
Observation: o resultado da ação
... (este Thought/Action/Action Input/Observation pode se repetir várias vezes)
Thought: eu sei a resposta final
Final Answer: a resposta final à pergunta original

Comece!

Question: {input}
{agent_scratchpad}
""")

# Inicializa a Memória (ConversationBufferMemory)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Cria o Agente ReAct com Memória
agent_with_memory = create_react_agent(llm, tools, prompt_template_with_memory)
agent_executor_with_memory = AgentExecutor(agent=agent_with_memory, tools=tools, verbose=True, memory=memory)


# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    print("\n--- Teste de Conexão do LLM (reutilizando do Capítulo 5) ---")
    try:
        response_llm = llm.invoke("Qual é a capital da França?")
        print(f"Resposta do LLM: {response_llm.content}")
    except Exception as e:
        print(f"Erro ao chamar o LLM: {e}")
        print("Verifique sua chave de API da OpenAI e sua conexão com a internet.")
        exit()

    print("\n\n--- Teste do Agente COM Memória de Longo Prazo (RAG) ---")
    print("Vamos testar as novas capacidades do agente de consultar nossa base de conhecimento interna.")

    # --- Exemplo 1: Perguntar sobre Políticas Internas ---
    try:
        print("\nUsuário: Qual a política de férias da empresa?")
        agent_executor_with_memory.invoke({"input": "Qual a política de férias da empresa?"})

        print("\nUsuário: Existe algum subsídio para desenvolvimento profissional?")
        agent_executor_with_memory.invoke({"input": "Existe algum subsídio para desenvolvimento profissional?"})
        
        print("\nUsuário: Qual o limite de reembolso para refeições?")
        agent_executor_with_memory.invoke({"input": "Qual o limite de reembolso para refeições?"})

        print("\nUsuário: Qual o modelo de trabalho adotado pela empresa?")
        agent_executor_with_memory.invoke({"input": "Qual o modelo de trabalho adotado pela empresa?"})

    except Exception as e:
        print(f"Erro ao interagir com o agente e base de conhecimento: {e}")
        print("Verifique se o script 'setup_chromadb.py' foi executado e a pasta 'chroma_db' existe.")
        
    print("\n--- Testando combinação de conhecimento interno e externo ---")
    try:
        print("\nUsuário: Qual a política de trabalho remoto e quem é o atual CEO da OpenAI?")
        agent_executor_with_memory.invoke({"input": "Qual a política de trabalho remoto e quem é o atual CEO da OpenAI?"})
    except Exception as e:
        print(f"Erro ao combinar conhecimentos: {e}")

    # --- Testes de Ações do Capítulo 6 (mantidos para referência) ---
    print("\n\n--- Testes de Ações (Capítulo 6 - Mantidos para Referência) ---")

    # --- Exemplo 2: Enviar E-mail ---
    try:
        print("\nUsuário: Por favor, envie um e-mail para 'test@example.com' com o assunto 'Relatório Diário' e o corpo 'Segue o relatório de vendas de hoje.'.")
        agent_executor_with_memory.invoke({"input": "Por favor, envie um e-mail para 'test@example.com' com o assunto 'Relatório Diário' e o corpo 'Segue o relatório de vendas de hoje.'."})
    except Exception as e:
        print(f"Erro ao executar ação de e-mail: {e}")

    # --- Exemplo 3: Criar Evento de Calendário (data dinâmica) ---
    try:
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        start_time = tomorrow.strftime('%Y-%m-%d 10:00')
        end_time = tomorrow.strftime('%Y-%m-%d 11:00')
        print(f"\nUsuário: Agende uma reunião 'Alinhamento de Projeto' para {start_time} a {end_time}, com 'ana@empresa.com, joao@empresa.com'.")
        agent_executor_with_memory.invoke({"input": f"Agende uma reunião 'Alinhamento de Projeto' para {start_time} a {end_time}, com 'ana@empresa.com, joao@empresa.com'."})
    except Exception as e:
        print(f"Erro ao executar ação de calendário: {e}")
