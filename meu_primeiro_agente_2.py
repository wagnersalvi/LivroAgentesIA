# meu_primeiro_agente.py

# --- 1. Importações Necessárias ---
from dotenv import load_dotenv
import os

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
    post_slack_message_function
)

# --- 2. Preparando o Terreno: Configuração do Ambiente e Chaves de API ---
load_dotenv() # Carrega as variáveis de ambiente do arquivo .env

# Verifica e obtém as chaves de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENAI_API_KEY:
    print("Erro: A chave OPENAI_API_KEY não está configurada no arquivo .env")
    exit()
# SerpAPI_API_KEY é opcional se não for usar a ferramenta de busca, mas vamos manter por compatibilidade
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
    # Nova Ferramenta para Envio de E-mail
    Tool(
        name="Send Email",
        func=send_email_function,
        description="Útil para enviar e-mails. Parâmetros: recipient_email (str), subject (str), body (str)."
    ),
    # Nova Ferramenta para Criar Evento de Calendário
    Tool(
        name="Create Calendar Event",
        func=create_calendar_event_function,
        description="Útil para agendar um evento no calendário. Parâmetros: title (str), start_time (str YYYY-MM-DD HH:MM), end_time (str YYYY-MM-DD HH:MM), attendees (str, e-mails separados por vírgula), description (str, opcional)."
    ),
    # Nova Ferramenta para Verificar Disponibilidade no Calendário
    Tool(
        name="Check Calendar Availability",
        func=check_calendar_availability_function,
        description="Útil para verificar a disponibilidade de participantes para um evento. Parâmetros: start_time (str YYYY-MM-DD HH:MM), end_time (str YYYY-MM-DD HH:MM), attendees (str, e-mails separados por vírgula)."
    ),
    # Nova Ferramenta para Postar no Slack
    Tool(
        name="Post Slack Message",
        func=post_slack_message_function,
        description="Útil para enviar mensagens para canais do Slack. Parâmetros: channel (str, nome do canal sem #), message (str)."
    )
]

# --- 5. Montando o Agente: O Coração do Nosso Primeiro Sistema Autônomo ---

# Prompt do Agente (com placeholder para chat_history)
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

    print("\n\n--- Testes do Agente COM Memória e Novas Ferramentas ---")
    print("Vamos testar as novas capacidades de ação do agente.")

    # --- Exemplo 1: Enviar E-mail ---
    try:
        print("\nUsuário: Por favor, envie um e-mail para 'test@example.com' com o assunto 'Relatório Diário' e o corpo 'Segue o relatório de vendas de hoje.'.")
        agent_executor_with_memory.invoke({"input": "Por favor, envie um e-mail para 'test@example.com' com o assunto 'Relatório Diário' e o corpo 'Segue o relatório de vendas de hoje.'."})
    except Exception as e:
        print(f"Erro ao executar ação de e-mail: {e}")

    # --- Exemplo 2: Criar Evento de Calendário ---
    try:
        print("\nUsuário: Agende uma reunião 'Alinhamento de Projeto' para amanhã, 10:00 às 11:00, com 'ana@empresa.com, joao@empresa.com'.")
        # Note: 'amanhã' será interpretado pelo LLM. Ele precisará deduzir a data correta.
        # Para demonstração: ele vai usar a data atual + 1 dia.
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        start_time = tomorrow.strftime('%Y-%m-%d 10:00')
        end_time = tomorrow.strftime('%Y-%m-%d 11:00')

        # Para que o LLM não precise calcular a data, podemos passá-la diretamente no prompt.
        # Ou podemos ter uma ferramenta 'get_current_date'
        print(f"   (Assumindo amanhã é {tomorrow.strftime('%Y-%m-%d')})")
        agent_executor_with_memory.invoke({"input": f"Agende uma reunião 'Alinhamento de Projeto' para {start_time} a {end_time}, com 'ana@empresa.com, joao@empresa.com'."})
    except Exception as e:
        print(f"Erro ao executar ação de calendário: {e}")
        
    # --- Exemplo 3: Verificar Disponibilidade no Calendário ---
    try:
        print("\nUsuário: Verifique se 'carlos@empresa.com, maria@empresa.com' estão disponíveis entre '2024-12-10 14:00' e '2024-12-10 15:00'.")
        agent_executor_with_memory.invoke({"input": "Verifique se 'carlos@empresa.com, maria@empresa.com' estão disponíveis entre '2024-12-10 14:00' e '2024-12-10 15:00'."})
    except Exception as e:
        print(f"Erro ao verificar disponibilidade: {e}")

    # --- Exemplo 4: Postar Mensagem no Slack ---
    try:
        print("\nUsuário: Por favor, poste a mensagem 'Lembrete: reunião de equipe às 14h' no canal 'geral' do Slack.")
        agent_executor_with_memory.invoke({"input": "Por favor, poste a mensagem 'Lembrete: reunião de equipe às 14h' no canal 'geral' do Slack."})
    except Exception as e:
        print(f"Erro ao executar ação no Slack: {e}")

    # --- Exemplo 5: Combinando Ações (Agendamento Inteligente - demonstração inicial) ---
    print("\n--- Demonstração de Agendamento Inteligente ---")
    try:
        # Note que o LLM terá que orquestrar a verificação e a criação,
        # além de deduzir a data de amanhã se não for explícita.
        # Para este exemplo, vou dar uma data explícita para focar na orquestração.
        next_week = today + datetime.timedelta(days=7)
        proposed_start = next_week.strftime('%Y-%m-%d 14:00')
        proposed_end = next_week.strftime('%Y-%m-%d 15:00')
        attendees_for_smart_schedule = "julia@empresa.com, pedro@empresa.com"
        
        print(f"\nUsuário: Agende uma reunião de 'Brainstorm de Marketing' para a próxima semana, no dia {next_week.strftime('%Y-%m-%d')}, das 14:00 às 15:00, com {attendees_for_smart_schedule}. Se todos estiverem disponíveis, envie também um e-mail de confirmação.")
        agent_executor_with_memory.invoke({"input": f"Agende uma reunião de 'Brainstorm de Marketing' para a próxima semana, no dia {next_week.strftime('%Y-%m-%d')}, das 14:00 às 15:00, com {attendees_for_smart_schedule}. Se todos estiverem disponíveis, envie também um e-mail de confirmação."})

    except Exception as e:
        print(f"Erro na demonstração de agendamento inteligente: {e}")
        print("Verifique se o LLM foi capaz de orquestrar as ações.")
