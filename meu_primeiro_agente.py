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

# --- 2. Preparando o Terreno: Configuração do Ambiente e Chaves de API ---
load_dotenv() # Carrega as variáveis de ambiente do arquivo .env

# Verifica e obtém as chaves de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENAI_API_KEY:
    print("Erro: A chave OPENAI_API_KEY não está configurada no arquivo .env")
    exit()
if not SERPAPI_API_KEY:
    print("Erro: A chave SERPAPI_API_KEY não está configurada no arquivo .env")
    exit()

print("Chaves de API carregadas com sucesso!")

# --- 3. A Primeira Peça: Conectando-se ao Cérebro (o LLM) ---
# Inicializa o Modelo de Linguagem (LLM) da OpenAI
# Usamos 'gpt-3.5-turbo' por ser rápido e eficiente para este exemplo.
# A 'temperature' controla a aleatoriedade da saída (0.0 para mais determinismo, 1.0 para mais criatividade).
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# --- 4. Dando Olhos e Mãos ao Agente: Criando uma Ferramenta (Tool) ---
# Inicializa o wrapper da SerpAPI para buscas no Google
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

# Define a ferramenta 'Google Search' para o agente.
# A 'name' é o identificador e a 'description' é crucial para o LLM decidir quando usar a ferramenta.
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Útil para buscar informações gerais na internet, sobre pessoas, lugares, eventos, definições e fatos atuais."
    )
]

# --- 5. Montando o Agente: O Coração do Nosso Primeiro Sistema Autônomo ---

# 5.1. Prompt do Agente (sem memória, para o primeiro teste)
# Este prompt instrui o LLM sobre seu papel, as ferramentas disponíveis e o formato de raciocínio (ReAct).
prompt_template_no_memory = PromptTemplate.from_template("""
Você é um agente de IA útil e atencioso.
Seu objetivo é responder perguntas da melhor forma possível, utilizando as ferramentas disponíveis.
Você tem acesso às seguintes ferramentas:

{tools}

Para responder a uma pergunta, siga este processo:
1. Pense no que você precisa fazer.
2. Se precisar de uma ferramenta, use 'Action:' e 'Action Input:'.
3. Se tiver a resposta final, use 'Final Answer:'.

Formato do seu raciocínio e ações:
Question: a pergunta de entrada
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

# Cria o Agente ReAct (sem memória para este executor)
agent_no_memory = create_react_agent(llm, tools, prompt_template_no_memory)
# Cria o Executor do Agente (verbose=True para ver o raciocínio passo a passo)
agent_executor_no_memory = AgentExecutor(agent=agent_no_memory, tools=tools, verbose=True)


# --- 6. Adicionando Memória: Lembre-se do Passado para uma Conversa Contínua ---

# 6.1. Prompt do Agente com histórico de chat
# Adicionamos o placeholder {chat_history} para que o LLM tenha acesso às interações passadas.
prompt_template_with_memory = PromptTemplate.from_template("""
Você é um agente de IA útil e atencioso.
Seu objetivo é responder perguntas da melhor forma possível, utilizando as ferramentas disponíveis.
Você tem acesso às seguintes ferramentas:

{tools}

Aqui está o histórico da sua conversa com o usuário:
{chat_history}

Para responder a uma pergunta, siga este processo:
1. Pense no que você precisa fazer.
2. Se precisar de uma ferramenta, use 'Action:' e 'Action Input:'.
3. Se tiver a resposta final, use 'Final Answer:'.

Formato do seu raciocínio e ações:
Question: a pergunta de entrada
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

# 6.2. Inicializa a Memória (ConversationBufferMemory)
# 'memory_key' deve corresponder ao placeholder no prompt ('chat_history').
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 6.3. Cria o Agente ReAct com Memória
agent_with_memory = create_react_agent(llm, tools, prompt_template_with_memory)
# Cria o Executor do Agente com memória
agent_executor_with_memory = AgentExecutor(agent=agent_with_memory, tools=tools, verbose=True, memory=memory)


# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    print("\n--- Teste de Conexão do LLM ---")
    try:
        response_llm = llm.invoke("Qual é a capital da França?")
        print(f"Resposta do LLM: {response_llm.content}")
    except Exception as e:
        print(f"Erro ao chamar o LLM: {e}")
        print("Verifique sua chave de API da OpenAI e sua conexão com a internet.")
        exit() # Sai se o LLM não funcionar

    print("\n--- Teste do Agente SEM Memória ---")
    try:
        print("\nUsuário: Qual é a capital da Croácia e qual a sua população?")
        agent_executor_no_memory.invoke({"input": "Qual é a capital da Croácia e qual a sua população?"})

        print("\nUsuário: Qual o resultado de 15 * 23?")
        # O agente tentará usar a ferramenta de busca para isso,
        # pois é a única ferramenta disponível e ele busca por 'informações'.
        # Isso mostra a necessidade de ter ferramentas mais específicas (ex: calculadora)
        # ou prompts mais refinados para direcionar o uso.
        agent_executor_no_memory.invoke({"input": "Qual o resultado de 15 * 23?"})

    except Exception as e:
        print(f"Erro ao interagir com o agente sem memória: {e}")
        print("Verifique suas chaves de API e se as ferramentas estão funcionando corretamente.")
        # Não sai, para que possamos testar o agente com memória


    print("\n\n--- Teste do Agente COM Memória ---")
    print("Vamos testar uma conversa multi-turn:")
    try:
        # Primeira pergunta
        print("\nUsuário: Qual a capital da Croácia?")
        agent_executor_with_memory.invoke({"input": "Qual a capital da Croácia?"})

        # Segunda pergunta, que depende da primeira
        print("\nUsuário: E qual a moeda usada lá?")
        agent_executor_with_memory.invoke({"input": "E qual a moeda usada lá?"})
        
        # Terceira pergunta, mostrando que ele ainda mantém o contexto
        print("\nUsuário: Qual o principal ponto turístico de lá?")
        agent_executor_with_memory.invoke({"input": "Qual o principal ponto turístico de lá?"})

        # Quarta pergunta, um novo tópico para ver como o agente se adapta
        print("\nUsuário: Qual a população do Canadá?")
        agent_executor_with_memory.invoke({"input": "Qual a população do Canadá?"})


    except Exception as e:
        print(f"Erro ao interagir com o agente com memória: {e}")
        print("Verifique suas chaves de API e se as ferramentas estão funcionando corretamente.")
