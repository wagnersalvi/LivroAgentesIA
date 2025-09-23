# tools_module.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import random
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA # Para a Chain de RAG
from langchain_openai import ChatOpenAI
import os

# Exemplo SIMPLIFICADO de função para enviar e-mail.
# Em produção, usaria a API real do Gmail, Outlook, etc., com OAuth2.0
def send_email_function(recipient_email: str, subject: str, body: str) -> str:
    """
    Simula o envio de um e-mail para um destinatário.
    Em um ambiente real, esta função se conectaria a um servidor SMTP ou usaria uma API de e-mail (ex: Gmail API).
    Retorna uma mensagem de sucesso ou erro.
    """
    print(f"--- Simulação de Envio de E-mail ---")
    print(f"Para: {recipient_email}")
    print(f"Assunto: {subject}")
    print(f"Corpo: {body}")
    print(f"------------------------------------")

    # Apenas para demonstração: simulamos sucesso.
    # Em um cenário real, você teria a lógica de conexão SMTP aqui.
    if "@" not in recipient_email:
        return f"Erro: Endereço de e-mail inválido: {recipient_email}"
    return f"E-mail com assunto '{subject}' enviado com sucesso para {recipient_email}."

# Exemplo SIMPLIFICADO de funções para interagir com um calendário.
# Em produção, usaria a Google Calendar API, Outlook Calendar API, etc.
def create_calendar_event_function(title: str, start_time: str, end_time: str, attendees: str, description: str = "") -> str:
    """
    Simula a criação de um evento de calendário.
    Args:
        title (str): Título do evento.
        start_time (str): Data e hora de início do evento (ex: '2024-12-25 09:00').
        end_time (str): Data e hora de fim do evento (ex: '2024-12-25 10:00').
        attendees (str): Lista de e-mails dos participantes, separados por vírgula.
        description (str): Descrição do evento.
    Returns:
        str: Mensagem de confirmação ou erro.
    """
    print(f"--- Simulação de Criação de Evento de Calendário ---")
    print(f"Título: {title}")
    print(f"Início: {start_time}")
    print(f"Fim: {end_time}")
    print(f"Participantes: {attendees}")
    print(f"Descrição: {description}")
    print(f"--------------------------------------------------")

    # Simula validação de data
    try:
        start_dt = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M')
        end_dt = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M')
        if start_dt >= end_dt:
            return "Erro: A data/hora de início deve ser anterior à data/hora de término."
    except ValueError:
        return "Erro: Formato de data/hora inválido. Use 'AAAA-MM-DD HH:MM'."

    return f"Evento '{title}' agendado com sucesso de {start_time} a {end_time} com {attendees}."

def check_calendar_availability_function(start_time: str, end_time: str, attendees: str) -> str:
    """
    Simula a verificação de disponibilidade de participantes em um período.
    Em um ambiente real, esta função consultaria a API do calendário.
    Retorna uma string indicando a disponibilidade.
    """
    print(f"--- Simulação de Verificação de Disponibilidade ---")
    print(f"Início: {start_time}")
    print(f"Fim: {end_time}")
    print(f"Participantes: {attendees}")
    print(f"--------------------------------------------------")

    try:
        start_dt = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M')
        end_dt = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M')
        if start_dt >= end_dt:
            return "Erro: A data/hora de início deve ser anterior à data/hora de término."
    except ValueError:
        return "Erro: Formato de data/hora inválido. Use 'AAAA-MM-DD HH:MM'."
    
    # Simula disponibilidade aleatória para alguns participantes
    attendee_list = [a.strip() for a in attendees.split(',') if a.strip()]
    unavailable = []
    for attendee in attendee_list:
        if random.random() < 0.2: # 20% de chance de estar indisponível
            unavailable.append(attendee)
            
    if unavailable:
        return f"Os seguintes participantes estão indisponíveis entre {start_time} e {end_time}: {', '.join(unavailable)}. Sugira outro horário."
    else:
        return f"Todos os participantes ({attendees}) estão disponíveis entre {start_time} e {end_time}."

def post_slack_message_function(channel: str, message: str) -> str:
    """
    Simula o envio de uma mensagem para um canal do Slack.
    Em um ambiente real, esta função usaria a API do Slack.
    """
    print(f"--- Simulação de Post no Slack ---")
    print(f"Canal: #{channel}")
    print(f"Mensagem: {message}")
    print(f"---------------------------------")
    if not channel.strip():
        return "Erro: O canal do Slack não pode ser vazio."
    return f"Mensagem postada com sucesso no canal #{channel}."


# Certifique-se que o OPENAI_API_KEY está disponível como variável de ambiente
# ou passe-o como parâmetro para OpenAIEmbeddings e ChatOpenAI

def query_knowledge_base_function(query: str) -> str:
    """
    Consulta a base de conhecimento ChromaDB para obter informações relevantes.
    Parâmetros: query (str) - A pergunta a ser feita à base de conhecimento.
    """
    # A base de conhecimento deve ter sido persistida em ./chroma_db
    persist_directory = "./chroma_db"

    # Carrega o modelo de embeddings (mesmo usado para indexar)
    # Certifique-se que OPENAI_API_KEY está configurada no ambiente
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Carrega a base de dados vetorial existente
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)

    # Cria um retrievador (retriever) a partir do VectorDB
    retriever = vectordb.as_retriever()

    # Cria um LLM para a etapa de Geração (o mesmo LLM principal do agente ou um específico para RAG)
    llm_rag = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Cria a cadeia de RetrievalQA (RAG)
    qa_chain = RetrievalQA.from_chain_type(llm=llm_rag, chain_type="stuff", retriever=retriever)

    # Executa a query
    try:
        result = qa_chain.invoke({"query": query})
        return result['result']
    except Exception as e:
        return f"Erro ao consultar a base de conhecimento: {e}"



from textblob import TextBlob # pip install textblob
def analyze_sentiment(text: str) -> str:
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positivo"
    elif analysis.sentiment.polarity < 0:
        return "Negativo"
    else:
        return "Neutro"
# Adicionar ao agente:
# Tool(name="Sentiment Analyzer", func=analyze_sentiment, description="Útil para analisar o sentimento de um texto (Positivo, Negativo, Neutro).")


# Simulação: em um cenário real, usaria um LLM ou um serviço de geração de código
def generate_code_snippet(description: str, language: str) -> str:
    if language.lower() == "python":
        return f"# Python code for: {description}\ndef example_function():\n    pass"
    elif language.lower() == "javascript":
        return f"// Javascript code for: {description}\nfunction exampleFunction() {{}}"
    return "Linguagem não suportada para geração de snippet."
# Adicionar ao agente:
# Tool(name="Code Generator", func=generate_code_snippet, description="Gera um snippet de código em uma linguagem específica (python, javascript) dado uma descrição.")



import sqlite3 # Exemplo com SQLite
def query_database(sql_query: str) -> str:
    try:
        conn = sqlite3.connect('my_data.db') # Conexão com DB
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return str(results) # Retorna resultados como string
    except Exception as e:
        return f"Erro ao executar SQL: {e}"
# Adicionar ao agente:
# Tool(name="Database Query", func=query_database, description="Executa uma consulta SQL segura em um banco de dados interno. Útil para extrair dados brutos.")
