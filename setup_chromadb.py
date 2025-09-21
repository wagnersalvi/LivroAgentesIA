# setup_chromadb.py

from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Verifica a chave da OpenAI, necessária para os embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Erro: A chave OPENAI_API_KEY não está configurada no arquivo .env.")
    print("Esta chave é necessária para gerar embeddings para a base de conhecimento.")
    exit()

# 1. Carregar documentos
# Opcional: Instale 'unstructured' para suportar mais tipos de arquivos (PDFs, DOCX)
# pip install unstructured
loader = TextLoader("politicas_empresa.txt")
documents = loader.load()

# 2. Dividir documentos em chunks menores
# Isso é importante porque os embeddings funcionam melhor com pedaços menores e o LLM tem limite de contexto.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 3. Criar Embeddings
# Usamos OpenAIEmbeddings para converter o texto em vetores numéricos.
# Você pode usar outros modelos de embeddings, como Sentence Transformers da Hugging Face.
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 4. Armazenar os embeddings no ChromaDB
# Criamos um diretório persistente para o ChromaDB, para que os dados sejam salvos.
persist_directory = "./chroma_db"
vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist() # Garante que os dados sejam escritos no disco

print(f"Base de conhecimento salva em {persist_directory} com {len(docs)} documentos indexados.")
print("Execute este script sempre que houver mudanças nos documentos da base de conhecimento.")
