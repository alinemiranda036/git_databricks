# Estudo de Caso - App de RAG Para Busca Semântica em Documentos - Aplicação e Uso de Modelos de Embeddings e Bancos de Dados Vetoriais

# Importa o módulo os para acessar variáveis de ambiente e funções do sistema operacional
import os

# Importa o módulo tempfile para criar arquivos temporários no disco
import tempfile

# Importa o framework Streamlit para criar a interface web
import streamlit as st

# Importa a função load_dotenv para carregar variáveis de ambiente do arquivo .env
from dotenv import load_dotenv

# Importa a classe ChatGroq para utilizar modelos LLM hospedados na plataforma Groq
from langchain_groq import ChatGroq

# Importa a classe de embeddings do Hugging Face para geração de vetores
from langchain_huggingface import HuggingFaceEmbeddings

# Importa o carregador de documentos PDF da comunidade LangChain
from langchain_community.document_loaders import PyPDFLoader

# Importa o splitter de texto recursivo para quebrar o texto em chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importa a integração com o banco vetorial Chroma
from langchain_chroma import Chroma

# Importa o template de prompt para chats do LangChain Core
from langchain_core.prompts import ChatPromptTemplate

# Importa o RunnablePassthrough para repassar valores adiante no pipeline
from langchain_core.runnables import RunnablePassthrough

# Importa o parser de saída para converter a resposta do LLM em string
from langchain_core.output_parsers import StrOutputParser

# Define o nome do modelo de embeddings a ser utilizado
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Define o diretório onde a base do ChromaDB será persistida em disco
CHROMA_PERSIST_DIR = "chroma_db_persist"

# Define o nome da coleção dentro do ChromaDB
CHROMA_COLLECTION_NAME = "demonstrativos_financeiros"

# Definindo TOKENIZERS_PARALLELISM=false para o tokenizer não usar threads extras
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Carrega as variáveis de ambiente presentes no arquivo .env
load_dotenv()

# Verifica se a variável de ambiente GROQ_API_KEY está disponível
if "GROQ_API_KEY" not in os.environ:
    
    # Exibe uma mensagem de erro no Streamlit caso a chave não seja encontrada
    st.error("A GROQ_API_KEY não foi encontrada.")
    
    # Exibe instruções para o usuário configurar a chave da Groq
    st.info("Por favor, crie um arquivo .env e adicione sua chave API do Groq.")
    
    # Interrompe a execução da aplicação Streamlit
    st.stop()

# Aplica cache ao recurso para evitar re-inicialização do LLM a cada requisição
@st.cache_resource
def dsa_get_llm():
    
    """Inicializa e cacheia o modelo de linguagem (LLM) do Groq."""
    
    # Escreve no app que o LLM da Groq está sendo carregado
    st.write("Inicializando LLM do Groq...")
    
    # Retorna uma instância do ChatGroq configurada com o modelo especificado e temperatura zero (quanto maior mais criativo o modelo é logo nesse caso queremos mais objetividade portanto = 0)
    return ChatGroq(model_name = "llama-3.3-70b-versatile", temperature = 0)

# Aplica cache ao recurso de embeddings para não recarregar o modelo a cada chamada
@st.cache_resource
def dsa_get_embedding_model():

    """Inicializa e cacheia o modelo de embeddings do Hugging Face."""

    
    # Log visual indicando que o modelo de embeddings está sendo carregado
    st.write(f"Carregando modelo de embedding ({EMBEDDING_MODEL})...")
    
    # Define os argumentos do modelo, incluindo o uso de CPU como dispositivo (por ser pdf simples podemos usar CPU se fosse uma massa de arquivos usaríamos GPU para processamento em paralelo)
    model_kwargs = {'device': 'cpu'} 
    
    # Define argumentos de codificação, desabilitando a normalização dos embeddings (aqui como estamos com pdf mais simples desativamso embeddings, se fosse uma massa de arquivos usaríamos normalização para melhorar a qualidade dos embeddings)
    encode_kwargs = {'normalize_embeddings': False}
    
    # Cria e retorna a instância de HuggingFaceEmbeddings configurada
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

# Aplica cache ao processo de criação do Vector Store, exibindo um spinner durante o processamento
@st.cache_resource(show_spinner = "Processando PDF e criando Vector Store...")
def dsa_get_vector_store(_uploaded_file, embedding_model):
    
    # Função responsável por processar o PDF enviado e criar o vector store no Chroma (construindo o banco vetorial a partir dos chunks do PDF)

    # Verifica se um arquivo foi realmente enviado pelo usuário
    if _uploaded_file is not None:
        
        try:
            
            # Cria um arquivo temporário no disco com extensão .pdf
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp_file:
                
                # Escreve o conteúdo do arquivo enviado dentro do arquivo temporário
                tmp_file.write(_uploaded_file.getvalue())
                
                # Armazena o caminho do arquivo temporário
                tmp_file_path = tmp_file.name

            # Cria um loader de PDF baseado no caminho do arquivo temporário
            loader = PyPDFLoader(tmp_file_path)
            
            # Carrega todas as páginas do PDF como documentos
            docs = loader.load()

            # Cria o splitter de texto para dividir os documentos em chunks menores (chuck_size é o tamanho de caracteres da repartição e chunk_overlap é a sobreposição entre os chunks para manter contexto)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 200)
            
            # Aplica o splitter aos documentos carregados para gerar a lista de chunks
            splits = text_splitter.split_documents(docs)

            # Cria o vector store no Chroma a partir dos chunks gerados e do modelo de embeddings
            vectorstore = Chroma.from_documents(
                documents = splits,
                embedding = embedding_model,
                persist_directory = CHROMA_PERSIST_DIR,
                collection_name = CHROMA_COLLECTION_NAME
            )
            
            # Exibe mensagem de sucesso com a quantidade de chunks criados
            st.success(f"PDF processado! {len(splits)} chunks criados e salvos no ChromaDB.")
            
            # Remove o arquivo temporário do disco após o processamento
            os.remove(tmp_file_path)
            
            # Retorna um retriever configurado para buscar os k documentos mais relevantes
            return vectorstore.as_retriever(search_kwargs = {"k": 3})

        # Captura qualquer exceção que ocorra durante o processamento do PDF
        except Exception as e:
            
            # Exibe a mensagem de erro no Streamlit
            st.error(f"Erro ao processar o PDF: {e}")
            
            # Em caso de erro, retorna None para indicar falha
            return None
    
    # Caso nenhum arquivo tenha sido enviado, retorna None
    return None

# Função auxiliar para formatar os documentos recuperados em um único texto
def dsa_formata_docs(docs):

    # Junta o conteúdo de cada documento, separando os chunks com delimitadores visuais
    return "\n\n---\n\n".join([d.page_content for d in docs])

# Define configurações gerais da página no Streamlit (título, ícone e layout)
st.set_page_config(page_title = "Data Science Academy", page_icon = ":100:", layout = "wide")

# Define o título principal da aplicação na interface
st.title("Data Science Academy - Estudo de Caso")

# Define o subtítulo destacando o estudo de caso e o uso de RAG
st.title("🤖 App de RAG Para Busca Semântica em Documentos")

# Exibe um texto explicativo sobre a tecnologia usada no app
st.markdown(f"Use o poder do RAG com Groq, ChromaDB (Persistente) e Embeddings HuggingFace.")

# Inicializa (ou recupera do cache) o modelo de linguagem da Groq
llm = dsa_get_llm()

# Inicializa (ou recupera do cache) o modelo de embeddings do Hugging Face
embeddings = dsa_get_embedding_model()

# Cria a área lateral da interface (sidebar) para configurações e upload
with st.sidebar:
    
    # Título da seção de upload de documento
    st.header("Carregar Documento")
    
    # Cria o componente de upload de arquivo, restringindo a PDFs e um único arquivo
    uploaded_file = st.file_uploader(
        "Carregue seu demonstrativo financeiro (.pdf)", 
        type = "pdf",
        accept_multiple_files = False
    )
    
    # Exibe um aviso sobre o tempo de processamento do PDF na primeira execução
    st.warning("Nota: O processamento do PDF (criação de embeddings) pode demorar alguns minutos na primeira vez.")

    # Cria uma área expansível na sidebar para informações de suporte
    with st.sidebar.expander("🆘 Suporte / Fale conosco", expanded = False):
        
        # Exibe o email de suporte dentro do expander
        st.write("Se tiver dúvidas envie mensagem para suporte@datascienceacademy.com.br")

    # Exibe um aviso informativo sobre possíveis imprecisões nas respostas da IA
    st.sidebar.info("Aviso: IA pode gerar respostas imprecisas, incompletas ou erradas. Sempre verifique informações críticas antes de confiar totalmente no resultado.")
    
# Cria o cabeçalho da seção de perguntas do usuário
st.header("Faça sua Pergunta")

# Inicializa o retriever dentro do estado de sessão se ele ainda não existir (se o retriever não estiver na sessão ele recebe valor none)
if 'retriever' not in st.session_state:

    # Define o retriever inicialmente como None
    st.session_state.retriever = None

# Se o usuário enviou um arquivo, cria ou atualiza o vector store com base nesse PDF
if uploaded_file:
    
    # Chama a função de processamento do PDF e atualiza o retriever na sessão
    st.session_state.retriever = dsa_get_vector_store(uploaded_file, embeddings)

# Caso não haja upload, verifica se já existe um diretório persistido do Chroma
elif os.path.exists(CHROMA_PERSIST_DIR):
    
    # Garante que o retriever seja carregado somente uma vez a partir do disco
    if st.session_state.retriever is None: # Só carrega na primeira vez
        
        # Exibe mensagem informando que o vector store está sendo carregado do disco
        st.write("Carregando Vector Store persistido do disco...")
        
        # Cria uma instância do Chroma apontando para o diretório persistido e o modelo de embeddings (criando um canal para acesso ao banco vetorial)
        vectorstore = Chroma(
            persist_directory = CHROMA_PERSIST_DIR,
            embedding_function = embeddings,
            collection_name = CHROMA_COLLECTION_NAME
        )
        
        # Atualiza o retriever na sessão a partir do vector store carregado (e o vector store que era none agora recebe o valor do banco vetorial)
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})
        
        # Exibe mensagem de sucesso na sidebar
        st.sidebar.success("Vector Store local carregado!")

# Caso não haja upload nem diretório persistido, orienta o usuário a enviar um PDF
else:

    # Informa ao usuário que é necessário fazer upload de um documento para começar
    st.info("Por favor, faça o upload de um documento PDF para começar.")

# Verifica se já existe um retriever disponível na sessão
if st.session_state.retriever:
    
    # Define o template do prompt (contém as instruções par ao LLM - que processa o prompt para retorno de resposta em linguagem natural) usado no fluxo de RAG
    RAG_PROMPT_TEMPLATE = """
    Você é um assistente de IA especializado em análise financeira.
    Sua tarefa é responder perguntas sobre demonstrativos financeiros usando APENAS o contexto fornecido.
    Seja direto, preciso e baseie-se exclusivamente nos dados dos trechos.
    Se a informação não estiver no contexto, diga "A informação não foi encontrada no documento."

    Contexto:
    {context}

    Pergunta:
    {question}

    Resposta (em Português):
    """
    #O LLM sempre vai gerar respostas com ou sem contexto, utilizando o contexto estamos direcionando o LLM a buscar respostas apenas no contexto fornecido, evitando alucinações e respostas fora do escopo do documento.
    
    # Cria um ChatPromptTemplate a partir do texto do template definido
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Monta a cadeia de RAG combinando retriever, formatação, prompt, LLM e parser de saída (preenchendo o context com o documento recuperado)
    dsa_rag_chain = (
        {"context": st.session_state.retriever | dsa_formata_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Aqui o runnablepassthrough é usado para repassar a pergunta do usuário diretamente para o prompt, sem alterações. O retriever busca os documentos relevantes, que são formatados e passados como contexto para o LLM.


    # Cria o campo de texto para o usuário digitar a pergunta
    question = st.text_input("Faça sua pergunta. Ex: Qual foi o critério de reconhecimento de receita? ou Qual o fluxo de caixa operacional?", disabled = False)

    # Verifica se o usuário digitou alguma pergunta
    if question:
        
        # Exibe um spinner enquanto busca informações e gera a resposta
        with st.spinner("Fazendo a busca semântica e gerando resposta..."):
            
            try:
                
                # Invoca a cadeia de RAG passando a pergunta do usuário (onde a IA REALMENTE TRABALHA!!)
                answer = dsa_rag_chain.invoke(question)
                
                # Exibe um cabeçalho indicando que a resposta foi gerada
                st.success("Resposta Gerada:")
                
                # Mostra a resposta retornada pelo LLM
                st.write(answer)
                
                # Cria um expander para exibir os chunks de contexto que foram usados na resposta
                with st.expander("Ver chunks de contexto utilizados"):
                    
                    # Recupera os documentos mais relevantes para a pergunta
                    retrieved_docs = st.session_state.retriever.invoke(question)
                    
                    # Exibe a versão JSON dos documentos recuperados
                    st.json([doc.to_json() for doc in retrieved_docs])

            # Captura eventuais erros durante a execução da cadeia de RAG
            except Exception as e:

                # Exibe uma mensagem de erro caso algo falhe ao invocar a cadeia
                st.error(f"Ocorreu um erro ao invocar a cadeia RAG: {e}")

# Caso não haja retriever disponível, desabilita o campo de pergunta
else:
    
    # Cria um input desabilitado, orientando o usuário a fazer upload antes
    st.text_input("Faça sua pergunta...", disabled = True)





