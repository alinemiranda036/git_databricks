Repositório: git_databricks

Repositório com estudos de caso e projetos práticos em Machine Learning, Databricks/Unity Catalog e RAG (Retrieval-Augmented Generation) com LLMs. Reúne desde experimentos de IA generativa aplicada a documentos até modelos preditivos clássicos de detecção de fraude e AutoML.

🤖 Projetos de RAG (Retrieval-Augmented Generation)

dsa_app.py — App de RAG para Busca Semântica em Documentos Financeiros

Aplicação web em Streamlit que permite fazer upload de um demonstrativo financeiro em PDF e conversar com o documento em linguagem natural. Pipeline completo de RAG:

Ingestão: leitura do PDF (PyPDFLoader) e divisão em chunks com RecursiveCharacterTextSplitter (800 caracteres, overlap de 200).

Embeddings: geração de vetores com modelo multilíngue do Hugging Face (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

Banco vetorial: armazenamento e busca semântica com ChromaDB persistente em disco.

Geração de resposta: recuperação dos 3 chunks mais relevantes e geração da resposta com Groq (modelo llama-3.3-70b-versatile) via LangChain, usando um prompt que restringe o LLM a responder apenas com base no contexto recuperado (mitigando alucinação).

Interface: construída em Streamlit, com cache de recursos (@st.cache_resource) para evitar recarregar modelo e embeddings a cada interação.

Este é o projeto que está rodando em produção (demonstração publicada no LinkedIn). Requer uma GROQ_API_KEY própria em um arquivo .env para funcionar.

comparar_embeddings.py — Benchmark de Modelos de Embeddings

Script de avaliação que compara diferentes modelos de embeddings no mesmo documento e nas mesmas perguntas, medindo qual retorna os chunks mais relevantes — uma espécie de "AutoML" para escolha de embeddings. Compara:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (usado no app principal)

intfloat/multilingual-e5-base

BAAI/bge-m3

Para cada modelo, mede taxa de acerto (com base em um gabarito de palavras-chave esperadas), tempo de carregamento do modelo, tempo de indexação dos chunks e tempo médio de busca — gerando um ranking final. Usado para validar a escolha do modelo de embeddings do dsa_app.py.

📊 Projetos de Machine Learning (Databricks)

Fraude Detection - Credit Card.ipynb

Detecção de fraude em transações de cartão de crédito. Dados carregados de uma tabela do Unity Catalog (Databricks), tratamento de valores nulos, análise de correlação com a variável alvo, balanceamento de classes com SMOTE (imbalanced-learn) e classificação com Random Forest (scikit-learn), avaliada por relatório de classificação e matriz de confusão.

Fraud Detection - Transactions (Hotmart).ipynb

Detecção de fraude em transações financeiras (dados via Unity Catalog). Inclui análise exploratória de dados detalhada, tratamento de desbalanceamento de classes com SMOTE e avaliação de modelos de classificação com métricas de acurácia, matriz de confusão e classification report.

ML para prever valor de transação em Cartão de Crédito.ipynb

Modelo preditivo de regressão para estimar o valor (Amount) de transações de cartão de crédito a partir de dados do Unity Catalog. Contempla análise exploratória, estatísticas descritivas, tratamento de nulos e análise de correlação das variáveis com o alvo para seleção de features.

Automatizando Máquinas Preditivas - Pycaret.ipynb

Estudo de caso de AutoML com PyCaret, aplicado a um problema de classificação (aprovação de empréstimo pessoal). Demonstra a criação automatizada de um experimento de classificação, comparação de múltiplos algoritmos e interpretabilidade dos resultados com SHAP.

🛠️ Principais tecnologias

IA Generativa / RAG: LangChain · Groq (Llama 3.3 70B) · Hugging Face Embeddings · ChromaDB · Streamlit

Machine Learning: scikit-learn · PyCaret · imbalanced-learn (SMOTE) · SHAP

Dados e infraestrutura: Databricks · PySpark · Unity Catalog · pandas · Matplotlib · Seaborn

▶️ Como executar o app de RAG (dsa_app.py)

Instale as dependências:

pip install streamlit langchain langchain-groq langchain-huggingface langchain-community langchain-chroma python-dotenv pypdf


Crie um arquivo .env na raiz do projeto com sua chave da Groq:

GROQ_API_KEY="sua_chave_aqui"


Execute a aplicação:

streamlit run "dsa_app.py"


Os notebooks de Machine Learning foram desenvolvidos e executados em ambiente Databricks, utilizando tabelas do Unity Catalog como fonte de dados.
