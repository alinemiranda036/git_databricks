# Script para Comparar Modelos de Embeddings usando o Demonstrativo Financeiro como Teste
# Objetivo: testar diferentes modelos de embeddings no MESMO documento e nas MESMAS perguntas,
# pra ver qual retorna os chunks mais relevantes (mais próximo do que seria um "AutoML" pra embeddings)

# Importa o módulo os para lidar com caminhos de arquivo
import os

# Importa o módulo time para medir tempo de carregamento e busca de cada modelo
import time

# Importa o loader de PDF do LangChain
from langchain_community.document_loaders import PyPDFLoader

# Importa o splitter de texto para dividir o PDF em chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importa a classe de embeddings do Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings

# Importa a integração com o Chroma (banco vetorial temporário, só para este teste)
from langchain_chroma import Chroma

# Caminho do PDF que será usado como fonte para o teste (ajuste se necessário)
PDF_PATH = "demonstrativo_financeiro.pdf"

# Lista de modelos de embeddings que serão comparados
# Você pode adicionar ou remover modelos dessa lista livremente
MODELOS_PARA_TESTAR = [
    # Modelo atual usado no seu dsa_app.py (leve e rápido)
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

    # Modelo multilíngue de porte médio, bom equilíbrio entre qualidade e velocidade
    "intfloat/multilingual-e5-base",

    # Modelo mais robusto (mais pesado, pode demorar mais no CPU), mas com ótima performance em benchmarks
    "BAAI/bge-m3",
]

# Perguntas de teste baseadas no conteúdo real do PDF, com palavras-chave que DEVEM aparecer
# na resposta correta (isso funciona como um "gabarito" simplificado para medir acerto)
PERGUNTAS_TESTE = [
    {
        "pergunta": "Qual foi o critério de reconhecimento de receita?",
        "palavras_chave": ["prestados", "entregues"]
    },
    {
        "pergunta": "Qual o fluxo de caixa operacional?",
        "palavras_chave": ["3.100.000"]
    },
    {
        "pergunta": "Qual o lucro líquido da empresa?",
        "palavras_chave": ["2.450.000"]
    },
    {
        "pergunta": "Qual o valor do patrimônio líquido?",
        "palavras_chave": ["14.900.000"]
    },
    {
        "pergunta": "Quanto foi investido em novos equipamentos?",
        "palavras_chave": ["900.000"]
    },
]

# Função responsável por carregar o PDF e dividir em chunks (mesma lógica do seu dsa_app.py)
def carregar_e_dividir_pdf(caminho):

    # Cria o loader apontando para o caminho do PDF
    loader = PyPDFLoader(caminho)

    # Carrega todas as páginas do documento
    docs = loader.load()

    # Cria o splitter com os mesmos parâmetros usados no seu app principal
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 200)

    # Retorna a lista de chunks gerados
    return text_splitter.split_documents(docs)

# Função que testa um único modelo de embeddings: carrega, indexa, busca e mede resultados
def testar_modelo(nome_modelo, chunks):

    # Exibe cabeçalho identificando qual modelo está sendo testado
    print(f"\n{'=' * 70}")
    print(f"Testando modelo: {nome_modelo}")
    print(f"{'=' * 70}")

    # Marca o tempo inicial para medir o carregamento do modelo
    t0 = time.time()

    # Instancia o modelo de embeddings (isso baixa o modelo do Hugging Face na primeira vez)
    embedding_model = HuggingFaceEmbeddings(
        model_name = nome_modelo,
        model_kwargs = {'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': False}
    )

    # Calcula quanto tempo levou para carregar o modelo
    tempo_carregamento = time.time() - t0

    # Marca o tempo inicial para medir a indexação (criação dos vetores) dos chunks
    t0 = time.time()

    # Cria um vector store TEMPORÁRIO em memória, só para este teste (não é o mesmo do seu app principal)
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        collection_name = f"teste_{nome_modelo.replace('/', '_')}"
    )

    # Calcula quanto tempo levou para indexar todos os chunks
    tempo_indexacao = time.time() - t0

    # Cria o retriever configurado para retornar os 3 chunks mais relevantes
    retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})

    # Contador de quantas perguntas o modelo respondeu corretamente
    acertos = 0

    # Lista para armazenar o tempo de cada busca individual
    tempos_busca = []

    # Percorre cada pergunta de teste
    for teste in PERGUNTAS_TESTE:

        # Marca o tempo inicial da busca
        t0 = time.time()

        # Executa a busca semântica para a pergunta atual
        docs_recuperados = retriever.invoke(teste["pergunta"])

        # Registra o tempo que essa busca levou
        tempos_busca.append(time.time() - t0)

        # Junta o texto de todos os chunks recuperados em uma única string (em minúsculas para comparação)
        texto_recuperado = " ".join([d.page_content for d in docs_recuperados]).lower()

        # Verifica se PELO MENOS UMA das palavras-chave esperadas aparece no texto recuperado
        encontrou = any(palavra.lower() in texto_recuperado for palavra in teste["palavras_chave"])

        # Define o símbolo de status conforme o resultado
        status = "[OK]" if encontrou else "[FALHOU]"

        # Exibe o resultado da pergunta atual
        print(f"{status} {teste['pergunta']}")

        # Incrementa o contador de acertos se encontrou a informação esperada
        if encontrou:
            acertos += 1

    # Calcula a taxa de acerto em porcentagem
    taxa_acerto = (acertos / len(PERGUNTAS_TESTE)) * 100

    # Calcula o tempo médio de busca entre todas as perguntas
    tempo_medio_busca = sum(tempos_busca) / len(tempos_busca)

    # Remove a coleção de teste do Chroma para não deixar lixo acumulado
    vectorstore.delete_collection()

    # Retorna um dicionário com todas as métricas coletadas para este modelo
    return {
        "modelo": nome_modelo,
        "taxa_acerto": taxa_acerto,
        "tempo_carregamento": tempo_carregamento,
        "tempo_indexacao": tempo_indexacao,
        "tempo_medio_busca": tempo_medio_busca,
    }

# Função principal que orquestra todo o processo de comparação
def main():

    # Verifica se o arquivo PDF existe antes de começar
    if not os.path.exists(PDF_PATH):
        print(f"Arquivo não encontrado: {PDF_PATH}")
        print("Ajuste a variável PDF_PATH no início do script com o caminho correto.")
        return

    # Informa que o PDF está sendo carregado e dividido
    print("Carregando e dividindo o PDF em chunks...")

    # Executa o carregamento e divisão do PDF (feito uma única vez, reutilizado para todos os modelos)
    chunks = carregar_e_dividir_pdf(PDF_PATH)

    # Informa quantos chunks foram gerados
    print(f"{len(chunks)} chunks criados a partir do documento.")

    # Lista para armazenar os resultados de todos os modelos testados
    resultados = []

    # Percorre cada modelo definido na lista MODELOS_PARA_TESTAR
    for modelo in MODELOS_PARA_TESTAR:

        try:
            # Executa o teste completo para o modelo atual
            resultado = testar_modelo(modelo, chunks)

            # Adiciona o resultado à lista geral
            resultados.append(resultado)

        # Captura erros (ex: modelo não encontrado, falta de memória, etc.)
        except Exception as e:

            # Exibe o erro e continua para o próximo modelo
            print(f"Erro ao testar o modelo {modelo}: {e}")

    # Exibe o cabeçalho da tabela final de resultados
    print(f"\n\n{'=' * 70}")
    print("RESULTADO FINAL - RANKING DOS MODELOS")
    print(f"{'=' * 70}\n")

    # Ordena os resultados da maior para a menor taxa de acerto
    resultados_ordenados = sorted(resultados, key = lambda x: -x["taxa_acerto"])

    # Percorre os resultados ordenados e exibe cada linha da tabela
    for i, r in enumerate(resultados_ordenados, start = 1):

        print(f"{i}. {r['modelo']}")
        print(f"   Taxa de acerto: {r['taxa_acerto']:.0f}%")
        print(f"   Tempo de carregamento do modelo: {r['tempo_carregamento']:.2f}s")
        print(f"   Tempo de indexação dos chunks: {r['tempo_indexacao']:.2f}s")
        print(f"   Tempo médio por busca: {r['tempo_medio_busca']:.3f}s")
        print()

# Ponto de entrada do script: executa a função main() somente se o arquivo for rodado diretamente
if __name__ == "__main__":
    main()
