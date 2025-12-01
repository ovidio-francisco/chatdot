from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def main():
    # 1) Configurar o LLM local (Ollama) com 3.1
    llm = Ollama(
        #  model="llama3.1",
        model="qwen2.5:0.5b",
        request_timeout=120.0,              # dá um pouco mais de folga
        context_window=1024,
        #  context_window=2048,
        #  context_window=4096,
        base_url="http://127.0.0.1:11434",  # força o host padrão do daemon
        # keep_alive=0,                     # opcional: não manter modelo carregado
    )
    Settings.llm = llm

    # 2) Embeddings locais
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"  # menor pra testes
    )
    Settings.embed_model = embed_model

    # 3) Ler docs
    print(">>> Lendo arquivos da pasta data/ ...")
    documents = SimpleDirectoryReader(
        "data",
        exclude_hidden=False,   # <– libera arquivos que começam com ponto
        ).load_data()
    print(f"Carregados {len(documents)} documentos.\n")

    # 4) Índice vetorial
    print(">>> Criando índice vetorial ...")
    index = VectorStoreIndex.from_documents(documents)
    print("Índice criado com sucesso.\n")

    # 5) QueryEngine
    query_engine = index.as_query_engine()

    # 6) Pergunta
    print(">>> Fazendo pergunta de teste ao índice...\n")
    question = "Resuma o conteúdo dos documentos."
    print("Pergunta:", question)

    response = query_engine.query(question)

    print("\n=== Resposta do modelo (RAG) ===\n")
    print(response)




if __name__ == "__main__":
    main()
