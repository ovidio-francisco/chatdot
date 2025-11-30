from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

def main():
    # Configura o LLM padrão do LlamaIndex para usar o Ollama local
    Settings.llm = Ollama(
        model="llama3.1",   # mesmo nome que você usa no `ollama run`
        request_timeout=60.0,
        #  context_window=1024,   # <= contexto menor = menos memória
        context_window=2048,   # <= contexto menor = menos memória
    )

    # Chamada simples para testar o fluxo Python -> LlamaIndex -> Ollama
    prompt = "Qual seu modelo e versão?"
    #  prompt = "Olá, quem é você?"
    print(">>> Enviando prompt:", prompt)
    response = Settings.llm.complete(prompt)
    print("\n=== Resposta do modelo ===\n")
    print(response)

if __name__ == "__main__":
    main()

