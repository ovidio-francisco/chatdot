from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

def main():
    # Configura o LLM padrão do LlamaIndex para usar o Ollama local
    Settings.llm = Ollama(
        model="llama3.1",
        request_timeout=60.0,
        context_window=1024,  
        #  context_window=4096,  
    )

    prompt = "Qual seu modelo e versão?"
    print(">>> Enviando prompt:", prompt)
    response = Settings.llm.complete(prompt)
    print("\n=== Resposta do modelo ===\n")
    print(response)

if __name__ == "__main__":
    main()

