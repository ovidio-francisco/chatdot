from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

def main():
    llm = Ollama(
        model="llama3.2:1b",
        request_timeout=60.0,
        context_window=1024,   # <= contexto menor = menos memória
        #  temperature=0.1,
        #  keep_alive=0,          # não manter modelo carregado depois da resposta
    )

    Settings.llm = llm

    prompt = "Olá, quem é você?"
    print(">>> Enviando prompt:", prompt)

    response = llm.complete(prompt)

    # CompletionResponse geralmente tem .text
    try:
        text = response.text
    except AttributeError:
        text = str(response)

    print("\n=== Resposta do modelo ===\n")
    print(text)

if __name__ == "__main__":
    main()
