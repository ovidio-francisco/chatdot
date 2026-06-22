# Chatdot

Experimental Python project for testing local language models with LlamaIndex and Ollama.

The goal of this project is to explore a simple structure for querying local documents by creating a vector index from files stored in the `data/` folder and asking questions about their content.

This repository works as an initial experiment with local LLMs, embeddings, and information retrieval from local files.

## Features

- [x] Run local language models with Ollama.
- [x] Integrate Python code with LlamaIndex.
- [x] Test simple prompts sent to local LLMs.
- [x] Load local documents from the `data/` folder.
- [x] Create a vector index from the loaded documents.
- [x] Query documents using natural language questions.

## Technologies used

* Python
* LlamaIndex
* Ollama
* HuggingFace Embeddings


## Requirements

Before running the project, make sure you have installed:

* Python 3
* Ollama
* A local model available in Ollama

Examples of models used in the project tests:

```bash
ollama pull llama3.2:1b
ollama pull llama3.1
ollama pull qwen2.5:0.5b
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ovidio-francisco/chatdot.git
```

Access the project folder:

```bash
cd chatdot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

To run `main_files.py`, you may also need to install the LlamaIndex HuggingFace embeddings package:

```bash
pip install llama-index-embeddings-huggingface
```

## How to run

### Simple Ollama test

The `src/main.py` file sends a simple prompt to a local model configured in Ollama.

```bash
python src/main.py
```

### Test with another model

The `src/main31.py` file performs a similar test using another model configured in the code.

```bash
python src/main31.py
```

### Query local files

The `src/main_files.py` file reads documents from the `data/` folder, creates a vector index, and asks a question about the loaded content.

```bash
python src/main_files.py
```

By default, the script asks:

```text
Summarize the content of the documents.
```

## General workflow

The main workflow of the project can be summarized as follows:

1. Configure a local model using Ollama.
2. Configure an embedding model.
3. Read files from the `data/` folder.
4. Create a vector index from the documents.
5. Ask a question about the loaded content.
6. Print the answer in the terminal.

## Notes

This project is an initial experiment and does not yet include a graphical interface, API, or persistent vector index.

The models can be changed directly in the Python files by adjusting the `model` parameter.

Make sure the Ollama service is running before executing the scripts.

## Possible future improvements

-[ ] Create a chat interface for querying documents.
-[ ] Allow model selection through command-line arguments or environment variables.
-[ ] Move configuration values to a `.env` file.
-[ ] Persist the vector index to avoid rebuilding it on every execution.
-[ ] Improve the organization of the scripts.
-[ ] Create a command-line interface.
-[ ] Add error handling.
-[ ] Update `requirements.txt` with all required dependencies.
-[ ] Document examples of questions and answers.


