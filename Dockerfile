FROM python:3.10-slim

# Definir variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar llama-cpp-python com otimizações para CPU
ENV CMAKE_ARGS="-DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON"
RUN pip install --upgrade pip && \
    pip install llama-cpp-python flask flask_cors numpy python-dotenv

# Criar estrutura de diretórios para múltiplos modelos
WORKDIR /app
RUN mkdir -p /app/models

# Volume para modelos
VOLUME /app/models

# Criar o script do servidor multi-modelos
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import gc\n\
import json\n\
import time\n\
from flask import Flask, request, jsonify\n\
from flask_cors import CORS\n\
from llama_cpp import Llama\n\
\n\
app = Flask(__name__)\n\
CORS(app)\n\
\n\
# Dicionário para armazenar modelos carregados\n\
loaded_models = {}\n\
\n\
# Configurações padrão\n\
DEFAULT_CONFIG = {\n\
    "n_ctx": 4096,\n\
    "n_threads": 18,\n\
    "n_batch": 512\n\
}\n\
\n\
def get_available_models():\n\
    """Retorna lista de modelos disponíveis."""\n\
    models = []\n\
    for root, dirs, files in os.walk("/app/models"):\n\
        for file in files:\n\
            if file.endswith(".gguf"):\n\
                full_path = os.path.join(root, file)\n\
                relative_path = os.path.relpath(full_path, "/app/models")\n\
                models.append({\n\
                    "name": relative_path,\n\
                    "path": full_path,\n\
                    "loaded": relative_path in loaded_models\n\
                })\n\
    return models\n\
\n\
def load_model(model_name):\n\
    """Carrega um modelo na memória."""\n\
    for model in get_available_models():\n\
        if model["name"] == model_name:\n\
            if model_name in loaded_models:\n\
                return loaded_models[model_name]\n\
            \n\
            print(f"Carregando modelo: {model_name}")\n\
            try:\n\
                llm = Llama(\n\
                    model_path=model["path"],\n\
                    n_ctx=DEFAULT_CONFIG["n_ctx"],\n\
                    n_threads=DEFAULT_CONFIG["n_threads"],\n\
                    n_batch=DEFAULT_CONFIG["n_batch"]\n\
                )\n\
                loaded_models[model_name] = llm\n\
                print(f"Modelo {model_name} carregado com sucesso!")\n\
                return llm\n\
            except Exception as e:\n\
                print(f"Erro ao carregar modelo {model_name}: {e}")\n\
                return None\n\
    return None\n\
\n\
def unload_model(model_name):\n\
    """Remove um modelo da memória."""\n\
    if model_name in loaded_models:\n\
        del loaded_models[model_name]\n\
        gc.collect()  # Forçar coleta de lixo\n\
        return True\n\
    return False\n\
\n\
@app.route("/models", methods=["GET"])\n\
def list_models():\n\
    """Lista todos os modelos disponíveis."""\n\
    models = get_available_models()\n\
    return jsonify({\n\
        "models": models,\n\
        "loaded_models": list(loaded_models.keys())\n\
    })\n\
\n\
@app.route("/models/load", methods=["POST"])\n\
def load_model_endpoint():\n\
    """Carrega um modelo específico."""\n\
    data = request.json\n\
    model_name = data.get("model_name")\n\
    \n\
    if not model_name:\n\
        return jsonify({"error": "model_name é obrigatório"}), 400\n\
    \n\
    llm = load_model(model_name)\n\
    if llm:\n\
        return jsonify({"message": f"Modelo {model_name} carregado com sucesso"})\n\
    else:\n\
        return jsonify({"error": f"Não foi possível carregar o modelo {model_name}"}), 500\n\
\n\
@app.route("/models/unload", methods=["POST"])\n\
def unload_model_endpoint():\n\
    """Remove um modelo da memória."""\n\
    data = request.json\n\
    model_name = data.get("model_name")\n\
    \n\
    if not model_name:\n\
        return jsonify({"error": "model_name é obrigatório"}), 400\n\
    \n\
    if unload_model(model_name):\n\
        return jsonify({"message": f"Modelo {model_name} removido da memória"})\n\
    else:\n\
        return jsonify({"error": f"Modelo {model_name} não estava carregado"}), 404\n\
\n\
@app.route("/completion", methods=["POST"])\n\
def completion():\n\
    """Gera texto usando um modelo específico."""\n\
    data = request.json\n\
    model_name = data.get("model", None)\n\
    prompt = data.get("prompt", "")\n\
    max_tokens = data.get("n_predict", 128)\n\
    temperature = data.get("temperature", 0.7)\n\
    stop_sequences = data.get("stop", [])\n\
    \n\
    # Se não especificou o modelo, usa o primeiro disponível\n\
    if not model_name:\n\
        if loaded_models:\n\
            model_name = list(loaded_models.keys())[0]\n\
        else:\n\
            # Tenta carregar o primeiro modelo disponível\n\
            models = get_available_models()\n\
            if models:\n\
                model_name = models[0]["name"]\n\
                load_model(model_name)\n\
            else:\n\
                return jsonify({"error": "Nenhum modelo disponível"}), 404\n\
    \n\
    # Garante que o modelo está carregado\n\
    if model_name not in loaded_models:\n\
        llm = load_model(model_name)\n\
        if not llm:\n\
            return jsonify({"error": f"Não foi possível carregar o modelo {model_name}"}), 500\n\
    else:\n\
        llm = loaded_models[model_name]\n\
    \n\
    start_time = time.time()\n\
    \n\
    # Gerar resposta\n\
    try:\n\
        result = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop_sequences)\n\
        \n\
        elapsed_time = time.time() - start_time\n\
        print(f"Geração concluída em {elapsed_time:.2f}s usando {model_name}")\n\
        \n\
        return jsonify({\n\
            "content": result[\'choices\'][0][\'text\'],\n\
            "model": model_name,\n\
            "generation_time": elapsed_time\n\
        })\n\
    except Exception as e:\n\
        return jsonify({"error": f"Erro na geração: {str(e)}"}), 500\n\
\n\
@app.route("/health", methods=["GET"])\n\
def health():\n\
    return jsonify({\n\
        "status": "ok",\n\
        "loaded_models": list(loaded_models.keys())\n\
    })\n\
\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=8080, debug=False)\n' > /app/server.py

# Tornar o script executável
RUN chmod +x /app/server.py

# Expor porta do servidor
EXPOSE 8080

# Comando para iniciar o servidor
CMD ["python3", "/app/server.py"]