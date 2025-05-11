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
    pip install llama-cpp-python flask flask_cors numpy

# Criar diretório para modelos
WORKDIR /app
RUN mkdir -p /app/models

# Volume para modelos
VOLUME /app/models

# Criar o script do servidor
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import time\n\
from flask import Flask, request, jsonify\n\
from flask_cors import CORS\n\
from llama_cpp import Llama\n\
\n\
app = Flask(__name__)\n\
CORS(app)\n\
\n\
# Caminho para o modelo\n\
MODEL_PATH = "/app/models/model.gguf"\n\
\n\
# Verificar se o modelo existe\n\
if not os.path.exists(MODEL_PATH):\n\
    print(f"AVISO: Modelo não encontrado em {MODEL_PATH}")\n\
    # Tenta encontrar qualquer arquivo .gguf\n\
    gguf_files = []\n\
    for root, dirs, files in os.walk("/app/models"):\n\
        for file in files:\n\
            if file.endswith(".gguf"):\n\
                gguf_files.append(os.path.join(root, file))\n\
    if gguf_files:\n\
        MODEL_PATH = gguf_files[0]\n\
        print(f"Usando modelo alternativo: {MODEL_PATH}")\n\
    else:\n\
        print("ERRO: Nenhum modelo .gguf encontrado!")\n\
        print("Arquivos disponíveis em /app/models:")\n\
        os.system("ls -la /app/models")\n\
        exit(1)\n\
\n\
# Inicializar o modelo\n\
print(f"Carregando modelo: {MODEL_PATH}")\n\
try:\n\
    llm = Llama(\n\
        model_path=MODEL_PATH,\n\
        n_ctx=4096,\n\
        n_threads=18,\n\
        n_batch=512\n\
    )\n\
    print("Modelo carregado com sucesso!")\n\
except Exception as e:\n\
    print(f"Erro ao carregar o modelo: {e}")\n\
    exit(1)\n\
\n\
@app.route("/completion", methods=["POST"])\n\
def completion():\n\
    data = request.json\n\
    prompt = data.get("prompt", "")\n\
    max_tokens = data.get("n_predict", 128)\n\
    temperature = data.get("temperature", 0.7)\n\
    stop_sequences = data.get("stop", [])\n\
    \n\
    start_time = time.time()\n\
    \n\
    # Gerar resposta\n\
    result = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop_sequences)\n\
    \n\
    elapsed_time = time.time() - start_time\n\
    print(f"Tempo de geração: {elapsed_time:.2f}s para {len(result[\'choices\'][0][\'text\'])} caracteres")\n\
    \n\
    return jsonify({\n\
        "content": result[\'choices\'][0][\'text\'],\n\
        "generation_time": elapsed_time\n\
    })\n\
\n\
@app.route("/health", methods=["GET"])\n\
def health():\n\
    return jsonify({"status": "ok"})\n\
\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=8080, debug=False)\n' > /app/server.py

# Tornar o script executável
RUN chmod +x /app/server.py

# Expor porta do servidor
EXPOSE 8080

# Comando para iniciar o servidor
CMD ["python3", "/app/server.py"]