#!/usr/bin/env python3
import os
import gc
import json
import time
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)

# Dicionário para armazenar modelos carregados
loaded_models = {}

# Modelo padrão
default_model = "instruct/mistral-7b-instruct.gguf"

# Configurações padrão
DEFAULT_CONFIG = {
    "n_ctx": 4096,
    "n_threads": 18,
    "n_batch": 512
}

def get_available_models():
    """Retorna lista de modelos disponíveis."""
    models = []
    for root, dirs, files in os.walk("/app/models"):
        for file in files:
            if file.endswith(".gguf"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, "/app/models")
                models.append({
                    "id": relative_path,  # ID compatível com OpenAI
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "permission": [],
                    "root": relative_path,
                    "parent": None
                })
    return models

def load_model(model_id):
    """Carrega um modelo na memória."""
    model_path = None
    for root, dirs, files in os.walk("/app/models"):
        for file in files:
            if file.endswith(".gguf"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, "/app/models")
                if relative_path == model_id:
                    model_path = full_path
                    break
    
    if not model_path:
        return None
    
    if model_id in loaded_models:
        return loaded_models[model_id]
    
    print(f"Carregando modelo: {model_id}")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=DEFAULT_CONFIG["n_ctx"],
            n_threads=DEFAULT_CONFIG["n_threads"],
            n_batch=DEFAULT_CONFIG["n_batch"]
        )
        loaded_models[model_id] = llm
        print(f"Modelo {model_id} carregado com sucesso!")
        return llm
    except Exception as e:
        print(f"Erro ao carregar modelo {model_id}: {e}")
        return None

# OpenAI Compatible Endpoints

@app.route("/v1/models", methods=["GET"])
def list_models():
    """Lista todos os modelos disponíveis (compatível com OpenAI)."""
    models = get_available_models()
    return jsonify({
        "object": "list",
        "data": models
    })

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Chat completions endpoint compatível com OpenAI."""
    data = request.json
    model_id = data.get("model", default_model)
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)
    stop = data.get("stop", ["</s>"])
    stream = data.get("stream", False)
    
    # Converter mensagens para o formato Mistral
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"<s>[INST] {content} [/INST]")
        elif role == "user":
            prompt_parts.append(f"<s>[INST] {content} [/INST]")
        elif role == "assistant":
            prompt_parts.append(f"{content}</s>")
    
    prompt = "\n".join(prompt_parts)
    
    # Garantir que o modelo está carregado
    llm = loaded_models.get(model_id)
    if not llm:
        llm = load_model(model_id)
        if not llm:
            return jsonify({"error": "Model not found"}), 404
    
    # Gerar resposta
    start_time = time.time()
    response_id = str(uuid.uuid4())
    
    try:
        result = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
        content = result['choices'][0]['text'].strip()
        
        elapsed_time = time.time() - start_time
        
        # Resposta no formato OpenAI
        response = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len((prompt + content).split())
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/completions", methods=["POST"])
def completions():
    """Completions endpoint compatível com OpenAI."""
    data = request.json
    model_id = data.get("model", default_model)
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)
    stop = data.get("stop", ["</s>"])
    
    # Garantir que o modelo está carregado
    llm = loaded_models.get(model_id)
    if not llm:
        llm = load_model(model_id)
        if not llm:
            return jsonify({"error": "Model not found"}), 404
    
    # Gerar resposta
    start_time = time.time()
    response_id = str(uuid.uuid4())
    
    try:
        result = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
        content = result['choices'][0]['text']
        
        elapsed_time = time.time() - start_time
        
        # Resposta no formato OpenAI
        response = {
            "id": f"cmpl-{response_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "text": content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len((prompt + content).split())
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/v1/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "loaded_models": list(loaded_models.keys())
    })

if __name__ == "__main__":
    # Carregar modelo padrão na inicialização
    print(f"Carregando modelo padrão: {default_model}")
    load_model(default_model)
    
    print("Servidor compatível com OpenAI iniciando...")
    app.run(host="0.0.0.0", port=8080, debug=False)