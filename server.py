#!/usr/bin/env python3
import os
import gc
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)

loaded_models = {}

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
                    "name": relative_path,
                    "path": full_path,
                    "loaded": relative_path in loaded_models
                })
    return models

def load_model(model_name):
    """Carrega um modelo na memória."""
    for model in get_available_models():
        if model["name"] == model_name:
            if model_name in loaded_models:
                return loaded_models[model_name]
            
            print(f"Carregando modelo: {model_name}")
            try:
                llm = Llama(
                    model_path=model["path"],
                    n_ctx=DEFAULT_CONFIG["n_ctx"],
                    n_threads=DEFAULT_CONFIG["n_threads"],
                    n_batch=DEFAULT_CONFIG["n_batch"]
                )
                loaded_models[model_name] = llm
                print(f"Modelo {model_name} carregado com sucesso!")
                return llm
            except Exception as e:
                print(f"Erro ao carregar modelo {model_name}: {e}")
                return None
    return None

def unload_model(model_name):
    """Remove um modelo da memória."""
    if model_name in loaded_models:
        del loaded_models[model_name]
        gc.collect()  # Forçar coleta de lixo
        return True
    return False

@app.route("/models", methods=["GET"])
def list_models():
    """Lista todos os modelos disponíveis."""
    models = get_available_models()
    return jsonify({
        "models": models,
        "loaded_models": list(loaded_models.keys())
    })

@app.route("/models/load", methods=["POST"])
def load_model_endpoint():
    """Carrega um modelo específico."""
    data = request.json
    model_name = data.get("model_name")
    
    if not model_name:
        return jsonify({"error": "model_name é obrigatório"}), 400
    
    llm = load_model(model_name)
    if llm:
        return jsonify({"message": f"Modelo {model_name} carregado com sucesso"})
    else:
        return jsonify({"error": f"Não foi possível carregar o modelo {model_name}"}), 500

@app.route("/models/unload", methods=["POST"])
def unload_model_endpoint():
    """Remove um modelo da memória."""
    data = request.json
    model_name = data.get("model_name")
    
    if not model_name:
        return jsonify({"error": "model_name é obrigatório"}), 400
    
    if unload_model(model_name):
        return jsonify({"message": f"Modelo {model_name} removido da memória"})
    else:
        return jsonify({"error": f"Modelo {model_name} não estava carregado"}), 404

@app.route("/completion", methods=["POST"])
def completion():
    """Gera texto usando um modelo específico."""
    data = request.json
    model_name = data.get("model", None)
    prompt = data.get("prompt", "")
    max_tokens = data.get("n_predict", 128)
    temperature = data.get("temperature", 0.7)
    stop_sequences = data.get("stop", [])
    
    if not model_name:
        if loaded_models:
            model_name = list(loaded_models.keys())[0]
        else:
            models = get_available_models()
            if models:
                model_name = models[0]["name"]
                load_model(model_name)
            else:
                return jsonify({"error": "Nenhum modelo disponível"}), 404
    
    if model_name not in loaded_models:
        llm = load_model(model_name)
        if not llm:
            return jsonify({"error": f"Não foi possível carregar o modelo {model_name}"}), 500
    else:
        llm = loaded_models[model_name]
    
    start_time = time.time()
    
    try:
        result = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop_sequences)
        
        elapsed_time = time.time() - start_time
        print(f"Geração concluída em {elapsed_time:.2f}s usando {model_name}")
        
        return jsonify({
            "content": result['choices'][0]['text'],
            "model": model_name,
            "generation_time": elapsed_time
        })
    except Exception as e:
        return jsonify({"error": f"Erro na geração: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "loaded_models": list(loaded_models.keys())
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)