#!/usr/bin/env python3
import os
import gc  # Importante para o unload
import json
import time
import uuid
import re
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

# Função para limpar tags de formatação da saída
def clean_output(text):
    """Remove tags de formatação da saída do modelo."""
    # Remover tags de formatação comuns
    patterns = [
        r'<s>|</s>',                           # Mistral/Llama
        r'\[INST\]|\[/INST\]',                 # Mistral
        r'<\|im_start\|>.*?<\|im_end\|>',      # Qwen completo
        r'<\|im_start\|>|<\|im_end\|>',        # Qwen parcial
        r'<\|user\|>|<\|assistant\|>|<\|system\|>', # TinyLlama
        r'<\|.*?\|>',                          # Outras tags
        r'<<SYS>>.*?<</SYS>>'                  # Outras tags de sistema
    ]
    
    # Aplicar cada padrão para limpar o texto
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    # Remover espaços extras e linhas vazias duplas
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.strip()

def get_model_specific_params(model_id):
    """Retorna parâmetros específicos para cada modelo."""
    model_id = model_id.lower()
    
    # Parâmetros padrão
    params = {
        "temperature": 0.7,
        "max_tokens": 512,
        "stop": ["</s>"],
        "repeat_penalty": 1.1,
        "frequency_penalty": 0.0,
        "top_p": 0.95
    }
    
    # Tinyllama precisa de ajustes especiais
    if "tiny" in model_id:
        params.update({
            "temperature": 0.5,  # Um pouco mais criativo que antes
            "max_tokens": 250,   # Aumento para concluir frases
            "stop": ["<|user|>", "<|assistant|>", "</s>", "<|endoftext|>"],
            "repeat_penalty": 1.3,  # Aumentar penalidade de repetição
            "top_p": 0.92,  # Reduzir um pouco a dispersão
            "mirostat_mode": 1,  # Ativar Mirostat (controle dinâmico)
            "mirostat_tau": 5.0,  # Valor conservador
            "mirostat_eta": 0.1   # Valor padrão
        })
    # Qwen também precisa de ajustes
    elif "qwen" in model_id and ("1.5b" in model_id or "1_5" in model_id or "1-5" in model_id):
        params.update({
            "temperature": 0.5,
            "max_tokens": 200,
            "stop": ["<|im_end|>", "</s>", "<|endoftext|>", "<|im_start|>"],
            "repeat_penalty": 1.2,
            "frequency_penalty": 0.01
        })
    
    return params

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

# NOVA FUNÇÃO: Descarregar modelo da memória
def unload_model(model_id):
    """Remove um modelo da memória."""
    if model_id in loaded_models:
        print(f"Descarregando modelo: {model_id}")
        del loaded_models[model_id]
        gc.collect()  # Forçar coleta de lixo
        print(f"Modelo {model_id} descarregado com sucesso!")
        return True
    else:
        print(f"Modelo {model_id} não está carregado")
        return False

# Detectar o tipo de modelo para ajustes específicos
def get_model_type(model_id):
    """Identifica o tipo de modelo com base no nome."""
    model_id = model_id.lower()
    if "qwen" in model_id:
        return "qwen"
    elif "tiny" in model_id:
        return "tiny"
    elif "phi" in model_id:
        return "phi"
    elif "llama" in model_id:
        return "llama"
    elif "mistral" in model_id:
        return "mistral"
    elif "gemma" in model_id:
        return "gemma"
    elif "deepseek" in model_id:
        return "deepseek"
    else:
        return "default"

# Formatar com base no tipo de modelo
def format_prompt_for_model(messages, model_type):
    """Formata o prompt de acordo com o tipo do modelo."""
    system_msg = next((m.get('content', '') for m in messages if m.get('role') == 'system'), '')
    user_msgs = [m.get('content', '') for m in messages if m.get('role') == 'user']
    user_msg = user_msgs[-1] if user_msgs else ''
    
    if model_type == "qwen":
        return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    elif model_type == "tiny":
        return f"<|user|>\n{user_msg}\n<|assistant|>\n"
    elif model_type == "phi":
        return f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"
    elif model_type == "llama":
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    elif model_type == "mistral":
        return f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]"
    elif model_type == "gemma":
        return f"<start_of_turn>system\n{system_msg}<end_of_turn>\n<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"
    elif model_type == "deepseek":
        return f"<|im_start|>system\n{system_msg}\n<|im_end|>\n<|im_start|>user\n{user_msg}\n<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Formato genérico
        return f"System: {system_msg}\nUser: {user_msg}\nAssistant:"

# OpenAI Compatible Endpoints
@app.route("/v1/models", methods=["GET"])
def list_models():
    """Lista todos os modelos disponíveis (compatível com OpenAI)."""
    models = get_available_models()
    return jsonify({
        "object": "list",
        "data": models
    })

@app.route("/v1/models/load", methods=["POST"])
def load_model_endpoint():
    """Carrega um modelo específico na memória (compatível com OpenAI)."""
    data = request.json
    model_id = data.get("model", "")
    
    # Verificar se o model_id foi fornecido
    if not model_id:
        return jsonify({
            "error": {
                "message": "model é obrigatório",
                "type": "invalid_request_error",
                "param": "model",
                "code": "param_required"
            }
        }), 400
    
    # Tentar carregar o modelo
    model = load_model(model_id)
    
    if model:
        return jsonify({
            "id": f"load-{uuid.uuid4()}",
            "object": "model.load",
            "created": int(time.time()),
            "model": model_id,
            "success": True
        })
    else:
        return jsonify({
            "error": {
                "message": f"Não foi possível carregar o modelo '{model_id}'",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found"
            }
        }), 404

@app.route("/v1/models/unload", methods=["POST"])
def unload_model_endpoint():
    """Descarrega um modelo específico da memória (compatível com OpenAI)."""
    data = request.json
    model_id = data.get("model", "") # Usa o mesmo campo "model" da API OpenAI para consistência
    
    # Verificar se o model_id foi fornecido
    if not model_id:
        return jsonify({
            "error": {
                "message": "model é obrigatório",
                "type": "invalid_request_error",
                "param": "model",
                "code": "param_required"
            }
        }), 400
    
    # Tentar descarregar o modelo
    success = unload_model(model_id)
    
    if success:
        return jsonify({
            "id": f"unload-{uuid.uuid4()}",
            "object": "model.unload",
            "created": int(time.time()),
            "model": model_id,
            "success": True
        })
    else:
        return jsonify({
            "error": {
                "message": f"Modelo '{model_id}' não está carregado",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_loaded"
            }
        }), 404

# Permitir descarregar todos os modelos
@app.route("/v1/models/unload_all", methods=["POST"])
def unload_all_models_endpoint():
    """Descarrega todos os modelos da memória."""
    models_unloaded = list(loaded_models.keys())
    count = len(models_unloaded)
    
    # Limpar o dicionário de modelos
    loaded_models.clear()
    gc.collect()  # Forçar coleta de lixo
    
    return jsonify({
        "id": f"unload-all-{uuid.uuid4()}",
        "object": "model.unload_all",
        "created": int(time.time()),
        "models_unloaded": models_unloaded,
        "count": count,
        "success": True
    })

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Chat completions endpoint compatível com OpenAI."""
    data = request.json
    model_id = data.get("model", default_model)
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    
    # Obter parâmetros específicos para o modelo
    model_params = get_model_specific_params(model_id)
    
    # Permitir sobrescrever com parâmetros fornecidos pelo usuário
    max_tokens = data.get("max_tokens", model_params["max_tokens"])
    temperature = data.get("temperature", model_params["temperature"])
    stop = data.get("stop", model_params["stop"])
    repeat_penalty = data.get("repeat_penalty", model_params.get("repeat_penalty", 1.1))
    frequency_penalty = data.get("frequency_penalty", model_params.get("frequency_penalty", 0.0))
    top_p = data.get("top_p", model_params.get("top_p", 0.95))
    
    # Parâmetros do mirostat (se aplicável)
    mirostat_mode = model_params.get("mirostat_mode", 0)  # 0 = desativado
    mirostat_tau = model_params.get("mirostat_tau", 5.0)
    mirostat_eta = model_params.get("mirostat_eta", 0.1)
    
    # Identificar o tipo de modelo
    model_type = get_model_type(model_id)
    
    # Formatar prompt específico para o modelo
    prompt = format_prompt_for_model(messages, model_type)
    
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
        # Adicionar suporte a mirostat e outros parâmetros
        result = llm(
            prompt, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            stop=stop,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta
        )
        
        # Limpar a saída de tags de formatação
        raw_content = result['choices'][0]['text']
        content = clean_output(raw_content)
        
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
    
    # Identificar o tipo de modelo e ajustar parâmetros
    model_type = get_model_type(model_id)
    if model_type in ["qwen", "tiny"]:
        temperature = min(temperature, 0.3)
        max_tokens = min(max_tokens, 200)
    
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
        
        # Limpar a saída
        raw_content = result['choices'][0]['text']
        content = clean_output(raw_content)
        
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
    
    # Verifique se waitress está instalado, se não, use o servidor de desenvolvimento
    try:
        from waitress import serve
        print("Usando servidor de produção Waitress...")
        serve(app, host="0.0.0.0", port=8080, threads=8)
    except ImportError:
        try:
            import gunicorn
            print("AVISO: Para iniciar com Gunicorn, execute: gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 server:app")
            app.run(host="0.0.0.0", port=8080)
        except ImportError:
            print("AVISO: Executando com servidor de desenvolvimento Flask.")
            print("AVISO: Instale waitress ou gunicorn para produção: pip install waitress")
            app.run(host="0.0.0.0", port=8080)