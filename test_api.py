#!/usr/bin/env python3
"""
Cliente Python para o servidor llama.cpp
"""

import requests
import json
import time
import argparse

def query_llama(prompt, api_url="http://localhost:8080/completion", 
                n_predict=512, temp=0.7, stop_sequences=None):
    """
    Envia uma consulta para o servidor llama.cpp e retorna a resposta.
    
    Args:
        prompt: Texto a ser enviado para o modelo
        api_url: URL da API do servidor
        n_predict: Número máximo de tokens a serem gerados
        temp: Temperatura para amostragem
        stop_sequences: Lista de sequências para parar a geração
        
    Returns:
        Texto gerado pelo modelo
    """
    if stop_sequences is None:
        stop_sequences = ["\n\n"]
        
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temp,
        "stop": stop_sequences
    }
    
    print(f"Enviando prompt: {prompt[:50]}...")
    start_time = time.time()
    
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    
    elapsed_time = time.time() - start_time
    print(f"Tempo de resposta: {elapsed_time:.2f} segundos")
    
    if response.status_code == 200:
        result = response.json()
        return result.get("content", "")
    else:
        print(f"Erro: {response.status_code}")
        print(response.text)
        return None

def chat_format(message):
    """Formata a mensagem no formato de chat para o Mistral."""
    return f"<s>[INST] {message} [/INST]"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente para o servidor llama.cpp")
    parser.add_argument("--prompt", type=str, default="Quem foi Santos Dumont?",
                        help="Prompt para enviar ao modelo")
    parser.add_argument("--url", type=str, default="http://localhost:8080/completion",
                        help="URL da API")
    parser.add_argument("--tokens", type=int, default=512, 
                        help="Número máximo de tokens a gerar")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Temperatura para geração")
    
    args = parser.parse_args()
    
    formatted_prompt = chat_format(args.prompt)
    response = query_llama(
        formatted_prompt, 
        api_url=args.url,
        n_predict=args.tokens,
        temp=args.temp,
        stop_sequences=["</s>"]
    )
    
    if response:
        print("\n--- RESPOSTA ---")
        print(response)