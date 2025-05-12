#!/bin/bash

# Substitua pelo IP do seu servidor
SERVER_IP="localhost:8080"

echo "=== Testando Servidor LLama.cpp ==="
echo ""

echo "1. Verificando status do servidor..."
curl -s http://$SERVER_IP/health | jq
echo ""

echo "2. Listando modelos disponíveis..."
MODELS=$(curl -s http://$SERVER_IP/models)
echo $MODELS | jq
echo ""

# Verificar se existe modelo
MODEL_COUNT=$(echo $MODELS | jq '.models | length')
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "Nenhum modelo encontrado. Vamos baixar um modelo..."
    cd /etc/easypanel/projects/ai/llama-cpp/code/models/
    wget -q --show-progress https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O model.gguf
    echo "Modelo baixado!"
    echo ""
fi

echo "3. Carregando primeiro modelo..."
FIRST_MODEL=$(echo $MODELS | jq -r '.models[0].name')
echo "Carregando modelo: $FIRST_MODEL"
curl -s -X POST http://$SERVER_IP/models/load \
  -H "Content-Type: application/json" \
  -d "{\"model_name\": \"$FIRST_MODEL\"}"
echo ""
echo ""

echo "4. Aguardando modelo carregar (30 segundos)..."
sleep 30

echo "5. Testando geração de texto..."
curl -s -X POST http://$SERVER_IP/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<s>[INST] Escreva um poema sobre IA em 4 linhas [/INST]",
    "n_predict": 128,
    "temperature": 0.7,
    "stop": ["</s>"]
  }' | jq '.content'
echo ""

echo "6. Status final do servidor..."
curl -s http://$SERVER_IP/health | jq
echo ""

echo "=== Teste Concluído ==="