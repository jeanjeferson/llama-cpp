#!/bin/bash

# Criar diretório para modelos se não existir
mkdir -p models

# Baixar o modelo Mistral 7B Instruct
echo "Baixando Mistral 7B Instruct..."
wget -c https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O models/model.gguf

echo "Download concluído! O modelo está em: models/model.gguf"