#!/bin/bash

# Executar o container com o servidor llama.cpp
echo "Iniciando servidor llama.cpp..."

docker run -d \
  --name llama-cpp-server \
  --restart unless-stopped \
  -p 8080:8080 \
  -v $(pwd)/models:/app/llama.cpp/models \
  llama-cpp-server:latest \
  --ctx-size 4096 \
  -t 18 \
  --batch-size 512 \
  --parallel 4 \
  --mlock

echo "Servidor iniciado na porta 8080!"
echo "Acesse a API em: http://localhost:8080"
echo "Para ver os logs do servidor: docker logs -f llama-cpp-server"