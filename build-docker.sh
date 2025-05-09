#!/bin/bash

# Construir a imagem Docker
echo "Construindo imagem Docker para llama.cpp..."
docker build -t llama-cpp-server:latest .

echo "Imagem Docker construída com sucesso!"