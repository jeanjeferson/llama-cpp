FROM ubuntu:22.04

# Definir variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    wget \
    curl \
    libcurl4-openssl-dev \
    unzip \
    ninja-build \
    ccache \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Clone do repositório llama.cpp
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /app/llama.cpp

# Compilar com otimizações
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON
RUN cmake --build build

# Debug: listar estrutura de diretórios e executáveis
RUN echo "==== ESTRUTURA DE DIRETÓRIOS ====" && \
    ls -la /app/llama.cpp/build && \
    echo "==== ARQUIVOS EXECUTÁVEIS ====" && \
    find /app/llama.cpp -type f -executable | sort

# Instalar dependências Python
RUN pip3 install flask flask_cors numpy sentencepiece

# Criar diretório para modelos
WORKDIR /app/llama.cpp
RUN mkdir -p models

# Volume para modelos
VOLUME /app/llama.cpp/models

# Criar script de inicialização que busca o servidor
RUN echo '#!/bin/bash\n\
# Encontrar o executável do servidor\n\
SERVER_PATHS=(\n\
  "/app/llama.cpp/build/bin/server"\n\
  "/app/llama.cpp/build/server"\n\
  "/app/llama.cpp/server/build/server"\n\
  "$(find /app/llama.cpp -type f -executable -name "*server*" | grep -v "start-server" | head -n 1)"\n\
)\n\
\n\
# Verificar cada caminho possível\n\
for path in "${SERVER_PATHS[@]}"; do\n\
  if [ -f "$path" ] && [ -x "$path" ]; then\n\
    echo "Encontrado servidor em: $path"\n\
    # Executar o servidor com todos os argumentos\n\
    exec "$path" -m /app/llama.cpp/models/model.gguf --host 0.0.0.0 --port 8080 "$@"\n\
    exit 0\n\
  fi\n\
done\n\
\n\
# Se não encontrou, listar todos os executáveis para debug\n\
echo "ERRO: Não foi possível encontrar o executável do servidor!"\n\
echo "Executáveis disponíveis:"\n\
find /app/llama.cpp -type f -executable | sort\n\
echo "Conteúdo do diretório build:"\n\
ls -la /app/llama.cpp/build\n\
\n\
# Última tentativa: executar qualquer binário com "server" no nome\n\
LAST_RESORT=$(find /app/llama.cpp -type f -executable -name "*server*" | head -n 1)\n\
if [ -n "$LAST_RESORT" ]; then\n\
  echo "Tentativa final: $LAST_RESORT"\n\
  exec "$LAST_RESORT" -m /app/llama.cpp/models/model.gguf --host 0.0.0.0 --port 8080 "$@"\n\
else\n\
  exit 1\n\
fi\n' > /app/start-server.sh

# Tornar o script executável
RUN chmod +x /app/start-server.sh

# Expor porta do servidor
EXPOSE 8080

# Usar o script como ponto de entrada
ENTRYPOINT ["/app/start-server.sh"]

# Parâmetros padrão
CMD ["--ctx-size", "4096", "-t", "18", "--batch-size", "512", "--parallel", "4", "--mlock"]