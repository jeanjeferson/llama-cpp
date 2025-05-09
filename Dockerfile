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

# Instalar dependências do servidor manualmente
# Em vez de usar requirements.txt que pode não existir
RUN pip3 install flask flask_cors numpy sentencepiece

# Criar diretório para modelos
WORKDIR /app/llama.cpp
RUN mkdir -p models

# Volume para modelos
VOLUME /app/llama.cpp/models

# Expor porta do servidor
EXPOSE 8080

# Comando para iniciar o servidor
ENTRYPOINT ["/app/llama.cpp/build/bin/server", "-m", "/app/llama.cpp/models/model.gguf", "--host", "0.0.0.0", "--port", "8080"]

# Parâmetros padrão
CMD ["--ctx-size", "4096", "-t", "18", "--batch-size", "512", "--parallel", "4", "--mlock"]