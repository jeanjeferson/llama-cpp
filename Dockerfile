FROM python:3.10-slim

# Definir variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar llama-cpp-python com otimizações para CPU
ENV CMAKE_ARGS="-DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON"
RUN pip install --upgrade pip && \
    pip install llama-cpp-python flask flask_cors numpy python-dotenv

# Criar estrutura de diretórios
WORKDIR /app
RUN mkdir -p /app/models

# Copiar o script do servidor
COPY server.py /app/server.py

# Tornar o script executável
RUN chmod +x /app/server.py

# Volume para modelos
VOLUME /app/models

# Expor porta do servidor
EXPOSE 8080

# Comando para iniciar o servidor
CMD ["python3", "/app/server.py"]