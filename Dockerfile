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
    pip install llama-cpp-python flask flask_cors numpy python-dotenv waitress gunicorn psutil

# Criar estrutura de diretórios
WORKDIR /app
RUN mkdir -p /app/models

# Copiar o script do servidor
COPY openai-compatible-server.py /app/server.py

# Tornar o script executável
RUN chmod +x /app/server.py

# Volume para modelos
VOLUME /app/models

# Expor porta do servidor
EXPOSE 8080

# Modificar o final do arquivo server.py para usar Waitress
RUN sed -i '/if __name__ == "__main__":/,$ d' /app/server.py && \
    echo 'if __name__ == "__main__":\n\
    print(f"Carregando modelo padrão: {default_model}")\n\
    load_model(default_model)\n\
    \n\
    print("Servidor compatível com OpenAI iniciando com Waitress...")\n\
    from waitress import serve\n\
    serve(app, host="0.0.0.0", port=8080, threads=8, connection_limit=500, channel_timeout=300)' >> /app/server.py

# Comando padrão
CMD ["python3", "/app/server.py"]