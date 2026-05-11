# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Instala Python e dependências de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Aceita HF_TOKEN como build argument
ARG HF_TOKEN=""

# Copia e instala dependências em um diretório temporário
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Faz login no HuggingFace se HF_TOKEN for fornecido
RUN if [ -n "$HF_TOKEN" ]; then \
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"; \
    fi

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Instala Python na stage de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Aceita HF_TOKEN como build argument para a stage de runtime
ARG HF_TOKEN=""

# Copia apenas os pacotes instalados do builder
COPY --from=builder /root/.local /root/.local

# Atualiza PATH para incluir os pacotes do usuário
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_TOKEN=${HF_TOKEN} \
    TABPFN_TOKEN=${TABPFN_TOKEN}

# Copia código da aplicação
COPY . .

EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]