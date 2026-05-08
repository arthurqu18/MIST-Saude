# MIST_Download
Este repositório é para que aqueles que desejarem baixar nossa ferramenta MIST-Saúde.

## Pré-requisitos:

Antes de começar, você precisará ter instalado em sua máquina:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/) (incluído no Docker Desktop)
* Git

## Passos para rodar a ferramenta:

- Clone este repositório no diretório que deseje.

- TabPFN: o modelo TabPFN v2.5 exige autenticação para download e uso local. Sem a autenticação, pode ocorrer erro na inicialização.

### Passo 1: Aceitar os Termos na Hugging Face

1. Visite https://huggingface.co/Prior-Labs/tabpfn_2_5
2. Faça login na sua conta Hugging Face ou crie uma conta e confirme o email de verificação
3. Volte para https://huggingface.co/Prior-Labs/tabpfn_2_5 e clique em **"Agree and access repository"**

### Passo 2: Obter os Tokens Necessários

#### Token Hugging Face
1. Vá para https://huggingface.co/settings/tokens
2. Clique em **"New token"**
3. Crie um token com acesso **read**
4. Copie o token gerado e cole no devido espaço .env do passo 3

#### Token do TabPFN
1. Acesse https://ux.priorlabs.ai
2. Faça login ou registre-se
3. Aceite a licença na aba **Licenses**
4. Copie sua API Key em https://ux.priorlabs.ai/api/keys e cole no devido espaço .env do passo 3

### Passo 3: Configurar o Environment

Crie um arquivo chamado exatamente `.env` na raiz do projeto e coloque:

```bash
HF_TOKEN=seu_token_huggingface_aqui
TABPFN_TOKEN=sua_api_key_tabpfn_aqui
```

Substitua os valores pelos tokens obtidos nos sites acima.

> Observação: o `.env` e o `Dockerfile` já estão configurados para usar esses tokens durante a build e execução do container.

### Passo 4: Executar o Docker

Com os tokens configurados e o Docker inicializado, execute:

```bash
docker compose up -d --build
```

Após o build e execução:

- Acesse `localhost:3000`
- Para parar a aplicação:

```bash
docker compose down
```
