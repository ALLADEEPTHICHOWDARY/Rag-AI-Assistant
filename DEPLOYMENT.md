# Deploying to Azure Container Apps

This app is containerized and deploys to **Azure Container Apps** (serverless
containers, scales to zero when idle — free tier covers a portfolio-scale
project, see cost notes at the bottom).

## One-time setup

You need the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
installed and logged in (`az login`) before starting.

### 1. Create a resource group

```bash
az group create \
  --name rag-assistant-rg \
  --location eastus
```

### 2. Create an Azure Container Registry (ACR)

This is where your Docker image gets stored. Name must be globally unique,
lowercase, alphanumeric only.

```bash
az acr create \
  --resource-group rag-assistant-rg \
  --name ragassistantacr \
  --sku Basic
```

> If `ragassistantacr` is taken, pick another name — and update `ACR_NAME`
> in `.github/workflows/deploy-azure.yml` to match.

### 3. Create the Container Apps environment

```bash
az extension add --name containerapp --upgrade

az containerapp env create \
  --name rag-assistant-env \
  --resource-group rag-assistant-rg \
  --location eastus
```

### 4. Build and push the first image manually

(After this, GitHub Actions takes over for every future push.)

```bash
az acr build \
  --registry ragassistantacr \
  --image rag-ai-assistant:latest \
  .
```

### 5. Create the Container App

```bash
az containerapp create \
  --name rag-ai-assistant \
  --resource-group rag-assistant-rg \
  --environment rag-assistant-env \
  --image ragassistantacr.azurecr.io/rag-ai-assistant:latest \
  --target-port 8501 \
  --ingress external \
  --registry-server ragassistantacr.azurecr.io \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 0 --max-replicas 1
```

`--min-replicas 0` is what makes this free-tier-friendly — the container
shuts down completely when nobody's using it and costs nothing while idle.
The trade-off: the first request after idle time takes 15-30s to "wake up"
(cold start), which is normal for scale-to-zero and fine for a portfolio demo.

Azure will print a URL like `https://rag-ai-assistant.<random>.eastus.azurecontainerapps.io`
— that's your live link for your resume/README.

### 6. Wire up GitHub Actions for CI/CD

Create a service principal so GitHub can deploy on your behalf:

```bash
az ad sp create-for-rbac \
  --name rag-assistant-gh-actions \
  --role contributor \
  --scopes /subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/rag-assistant-rg \
  --sdk-auth
```

Copy the entire JSON output. In your GitHub repo:
**Settings → Secrets and variables → Actions → New repository secret**
Name it `AZURE_CREDENTIALS`, paste the JSON as the value.

That's it — every push to `main` now rebuilds the image and redeploys automatically.

## Cost notes

- Azure Container Apps gives every subscription 180,000 vCPU-seconds,
  360,000 GiB-seconds, and 2 million requests **free every month**.
- With `--min-replicas 0`, you're only billed while the container is
  actually handling a request — realistically $0/month for a demo project.
- To guarantee zero ongoing cost when you're done showcasing it:
  `az group delete --name rag-assistant-rg` removes everything.

## Running locally with Docker (no Azure needed)

```bash
docker build -t rag-ai-assistant .
docker run -p 8501:8501 rag-ai-assistant
```

Then open http://localhost:8501.
