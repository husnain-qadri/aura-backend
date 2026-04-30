# Aura Backend

Standalone Flask API for the Adaptive User Reading Assistant, configured for deployment on Google Cloud App Engine (Flexible).

## Prerequisites

- Python 3.14+
- Google Cloud SDK (`gcloud`) for deployment

## Local Development

```bash
# Create virtual environment
python3.14 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY="your-groq-api-key"
export FLASK_DEBUG=true

# Run the dev server
python main.py
```

The server starts on `http://localhost:8080`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/parse` | Upload and parse a PDF |
| POST | `/api/explain` | Explain selected text |
| POST | `/api/query` | Ask a question about a paper |
| POST | `/api/compare` | Compare two papers |
| POST | `/api/citation` | Verify a citation |
| POST | `/api/extract-references` | Extract and resolve references |
| POST | `/api/fetch-reference-pdf` | Download a reference PDF |
| POST | `/api/highlights` | Generate reading highlights |
| POST | `/api/reading-path` | Generate a guided reading path |
| POST | `/api/checkpoints` | Extract critical checkpoints |

## Deploy to GCP App Engine

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Set your Groq API key in app.yaml (env_variables section)
# Then deploy:
gcloud app deploy
```

For secrets, prefer GCP Secret Manager over hardcoding in `app.yaml`:

```bash
echo -n "your-key" | gcloud secrets create GROQ_API_KEY --data-file=-
```

Then reference it in `app.yaml` with:

```yaml
env_variables:
  GROQ_API_KEY: sm://YOUR_PROJECT_ID/GROQ_API_KEY
```

## Docker (local testing)

```bash
docker build -t aura-backend .
docker run -p 8080:8080 -e GROQ_API_KEY="your-key" aura-backend
```
