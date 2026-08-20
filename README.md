# VERA: AI-Powered Resume Screening Agent

VERA (Verified Evaluation & Ranking Assistant) is an intelligent, transparent, and explainable applicant tracking and resume screening engine. It moves beyond black-box semantic matching by utilizing a two-stage **BM25 lexical retrieval + Local LLM Judge** pipeline to score candidates against Job Descriptions (JDs).

Every score, match, and missing requirement is backed by a specific, inspectable quote from the candidate's resume, ensuring HR teams know exactly *why* a candidate was ranked the way they were.


## 🌟 Key Features
* **Explainable Scoring:** No ambiguous cosine similarity scores. VERA retrieves actual resume chunks (via BM25) and uses a local LLM to judge the evidence as *Direct*, *Related*, *Weak*, or *None*, providing a plain-text reason for every match.

* **7-Dimensional Job Fit Analysis:** Candidates are scored across Mandatory Skills, Relevant Experience, Education, Industry Keywords, Soft Skills, Job Title Match, and Preferred Skills.

* **Deterministic Safety Nets:** Built-in Python guardrails prevent LLM hallucinations. If a model invents a skill, fabricates a company, or misattributes a JD requirement to a candidate, the pipeline catches and drops it before scoring.

* **Hard Gate Filtering:** Automatically flags and filters out candidates who fail to meet a customizable threshold of mandatory requirements.

* **Batch Streaming UI:** The Next.js frontend receives resume ingestion results via progressive NDJSON streaming, preventing long loading screens for large batches.

* **100% Local & Private:** Powered by local Ollama models (e.g., `qwen2.5:3b`, `gemma3:1b`), ensuring candidate PII never leaves your infrastructure.

## 🏗️ Architecture & Tech Stack

**Backend (Python / FastAPI)**

* **API:** FastAPI, Pydantic, Uvicorn.
* **Database:** SQLite (Zero-setup, via `db.py`).
* **Extraction:** PyMuPDF, `pdf2image`, `pytesseract` (for OCR fallback).
* **Retrieval & Scoring:** `rank-bm25` (BM25Plus), `ollama` (Python client), `difflib` (deterministic string matching).

**Frontend (Next.js / React)**

* **Framework:** Next.js 14 (App Router).
* **Styling:** Pure CSS (`globals.css`) matching a modern dark-dashboard aesthetic.
* **State Management:** React Context (`providers.js`) with `localStorage` persistence.

## 🚀 Getting Started

### Prerequisites

Before running VERA, ensure you have the following installed on your system:

1. **Python 3.10+**
2. **Node.js 18+** & **npm**
3. **Ollama**: Installed and running locally. You must pull the required models:
   * `ollama pull qwen2.5:3b` (Used for structured extraction)
   * `ollama pull gemma3:1b` (Used for evidence judging)
4. **Tesseract OCR** & **Poppler**: Required for processing scanned PDFs/images.
   * *Mac:* `brew install tesseract poppler`
   * *Linux:* `sudo apt-get install tesseract-ocr poppler-utils`
   * *Windows:*
     * Install Tesseract via the [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki), or with Chocolatey: `choco install tesseract`
     * Install Poppler via Chocolatey: `choco install poppler`
     * Alternatively, download the [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) binaries and add the `bin/` folder to your system `PATH`
     * Verify both are on your `PATH` by running `tesseract --version` and `pdftoppm -v` in a new terminal


### 1. Backend Setup

Open a terminal and navigate to the `VERA-engine` directory:

**Mac / Linux:**

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn api:app --reload
```

**Windows (Command Prompt / PowerShell):**

```bat
:: Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt

:: Start the FastAPI server
uvicorn api:app --reload
```

The backend will run at `http://localhost:8000`. The SQLite database (`VERA.db`) and local `storage/` directories for resumes and JDs will be created automatically on startup.

### 2. Frontend Setup

Open a new terminal and navigate to the `VERA-frontend` directory:

**Mac / Linux:**

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.local.example .env.local
```

**Windows (Command Prompt):**

```bat
:: Install dependencies
npm install

:: Set up environment variables
copy .env.local.example .env.local
```

**Windows (PowerShell):**

```powershell
# Install dependencies
npm install
```

The frontend dashboard will be accessible at `http://localhost:3000`.


## 🐳 Running with Docker (Recommended)
 
1. **Configure Ollama for Docker access (macOS):**
```bash
   launchctl setenv OLLAMA_HOST "0.0.0.0"
   # Quit Ollama from menu bar and restart it
```
 
2. **Build the image:**
```bash
   cd vera-engine
   docker build -t resume-analyzer-backend .
```
 
3. **Run the container with persistent volumes & live reload:**
```bash
   docker run -d \
     --name ai-backend \
     -p 8000:8000 \
     -v "$(pwd):/app" \
     -e OLLAMA_HOST="http://host.docker.internal:11434" \
     -e TALENTLENS_FRONTEND_ORIGINS="*" \
     resume-analyzer-backend \
     uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
 
 
## 🌐 Cloud Deployment (Vercel + Local Tunnel)
 
1. **Expose Local Backend:**
```bash
   ngrok http 8000
```
 
2. **Deploy Frontend on Vercel:**
   * Import repo and set Root Directory to `vera-frontend`.
   * Set Environment Variable:
     * `NEXT_PUBLIC_API_BASE_URL`: `https://your-ngrok-url.ngrok-free.dev`
3. **Configure Backend CORS:**
   * Pass your Vercel deployment URL into `TALENTLENS_FRONTEND_ORIGINS`.

 
## ⚙️ Environment Variables Reference
 
| Variable | Scope | Description | Default |
| :--- | :--- | :--- | :--- |
| `NEXT_PUBLIC_API_BASE_URL` | Frontend | Base URL for FastAPI backend | `http://localhost:8000` |
| `OLLAMA_HOST` | Backend | Host address for Ollama instance | `http://127.0.0.1:11434` |
| `TALENTLENS_FRONTEND_ORIGINS` | Backend | Allowed CORS origins (comma-separated) | `http://localhost:3000` |
 

## 📖 Usage & Workflows

1. **Screen Candidates (`/screen`)**
   * Upload a Job Description (PDF/DOCX) or select an existing one from the library. The engine will extract the role title, experience required, and atomic skills.
   * Upload a batch of candidate resumes.
   * Click "Run Analysis" to score the current batch against the selected JD.

2. **Candidate Ranking (`/results/[roleId]`)**
   * View the ranked table of all candidates scored against a specific role.
   * Filter by *All*, *85+ (Excellent)*, *80+ (Strong)*, or *Mandatory pass*.
   * Quickly glance at the category breakdown (Mandatory, Experience, Education, Soft Skills, etc.) natively in the table.

3. **Candidate Detail (`/candidate/[candidateId]`)**
   * Click on any candidate to view their comprehensive **Job Fit** scorecard.
   * Review the **Evidence Panel** to see exactly which skills matched (via exact match or LLM judgment) and read the judge's reasoning.
   * Review extracted experience, parsed education, and additional candidate skills that were not consumed by the JD requirements.


## 📊 Scoring Methodology

The final Job Fit score (0-100%) is calculated across 7 weighted categories (configured in `scorer.py`):

1. **Mandatory Skills (25%)**: Must clear the minimum contribution threshold. Evaluated via exact word-boundary match or BM25+LLM evidence.
2. **Relevant Experience (25%)**: Deterministic calculation of total years worked vs. JD minimum requirements.
3. **Education (20%)**: Degree and field matching against JD requirements.
4. **Industry Keywords (10%)**: Domain-specific terminology matches (e.g., "Fintech", "Healthcare").
5. **Soft Skills (10%)**: Interpersonal and workflow requirements.
6. **Job Title Match (5%)**: Deterministic similarity bypass (>= 50% match) or LLM evaluation of the candidate's recent roles against the target role.
7. **Preferred Skills (5%)**: "Nice-to-have" technical tools and frameworks.

**The Hard Gate:** By default, if a candidate is missing more than **6** Mandatory Skills, their application is flagged as `hard_gate_failed`, excluding them from the primary ranked pool.
