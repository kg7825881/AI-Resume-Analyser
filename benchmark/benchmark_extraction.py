"""
benchmark_extraction.py

Compares qwen2.5:3b-instruct vs phi3.5-mini for structured resume/JD
extraction on your local Ollama setup. Measures:
  - wall-clock latency per call
  - JSON validity (does it parse cleanly, no markdown fences etc.)
  - field completeness (how many expected top-level fields were populated)
  - a rough token/sec throughput estimate

Usage:
    1. Pull both models first:
         ollama pull qwen2.5:3b-instruct
         ollama pull phi3.5-mini

    2. Drop 2-3 real resume texts and one JD text into the SAMPLES dict
       below (or point RESUME_DIR / JD_PATH at real files - see the
       load_samples() function to swap in your own extractor.py's
       text-loading logic instead of the plain read_text() used here).

    3. Run:
         python benchmark_extraction.py

    Results are printed as a table and also written to
    benchmark_results.json so you can compare runs over time.
"""

import json
import time
import statistics
from pathlib import Path
from datetime import datetime, timezone

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELS = ["qwen2.5:3b-instruct", "phi3.5-mini"]
NUM_RUNS_PER_SAMPLE = 3  # run each sample this many times per model to smooth out noise

# Expected top-level fields we check for completeness scoring.
# Adjust this to match whatever schema your extractor.py actually asks for.
EXPECTED_FIELDS = [
    "name",
    "skills",
    "experience",
    "education",
    "certifications",
    "projects",
]

EXTRACTION_PROMPT = """You are a resume parsing engine. Extract structured data from the
resume text below and return ONLY valid JSON, no markdown fences, no commentary.

Schema:
{{
  "name": string,
  "skills": [string],
  "experience": [{{"title": string, "company": string, "duration": string, "description": string}}],
  "education": [{{"degree": string, "institution": string, "year": string}}],
  "certifications": [string],
  "projects": [{{"name": string, "description": string, "technologies": [string]}}]
}}

Resume text:
---
{resume_text}
---

Return ONLY the JSON object.
"""

# --- Replace these with real resume text pulled from your samples folder ---
SAMPLES = {
    "sample_1_short": """
  "candidate_name": "VIBHUTI SHARMA",
  "skills": [
    "Data Scientist",
    "Product Owner",
    "AI Platform Development",
    "MLOps",
    "GenAI Platform Development",
    "Agentic AI",
    "OCR/Document Processing",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing (NLP)",
    "Python",
    "FastAPI",
    "RAG",
    "Azure OpenAI",
    "Google ADK",
    "Crew AI",
    "DeepSeek",
    "Stable Diffusion",
    "GPT variants",
    "Grok",
    "LLMs",
    "FAISS",
    "Databricks",
    "Snowflake",
    "AWS S3",
    "Azure Storage",
    "GCP",
    "SQL",
    "GitHub",
    "Numpy",
    "Pandas",
    "Scikit-learn",
    "Tensorflow",
    "Keras",
    "H2O",
    "Plotly",
    "PySpark",
    "MongoDB",
    "Postgres",
    "Docker",
    "Selenium",
    "BeautifulSoup",
    "Deep Learning",
    "Computer Vision",
    "Statistical Analysis",
    "Processing Large Datasets",
    "Data Visualization"
  ],
  "total_years_experience": 10,
  "experience": [
    {
      "title": "Team Lead in Data Science at UPL",
      "company": "UPL Ltd",
      "years": 7,
      "domain": "Agentic AI, MLOps, OCR/Document Processing, Machine Learning, Computer Vision, Natural Language Processing (NLP)",
      "description": "Led and hands-on developed the platform for UPL Genie, a secure enterprise GenAI platform. Designed, built, and rolled out platform RBAC, breaking down work into sprint-ready tasks and aligning teams on access rules for apps/features/data. Used adoption and usage analytics to reprioritise the roadmap toward high-impact capabilities."
    },
    {
      "title": "Technical PM / AI PM / Platform PM",
      "company": "UPL Ltd",
      "years": 2,
      "domain": "Agentic AI, MLOps, OCR/Document Processing, Machine Learning, Computer Vision, Natural Language Processing (NLP)",
      "description": "Led and hands-on developed the platform for UPL Genie, a secure enterprise GenAI platform. Designed, built, and rolled out platform RBAC, breaking down work into sprint-ready tasks and aligning teams on access rules for apps/features/data."
    },
    {
      "title": "Data Scientist | Analytics Consultant",
      "company": "EXL",
      "years": 1,
      "domain": "Agentic AI, MLOps, OCR/Document Processing, Machine Learning, Computer Vision, Natural Language Processing (NLP)",
      "description": "Developed claims coverage predictive model to optimize coverage decisions for a top US insurer. Achieved a 10% reduction in overwrite costs, 13% improvement in decision accuracy, and an 18% reduction in false alerts."
    },
    {
      "title": "Data Scientist",
      "company": "GoMechanic",
      "years": 2,
      "domain": "Agentic AI, MLOps, OCR/Document Processing, Machine Learning, Computer Vision, Natural Language Processing (NLP)",
      "description": "Built & annotated multiple datasets of 7000+ images to facilitate accurate model training. Designed and implemented multiple Computer Vision models to tackle distinct challenges, including Car Orientation Detection, Car Color Detection, Car Part Detection, and Damage Type Detection."
    },
    {
      "title": "Junior Research Fellow",
      "company": "IIT Jodhpur",
      "years": 1,
      "domain": "Agentic AI, MLOps, OCR/Document Processing, Machine Learning, Computer Vision, Natural Language Processing (NLP)",
      "description": "Implemented Neural Architecture Search to discover the optimal CNN architecture for a given dataset. Achieved a successful balance between accuracy and parameter count using a Genetic Algorithm-based search algorithm."
    },
    {
      "title": "Engineer Trainee",
      "company": "Infosys",
      "years": 1,
      "domain": "Agentic AI, MLOps, OCR/Document Processing, Machine Learning, Computer Vision, Natural Language Processing (NLP)",
      "description": "Built early foundation in software delivery, SDLC discipline, and cross-team execution in an enterprise environment."
    }
  ],
  "education": [
    {
      "degree_level": "MTECH",
      "field": "COMPUTER SCIENCE ENGINEERING",
      "institution": "JUIT Waknaghat"
    },
    {
      "degree_level": "BTECH",
      "field": "COMPUTER SCIENCE ENGINEERING",
      "institution": "DTU Delhi (formerly, Delhi College of Engineering)"
    },
    {
      "degree_level": "MHRD Scholarship recipient-2",
      "field": "",
      "institution": "IIT Jodhpur"
    }
  ],
  "certifications": [
    "Digital Product Management (Coursera)",
    "AI Product Management (Coursera)",
    "Demand Forecasting Automation (Coursera)"
  ],
  "projects": [
    {
      "title": "UPL Genie - Enterprise GenAI Platform",
      "description": "Owned end-to-end product delivery for automated regulatory dossier generation used prior to country-wise filings. Partnered closely with country regulatory teams, aligning daily on requirements, priorities, and rollout readiness."
    },
    {
      "title": "OCR + Invoice Processing (Document AI)",
      "description": "Built OCR pipelines for processing submitted invoices and extracting structured data. Reduced manual processing effort and improved downstream finance accuracy."
    }
  ],
  "candidate_id": "45d670eb-9f18-4430-93af-94598be0ddb6",
  "document_id": "a7f05fde-afb0-4d59-9d58-1e6bd281f2d1",
  "file_name": "Naukri_VibhutiSharma[5y_6m].pdf",
  "extraction_method": "pdf_text",
  "extraction_warnings": [],
  "uploaded_at": "2026-08-13T03:19:48.105238+00:00"
}
""",
    "sample_2_longer": """
Rahul Verma
Senior Data Scientist

Summary: 6+ years building ML pipelines for fraud detection and recommendation
systems. Strong background in NLP and classical ML.

Technical Skills:
- Languages: Python, SQL, R
- ML: scikit-learn, XGBoost, PyTorch, TensorFlow
- Data: Spark, Airflow, Snowflake
- Cloud: AWS (SageMaker, S3, Lambda)

Work Experience:
Senior Data Scientist, FinSecure Analytics (2021-Present)
- Led a team of 3 building a real-time fraud detection model, reducing false
  positives by 22%
- Designed feature pipelines processing 5M+ transactions daily using Spark

Data Scientist, RetailIQ (2018-2021)
- Built recommendation engine improving click-through rate by 15%
- Automated model retraining pipeline with Airflow

Projects:
Fraud Graph Analysis - built a graph-based anomaly detection prototype using
NetworkX and community detection to flag collusive fraud rings.

Education:
M.S. Data Science, IIT Bombay, 2018
B.S. Statistics, Pune University, 2016

Certifications: TensorFlow Developer Certificate, AWS Machine Learning Specialty
""",
}
# ----------------------------------------------------------------------------


def call_model(model: str, prompt: str) -> tuple[str, float]:
    """Call Ollama chat endpoint, return (response_text, latency_seconds)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": 1024},
    }
    start = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    text = data.get("message", {}).get("content", "")
    return text, elapsed


def try_parse_json(text: str) -> dict | None:
    """Strip common markdown fences and attempt to parse JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # drop a leading 'json' language tag if present
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # last resort: find the first { ... last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def completeness_score(parsed: dict | None) -> float:
    if not parsed:
        return 0.0
    populated = 0
    for field in EXPECTED_FIELDS:
        val = parsed.get(field)
        if val not in (None, "", [], {}):
            populated += 1
    return populated / len(EXPECTED_FIELDS)


def run_benchmark():
    results = []
    for model in MODELS:
        print(f"\n=== Benchmarking {model} ===")
        model_latencies = []
        model_valid_json = 0
        model_completeness = []
        total_calls = 0

        for sample_name, resume_text in SAMPLES.items():
            prompt = EXTRACTION_PROMPT.format(resume_text=resume_text.strip())
            for run_i in range(NUM_RUNS_PER_SAMPLE):
                total_calls += 1
                try:
                    text, elapsed = call_model(model, prompt)
                except requests.exceptions.RequestException as e:
                    print(f"  [{sample_name} run {run_i+1}] ERROR calling {model}: {e}")
                    continue

                parsed = try_parse_json(text)
                is_valid = parsed is not None
                completeness = completeness_score(parsed)

                model_latencies.append(elapsed)
                model_valid_json += int(is_valid)
                model_completeness.append(completeness)

                print(
                    f"  [{sample_name} run {run_i+1}] "
                    f"latency={elapsed:.2f}s  valid_json={is_valid}  "
                    f"completeness={completeness:.0%}"
                )

        summary = {
            "model": model,
            "total_calls": total_calls,
            "avg_latency_sec": round(statistics.mean(model_latencies), 2) if model_latencies else None,
            "median_latency_sec": round(statistics.median(model_latencies), 2) if model_latencies else None,
            "json_validity_rate": round(model_valid_json / total_calls, 2) if total_calls else None,
            "avg_field_completeness": round(statistics.mean(model_completeness), 2) if model_completeness else None,
        }
        results.append(summary)

    print("\n\n=== SUMMARY ===")
    header = f"{'Model':<22}{'Avg Latency(s)':<16}{'Median(s)':<12}{'JSON Valid %':<14}{'Field Complete %':<18}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<22}"
            f"{str(r['avg_latency_sec']):<16}"
            f"{str(r['median_latency_sec']):<12}"
            f"{str(int((r['json_validity_rate'] or 0) * 100)) + '%':<14}"
            f"{str(int((r['avg_field_completeness'] or 0) * 100)) + '%':<18}"
        )

    out_path = Path("benchmark_results.json")
    out_data = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nFull results written to {out_path.resolve()}")


if __name__ == "__main__":
    run_benchmark()
