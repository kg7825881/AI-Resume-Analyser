"""
calibrate_embeddings.py — run this once after pulling nomic-embed-text to find real
SEM_LOW/SEM_HIGH and TEXT_SIM_LOW/TEXT_SIM_HIGH values for matcher.py.

The thresholds currently in matcher.py were calibrated for qwen3-embedding:8b — a
different model's cosine similarity distribution, likely wrong for nomic-embed-text.
This script prints raw cosine similarity for known match / near-match / no-match pairs
so you can pick real cutoffs instead of guessing, the same way the qwen3 thresholds
were originally derived from real test-run data mid-conversation.

Usage: python3 calibrate_embeddings.py
"""

from matcher import get_embedding, cosine_similarity

print("=== SKILL-TO-SKILL PAIRS (calibrates SEM_LOW / SEM_HIGH) ===\n")

skill_pairs = [
    ("Python", "Python"),                          # identical -> expect very high
    ("Python", "Python (Advanced)"),                # near-identical phrasing
    ("Docker", "Kubernetes"),                       # related but distinct
    ("Docker", "Containerization"),                 # near-synonym
    ("Machine Learning", "Deep Learning"),           # related, overlapping field
    ("Python", "Adobe Photoshop"),                   # unrelated -> expect low
    ("SQL", "Data Analysis"),                        # loosely related
]

for a, b in skill_pairs:
    sim = cosine_similarity(get_embedding(a), get_embedding(b))
    print(f"  {a!r:28} vs {b!r:28} -> {sim:.3f}")

print("\n=== FREE-TEXT PAIRS (calibrates TEXT_SIM_LOW / TEXT_SIM_HIGH) ===\n")

text_pairs = [
    (
        "Build and deploy machine learning models for classification and NLP tasks.",
        "Developed a CNN-based deep learning model to classify chest X-ray images into "
        "Normal vs. Pneumonia categories, achieving 89% validation accuracy.",
    ),  # genuinely relevant JD-vs-project pair -> expect this to land near/above TEXT_SIM_HIGH
    (
        "Machine Learning Engineer. Key skills: Python, PyTorch, TensorFlow.",
        "Orchestrated 5+ inter-college sports tournaments for 500+ participants.",
    ),  # irrelevant pair -> expect this near/below TEXT_SIM_LOW
    (
        "Build and deploy machine learning models.",
        "Analyzed insurance subrogation claims to create training data for AI-powered claim assistance.",
    ),  # partially relevant (AI-adjacent but not core ML engineering) -> expect somewhere in between
]

for a, b in text_pairs:
    sim = cosine_similarity(get_embedding(a), get_embedding(b))
    print(f"  sim = {sim:.3f}")
    print(f"    A: {a}")
    print(f"    B: {b}\n")

print("=== HOW TO USE THIS ===")
print("Pick SEM_LOW just below your weakest genuine-match skill pair's score, and SEM_HIGH")
print("just below your strongest genuine-match pair's score (leave room for real variation).")
print("Do the same for TEXT_SIM_LOW/HIGH using the free-text pairs. Update the four constants")
print("at the top of matcher.py with the real numbers you see above.")
