# Resume–Job Description Matcher

A small NLP project that compares a resume with a job description. It shows:

1. **Match score (0–100%)**: TF-IDF vectors + cosine similarity
2. **Matched keywords**: important JD terms that are also in the resume
3. **Missing keywords**: important JD terms the resume doesn't mention
4. *(Optional)* **Semantic score**: sentence embeddings (`all-MiniLM-L6-v2`), so you can compare word overlap with meaning overlap

## Project structure

```
app.py                     # Whole app: NLP pipeline + Streamlit UI
requirements.txt
samples/                   # Pre-packed samples for quick demo
  - resume.txt             # Sample candidate profile (text format)
  - job_description.txt    # Junior ML Engineer opening

demo_data/samples/         # Additional structured sample data by role
  - python_developer.jdxmd # Senior Python/NLP engineer JD
  - datascientist_resume.md # Mid-level NLP-focused resume
  - data_engineer_resume.txt   # ETL-pipeline-heavy profile
  - content_writer.jdtxt       # Prompt-writing content spec

VIVA_CHEAT_SHEET.md       # 5 key NLP concepts to explain live
docs/DEMO_GUIDE.md        # Step-by-step demo walkthrough for presenters
docs/VIVA_NOTES.md        # Detailed answers for viva-style Q&A session

demo_data/scripts/        # Lightweight smoke tests and utilities
  - smoke_test.sh         # Quick validation script
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`sentence-transformers` is only needed for the optional semantic score. It also
installs PyTorch, which is a large download. If you skip it, everything else
still works.

On first run, the app downloads the NLTK data it needs (punkt, stopwords,
wordnet). This happens automatically.

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501 and:

1. Paste the resume and JD, or upload them as `.txt`/`.pdf`. The files in `samples/` work well.
2. *(Optional)* Turn on the semantic score toggle.
3. Click **Analyze**.
4. Open **"How it works"** to see the tokens after each preprocessing stage.

## The NLP pipeline (in `app.py`)

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `preprocess(text)` | lowercase → remove punctuation/numbers → tokenize → remove stopwords → lemmatize → remove domain stopwords |
| 2 | `vectorize(resume, jd)` | TF-IDF vectors for both documents (scikit-learn) |
| 3 | `similarity(vec1, vec2)` | cosine similarity × 100 |
| 4 | `keyword_gap(resume, jd)` | top TF-IDF terms of the JD → split into matched / missing |
| ★ | `semantic_similarity(...)` | cosine similarity of sentence embeddings |

**Note on `keyword_gap`:** JD keywords are ranked with TF-IDF computed across
the lines of the JD itself, not across the resume and JD together. With only
two documents, IDF lowers the weight of every word that appears in both. That
would push skills the candidate *already has* to the bottom, so they would
never show up as "matched".
