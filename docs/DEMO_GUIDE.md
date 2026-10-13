# Demo Guide: Resume–Job Description Matcher

## Quick Start for Presenters

This guide walks you through demonstrating the NLP matcher to an audience.

---

## Before the Demo

1. **Install dependencies** (if running from scratch):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```

   The app will open at http://localhost:8501 in your browser.

---

## Demo Flow (≈8–10 minutes total)

### Slide 1: Hook the Audience (30 seconds)

**Goal:** Show that matching resumes to jobs is actually non-trivial.

**Say:**
> "Imagine hiring a junior ML engineer for text analytics. How do we know if a candidate has the right skills? Is this just keyword matching, or do we need something smarter?"

---

### Slide 2: Walk Through Inputs (1 minute)

**Goal:** Load the sample data using either:

- **Upload**: Click "Upload Files", select `samples/job_description.txt` and `samples/resume.txt`.
- **Paste**: Type short text directly into the text area.

**Say:**
> "Here's a real JD for a Junior NLP Engineer. And here's a candidate's resume. Let's see how our NLP pipeline compares them."

---

### Slide 3: Explain Preprocessing (2 minutes)

**Goal:** Open **"How it works"** tab to show tokenization, lemmatization, and stopword removal.

**What the audience should understand:**
- Lowercase, punctuation stripped, then tokenized.
- Stopwords like "the", "and" are removed.
- Lemmatization maps variations (e.g., "designing" → "design").
  - Custom domain stopwords also removed ("python code" is kept).

**Say:**
> "Before we compare, text must be normalized. Here you can see tokens going through preprocessing: lowercased, punctuation cleaned, stopwords dropped, and words lemmatized."

---

### Slide 4: Show TF–IDF & Cosine Similarity (2 minutes)

**Goal:** Click **"Analyze"** to compute match score.

**What's computed:**
- `vectorize()`: TF-IDF across the JD itself for keyword weighting.
- `similarity()`: cosine similarity between resume and JD vectors = final score (0–100%).

**Say:**
> "Our vectorizer builds TF-IDF features from the job description, emphasizing rare terms like 'named entity recognition' over common ones like 'Python'. Cosine similarity then gives us a score of X%."

---

### Slide 5: Discuss Matched & Missing Keywords (1 minute)

**Goal:** Point out matched vs missing JD keywords.

**Why it matters:**
- JD keywords are sorted by TF-IDF weight in the job posting alone.
- This prevents inflated "matched" scores from words appearing in both docs.

**Say:**
> "Here are critical JD terms our model found in the resume (matched) versus what's absent (missing). Because we rank on TF-IDF within only the JD, skills already present show up correctly as 'matched'."

---

### Slide 6: Optional Semantic Similarity Toggle (2 minutes)

**Goal:** Turn on semantic score if `sentence-transformers` is installed.

**What's computed:**
- Sentence embeddings using a pre-trained transformer.
- Cosine similarity comparing resume and JD at the sentence level.

**Say:**
> "Beyond keyword overlap, we can also embed both texts semantically and compute another similarity score. This captures meaning—even if wording differs—by measuring vector space distance."

---

## Common Pitfalls to Watch For

| Issue                         | Fix                                                    | Why it matters                                              |
|-------------------------------|---------------------------------------------------------|-------------------------------------------------------------|
| No internet                  | Install offline NLTK data (`python -m nltk.downloader`) | PyTorch & embeddings won't load without download.           |
| `SentenceTransformer` not installed | `pip install sentence-transformers`                   | Semantic toggle needs it; optional but nice to have.        |
| Large resume/JD paste         | Split into multiple files or paste sections            | Long text can strain memory during tokenization.             |

---

## Wrap-Up (30 seconds)

**Say:**
> "The pipeline normalizes text, extracts weighted keywords via TF-IDF, and computes both vector cosine similarity and optional semantic overlap to give a 0–100% match score — plus keyword gaps for quick review."
