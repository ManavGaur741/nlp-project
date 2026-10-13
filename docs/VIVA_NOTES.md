# Viva Notes: Deep-Dive Answers for Q&A Session

Use these notes to prepare detailed explanations of the system's NLP components.

---

## Core Concepts (From `VIVA_CHEAT_SHEET.md`)

### 1. What is TF–IDF and why do we use it in `vectorize()`?

**Answer:**
TF-IDF (Term Frequency–Inverse Document Frequency) weights terms by how important they are to a document relative to the other documents being compared. In our single-doc scenario, we compute IDF across the job description lines themselves so that words like "BERT" or "tokenization" get higher weight than stop-like words. This emphasizes specialized skills rather than common ones when ranking matched/missing keywords.

---

### 2. Why lemmatize instead of just lowercasing?

**Answer:**
Lemmatization maps inflected forms to a single canonical base form: e.g., "preprocessing", "processed" → "process". This clusters all variants into one term, boosting TF-IDF counts and making semantic search more accurate than simple lowercasing which only normalizes case but doesn't unify word forms.

---

### 3. How does the cosine similarity score relate to a final percentage?

**Answer:**
Cosine similarity ranges from -1 to +1 for vectors in high-dimensional space: 0 means orthogonal (perpendicular), +1 means identical direction. In our pipeline we compute `vec1.dot(vec2) / norms`, then multiply by 100% to give a match score of 0–100%. A positive score reflects that TF-IDF weights are all non-negative, so dot products stay in the 0..+1 range (not -1...).

---

### 4. What is keyword_gap(resume, jd) and how does it compute matched vs missing?

**Answer:**
`keyword_gap()` identifies important terms by sorting them by TF-IDF value computed *within the JD alone*. Terms in the resume whose IDF-weight matches exceed a threshold show up as "matched"; remaining critical JD terms that don't appear in the resume are "missing". By computing IDF only on the JD (not both docs together), skills the candidate already has correctly appear at the top and aren't pushed to missing merely because they appear in the JD text.

---

### 5. In the demo, can I disable the semantic score if sentence-transformers is not installed?

**Answer:**
Yes! The `app.py` code checks for the `sentence_transformers` module at import time; if missing it skips loading and uses only TF-IDF vectors. When you toggle the "semantic score" checkbox in the UI, it will show "not available" or skip adding to results gracefully. Just ensure `requirements.txt` comment is correct:

- For minimal install: omit `sentence-transformers`.
- To enable semantic features: uncomment the line in `requirements.txt`.

---

## Extended Q&A Prep (For Advanced Presentations)

### 6. What NLTK stopwords are used for our NLP domain?

**Answer:**
NLTK's built-in English_stopwords plus a custom set to retain programming terms ("python", "code", "git") that would otherwise be flagged as unimportant. For text classification tasks we remove common function words but keep technical keywords critical to the task.

---

### 7. How does our system handle long resumes? Does it truncate text?

**Answer:**
The NLP pipeline currently tokenizes all loaded text without hard truncation. However, very large files may hit memory constraints during TF-IDF computation. For production systems you might implement chunking: split texts into fixed-size windows; aggregate scores with averaging.

---

### 8. Why do we compute vectorizer on the JD and not both docs together?

**Answer:**
TF-IDF's inverse document factor penalizes words appearing in every input doc; if we vectorize both resume AND JD simultaneously, any skill the candidate already has (which appears in both) gets a low IDF weight. By computing TF-IDF only within the JD, all critical JD terms stay high-weighted; those that appear in the candidate's profile rank as "matched" and those missing stay clearly identified for hiring managers.

---

### 9. How could we extend this for multi-language resumes?

**Answer:**
NLTK's built-in stopword sets are English-only. For multilingual support:
- Load language-specific stopwords (e.g., NLTK French, Spanish datasets).
- Switch tokenizer to `spaCy` with appropriate language models (`en_core_web_sm`, `de_core_news_sm`).
- For embeddings, replace the English sentence-transformers model with BERT-base-multilingual-cased if available on Hugging Face.

---

### 10. What is a typical workflow for a new candidate's resume?

**Answer:**
1. **Upload**: Drag & drop `.txt`/`.pdf` file into "Upload Files" or paste raw text.
2. **Preprocess**: App tokenizes, lowercases, removes stopwords and lemmatizes on the fly.
3. **Vectorize**: TF-IDF vectors computed for both JD and resume (JD IDF-only strategy).
4. **Match & Compare**: Cosine similarity yields score; matched/missing keyword lists printed.
5. **Semantic (optional)**: If enabled, sentence embeddings add secondary semantic overlap metric.

---

This document covers core viva-style questions with technical depth for an audience of engineers hiring candidates or students presenting the system.
