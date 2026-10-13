"""
Resume–Job Description Matcher
==============================
A simple NLP project that compares a resume with a job description (JD).

Pipeline:
    1. preprocess()   -> clean the raw text into meaningful tokens
    2. vectorize()    -> turn both documents into TF-IDF vectors
    3. similarity()   -> cosine similarity between the vectors, as a percentage
    4. keyword_gap()  -> which important JD keywords the resume has / is missing
    (bonus) semantic_similarity() -> sentence-embedding score for comparison

Run with:  streamlit run app.py
"""

import re
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# One-time setup: download the NLTK data we need (skipped if already present)
# ---------------------------------------------------------------------------
for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),   # needed by newer NLTK versions
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.data.find(path + ".zip")  # e.g. wordnet is kept as a .zip
        except LookupError:
            nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# Domain-specific stopwords: words that appear in almost every resume/JD but
# say nothing about actual skills. Removing them keeps the keyword lists useful.
# Written in base form, so we remove them AFTER lemmatizing
# ("requirements" -> "requirement" -> removed).
DOMAIN_STOP_WORDS = {
    "experience", "requirement", "responsibility", "role", "using", "work",
    "nice", "strong", "skill", "knowledge", "ability", "looking", "job",
    "title", "year", "including", "related", "field", "familiarity", "etc",
}
LEMMATIZER = WordNetLemmatizer()

# How many top JD keywords to inspect in the keyword-gap analysis
TOP_N_KEYWORDS = 30


# ===========================================================================
# STEP 1: PREPROCESSING
# ===========================================================================
def preprocess(text, return_stages=False):
    """
    Clean raw text and return a list of meaningful tokens.

    WHY: Raw text is noisy. "Python", "python," and "PYTHON" should all count
    as the same word, and words like "the" or "and" carry no information
    about skills. Cleaning first makes the later comparison fair.

    If return_stages=True, also returns a dict with the output of every
    stage (used by the "How it works" section in the UI).
    """
    stages = {}

    # 1a. Lowercase -> "Python" and "python" become the same token
    text = text.lower()
    stages["1. Lowercased"] = text

    # 1b. Remove punctuation and numbers -> they rarely signal skills and
    #     would otherwise create junk tokens like "2023" or "),"
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    stages["2. Punctuation & numbers removed"] = text

    # 1c. Tokenize -> split the string into individual words
    tokens = word_tokenize(text)
    stages["3. Tokenized"] = tokens

    # 1d. Remove stopwords -> drop very common words ("the", "is", "and")
    #     and 1-letter leftovers, which add noise but no meaning
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    stages["4. Stopwords removed"] = tokens

    # 1e. Lemmatize -> reduce words to their dictionary form
    #     ("models" -> "model", "analyses" -> "analysis") so that different
    #     forms of the same word are matched together
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    stages["5. Lemmatized"] = tokens

    # 1f. Remove domain stopwords (generic resume/JD words, see top of file)
    tokens = [t for t in tokens if t not in DOMAIN_STOP_WORDS]
    stages["6. Domain stopwords removed (final)"] = tokens

    if return_stages:
        return tokens, stages
    return tokens


# ===========================================================================
# STEP 2: VECTORIZATION (TF-IDF)
# ===========================================================================
def vectorize(resume_tokens, jd_tokens):
    """
    Convert both token lists into TF-IDF vectors.

    WHY: Computers compare numbers, not words. TF-IDF gives each word a
    weight that is high when the word is frequent in a document (TF) but
    rare across documents (IDF), so important terms count more than
    generic ones.

    Returns (resume_vector, jd_vector, fitted_vectorizer).
    """
    # Our tokens are already cleaned, so we join them back into strings
    # and tell the vectorizer not to lowercase/strip again.
    docs = [" ".join(resume_tokens), " ".join(jd_tokens)]

    # We use single words (unigrams). Adding ngram_range=(1, 2) would also
    # capture phrases like "machine learning", at the cost of noisier keywords.
    vectorizer = TfidfVectorizer(lowercase=False)
    matrix = vectorizer.fit_transform(docs)  # row 0 = resume, row 1 = JD

    return matrix[0], matrix[1], vectorizer


# ===========================================================================
# STEP 3: SIMILARITY (cosine)
# ===========================================================================
def similarity(vec1, vec2):
    """
    Cosine similarity between two vectors, returned as a percentage (0–100).

    WHY: Cosine similarity measures the ANGLE between two vectors, not their
    length. So a long resume and a short JD can still score highly if they
    talk about the same things in similar proportions.
    """
    score = cosine_similarity(vec1, vec2)[0][0]
    return round(float(score) * 100, 2)


# ===========================================================================
# STEP 4: KEYWORD GAP
# ===========================================================================
def keyword_gap(resume_text, jd_text, top_n=TOP_N_KEYWORDS):
    """
    Find the most important JD keywords and split them into those that
    appear in the resume (matched) and those that don't (missing).

    WHY: A single score says "how similar", but not "what to fix".
    Listing missing keywords gives the candidate actionable feedback.

    Returns (matched_list, missing_list).
    """
    # --- Rank JD keywords by TF-IDF *within the JD itself* ---
    # WHY not reuse the resume-vs-JD TF-IDF from vectorize()? With only two
    # documents, IDF gives LOWER weight to words found in both, so skills the
    # resume already has (e.g. "python") would sink to the bottom and never
    # show up as "matched". Instead we treat each line of the JD as a mini
    # document: words that appear in a few specific lines (skills, tools)
    # score high, generic words repeated everywhere score lower.
    jd_lines = [" ".join(preprocess(line)) for line in jd_text.splitlines()]
    jd_lines = [line for line in jd_lines if line]  # drop empty lines

    # norm=None: don't scale each line to length 1, otherwise a one-word
    # heading line like "Requirements" would get an unfairly high score.
    vectorizer = TfidfVectorizer(lowercase=False, norm=None)
    matrix = vectorizer.fit_transform(jd_lines)
    term_scores = matrix.sum(axis=0).A1  # total TF-IDF weight of each term
    ranked = sorted(zip(vectorizer.get_feature_names_out(), term_scores),
                    key=lambda x: x[1], reverse=True)
    top_jd_terms = [term for term, score in ranked[:top_n]]

    # --- Check each top keyword against the resume's tokens ---
    # A set gives fast "is this word in the resume?" lookups.
    resume_vocab = set(preprocess(resume_text))
    matched = [t for t in top_jd_terms if t in resume_vocab]
    missing = [t for t in top_jd_terms if t not in resume_vocab]
    return matched, missing


# ===========================================================================
# BONUS: SEMANTIC SIMILARITY (sentence embeddings)
# ===========================================================================
@st.cache_resource(show_spinner="Loading sentence-transformer model (first time only)...")
def load_embedding_model():
    """Load the model once and keep it in memory (it takes a few seconds)."""
    from sentence_transformers import SentenceTransformer  # heavy import, so only when needed
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(resume_text, jd_text):
    """
    Similarity based on MEANING rather than exact words.

    WHY: TF-IDF only matches identical words — it can't tell that
    "ML engineer" and "machine learning developer" mean the same thing.
    A sentence-embedding model maps text to a vector that captures meaning,
    so synonyms and paraphrases end up close together.

    Note: we pass the RAW text here — the model was trained on natural
    sentences, so stopwords and word order actually help it.
    """
    model = load_embedding_model()
    embeddings = model.encode([resume_text, jd_text])
    return similarity(embeddings[0:1], embeddings[1:2])


# ===========================================================================
# Helpers for the UI
# ===========================================================================
def read_uploaded_file(uploaded_file):
    """Extract plain text from an uploaded .txt or .pdf file."""
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def render_tags(words, color):
    """Show a list of words as coloured 'pill' tags using a bit of HTML."""
    if not words:
        st.write("_None_")
        return
    tags = " ".join(
        f"<span style='background:{color};color:white;padding:4px 10px;"
        f"border-radius:12px;margin:3px;display:inline-block;'>{w}</span>"
        for w in words
    )
    st.markdown(tags, unsafe_allow_html=True)


def text_input_column(label, key):
    """A text area plus an optional file uploader. Uploaded file wins if given."""
    st.subheader(label)
    uploaded = st.file_uploader(f"Upload {label} (.txt or .pdf)", type=["txt", "pdf"], key=f"{key}_file")
    pasted = st.text_area(f"...or paste the {label} here", height=300, key=f"{key}_text")
    return read_uploaded_file(uploaded) if uploaded else pasted


# ===========================================================================
# STREAMLIT UI
# ===========================================================================
def main():
    st.set_page_config(page_title="Resume–JD Matcher", page_icon="📄", layout="wide")
    st.title("📄 Resume – Job Description Matcher")
    st.caption("TF-IDF + cosine similarity, with an optional semantic (embedding) score")

    left, right = st.columns(2)
    with left:
        resume_text = text_input_column("Resume", "resume")
    with right:
        jd_text = text_input_column("Job Description", "jd")

    use_semantic = st.toggle(
        "Also compute semantic score (sentence-transformers, all-MiniLM-L6-v2)"
    )

    if not st.button("🔍 Analyze", type="primary"):
        return

    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please provide both a resume and a job description.")
        return

    # ---- Run the pipeline ----
    resume_tokens, resume_stages = preprocess(resume_text, return_stages=True)
    jd_tokens, jd_stages = preprocess(jd_text, return_stages=True)

    if not resume_tokens or not jd_tokens:
        st.warning("After preprocessing, one of the texts has no meaningful words left.")
        return

    resume_vec, jd_vec, vectorizer = vectorize(resume_tokens, jd_tokens)
    tfidf_score = similarity(resume_vec, jd_vec)
    matched, missing = keyword_gap(resume_text, jd_text)

    # ---- Scores ----
    st.divider()
    st.header("Results")
    score_cols = st.columns(2 if use_semantic else 1)
    with score_cols[0]:
        st.metric("Lexical match (TF-IDF)", f"{tfidf_score:.1f}%")
        st.progress(min(tfidf_score / 100, 1.0))

    if use_semantic:
        with score_cols[1]:
            try:
                sem_score = semantic_similarity(resume_text, jd_text)
                st.metric("Semantic match (embeddings)", f"{sem_score:.1f}%")
                st.progress(min(max(sem_score, 0) / 100, 1.0))
            except ImportError:
                st.error("sentence-transformers is not installed. Run: pip install sentence-transformers")
        st.info(
            "Lexical score counts shared exact words; semantic score compares meaning. "
            "Semantic scores are usually higher because related words (e.g. 'ML' vs "
            "'machine learning') still count as similar."
        )

    # ---- Keywords ----
    kw_left, kw_right = st.columns(2)
    with kw_left:
        st.subheader(f"✅ Matched keywords ({len(matched)})")
        render_tags(matched, "#2e7d32")
    with kw_right:
        st.subheader(f"❌ Missing keywords ({len(missing)})")
        render_tags(missing, "#c62828")

    # ---- How it works (for explaining the pipeline) ----
    with st.expander("🔬 How it works — see each pipeline stage"):
        st.markdown(
            "**Step 1 – Preprocessing.** Each document goes through the stages below. "
            "Watch how the text shrinks to only meaningful, normalised words."
        )
        # Tabs (not a radio button) so switching doesn't re-run the app
        # and wipe the results.
        for tab, stages in zip(st.tabs(["Resume", "Job Description"]), [resume_stages, jd_stages]):
            with tab:
                for stage_name, output in stages.items():
                    st.markdown(f"**{stage_name}**")
                    if isinstance(output, list):
                        st.caption(f"{len(output)} tokens")
                        st.code(", ".join(output[:150]) + (" ..." if len(output) > 150 else ""))
                    else:
                        st.code(output[:1500] + (" ..." if len(output) > 1500 else ""))

        st.markdown(
            f"**Step 2 – TF-IDF.** The vocabulary has **{len(vectorizer.get_feature_names_out())}** "
            "unique words. Each document becomes a vector of that length."
        )
        st.markdown(
            f"**Step 3 – Cosine similarity.** The angle between the two vectors gives "
            f"**{tfidf_score:.1f}%**."
        )
        st.markdown(
            f"**Step 4 – Keyword gap.** The JD is split into lines and "
            f"TF-IDF is computed across those lines to find its top {TOP_N_KEYWORDS} keywords. "
            "Each keyword is then looked up in the resume tokens: present → matched, absent → missing."
        )


if __name__ == "__main__":
    main()
