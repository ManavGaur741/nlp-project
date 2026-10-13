"""
analysis_api.py
===============
Backend NLP Analysis API module for Resume–Job Description matching.

Exposes:
    analyze_resume(resume_text: str, job_description: str) -> dict

Expected contract keys:
    - overall_score (float/int 0-100)
    - tfidf_score (float/int 0-100)
    - semantic_score (float/int 0-100 or None)
    - skill_score (float/int 0-100)
    - matched_skills (list of str)
    - missing_skills (list of str)
    - suggestions (list of str)
"""

import re
import string
from typing import Dict, List, Tuple, Any, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK resources are available
for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.data.find(path + ".zip")
        except LookupError:
            nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

DOMAIN_STOP_WORDS = {
    "experience", "requirement", "responsibility", "role", "using", "work",
    "nice", "strong", "skill", "knowledge", "ability", "looking", "job",
    "title", "year", "including", "related", "field", "familiarity", "etc",
}
LEMMATIZER = WordNetLemmatizer()
TOP_N_KEYWORDS = 30

# Lazy-loaded sentence transformer model
_EMBEDDING_MODEL = None


def get_embedding_model():
    """Lazy-load sentence-transformers model to save memory until requested."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBEDDING_MODEL = False
    return _EMBEDDING_MODEL if _EMBEDDING_MODEL is not False else None


def preprocess(text: str, return_stages: bool = False) -> Any:
    """
    Clean raw text into meaningful normalized tokens.
    """
    stages = {}
    text_lower = text.lower()
    stages["1. Lowercased"] = text_lower

    text_no_punct = re.sub(f"[{re.escape(string.punctuation)}]", " ", text_lower)
    text_no_punct = re.sub(r"\d+", " ", text_no_punct)
    stages["2. Punctuation & numbers removed"] = text_no_punct

    tokens = word_tokenize(text_no_punct)
    stages["3. Tokenized"] = tokens

    tokens_no_stops = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    stages["4. Stopwords removed"] = tokens_no_stops

    lemmatized = [LEMMATIZER.lemmatize(t) for t in tokens_no_stops]
    stages["5. Lemmatized"] = lemmatized

    final_tokens = [t for t in lemmatized if t not in DOMAIN_STOP_WORDS]
    stages["6. Domain stopwords removed (final)"] = final_tokens

    if return_stages:
        return final_tokens, stages
    return final_tokens


def vectorize(resume_tokens: List[str], jd_tokens: List[str]) -> Tuple[Any, Any, TfidfVectorizer]:
    """Convert token lists into TF-IDF document vectors."""
    docs = [" ".join(resume_tokens), " ".join(jd_tokens)]
    vectorizer = TfidfVectorizer(lowercase=False)
    matrix = vectorizer.fit_transform(docs)
    return matrix[0], matrix[1], vectorizer


def similarity(vec1: Any, vec2: Any) -> float:
    """Compute cosine similarity as a percentage between 0 and 100."""
    score = cosine_similarity(vec1, vec2)[0][0]
    return round(float(score) * 100, 2)


def keyword_gap(resume_text: str, jd_text: str, top_n: int = TOP_N_KEYWORDS) -> Tuple[List[str], List[str]]:
    """
    Rank JD keywords by line-level TF-IDF and compare against resume tokens.
    """
    jd_lines = [" ".join(preprocess(line)) for line in jd_text.splitlines()]
    jd_lines = [line for line in jd_lines if line]
    if not jd_lines:
        return [], []

    vectorizer = TfidfVectorizer(lowercase=False, norm=None)
    matrix = vectorizer.fit_transform(jd_lines)
    term_scores = matrix.sum(axis=0).A1
    ranked = sorted(
        zip(vectorizer.get_feature_names_out(), term_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    top_jd_terms = [term for term, _ in ranked[:top_n]]

    resume_vocab = set(preprocess(resume_text))
    matched = [t for t in top_jd_terms if t in resume_vocab]
    missing = [t for t in top_jd_terms if t not in resume_vocab]
    return matched, missing


def semantic_similarity(resume_text: str, jd_text: str) -> Optional[float]:
    """Calculate semantic similarity via sentence embeddings."""
    model = get_embedding_model()
    if model is None:
        return None
    try:
        embeddings = model.encode([resume_text, jd_text])
        return similarity(embeddings[0:1], embeddings[1:2])
    except Exception:
        return None


def generate_suggestions(
    matched_skills: List[str],
    missing_skills: List[str],
    tfidf_score: float,
    semantic_score: Optional[float],
    skill_score: float,
) -> List[str]:
    """Generate prioritized, actionable resume improvement recommendations."""
    suggestions = []

    # Missing keywords recommendations
    if missing_skills:
        top_missing = missing_skills[:6]
        skills_str = ", ".join(f"'{s}'" for s in top_missing)
        suggestions.append(
            f"Target critical missing skills: Prioritize incorporating key role keywords such as {skills_str} into your technical skills and project descriptions."
        )

    # Lexical alignment feedback
    if tfidf_score < 30:
        suggestions.append(
            "Align vocabulary with the job description: Your lexical score is below 30%. Mirror specific industry terms and action verbs used in the job posting to improve ATS ranking."
        )
    elif tfidf_score < 50:
        suggestions.append(
            "Strengthen core terminology: Boost keyword density around core responsibilities to improve exact term matches."
        )

    # Semantic vs Lexical comparison
    if semantic_score is not None:
        if semantic_score > (tfidf_score + 15):
            suggestions.append(
                "Good conceptual alignment: Your resume conveys relevant concepts, but lacks the exact phrasing of the job posting. Rephrase existing project bullet points with the employer's specific vocabulary."
            )
        elif semantic_score < 40:
            suggestions.append(
                "Contextual depth: Expand on project scope and methodologies to demonstrate deeper relevance to the job domain."
            )

    # Project bullet points & quantification
    suggestions.append(
        "Quantify project achievements: Describe projects using the Google XYZ formula ('Accomplished [X] as measured by [Y] by doing [Z]') with concrete metrics (e.g., accuracy, latency, scale)."
    )

    # Skill section formatting
    if len(missing_skills) > 8:
        suggestions.append(
            "Structure dedicated skill categories: Organize your skills into clean buckets (e.g., 'Core NLP & ML', 'Frameworks & Libraries', 'DevOps & Cloud') to clearly highlight competencies."
        )

    return suggestions


def analyze_resume(
    resume_text: str,
    job_description: str,
    include_semantic: bool = True,
) -> Dict[str, Any]:
    """
    Backend contract:
        analyze_resume(resume_text: str, job_description: str) -> dict

    Expected keys:
        overall_score, tfidf_score, semantic_score, skill_score,
        matched_skills, missing_skills, suggestions.
    """
    if not resume_text.strip() or not job_description.strip():
        return {
            "overall_score": 0.0,
            "tfidf_score": 0.0,
            "semantic_score": None,
            "skill_score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "suggestions": ["Please provide both resume text and a job description to perform analysis."],
        }

    # Preprocessing
    resume_tokens = preprocess(resume_text)
    jd_tokens = preprocess(job_description)

    if not resume_tokens or not jd_tokens:
        return {
            "overall_score": 0.0,
            "tfidf_score": 0.0,
            "semantic_score": None,
            "skill_score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "suggestions": ["One of the documents contains insufficient recognizable words after text cleaning."],
        }

    # Lexical TF-IDF Cosine Similarity
    resume_vec, jd_vec, _ = vectorize(resume_tokens, jd_tokens)
    tfidf_score = similarity(resume_vec, jd_vec)

    # Skill matching gap
    matched_skills, missing_skills = keyword_gap(resume_text, job_description)
    total_skills = len(matched_skills) + len(missing_skills)
    skill_score = round((len(matched_skills) / total_skills * 100), 1) if total_skills > 0 else 0.0

    # Semantic embedding score
    semantic_score = None
    if include_semantic:
        semantic_score = semantic_similarity(resume_text, job_description)

    # Overall compatibility score
    if semantic_score is not None:
        # Weighted composite: 35% Lexical, 35% Semantic, 30% Skill Coverage
        overall_score = round(
            0.35 * tfidf_score + 0.35 * semantic_score + 0.30 * skill_score, 1
        )
    else:
        # Weighted composite without semantic: 50% Lexical, 50% Skill Coverage
        overall_score = round(0.50 * tfidf_score + 0.50 * skill_score, 1)

    overall_score = max(0.0, min(100.0, overall_score))

    # Actionable suggestions
    suggestions = generate_suggestions(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        tfidf_score=tfidf_score,
        semantic_score=semantic_score,
        skill_score=skill_score,
    )

    return {
        "overall_score": overall_score,
        "tfidf_score": tfidf_score,
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "suggestions": suggestions,
    }
