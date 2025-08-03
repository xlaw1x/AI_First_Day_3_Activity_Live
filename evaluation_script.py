import os, json, re, difflib
from collections import Counter
from typing import List, Tuple

import pandas as pd
import numpy as np
import spacy
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz

# Load NLP and embedding models
nlp = spacy.load('en_core_web_sm')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS

# TF-IDF vectorizer (fitted per speaker turn)
vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents='unicode')

def collapse_contractions(text: str) -> str:
    # Collapse spaced contractions like "ma ' am" to "ma'am"
    text = re.sub(r"\b([a-zA-Z]{1,3})\s*['`\u2019\u2018\u201B\u2032\u2035\uFF07\u275B\u275C\u02BC]+\s*([a-zA-Z]{1,3})\b", r"\1'\2", text)
    # Normalize dash-like characters (e.g., em-dash, en-dash) to a space
    text = re.sub(r"[\u2012-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", text)  # replace dash variants with hyphen
    text = re.sub(r"\s*-\s*", " ", text)  # collapse hyphen to space for short phrases
    return text

def clean_text_for_semantics(text: str) -> str:
    text = unidecode(text)
    text = normalise(text)
    text = collapse_contractions(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_for_tfidf(text: str) -> str:
    text = unidecode(text)
    text = normalise(text)
    text = collapse_contractions(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalise(text: str) -> str:
    text = re.sub(r'^[QA]:\s+', '', text, flags=re.M)
    text = re.sub(r'\d{1,2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d+', '<NUM>', text)
    return text

def tokens(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r'[A-Za-z0-9<>]+', text)]

def length_ratio(src: str, out: str) -> float:
    """Calculate length ratio between output and source"""
    src_tokens = tokens(src)
    out_tokens = tokens(out)
    if len(src_tokens) == 0:
        return 1.0 if len(out_tokens) == 0 else float('inf')
    return len(out_tokens) / len(src_tokens)

def unique_token_recall(src: str, out: str) -> float:
    """Calculate unique token recall score"""
    src_tokens = set(tokens(src))
    out_tokens = set(tokens(out))
    if len(src_tokens) == 0:
        return 1.0
    return len(src_tokens & out_tokens) / len(src_tokens)

def jaccard_nonstop(src: str, out: str) -> float:
    """Calculate Jaccard similarity excluding stop words"""
    src_tokens = set(t for t in tokens(src) if t not in STOP_WORDS)
    out_tokens = set(t for t in tokens(out) if t not in STOP_WORDS)
    if len(src_tokens) == 0 and len(out_tokens) == 0:
        return 1.0
    if len(src_tokens) == 0 or len(out_tokens) == 0:
        return 0.0
    return len(src_tokens & out_tokens) / len(src_tokens | out_tokens)

def get_entity_recall_score(src: str, out: str) -> float:
    """Calculate entity recall score"""
    s_ents = set(entities(src))
    if len(s_ents) == 0:
        return 1.0
    o_ents = set(entities(out))
    return len(s_ents & o_ents) / len(s_ents)

def get_long_detection_score(src: str, out: str, run: int = 4) -> float:
    """Calculate long deletion detection score (0 if long deletion detected, 1 otherwise)"""
    sm = difflib.SequenceMatcher(None, tokens(src), tokens(out))
    run_len = 0
    for op, i1,i2,j1,j2 in sm.get_opcodes():
        if op == 'delete':
            run_len = max(run_len, i2 - i1)
    return 0.0 if run_len >= run else 1.0

def semantic_alignment(src: str, out: str) -> float:
    src_clean = clean_text_for_semantics(src)
    out_clean = clean_text_for_semantics(out)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', src_clean) if s.strip()]
    if not sentences:
        return 1.0
    emb_src = embedder.encode(sentences, normalize_embeddings=True)
    emb_out = embedder.encode([out_clean], normalize_embeddings=True)
    sims = cosine_similarity(emb_src, emb_out).max(axis=1)
    return sims.mean()

def fit_corpus(corpus: List[str]):
    cleaned = [clean_text_for_tfidf(text) for text in corpus]
    filtered = [text for text in cleaned if len(tokens(text)) > 2]
    if filtered:
        vectoriser.fit(filtered)
    else:
        vectoriser.fit(["dummy fallback text"])

def tfidf_top_k(text: str, k: int = 30) -> List[str]:
    text = clean_text_for_tfidf(text)
    vec = vectoriser.transform([text]).toarray()[0]
    vocab = vectoriser.get_feature_names_out()
    idx = vec.argsort()[-k:]
    return [vocab[i] for i in idx if vec[i] > 0]

def get_tfidf_coverage_score(src: str, out: str, k: int = 30) -> float:
    src_n = clean_text_for_tfidf(src)
    out_n = clean_text_for_tfidf(out)
    out_text = ' '.join(tokens(out_n))
    must_keep = set(tfidf_top_k(src, k))
    if not must_keep:
        return 1.0
    present = sum(1 for tok in must_keep if tok in out_text or fuzz.partial_ratio(tok, out_text) > 90)
    return present / len(must_keep)

def tfidf_coverage(src: str, out: str, k: int = 30, thresh: float = 0.7) -> bool:
    return get_tfidf_coverage_score(src, out, k) >= thresh

def entities(text: str) -> List[Tuple[str,str]]:
    doc = nlp(text)
    return [(unidecode(e.text.strip()), e.label_) for e in doc.ents if e.label_ in {'PERSON','ORG','GPE','DATE','LAW'}]

def entity_recall(src: str, out: str, thresh: float = 0.95) -> bool:
    s_ents = set(entities(src))
    if len(s_ents) < 3: return True
    o_ents = set(entities(out))
    return len(s_ents & o_ents) / len(s_ents) >= thresh

def spans_deleted(src: str, out: str, run: int = 4) -> bool:
    sm = difflib.SequenceMatcher(None, tokens(src), tokens(out))
    run_len = 0
    for op, i1,i2,j1,j2 in sm.get_opcodes():
        if op == 'delete':
            run_len = max(run_len, i2 - i1)
    return run_len >= run

def has_unwanted_edits(src: str, out: str,
                       len_low=0.85, len_high=1.15,
                       uniq_thresh=0.9, jac_thresh=0.8):
    src_n = normalise(src)
    out_n = normalise(out)

    if src.strip() == out.strip():
        return False, None
    if not (len_low <= length_ratio(src_n, out_n) <= len_high):
        return True, 'length_ratio'
    if unique_token_recall(src_n, out_n) < uniq_thresh:
        return True, 'unique_token_recall'
    if jaccard_nonstop(src_n, out_n) < jac_thresh:
        return True, 'jaccard'
    if not tfidf_coverage(src_n, out_n):
        return True, 'tfidf_coverage'
    if not entity_recall(src, out):
        return True, 'entity_recall'
    if semantic_alignment(src_n, out_n) < 0.8:
        return True, 'semantic_alignment'
    if spans_deleted(src_n, out_n):
        return True, 'long_deletion'

    return False, None

def compare_and_fuzz(df_row, threshold=75):
    asr_df = df_row["speaker_turns_df"]
    llm_df = df_row["llm_turns_df"]

    asr_speakers = asr_df["speaker"].astype(str).str.upper().tolist()
    llm_speakers = llm_df["speaker"].str.replace("Speaker ", "", case=False).str.upper().tolist()

    if asr_speakers != llm_speakers:
        print(f"[SKIP] Job {df_row['job_id']}: Speaker sequences do not match.")
        return None

    merged = pd.DataFrame({
        "speaker": asr_speakers,
        "asr_text": asr_df["text"],
        "llm_text": llm_df["text"]
    })

    merged["fuzzy_ratio"] = merged.apply(
        lambda row: fuzz.ratio(row["asr_text"], row["llm_text"]), axis=1
    )
    merged["token_sort_ratio"] = merged.apply(
        lambda row: fuzz.token_sort_ratio(row["asr_text"], row["llm_text"]), axis=1
    )
    merged["below_threshold"] = merged["token_sort_ratio"] < threshold

    tfidf_scores = []
    semantic_scores = []
    unique_token_recall_scores = []
    jaccard_scores = []
    length_ratio_scores = []
    entity_recall_scores = []
    long_detection_scores = []
    reasons = []

    for _, row in merged.iterrows():
        SRC = row["asr_text"]
        OUT_BAD = row["llm_text"]
        fit_corpus([SRC])
        
        # Calculate all scores
        tfidf_score = get_tfidf_coverage_score(SRC, OUT_BAD)
        semantic_score = semantic_alignment(SRC, OUT_BAD)
        unique_token_recall_score = unique_token_recall(normalise(SRC), normalise(OUT_BAD))
        jaccard_score = jaccard_nonstop(normalise(SRC), normalise(OUT_BAD))
        length_ratio_score = length_ratio(normalise(SRC), normalise(OUT_BAD))
        entity_recall_score = get_entity_recall_score(SRC, OUT_BAD)
        long_detection_score = get_long_detection_score(normalise(SRC), normalise(OUT_BAD))
        
        flagged, reason = has_unwanted_edits(SRC, OUT_BAD)
        
        tfidf_scores.append(tfidf_score)
        semantic_scores.append(semantic_score)
        unique_token_recall_scores.append(unique_token_recall_score)
        jaccard_scores.append(jaccard_score)
        length_ratio_scores.append(length_ratio_score)
        entity_recall_scores.append(entity_recall_score)
        long_detection_scores.append(long_detection_score)
        reasons.append((flagged, reason))

    merged["has_unwanted_edit"] = [f for f, _ in reasons]
    merged["edit_reason"] = [r for _, r in reasons]
    merged["tfidf_score"] = tfidf_scores
    merged["semantic_score"] = semantic_scores
    merged["unique_token_recall_score"] = unique_token_recall_scores
    merged["jaccard_score"] = jaccard_scores
    merged["length_ratio_score"] = length_ratio_scores
    merged["entity_recall_score"] = entity_recall_scores
    merged["long_detection_score"] = long_detection_scores

    print(f"[INFO] Job {df_row['job_id']}: "
          f"{merged['below_threshold'].sum()}/{len(merged)} below fuzz threshold; "
          f"{merged['has_unwanted_edit'].sum()} unwanted edits.")

    return merged

# Main execution
df = load_all_jobs()
df = df.dropna(subset=["asr_text", "llm_corrections"]).reset_index(drop=True)

df["speaker_turns_df"] = df["asr_text"].apply(lambda x: extract_speaker_turns(x) if x else pd.DataFrame())
df["llm_turns_df"] = df["llm_corrections"].apply(lambda x: extract_llm_turns(x) if x else pd.DataFrame())

results = []
for idx, row in df.iterrows():
    result = compare_and_fuzz(row)
    if result is not None:
        results.append({
            "job_id": row["job_id"],
            "fuzz_results": result
        })

results_df = pd.DataFrame(results)