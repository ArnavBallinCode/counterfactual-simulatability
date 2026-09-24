"""
Simulation Generality (Diversity) Evaluation Module
Computes pairwise BLEU-4 diversity, Sentence-Transformers semantic cosine diversity,
and token-level Jaccard diversity across simulatable counterfactuals.
"""

import json
import re
import string
import numpy as np
import torch
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

_sentence_model = None
_en_stopwords = None

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        print('Loading sentence transformer: sentence-transformers/all-mpnet-base-v2...')
        _sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
    return _sentence_model

def get_stopwords():
    global _en_stopwords
    if _en_stopwords is None:
        _en_stopwords = set(stopwords.words('english'))
    return _en_stopwords

def calculate_pairwise_bleu(texts):
    if len(texts) <= 1:
        return np.nan
    bleu = BLEU(effective_order=True)
    scores = []
    for i in range(len(texts)):
        for j in range(len(texts)):
            if i != j:
                s = bleu.sentence_score(texts[i], [texts[j]]).score / 100.0
                scores.append(s)
    return float(np.mean(scores)) if scores else np.nan

def calculate_pairwise_cosine(texts):
    if len(texts) <= 1:
        return np.nan
    st_model = get_sentence_model()
    embeddings = torch.tensor(st_model.encode(texts, batch_size=64, show_progress_bar=False))
    sims = torch.nn.functional.cosine_similarity(embeddings[:, :, None], embeddings.t()[None, :, :])
    sims.fill_diagonal_(torch.nan)
    return float(torch.nanmean(sims).item())

def calculate_pairwise_jaccard(texts):
    if len(texts) <= 1:
        return np.nan
    stop_words = get_stopwords()

    def tokenize(t):
        cleaned = re.sub(' +', ' ', t.translate(str.maketrans('', '', string.punctuation)).strip().lower())
        return set(w for w in cleaned.split() if w not in stop_words and len(w) > 0)

    bows = [tokenize(t) for t in texts]
    jaccards = []
    for i in range(len(bows)):
        for j in range(len(bows)):
            if i != j:
                union_len = len(bows[i].union(bows[j]))
                if union_len > 0:
                    jaccards.append(len(bows[i].intersection(bows[j])) / union_len)
    return float(np.mean(jaccards)) if jaccards else np.nan

def compute_simulation_generality(sim_records):
    # Generality = 1 - average pairwise similarity over simulatable counterfactuals C*
    bleu_gens, cosine_gens, jaccard_gens = [], [], []

    for ex in sim_records:
        simulatable_qns = [
            item['question'] for item in ex['simulations']
            if item['sim_prediction'] in ['yes', 'no']
        ]

        if len(simulatable_qns) > 1:
            p_bleu = calculate_pairwise_bleu(simulatable_qns)
            p_cos = calculate_pairwise_cosine(simulatable_qns)
            p_jaccard = calculate_pairwise_jaccard(simulatable_qns)

            if not np.isnan(p_bleu):
                bleu_gens.append(1.0 - p_bleu)
            if not np.isnan(p_cos):
                cosine_gens.append(1.0 - p_cos)
            if not np.isnan(p_jaccard):
                jaccard_gens.append(1.0 - p_jaccard)

    return {
        'bleu_generality': round(float(np.mean(bleu_gens)), 3) if bleu_gens else 0.0,
        'cosine_generality': round(float(np.mean(cosine_gens)), 3) if cosine_gens else 0.0,
        'jaccard_generality': round(float(np.mean(jaccard_gens)), 3) if jaccard_gens else 0.0,
        'num_evaluable_explanations': len(bleu_gens)
    }
