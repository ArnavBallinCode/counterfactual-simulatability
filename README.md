# Paper Replication: Counterfactual Simulatability of Natural Language Explanations

**Course**: DA351 – Explainable AI (XAI)  
**Phase**: Phase 1 – Research Paper Replication (20 Marks)  
**Selected Paper**: *"Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations"* (Chen et al., EMNLP 2023)  
**Official Paper URL**: https://arxiv.org/abs/2307.08678  
**Reference Code**: https://github.com/yandachen/CounterfactualSimulatability  
**Evaluation Dataset**: [StrategyQA](https://allenai.org/data/strategyqa) (Implicit multi-hop reasoning)

---

## 1. Overview & Objective

This repository contains the complete, reproducible implementation for replicating the empirical findings of Chen et al. (2023). 

Large Language Models (LLMs) frequently output persuasive and fluent explanations for their decisions. However, **plausibility does not equal behavioral faithfulness**. This project implements **Counterfactual Simulatability**, an evaluation framework that measures whether an explanation enables an observer (human or proxy simulator) to accurately predict how the model will behave on diverse, related counterfactual questions.

### Key Hypotheses & Target Results Replicated:
1. **Simulation Precision (Table 4)**: Evaluates whether a larger, more capable model produces explanations with higher simulation precision than a smaller model under both Chain-of-Thought (CoT) and Post-Hoc reasoning.
2. **Simulation Generality (Table 5)**: Measures the diversity/breadth of the predictable counterfactual space using three orthogonal diversity metrics ($1 - \text{similarity}$):
   - **BLEU Diversity**: $1 - \text{pairwise BLEU}$
   - **Cosine Diversity**: $1 - \text{pairwise cosine similarity}$ via Sentence-Transformers (`all-mpnet-base-v2`)
   - **Jaccard Diversity**: $1 - \text{pairwise Jaccard similarity}$ over stopword-filtered word bags
3. **Task Accuracy vs. Precision (Table 6)**: Tests whether task accuracy directly translates to simulation precision.

---

## 2. Experimental Setup & Reproducibility

* **Evaluated Baseline Model**: `llama-3.1-8b-instant` (Open-source proxy for GPT-3.5)
* **Evaluated Flagship Model**: `llama-3.3-70b-versatile` (Open-source proxy for GPT-4)
* **Simulator Model**: `llama-3.3-70b-versatile` (Replicating paper Section 5.1 / Table 2 automated simulator proxy)
* **Dataset**: StrategyQA test questions extracted directly from the paper's experimental suite.
* **Random Seed**: Fixed to `42` across all sampling and data splits.
* **API Provider**: Free Groq Cloud API with local SQLite persistent caching to prevent redundant requests and respect rate limits.

---

## 3. Project Structure

```text
replication/
├── .env                      # API keys (GROQ_API_KEY)
├── .env.example              # Template for environment variables
├── .gitignore                # Git exclusions
├── requirements.txt          # Python dependencies
├── README.md                 # Complete documentation & execution guide
├── AI_USAGE.md               # Course compliance document for AI tools
├── replication_summary.md    # 1-page replication findings & comparison
├── run_pipeline.py           # End-to-end master execution pipeline
├── data/
│   ├── strategyqa_raw/       # StrategyQA test set questions
│   └── processed/            # Cleaned splits
├── prompts/
│   ├── cot_prompt.txt        # Few-shot Chain-of-Thought prompt
│   ├── posthoc_prompt.txt    # Few-shot Post-Hoc prompt
│   ├── counterfactual_prompt.txt # Few-shot counterfactual generator prompt
│   ├── simulation_prompt.txt # Few-shot simulator judgment prompt
│   └── prompts.json          # Complete JSON prompt suite from paper
├── src/
│   ├── config.py             # Model names, paths, seed, rate limit settings
│   ├── groq_client.py        # Cached Groq client with backoff and retry
│   ├── test_connection.py    # Connectivity and model verification script
│   ├── load_strategyqa.py    # StrategyQA data loader
│   ├── generate_cot.py       # CoT & Post-Hoc explanation generator
│   ├── generate_counterfactuals.py # Counterfactual question generator
│   ├── run_simulation.py     # Simulator execution
│   ├── get_actual_answers.py # Actual target answer evaluation
│   ├── calculate_precision.py# Table 4 Simulation Precision metric
│   ├── calculate_generality.py# Table 5 Generality metrics (BLEU, Cosine, Jaccard)
│   ├── aggregate_results.py  # Results compilation & statistical significance tests
│   └── plot_results.py       # Visualization generator (publication-ready figures)
└── outputs/
    ├── groq_cache.sqlite     # Persistent response cache
    ├── explanations.jsonl    # Raw generated explanations
    ├── counterfactuals.jsonl # Raw generated counterfactual questions
    ├── simulations.jsonl     # Simulator predictions
    ├── actual_answers.jsonl  # True model answers to counterfactuals
    ├── metrics.json          # Final aggregated metrics
    └── figures/              # Generated plots (.png)
```

---

## 4. Installation & Quickstart

### Step 1: Clone Repository & Install Dependencies
```bash
git clone <your-repo-url>
cd replication
pip install -r requirements.txt
```

### Step 2: Configure Environment
Copy `.env.example` to `.env` and insert your free Groq API key:
```bash
# In replication/.env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### Step 3: Test API Connectivity & Model Availability
```bash
python src/test_connection.py
```

### Step 4: Run the Complete Replication Pipeline
Execute the full pipeline across all models and explanation styles:
```bash
# Run full replication (30 questions x 10 counterfactuals x 4 settings = 1,200 counterfactual evaluations):
python run_pipeline.py --limit 30 --num_cfs 10

# Or run a quick 5-question pilot run:
python run_pipeline.py --limit 5 --num_cfs 10
```

---

## 5. Deliverables Checklist

- [x] Complete Python codebase implementing model, XAI counterfactual generation, simulator, and metrics.
- [x] Reproducibility: `requirements.txt`, random seed (`42`), dataset link, and run instructions.
- [x] Result comparison with original paper (`replication_summary.md` and `outputs/figures/`).
- [x] Academic Integrity declaration (`AI_USAGE.md`).

---

## 6. Future Work & Limitations

### Limitations of this Replication
- **Model proxies**: We used open-source Llama models via Groq as free-tier proxies for the GPT-3.5/GPT-4 models in the original paper. Behavioural differences between model families may explain small divergences in results.
- **Dataset scale**: We evaluated on a subset of StrategyQA due to API rate limits. The full paper used a larger evaluation sweep.
- **Simulator faithfulness**: The automated simulator is itself an LLM, introducing its own errors into the precision/generality estimates.

### Promising Future Directions
1. **Multi-dataset evaluation** – Extend beyond StrategyQA to CommonsenseQA, ARC, and OpenBookQA to test whether counterfactual simulatability generalises across reasoning types.
2. **Human-in-the-loop simulation** – Replace the LLM proxy simulator with crowd-sourced human annotators on a subset to validate automated simulator accuracy.
3. **Fine-tuning for faithfulness** – Investigate whether RLHF or DPO training with a simulatability reward signal can directly improve explanation faithfulness scores.
4. **Cross-lingual simulatability** – Test whether explanations in English remain simulatable when counterfactual questions are posed in other languages.
5. **Metric calibration** – Explore alternative diversity metrics (BERTScore, STS-B) for the generality computation and study their correlation with human judgements.
