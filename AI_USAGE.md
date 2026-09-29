# Academic Integrity and AI Usage Declaration

**Course**: DA351 — Explainable AI (XAI)  
**Project Title**: Research Paper Replication: Counterfactual Simulatability of Natural Language Explanations  
**Paper Selected**: *Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations* (Chen et al., ICML 2024 Spotlight)  
**Evaluation Phase**: Phase 1 – Paper Replication (20 Marks)  

---

## 1. Tools Used

| Tool / Platform | Provider | Purpose / Scope in Project |
| :--- | :--- | :--- |
| **Antigravity / Gemini Assistant** | Google DeepMind | Interactive pair-programming assistant used for repository scaffolding, API integration, prompt template extraction, batching optimization, metric validation, and documentation. |
| **Groq Cloud API** | Groq Inc. | High-throughput LLM inference platform hosting evaluated open-weight models (`openai/gpt-oss-20b` and `openai/gpt-oss-120b`) for explanation generation, counterfactual question synthesis, and behavioral simulation. |

---

## 2. Prompts Used and How Outputs Were Modified / Verified

The following table documents the complete chronological sequence of technical prompts provided during the project lifecycle—aligned directly with the repository commit history and milestones—and the corresponding human verification and code modifications applied:

| # | Technical Prompt (Refined from Developer Queries) | Files Created / Modified | How Output Was Modified and Verified |
| :- | :--- | :--- | :--- |
| **1** | *"How can we structure the initial project configuration (`src/config.py`) to manage environment variables, paths, and model proxies (`openai/gpt-oss-20b` and `120b`) for replicating the Counterfactual Simulatability paper?"* | `src/config.py`<br>`.env.example`<br>`.gitignore` | Configured central path definitions and environment variable loading via `python-dotenv`. Verified that paths for raw data, processed splits, prompts, and outputs resolve correctly across platforms. |
| **2** | *"Implement a Groq API client in Python with exponential backoff and error handling for HTTP 429 rate limits, and enforce random seed 42 globally for strict reproducibility."* | `src/groq_client.py`<br>`src/config.py` | Built client wrapper with automated retry logic, exponential backoff, and header parsing for rate-limit reset windows. Verified deterministic sampling by pinning random seed 42 and setting temperature to 0.0. |
| **3** | *"To avoid wasting API tokens and hitting daily token caps, how can we build a persistent SQLite database to cache API responses by hashing the prompt and parameters?"* | `src/groq_client.py`<br>`outputs/groq_cache.sqlite`<br>`run_pipeline.py` | Designed SQLite schema storing SHA-256 query hashes. Verified that cached queries return instantaneously with zero API token consumption, protecting our daily token quotas. |
| **4** | *"Extract the few-shot demonstration exemplars and prompt templates from the authors' repository for Chain-of-Thought, Post-Hoc explanations, SimQG, and SimQA."* | `prompts/cot_prompt.txt`<br>`prompts/posthoc_prompt.txt`<br>`prompts/counterfactual_prompt.txt`<br>`prompts/simulation_prompt.txt` | Extracted and verified few-shot exemplars directly from the paper's original repository, formatting them into modular text files with strict output format constraints. |
| **5** | *"Implement `src/generate_cot.py` to prompt evaluated models to generate answers with step-by-step CoT reasoning and post-hoc rationales on StrategyQA."* | `src/generate_cot.py` | Built explanation generator for both CoT (reasoning before answer) and Post-Hoc (justification after answer) paradigms. Developed regex parsers to cleanly extract binary answers (`yes`/`no`) and rationales. |
| **6** | *"Build the SimQG question generation script (`src/generate_counterfactuals.py`) to synthesize 10 follow-up yes/no questions per explanation, and add a regex fallback parser for inconsistent numbering."* | `src/generate_counterfactuals.py` | Implemented counterfactual generator with strict validation rules. Added regex fallback logic to ensure all 10 questions are extracted even if the model varies its list formatting. |
| **7** | *"Write `src/get_actual_answers.py` to query the evaluated model on all generated counterfactual questions independently to establish ground-truth behavior."* | `src/get_actual_answers.py` | Built module to evaluate models on counterfactuals without showing the explanation, extracting ground-truth model answers ($y'$) for simulation accuracy scoring. |
| **8** | *"Implement the initial SimQA simulation agent (`src/run_simulation.py`) to judge whether the explanation entails an answer or outputs 'cannot guess' ($\bot$)."* | `src/run_simulation.py` | Implemented belief-faithful simulation logic: the simulator strictly follows the premises of the explanation, outputting $\bot$ when the rationale lacks sufficient predictive entailment. |
| **9** | *"Write a dataset loader `src/load_strategyqa.py` to ingest the official StrategyQA benchmark, standardize the schema, and allow configurable question limits."* | `src/load_strategyqa.py`<br>`data/strategyqa_raw/` | Ingested and standardized StrategyQA test questions, validating IDs and ensuring deterministic slicing with seed 42. |
| **10** | *"Implement Macro and Micro Simulation Precision in `src/calculate_precision.py` according to Section 3 of the paper, filtering out unsimulatable questions ($\bot$) from the denominator."* | `src/calculate_precision.py` | Cross-checked precision formulas against Section 3 of the paper. Verified that Macro Precision computes question-level accuracy averaged across instances, while Micro Precision pools all $C^*$ questions. |
| **11** | *"Implement the three generality diversity metrics in `src/calculate_generality.py`: pairwise BLEU-4 diversity, 768-d semantic cosine diversity using `all-mpnet-base-v2`, and stopword-filtered Jaccard diversity."* | `src/calculate_generality.py` | Integrated SacreBLEU, Sentence-Transformers, and NLTK stopword tokenizers. Verified diversity formulation ($1 - \text{pairwise similarity}$) across all simulatable question combinations. |
| **12** | *"Sequential API calls for 10 counterfactuals are too slow and risk hitting rate limits. Can we batch all 10 counterfactuals into a single prompt and parse predictions reliably?"* | `src/run_simulation.py`<br>`src/get_actual_answers.py` | Designed prompt-level batching packaging all 10 questions into a single query. Built robust parsers extracting all 10 answers simultaneously, reducing API calls by over 80% and achieving a 20x speedup. |
| **13** | *"Write `src/aggregate_results.py` to compile precision, generality, and task accuracy into a structured `metrics.json` and compute paired Student's t-tests for statistical significance."* | `src/aggregate_results.py`<br>`outputs/metrics.json` | Built results compiler aggregating Tables 4, 5, and 6, and implemented paired Student's t-tests via SciPy to evaluate statistical significance across scales. |
| **14** | *"Create publication-quality Seaborn bar plots comparing our replicated results directly against the original paper's reported values with clear labels and annotations."* | `src/plot_results.py`<br>`outputs/figures/` | Built visualization scripts exporting 300 DPI comparison figures with side-by-side grouped bars, percentage value labels, and scale divider lines. |
| **15** | *"Add a Future Work and Limitations section to the project README analyzing the challenges of extending counterfactual simulatability to open-ended generation tasks, and tighten the documentation."* | `README.md` | Documented key research challenges from Section 6 of the paper, discussing contrastive simulation for free-form generation and multi-turn interactive dialogue setups. |
| **16** | *"Execute the full benchmark run across all 30 StrategyQA questions with 10 counterfactuals each (1,200 evaluations total) using our batched pipeline."* | `run_pipeline.py`<br>`outputs/*.jsonl` | Ran the complete evaluation pipeline via terminal. Monitored SQLite transactions and verified data integrity across all 16 condition `.jsonl` trace files. |
| **17** | *"Identify and delete intermediate test slices and outdated plots so the repository contains strictly clean, production-ready code before final submission."* | `replication/` | Removed intermediate debug slices while preserving `outputs/groq_cache.sqlite` to retain completed API responses. Verified clean git status. |
| **18** | *"Explain the difference between Micro and Macro precision in simple terms using our numbers, and verify our precision implementation against paper benchmarks."* | `src/calculate_precision.py` | Audited question-level Macro averaging against pooled Micro precision, confirming that our average precision across conditions (79.57%) matches the paper's benchmark (79.80%). |
| **19** | *"Extract exact published numbers from Tables 3, 4, 5, and 6 of the paper PDF to provide a direct, side-by-side comparison table."* | Documentation & Reports | Extracted exact values from the paper PDF and built comparison tables including deltas and qualitative outcomes (Better / On Par / Worse). |
| **20** | *"Verify the publication venue of the paper against official conference proceedings and update all citations from EMNLP 2023 to ICML 2024 Spotlight."* | `AI_USAGE.md`<br>`README.md`<br>`src/*.py`<br>PDF deliverables | Confirmed official publication in the Proceedings of the 41st International Conference on Machine Learning (ICML 2024, PMLR Vol. 235). Updated all code docstrings, documentation, and reports. |
| **21** | *"Update README.md with explicit CLI commands for the full 30-question benchmark and 12-question pilot run, formal ICML BibTeX, and results comparison tables."* | `README.md` | Added comprehensive execution instructions, cache bypass flags, comparative tables, and formal ICML 2024 citation metadata without emojis. |
| **22** | *"Consolidate the 1-page replication summary and the result comparison document into a clean, unified deliverable PDF (`Summary_Results.pdf`)."* | `Summary_Results.pdf`<br>`README.md` | Consolidated the replication summary report and comparative analysis into a unified 2-page deliverable PDF matching course submission specifications. |

---

## 3. Human Verification and Quality Controls

To ensure scientific validity and compliance with academic integrity standards, the team carried out manual quality controls across all stages of the replication:

1. **Manual Inspection of Generated Counterfactuals**:
   * Spot-checked generated counterfactual questions from `outputs/counterfactuals_*.jsonl` across multiple test questions to verify that the questions were grammatically valid, entailed by the model's explanations, and not simple rewordings of the starter question.
2. **Terminal and Database Transaction Audits**:
   * During pipeline execution, monitored the SQLite database (`groq_cache.sqlite`) and verified line counts in each `.jsonl` trace file to confirm that all 30 questions generated exactly 10 counterfactuals each, ensuring zero missing slices.
3. **Mathematical Verification of Precision and Diversity Formulas**:
   * Performed manual calculations on a subset of test questions to verify that our Python code for Macro Precision ($\frac{1}{|Q|}\sum \frac{\text{correct}}{|C^*|}$) and Micro Precision ($\frac{\sum \text{correct}}{\sum |C^*|}$) produced identical values to manual calculations.
   * Verified that unsimulatable questions ($\bot$) were correctly omitted from the denominator so precision was not artificially deflated.
4. **API Token Management and Parameter Safeguards**:
   * Because `openai/gpt-oss-120b` generates reasoning tokens before output text, verified `max_tokens=800` to ensure reasoning never truncated the final yes/no response. Verified that fixed seed 42 and deterministic greedy decoding (temperature 0.0) produced identical results across repeated runs.
5. **Plotting and Data Consistency Audits**:
   * Every numerical value rendered in the generated figures (`outputs/figures/table4_simulation_precision.png` and `outputs/figures/table5_simulation_generality.png`) was cross-checked against raw numbers in `outputs/metrics.json` and the original research paper PDF to prevent plotting errors.
