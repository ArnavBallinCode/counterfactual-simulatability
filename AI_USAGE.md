# Academic Integrity and AI Usage Declaration

**Course**: DA351 — Explainable AI (XAI)  
**Project Title**: Research Paper Replication: Counterfactual Simulatability of Natural Language Explanations  
**Paper Selected**: *Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations* (Chen et al., ICML 2024 Spotlight)  
**Evaluation Phase**: Phase 1 – Paper Replication (20 Marks)  

---

## 1. Tools Used

| Tool / Platform | Provider | Purpose / Scope in Project |
| :--- | :--- | :--- |
| **Antigravity / Gemini Assistant** | Google DeepMind | Pair-programming assistant used for code development, debugging rate limits, designing batching optimization, implementing metric equations, and validating outputs. |
| **Groq Cloud API** | Groq Inc. | High-throughput LLM inference platform hosting evaluated open-weight models (`openai/gpt-oss-20b` and `openai/gpt-oss-120b`) for explanation generation, counterfactual question synthesis, and behavioral simulation. |

---

## 2. Prompts Used and How Outputs Were Modified / Verified

The following table records the conversational prompts provided during project implementation, broken down into two-step conversational query pairs matching our natural interaction workflow, along with the specific files created/modified and verification steps applied:

| # | Conversational User Prompt | Files Created / Modified | How Output Was Modified and Verified |
| :- | :--- | :--- | :--- |
| **1** | *"how should we structure the project folders and config for this replication paper?"* | `src/config.py`<br>`.env.example` | Established clean repository modularity dividing data, prompts, source logic, and outputs. Verified path variables dynamically resolve across platforms. |
| **2** | *"make config.py with central paths and .env loading, and add a .gitignore"* | `src/config.py`<br>`.gitignore` | Configured `python-dotenv` environment variable loading and created `.gitignore` to prevent API keys and raw data caches from being committed. |
| **3** | *"how do we connect to groq? write a client with retries and exponential backoff for rate limits"* | `src/groq_client.py` | Built API wrapper with exponential backoff, retry loops, and rate limit header parsing. Audited error-handling logic for HTTP 429 status codes. |
| **4** | *"also set seed 42 globally and temp 0 so all outputs are deterministic"* | `src/config.py`<br>`src/groq_client.py` | Hardcoded random seed 42 across Python's `random`, NumPy, and Groq inference parameters with `temperature=0.0` for full scientific reproducibility. |
| **5** | *"wait groq api will hit rate limits if we re-run prompts, can we add an sqlite cache?"* | `src/groq_client.py`<br>`outputs/groq_cache.sqlite` | Implemented local SQLite persistence. Verified that cached queries return in <1ms without consuming any network tokens or triggering rate limits. |
| **6** | *"make sure the cache checks sha256 hashes of queries so cached prompts return instantly"* | `src/groq_client.py` | Designed SHA-256 query hashing for prompt-response pairs. Verified hash collision resistance and instant local lookups across repeated runs. |
| **7** | *"wire up run_pipeline.py so we can run the whole pipeline from terminal with a --limit argument"* | `run_pipeline.py` | Built master CLI coordinator with `argparse` flags (`--limit`, `--num_cfs`, `--no_cache`) to allow running test subsets before full execution. |
| **8** | *"write load_strategyqa.py to load strategyqa test questions and save them into data/processed/"* | `src/load_strategyqa.py`<br>`data/strategyqa_raw/` | Implemented data parser to extract benchmark questions, validate JSON schemas, and output clean evaluation subsets with unique IDs. |
| **9** | *"extract the cot and post-hoc prompt templates with few-shot examples from the paper repo"* | `prompts/cot_prompt.txt`<br>`prompts/posthoc_prompt.txt` | Extracted the authors' 7-shot reasoning exemplars, verifying that CoT prompts end with `"So the answer is yes/no."` and post-hoc prompts seek justifications. |
| **10** | *"now extract the simqg counterfactual generation and simqa simulation prompts from the paper"* | `prompts/counterfactual_prompt.txt`<br>`prompts/simulation_prompt.txt` | Formatted prompt instructions ensuring SimQG generates 10 yes/no counterfactuals and SimQA strictly grounds simulations on the model's explanations. |
| **11** | *"implement generate_cot.py to get model answers and explanations for cot and post-hoc"* | `src/generate_cot.py` | Programmed batch generation across all evaluation questions. Verified that both `gpt-oss-20b` and `gpt-oss-120b` generate complete rationales. |
| **12** | *"make sure regex parsers correctly separate the explanation text from the final yes/no answer"* | `src/generate_cot.py` | Added regex pattern matching to cleanly isolate the final predicted label (`yes`/`no`) from the preceding chain-of-thought or post-hoc explanation. |
| **13** | *"write generate_counterfactuals.py to prompt the model to generate 10 follow-up yes/no questions for each explanation"* | `src/generate_counterfactuals.py` | Implemented counterfactual question generator (`SimQG`). Passed original question, predicted answer, and explanation as context. |
| **14** | *"some generated questions have weird formatting or missing numbers, add regex fallback to catch all 10 questions"* | `src/generate_counterfactuals.py` | Added fallback regex parser that extracts questions even when models omit numerical indices or include conversational preamble. |
| **15** | *"now write get_actual_answers.py to ask the model the 10 counterfactual questions directly without seeing any explanation"* | `src/get_actual_answers.py` | Built module querying the target model on counterfactual questions in isolation to establish ground-truth behavior labels ($y'$). |
| **16** | *"implement run_simulation.py where the simulator guesses the model's answer using only explanation, or outputs 'cannot guess'"* | `src/run_simulation.py` | Programmed simulation module ($\text{SimQA}$) with the unsimulatable filter ($\bot$), ensuring uncertain predictions are marked as unsimulatable. |
| **17** | *"how is simulation precision calculated in section 3 of the paper? write calculate_precision.py"* | `src/calculate_precision.py` | Implemented Equation 1 from the paper. Verified that unsimulatable counterfactuals ($\bot$) are excluded from the denominator. |
| **18** | *"what is the difference between macro and micro precision here? implement both so we track prompt-level and overall accuracy"* | `src/calculate_precision.py` | Programmed Macro Precision (averaging question-level accuracies) and Micro Precision (pooling all simulatable counterfactuals across the dataset). |
| **19** | *"implement the generality metric using pairwise bleu-4 diversity across counterfactual questions"* | `src/calculate_generality.py` | Integrated SacreBLEU to compute sentence-level BLEU-4 between all pairs of counterfactuals per question, taking $1 - \text{BLEU}$ as diversity. |
| **20** | *"extend generality to also compute sentence-transformers cosine diversity with all-mpnet-base-v2 and jaccard token diversity"* | `src/calculate_generality.py` | Implemented embedding cosine diversity ($1 - \text{cosine similarity}$) and NLTK stopword-filtered token Jaccard diversity ($1 - \text{Jaccard}$). |
| **21** | *"the simulation is running too slow calling the api 10 times per question, cant u make it faster using any techniques"* | `src/run_simulation.py`<br>`src/get_actual_answers.py` | Diagnosed serial network latency bottleneck. Proposed and developed prompt-level batching to query all 10 counterfactuals in a single API call. |
| **22** | *"implement prompt batching in run_simulation.py and get_actual_answers.py so all 10 counterfactuals are evaluated in a single api call"* | `src/run_simulation.py`<br>`src/get_actual_answers.py` | Deployed prompt batching with structured numbered output parsing, reducing network round-trips by >80% and speeding up runs from 4 hours to 40 minutes. |
| **23** | *"write aggregate_results.py to compile precision, generality, and task accuracy into metrics.json"* | `src/aggregate_results.py`<br>`outputs/metrics.json` | Built summary aggregator compiling precision, generality, and task accuracy across all 4 experimental conditions into a unified JSON file. |
| **24** | *"add paired t-test in aggregate_results.py to test if the precision difference between 120b and 20b is statistically significant"* | `src/aggregate_results.py` | Added SciPy `ttest_rel` comparing question-level precision between models, computing t-statistics and p-values to evaluate statistical significance. |
| **25** | *"generate a bar plot for table 4 simulation precision comparing our replicated 20b and 120b models against gpt-3.5 and gpt-4 from the paper"* | `src/plot_results.py`<br>`outputs/figures/table4_simulation_precision.png` | Created side-by-side grouped bar plot comparing original paper figures (GPT-3.5, GPT-4) with replicated results across CoT and Post-Hoc conditions. |
| **26** | *"improve the table 4 plot: add percentage labels on top of each bar, separate small vs large models, and use seaborn with 300 dpi"* | `src/plot_results.py`<br>`outputs/figures/table4_simulation_precision.png` | Enhanced plot aesthetics using Seaborn styling, added bold numerical percentages above bars, inserted model category dividing line, and exported 300 DPI raster. |
| **27** | *"now generate the table 5 generality plot showing bleu, cosine, and jaccard diversity across all 4 experimental conditions"* | `src/plot_results.py`<br>`outputs/figures/table5_simulation_generality.png` | Generated grouped bar visualization of counterfactual diversity across BLEU, Cosine, and Jaccard metrics for all 4 experimental conditions. |
| **28** | *"make sure table 5 plot uses clean grouped bars with distinct palettes and high resolution for the report"* | `src/plot_results.py`<br>`outputs/figures/table5_simulation_generality.png` | Adjusted color palette, legend placement, and axis limits to ensure visual clarity when embedded in the replication report. |
| **29** | *"give command i will run in my terminal so i get to see everything and u check whether that work is being done perfectly"* | `run_pipeline.py`<br>`outputs/groq_cache.sqlite` | Formulated CLI command `python run_pipeline.py --limit 30 --num_cfs 10`. Monitored terminal stdout and SQLite cache inserts throughout execution. |
| **30** | *"before running it delete all unnecessary files like which were generated due to previous runs and old figures, keep only the cache"* | `replication/` workspace | Cleaned out intermediate `.jsonl` trace files, old figure exports, and temporary splits while safely preserving `groq_cache.sqlite`. |
| **31** | *"have we run all experiments needed? check and confirm whether all 30 questions and output files are complete"* | `outputs/*.jsonl`<br>`outputs/metrics.json` | Audited line counts across all 16 condition trace files, verifying that all 30 questions and 1,200 counterfactual evaluations completed successfully. |
| **32** | *"is table 5 and 6 doesn't have paper results because u haven't put in table? extract exact numbers from the paper pdf"* | `LLM Self-Explanations Research Paper.pdf` | Extracted exact values from Tables 3, 4, 5, and 6 directly from the original paper PDF to populate benchmark comparisons. |
| **33** | *"create side-by-side comparison tables for table 4, 5, and 6 showing paper results, our replication, delta, and whether we are better or on par"* | Project documentation & deliverables | Built markdown comparison tables showing original paper metrics, replicated values, percentage point deltas, and qualitative parity assessments. |
| **34** | *"the original paper is actually icml 2024 spotlight, not emnlp. update the venue citation across the codebase, readme, and reports"* | `AI_USAGE.md`<br>`README.md`<br>`src/*.py`<br>Deliverables | Verified official publication venue in PMLR Vol. 235 (ICML 2024 Spotlight). Updated venue citations across all docstrings, README, and reports. |
| **35** | *"now update readme include commadns also form whcih we ran copnterfa ual 30 12 one and icml citation too no emojis too"* | `README.md` | Added execution commands for 30-question full replication and 12-question subset, detailed result comparison tables, and formal ICML BibTeX citation. |
| **36** | *"consolidate the 1-page replication summary and result comparison into a clean unified pdf deliverable"* | `Summary_Results.pdf`<br>`README.md` | Combined the 1-page replication summary and result comparison into a 2-page deliverable PDF (`Summary_Results.pdf`) and updated repository links. |

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
