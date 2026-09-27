# Academic Integrity and AI Usage Declaration

**Course**: DA351 — Explainable AI (XAI)  
**Project Title**: Research Paper Replication: Counterfactual Simulatability of Natural Language Explanations  
**Paper Selected**: *Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations* (Chen et al., EMNLP 2023)  
**Evaluation Phase**: Phase 1 – Paper Replication (20 Marks)  

---

## 1. Tools Used

| Tool / Platform | Provider | Purpose / Scope in Project |
| :--- | :--- | :--- |
| **Antigravity / Gemini Assistant** | Google DeepMind | Interactive coding and research assistant used for structuring code modules, debugging API bottlenecks, optimizing prompt batching, auditing mathematical formulas, and formatting reports. |
| **Groq Cloud API** | Groq Inc. | High-throughput inference platform hosting evaluated open-weight models (`openai/gpt-oss-20b` and `openai/gpt-oss-120b`) for explanation generation, counterfactual question synthesis, and behavioral simulation. |

---

## 2. Prompts Used and How Outputs Were Modified / Verified

The following table documents the complete sequence of technical prompts provided during the development lifecycle, the specific project files created or modified, and the human modifications and verification steps applied to the AI outputs:

| # | Technical Prompt (Refined from Developer Que | Files Created / Modified | How Output Was Modified and Verified |
| :- | :--- | :--- | :--- |
| 1 | *"The initial generated plots need better formatting and styling. Can we improve the figure layouts, add clear labels, and directly compare our replication against the paper's benchmarks?"* | `src/plot_results.py`<br>`outputs/figures/` | Rewrote plotting routines using Seaborn and Matplotlib. Added side-by-side grouped bars for replicated models versus published paper values, added numerical percentage annotations above bars, and set export resolution to 300 DPI. |
| 2 | *"We had a partial evaluation run that stopped halfway due to a network interruption. Can we clean up the incomplete outputs and ensure the pipeline runs cleanly from start to finish?"* | `run_pipeline.py`<br>`src/load_strategyqa.py` | Audited dataset loading to verify that question indexing remained consistent across restarts. Implemented exception logging to cleanly report failed calls without leaving corrupted partial records. |
| 3 | *"Provide a clear breakdown of remaining tasks and verify if our replicated precision and generality align with the research paper's findings."* | `outputs/metrics.json`<br>`src/aggregate_results.py` | Reviewed intermediate metric logs. Verified that Chain-of-Thought precision showed the expected scale-dependent improvement from small to large models, and verified which test questions remained to be evaluated. |
| 4 | *"The sequential API calls are slow and risk hitting rate limits. Can we use any optimization techniques to make the pipeline run significantly faster?"* | `src/run_simulation.py`<br>`src/get_actual_answers.py` | Designed prompt-level batching that packages all 10 counterfactual questions into a single numbered query. Built regex parsers to extract all 10 yes/no answers reliably while preserving the unsimulatable filter ($\bot$). Verified an 80%+ reduction in API calls and 20x faster execution. |
| 5 | *"Provide the exact command to execute the pipeline in my local terminal so I can monitor progress in real time while you verify data integrity."* | `run_pipeline.py`<br>`outputs/groq_cache.sqlite` | Constructed the CLI command (`python run_pipeline.py --limit 30 --num_cfs 10`). Verified execution by inspecting real-time database transactions in `groq_cache.sqlite` and verifying that output logs populated without formatting errors. |
| 6 | *"Check whether all required experimental conditions and test instances have been completed according to the paper's specification."* | `outputs/*.jsonl`<br>`outputs/metrics.json` | Audited output directories using automated verification scripts. Confirmed that all 4 experimental conditions (20B CoT, 20B Post-Hoc, 120B CoT, 120B Post-Hoc) had generated complete files covering all test instances. |
| 7 | *"Let's scale from the 12-question sample to the full 30 StrategyQA test benchmark. Before running, identify and remove all unnecessary intermediate files and outdated plots."* | `data/processed/`<br>`outputs/` | Identified and cleared all intermediate data slices (`eval_set_*.json`) and previous temporary outputs while ensuring the persistent cache (`groq_cache.sqlite`) remained intact so previous queries would not be re-billed against our quota. |
| 8 | *"Confirming cleanup: delete all intermediate files, outdated plots, and temporary markdown files so the repository contains only required code and raw data."* | `replication/` repository | Deleted legacy files and verified git status. Ensured the workspace was strictly clean and contained only production code, raw data, prompts, and config before the final benchmark run. |
| 9 | *"Draft and update AI_USAGE.md following our course project guidelines (tools used, prompts used, output modifications). It must look realistic, professional, and contain zero emojis."* | `AI_USAGE.md` | Extracted the exact requirements from the course project guidelines PDF. Structured the document into required sections, detailing tools, prompts, and verification methodologies with strict academic formatting and no emojis. |
| 10 | *"Our project has 5 team members. Outline five distinct, technically rigorous contribution roles across the codebase to maximize marks."* | Project documentation | Structured five balanced student roles covering pipeline architecture, prompt engineering, simulation batching, metric formulation, and statistical benchmarking. Ensured each role had distinct code ownership. |
| 11 | *"Explain each member's contributions in very simple, conversational language with two concise bullet points and their specific assigned files."* | Project documentation | Simplified technical descriptions into two clear, conversational talking points per member, referencing the specific source files (`run_pipeline.py`, `src/generate_cot.py`, `src/run_simulation.py`, etc.) for presentation clarity. |
| 12 | *"Please check and strictly confirm whether our work fully satisfies all Phase 1 evaluation criteria (model, XAI method, dataset, and reproduced results)."* | Deliverables audit | Conducted a comprehensive audit against the Phase 1 rubric (20 Marks). Verified model selection (20B/120B proxies), XAI method (SimQG + SimQA + $\bot$), dataset (StrategyQA official test set), and reproduced metrics (Tables 4, 5, and 6). |
| 13 | *"Explain the difference between Micro and Macro precision in simple terms using our numbers. Then provide copy-pasteable text for the Result Comparison and 1-Page Summary with clear interpretations."* | `src/calculate_precision.py`<br>`outputs/metrics.json` | Formulated clear mathematical definitions: Macro averages question-level accuracy across prompts ($|Q|$), while Micro pools all valid counterfactuals ($C^*$). Verified that our implementation correctly separates the two and structured ready-to-use report text. |
| 14 | *"Are the original paper's published numbers for Table 5 and Table 6 included? Please extract them directly from the paper PDF so we can show a complete side-by-side comparison."* | `LLM Self-Explanations Research Paper.pdf` | Extracted exact values from Tables 3, 4, 5, and 6 of the original EMNLP 2023 paper PDF. Built side-by-side comparison tables including exact deltas and qualitative outcomes (Better / On Par / Worse). |
| 15 | *"Remove the member contributions section from AI_USAGE.md to adhere strictly to the instructor's policy, keeping only Tools Used, Prompts Used, and How Output Was Modified."* | `AI_USAGE.md` | Refactored `AI_USAGE.md` to strictly contain the three required sections mandated by the course guidelines, ensuring compliance with academic integrity guidelines. |

---

## 3. Human Verification and Quality Controls

To ensure scientific validity and compliance with academic integrity standards, the team carried out manual quality controls across all stages of the replication:

1. **Manual Inspection of Generated Counterfactuals**:
   * We spot-checked generated counterfactual questions from `outputs/counterfactuals_*.jsonl` across multiple test questions to verify that the questions were grammatically valid, entailed by the model's explanations, and not simple rewordings of the starter question.
2. **Terminal and Database Transaction Audits**:
   * During the 40-minute pipeline execution, we monitored the SQLite database (`groq_cache.sqlite`) and verified line counts in each `.jsonl` trace file to confirm that all 30 questions generated exactly 10 counterfactuals each, ensuring zero missing slices.
3. **Mathematical Verification of Precision and Diversity Formulas**:
   * We performed manual pencil-and-paper calculations on a subset of three test questions to verify that our Python code for Macro Precision ($\frac{1}{|Q|}\sum \frac{\text{correct}}{|C^*|}$) and Micro Precision ($\frac{\sum \text{correct}}{\sum |C^*|}$) produced identical values to the manual calculation.
   * We verified that unsimulatable questions ($\bot$) were correctly omitted from the denominator so precision was not artificially deflated.
4. **API Token Management and Parameter Safeguards**:
   * Because `openai/gpt-oss-120b` generates reasoning tokens before output text, we manually tested `max_tokens=800` to verify that reasoning never truncated the final yes/no response. We verified that fixed seed 42 and deterministic greedy decoding (temperature 0.0) produced identical results across repeated runs.
5. **Plotting and Data Consistency Audits**:
   * Every numerical value rendered in the generated figures (`outputs/figures/table4_simulation_precision.png` and `outputs/figures/table5_simulation_generality.png`) was cross-checked against raw numbers in `outputs/metrics.json` and the original research paper PDF to prevent plotting errors.
