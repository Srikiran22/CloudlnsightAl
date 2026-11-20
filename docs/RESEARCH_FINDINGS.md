# TabuLens — Research Findings, Best Practices and Product Roadmap

Companion to PROJECT.md. Research scope is the concepts and product direction behind TabuLens, not a line-by-line code-defect audit. Research order: Exa → Firecrawl → TinyFish / Reddit. Reviewed 2026-11-20.

## 1. Research method and limitations

Repository mapping preceded research: recursive tree, app entry point, workflow pages, core utilities, tests, configuration, architecture/methodology docs and commit history. The README was not used as the source of truth.

EFT order:
1. Exa: first connected account returned HTTP 402 (credit exhaustion). A second already-connected account, Gmail-VIT, was tried and succeeded. No account was created and no restriction bypassed.
2. Firecrawl: searched primary documentation and developer/research sources, then retrieved selected official pages.
3. TinyFish: searched Reddit practitioner discussions and fetched selected original threads.

The expanded pass covered about twenty Exa topic searches, twenty Firecrawl developer-documentation searches, ten TinyFish Reddit searches and selected page fetches. This is broad research, not a mathematically exhaustive survey of every paper, product or industry implementation. Reddit is qualitative practitioner experience, not peer-reviewed evidence.

## 2. Central product conclusion

TabuLens should become a trustworthy tabular-data decision workbench, not a feature-count contest. Its strongest differentiators are bounded local ingestion, data-quality/statistical exploration, reproducible preprocessing and evaluation, explainable distribution-change diagnostics, decision-aware classification, trust-checked model export and reports that preserve dataset context.

The next improvements should increase evidence quality: what changed in the data, how large the change is, whether it matters to a particular model, what objective selected the threshold, and whether the exact result can be reproduced.

## 3. Findings by concept

### 3.1 Data quality: explicit contracts beat generic scores

Current foundation: hygiene index, missingness/uniqueness summaries, schema comparison, duplicate checks, parser limits and profiling.

TensorFlow Data Validation (TFDV) uses schemas to describe expected feature presence, types, domains and value counts, then validates incoming data against those expectations. It distinguishes stable schema expectations from dataset statistics that vary between samples. It also supports training-serving skew and temporal drift comparisons. Sources: https://www.tensorflow.org/tfx/guide/tfdv and https://www.tensorflow.org/tfx/data_validation/get_started.

Recommendations:
1. Add an editable dataset contract: required columns, data types, allowed categories, nullability, numeric bounds, uniqueness keys and optional business rules.
2. Validate against the contract before cleaning/training.
3. Separate hard errors (missing required column, wrong type, invalid key) from warnings (unexpected category, missingness increase, distribution change).
4. Show violation counts and example rows subject to redaction.
5. Version the contract and record whether a schema change was accepted.

Avoid presenting one “quality = 94%” score as objective correctness. Completeness and uniqueness do not prove validity, representativeness or fitness for a business question.

Best differentiating feature: a lightweight local Data Contract / Schema Guard exportable as JSON or YAML.

### 3.2 Drift: separate significance, effect size and operational impact

Current foundation: numeric two-sample KS, Benjamini–Hochberg FDR, categorical TVD, minimum sample-size checks, schema/missingness/mean flags and unseen-category reporting.

Distinguish:
- Covariate/feature drift: P(X) changes.
- Label/prior drift: P(Y) changes.
- Concept drift: P(Y|X) changes.
- Performance degradation: quality on labelled outcomes worsens.

Feature-only statistics can reveal changes in P(X), but cannot prove concept drift or model-accuracy loss.

A 2024 Nature Communications study using real-world medical imaging found drift detection depended strongly on sample size and population characteristics; performance monitoring alone was not a sufficient proxy for input drift. https://www.nature.com/articles/s41467-024-46142-w. A 2024 benchmark of unsupervised concept-drift detectors found methods differ by target metric and assumptions, and some common evaluation proxies can be biased. https://link.springer.com/article/10.1007/s41060-024-00620-y.

Recommendations:
1. Display raw/adjusted p-values, KS statistic, reference/current sample sizes and number of tests.
2. Show effect size separately from statistical significance.
3. Add confidence intervals or bootstrap uncertainty where feasible.
4. Track drift across time windows, not only one baseline-vs-current comparison.
5. Include prediction distribution, missingness and unseen-category rates.
6. Add labelled performance evaluation when outcomes become available.
7. Let users choose/version the reference dataset.

Avoid treating a KS p-value as magnitude, tiny significant shifts as incidents, feature shift as proof of accuracy loss, or drift flags as automatic retraining triggers. Do not present TVD=0.20 as a universal business threshold.

Evaluate PSI, Jensen–Shannon divergence, Wasserstein distance, chi-square, classifier-based two-sample tests and multivariate detectors only where they add a useful diagnostic. TFDV documents L-infinity distance for categorical distributions and approximate Jensen–Shannon divergence for numeric drift, and notes thresholds require domain knowledge and iteration: https://www.tensorflow.org/tfx/guide/tfdv.

### 3.3 Validation: split according to how the data is generated

Current foundation: Pipeline + ColumnTransformer, median/mode imputation, scaling, one-hot encoding, stratified/K-fold validation, TimeSeriesSplit option, dummy baselines and out-of-fold probabilities.

scikit-learn explicitly recommends Pipeline to ensure learned transformations are fitted on the correct training subset during cross-validation and hyperparameter tuning: https://scikit-learn.org/stable/common_pitfalls.html.

Recommendations:
1. Expose split strategies with explanations:
   - StratifiedKFold for IID classification.
   - KFold for IID regression.
   - GroupKFold / StratifiedGroupKFold for repeated people, customers, devices or sites.
   - TimeSeriesSplit / rolling-origin for chronological prediction.
2. Let users select timestamp and entity/group columns; do not silently infer row order is time.
3. Explain gap/embargo when time windows or labels overlap.
4. Keep a final untouched test set; tune thresholds on validation/OOF data.
5. Show split policy, counts, class distributions and dropped rows.
6. Add repeated-seed or uncertainty reporting for small datasets.

Avoid random splitting for repeated entities or time-dependent records when it leaks entity/future information and inflates generalization estimates.

### 3.4 Decision thresholds: connect the cutoff to costs

Current foundation: binary threshold tuning from OOF probabilities, precision/recall/F1/MCC recalculation and persistence into batch inference.

scikit-learn treats threshold selection as a post-training decision step and documents cross-validated threshold tuning and probability calibration: https://scikit-learn.org/stable/modules/classification_threshold.html and https://scikit-learn.org/stable/modules/calibration.html.

Recommendations:
1. Add false-positive/false-negative cost inputs or a cost matrix.
2. Allow operational constraints such as recall ≥ 90%, precision ≥ 80% or minimum expected cost.
3. Show confusion-matrix counts, precision, recall, F1, MCC, balanced accuracy and expected cost.
4. Add calibration curve/Brier score before describing probabilities as calibrated risk.
5. Show threshold stability across folds/bootstrap samples.
6. Persist objective, validation fingerprint and threshold in model metadata.

Avoid calling maximum F1 universally optimal. Choose on validation data and evaluate on separate test data.

### 3.5 Model comparison: credible tabular benchmarks, not an algorithm zoo

Current foundation: logistic regression, decision tree, random forest, gradient boosting variants and dummy baselines.

Recommendations:
1. Evaluate CatBoost or HistGradientBoosting as practical next candidates; add XGBoost/LightGBM only if dependency and installation cost are justified.
2. Retain simple interpretable and dummy baselines.
3. Use identical splits, preprocessing and metrics for all candidates.
4. Add compute budgets, trial limits, timeouts and cancellation.
5. Report training time, prediction latency, memory and artifact size with predictive scores.
6. Build reproducible benchmark mode using documented public datasets and fixed seeds.

Avoid adding deep learning or tabular foundation models for hype. A single accuracy score on synthetic data is not evidence of superiority.

### 3.6 Feature importance: explain the meaning of the number

Current foundation: estimator-provided impurity importance or coefficient magnitude, with encoded categories grouped back to source features.

scikit-learn documents that impurity-based tree importance can favor high-cardinality features and can be high for features used to overfit. Permutation importance is a model-agnostic alternative: https://scikit-learn.org/stable/modules/permutation_importance.html.

Recommendations:
1. Add repeated permutation importance on held-out data with uncertainty.
2. Label method and scope: impurity-based, coefficient magnitude or permutation score drop.
3. Warn correlated features can share/mask importance.
4. Add optional partial dependence/ICE.
5. Add SHAP only if runtime/dependency cost is justified and document background and correlation assumptions.

Avoid calling importance causal or scientific importance; do not treat methods as interchangeable.

### 3.7 Missing values and outliers: preserve evidence before changing data

Current foundation: imputation, IQR and Z-score diagnostics, distribution shape and feature transforms.

Missingness may be MCAR, MAR or MNAR; the mechanism affects inference. Single imputation can distort variance. Outliers can be errors or genuine rare events. Sources: 2025 data-cleaning primer https://pmc.ncbi.nlm.nih.gov/articles/PMC12721946/ and scikit-learn imputation guide https://scikit-learn.org/stable/modules/impute.html.

Recommendations:
1. Show missingness by feature and, where useful, group/time.
2. Offer missingness indicators; fit imputation inside the pipeline.
3. Explain median/mode imputation is a baseline, not a universal solution.
4. Show before/after distributions and counts for every cleaning action.
5. Add robust MAD scores alongside IQR/Z-score.
6. Let users mark outliers investigate/exclude/cap/retain; preserve an audit log.
7. Provide sensitivity analysis with and without reviewed outliers.

Avoid deleting outliers merely because they are far from the mean, silently dropping missing rows, imputing before splitting, or assuming Z-scores suit highly skewed/multimodal data.

### 3.8 Data lineage and experiment reproducibility

Current foundation: dataset fingerprints, model metadata, random seeds, dependency/config files and a local verification runner.

MLflow Tracking records parameters, metrics and artifacts; dataset tracking links data inputs to runs; Model Registry records model versions and lineage. Sources: https://mlflow.org/docs/latest/ml/ , https://mlflow.org/docs/latest/ml/dataset/ and https://mlflow.org/docs/latest/ml/model-registry/.

Recommendations:
1. Build a compact local experiment ledger before requiring hosted tracking.
2. Store run ID/time, Git commit, dataset fingerprint, target, feature list, split policy, seed, preprocessing config, estimator params, metrics, threshold objective, package versions and artifact checksum.
3. Record transformations and source/derived fingerprints.
4. Export a rerunnable JSON config and command.
5. Compare runs in a table.
6. Add optional local MLflow integration only when needed.

Avoid tracking only a filename and one score; do not upload private datasets/row-level examples by default.

### 3.9 Model artifact security and portability

Current foundation: joblib bundle, HMAC-SHA256 verification before deserialization, standalone inference export, metadata and feature-schema checks.

Python's pickle documentation warns that unpickling malicious/tampered data can execute arbitrary code and recommends signing to detect tampering: https://docs.python.org/3/library/pickle.html. scikit-learn describes ONNX, skops.io and pickle/joblib trade-offs: https://scikit-learn.org/stable/model_persistence.html.

Recommendations:
1. Keep signature verification before joblib.load.
2. Protect the signing key outside the artifact bundle; document threat model, recovery and rotation.
3. Add manifest with artifact hash, signature algorithm/key ID, versions, input schema, training fingerprint and creation time.
4. Explain that HMAC proves possession of a key, not model accuracy or benign behavior.
5. Evaluate skops.io for supported models and inspect untrusted types.
6. Evaluate ONNX for supported inference-only portability.
7. Test loading in a clean, pinned environment.

Avoid claiming HMAC makes pickle intrinsically safe; never commit the key or ship the key alongside the artifact as proof of third-party authenticity.

### 3.10 File ingestion: retain bounded-parser identity

Current foundation: byte/row/column/cell/memory limits, chunked CSV, bounded JSON/Parquet, Excel ZIP checks, XML DTD/XXE rejection, path containment, atomic writes and CSV formula sanitization.

Recommendations:
1. Keep format-specific parsers and central resource policy.
2. Enforce compressed/uncompressed archive limits, member count, nesting and parser time budgets.
3. Keep parsing separate from UI; test malformed inputs at parser boundaries.
4. Explain whether data was truncated, sampled or rejected.
5. Test spreadsheet formula triggers including whitespace/control prefixes.
6. Use read-only SQLite connections and row/table limits; do not execute user SQL by default.
7. Add fuzz/property-based tests for malformed CSV, JSON, XML, ZIP and Parquet metadata.

Avoid trusting extension/MIME alone, reading untrusted files fully before limits, enabling external XML entities or assuming local files are automatically safe.

### 3.11 Privacy and optional AI integrations

Current foundation: local-first core and heuristic sensitive-column screening. Recent refactoring removed external Gemini/GenAI dependencies and the AST sandbox.

Recommendations:
1. Keep core workflow offline and functional without API keys.
2. Label PII screening heuristic; show reason; do not claim anonymization.
3. Make any future external LLM feature opt-in and disabled by default.
4. Redact before prompting and show exact outbound payload for approval.
5. Never send whole datasets by default; prefer aggregate summaries/user-selected fields.
6. Log provider/model/prompt version/consent without raw private data.
7. Preserve a local-only path.

Avoid adding a chatbot to make the project look more “AI”. Do not claim privacy/compliance without a threat model and documented data flows.

### 3.12 Reports and product experience

Current foundation: PDF reports, presets, charts, sensitive-column exclusions and dataset fingerprint checks.

Recommendations:
1. Add reproducibility header: dataset fingerprint, dimensions, date, app version, run ID, target, split and package versions.
2. Separate observation, statistical evidence, practical interpretation and next action.
3. Include caveats, sample sizes and missing-data coverage.
4. Export machine-readable JSON alongside PDF.
5. Provide one-page executive summary plus methodology appendix.
6. Preserve configurable redaction and list excluded columns.

Avoid fabricated confidence, invented business impact, generic “AI says” filler or charts without sample-size/context.

## 4. Reddit practitioner findings

These are qualitative practitioner discussions, not standards. They surface operational lessons and disagreements; use them alongside primary documentation.

### Drift is not the same as model-health monitoring
Practitioners discuss feature statistics, multivariate drift, outliers, prediction distributions and labelled performance. One monitoring discussion describes drift as potentially harmful or harmless depending on whether predictions become incorrect. A predictive-maintenance practitioner describes discussing drift with clients rather than automatically retraining.
- https://www.reddit.com/r/MachineLearning/comments/ppbjqw/d_how_to_maintain_ml_models/
- https://www.reddit.com/r/MachineLearning/comments/i47gka/d_what_metrics_do_you_monitor_for_your_models_in/

Product implication: historical baselines, investigation workflow and delayed-label evaluation; not drift→retrain automation.

### Leakage can invalidate impressive results
A leakage discussion recommends keeping test data isolated from iterative development and calls out preprocessing statistics from test data and temporal leakage:
https://www.reddit.com/r/MachineLearning/comments/s9axn7/d_how_often_you_fall_into_data_leakage/

Product implication: visible split policy, group/time splits and an untouched final test set.

### Thresholds should follow the problem's cost structure
Practitioners warn against optimizing metrics without understanding false-positive and false-negative costs:
https://www.reddit.com/r/MachineLearning/comments/v15qq8/d_finding_an_optimal_threshold_in_multiclass/

Product implication: cost matrix and operating constraints are more useful than another generic threshold slider.

### Outliers can be consequential observations
A data-science discussion describes outliers that revealed product/SLA failures and stresses distinguishing junk from genuine rare events:
https://www.reddit.com/r/datascience/comments/12uzusd/not_all_outliers_are_wrong/

Product implication: reversible outlier handling and sensitivity analysis.

### Experiment tracking is about reproducibility
Practitioners mention MLflow, DVC, Git, structured logs and experiments.md with config/Git hash. The common need is linking metrics to exact code, data and configuration:
https://www.reddit.com/r/MachineLearning/comments/1gpc3cv/d_how_do_you_keep_track_of_experiments_history/

Product implication: local experiment manifest first; tracking platform when justified.

### Data contracts encode business assumptions
A discussion emphasizes data scientists should help turn domain expectations into explicit testable assertions:
https://www.reddit.com/r/datascience/comments/1l8xqgf/data_scientists_need_to_know_about_data_contracts/

Product implication: user-authored contracts are a natural differentiator.

### Feature importance is easy to over-interpret
Discussions disagree about SHAP, permutation and tree-native importance; rankings depend on model, metric, correlations and data distribution:
https://www.reddit.com/r/datascience/comments/15ft0xx/help_in_interpreting_why_the_model_gave_the/

Product implication: label method/limitations; never equate importance with causality.

## 5. Prioritized product roadmap

### P0 — Make existing decisions defensible
1. Data Contract / Schema Guard: required columns, types, ranges, domains, nullability, uniqueness; local versioning and validation.
2. Drift interpretation: p-value, effect size, sample size and impact separated; temporal trend; feature vs concept drift vs labelled performance.
3. Validation strategy: GroupKFold/StratifiedGroupKFold; explicit time/group fields; split visualization.
4. Cost-aware thresholds: costs/constraints, calibration, threshold stability and provenance.
5. Experiment manifest: dataset fingerprint, Git SHA, seed, split, feature set, params, package versions, metrics and artifact signature.

### P1 — Add depth where it changes decisions
6. Permutation importance and correlated-feature warnings.
7. MAD outlier diagnostics and reversible sensitivity analysis.
8. Missingness indicators and grouped/time analysis.
9. Lineage from raw data through cleaning/features/training/inference.
10. Public real-world benchmark mode with resource measurements.

### P2 — Extend deployment without abandoning local-first
11. Optional local MLflow tracking.
12. ONNX export for supported inference pipelines.
13. skops persistence for supported estimators.
14. CLI batch inference with structured output and exit codes.
15. Scheduled local drift job that archives comparisons.
16. Optional cloud storage only if requested; preserve offline mode.

## 6. Deliberately avoid

- General chatbot/LLM wrapper as substitute for product depth.
- Algorithms without benchmark, user need and fair evaluation.
- Automatic retraining without approval, validation, rollback and monitoring.
- Equating drift with model failure.
- Accuracy-only evaluation for imbalanced data.
- Threshold tuning on final test data.
- Preprocessing fitted on held-out data.
- Random split for repeated entities or time-dependent records without leakage analysis.
- Automatic outlier/missing-row deletion.
- Calling heuristic PII screening anonymization.
- Calling HMAC a sandbox or proof of model quality.
- Production/enterprise/privacy/performance claims without measured evidence.
- Dependency-heavy infrastructure when a local manifest or optional integration solves the need.

## 7. Experiments that produce credible portfolio evidence

| Experiment | Measure | Deliverable |
|---|---|---|
| Drift sensitivity | Detection rate, false alerts, effect size/runtime by sample size | Drift calibration report |
| Validation leakage | Random vs group/time split on repeated entities | Generalization-gap report |
| Threshold decision | Expected cost and precision/recall across cutoffs | Decision-policy comparison |
| Model benchmark | Score, runtime, memory, model size, inference latency | Reproducible leaderboard |
| Ingestion stress | Peak memory, time, malformed-input rejection | Resource-boundary benchmark |
| Portability | Prediction parity across bundle/script/optional ONNX | Inference parity report |
| Data contract | Inject missing columns, invalid categories, range violations | Contract validation report |
| Outlier sensitivity | Metrics with/without reviewed outliers | Robustness report |

For every benchmark publish dataset source/license, split policy, seed, hardware, package versions, metric definition and raw result. Never fabricate numbers.

Suggested public datasets: UCI Adult/Census Income (document sensitive attributes/fairness caveats), UCI Bank Marketing, UCI Bike Sharing (temporal evaluation), OpenML tabular classification/regression suites, and a properly licensed public churn dataset. Retain controlled synthetic shifts for drift tests but label them synthetic.

## 8. Resume and interview narrative

One-sentence story: “TabuLens is a local-first tabular data science workbench connecting data-quality checks, statistical drift analysis, leakage-aware model evaluation, cost-sensitive decisions and portable inference in one reproducible workflow.”

Strong interview topics:
- Why KS needs multiple-testing correction.
- Why p-values and effect sizes differ.
- Why feature drift does not prove concept drift or accuracy loss.
- Why learned preprocessing belongs inside CV pipelines.
- Why group/time splits depend on data generation.
- Why a threshold is a business decision.
- Why joblib/pickle is executable content.
- What HMAC proves and does not prove.
- Why bounded ingestion and local-first are product constraints.

Resume bullet (add metrics only after measuring):
“Developed TabuLens, a local-first tabular analytics and ML workbench with bounded multi-format ingestion, statistical drift auditing (KS with Benjamini–Hochberg FDR and categorical TVD), leakage-aware scikit-learn pipelines, cross-validation baselines, cost-aware threshold tuning, HMAC-verified model artifacts, batch inference and standalone Python export.”

## 9. Research source index

### Official documentation / primary sources
- scikit-learn common pitfalls/data leakage: https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- Threshold tuning: https://scikit-learn.org/stable/modules/classification_threshold.html
- Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- Model persistence: https://scikit-learn.org/stable/model_persistence.html
- Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- Missing-value imputation: https://scikit-learn.org/stable/modules/impute.html
- Python pickle security warning: https://docs.python.org/3/library/pickle.html
- TensorFlow Data Validation: https://www.tensorflow.org/tfx/guide/tfdv
- TFDV getting started: https://www.tensorflow.org/tfx/data_validation/get_started
- MLflow tracking/model lifecycle: https://mlflow.org/docs/latest/ml/
- MLflow dataset tracking: https://mlflow.org/docs/latest/ml/dataset/
- MLflow Model Registry: https://mlflow.org/docs/latest/ml/model-registry/
- Google Research Data Cards Playbook: https://sites.research.google/datacardsplaybook/
- Docker build best practices: https://docs.docker.com/build/building/best-practices/
- GitHub Actions secure use: https://docs.github.com/en/actions/reference/security/secure-use
- ONNX: https://onnx.ai/
- Nature Communications 2024 drift study: https://www.nature.com/articles/s41467-024-46142-w
- IJDSA 2024 unsupervised concept-drift benchmark: https://link.springer.com/article/10.1007/s41060-024-00620-y
- 2025 missing data/outlier primer: https://pmc.ncbi.nlm.nih.gov/articles/PMC12721946/

### Reddit practitioner discussions
- Model maintenance: https://www.reddit.com/r/MachineLearning/comments/ppbjqw/d_how_to_maintain_ml_models/
- Production monitoring metrics: https://www.reddit.com/r/MachineLearning/comments/i47gka/d_what_metrics_do_you_monitor_for_your_models_in/
- Drift methods: https://www.reddit.com/r/datascience/comments/1h6wmn2/data_drift_detection_methods_aside_from_changes/
- Data leakage: https://www.reddit.com/r/MachineLearning/comments/s9axn7/d_how_often_you_fall_into_data_leakage/
- Threshold selection: https://www.reddit.com/r/MachineLearning/comments/v15qq8/d_finding_an_optimal_threshold_in_multiclass/
- Outliers: https://www.reddit.com/r/datascience/comments/12uzusd/not_all_outliers_are_wrong/
- Experiment tracking: https://www.reddit.com/r/MachineLearning/comments/1gpc3cv/d_how_do_you_keep_track_of_experiments_history/
- Data contracts: https://www.reddit.com/r/datascience/comments/1l8xqgf/data_scientists_need_to_know_about_data_contracts/
- Feature importance interpretation: https://www.reddit.com/r/datascience/comments/15ft0xx/help_in_interpreting_why_the_model_gave_the/

### EFT execution summary
Exa's alternate connected account was used to expand searches across data validation/contracts, drift taxonomy/metrics, leakage/splits, threshold/calibration, tabular model benchmarking, feature importance, missingness/outliers, experiment tracking, model security, ingestion, privacy, monitoring, documentation and Docker/CI.
Firecrawl searched official scikit-learn, MLflow, TFDV, Evidently, Great Expectations, OWASP, Python, joblib/skops, ONNX, Docker, GitHub Actions, Google Research and academic sources.
TinyFish searched Reddit discussions on drift, leakage, thresholds, outliers, experiment tracking, contracts, explainability, persistence and privacy; selected original threads were fetched and read. Reddit findings are qualitative and cross-checked against primary sources where possible.

## 10. Completion checklist for future features

- Feature exists in UI/CLI and is reachable.
- Method and assumptions are documented.
- Tests cover normal, edge and adversarial cases.
- Public/synthetic benchmark demonstrates behavior.
- Output includes sample sizes, data fingerprint and configuration.
- Failure states and limitations are visible.
- No unsupported production/security/privacy/performance claims.
- Resume bullet updated only after demonstration and measurement.


## 11. Repository re-verification delta — 2026-11-20

The second repository pass expanded the feature inventory and exposed several mismatches that should affect both how the project is described and how it is developed.

### 11.1 Added implementation facts

- `utils/preprocessing.py` is a separate cleaning utility. Numeric strategies are mean, median or zero; categorical strategies are mode or Unknown; datetime strategies are forward-fill, forward-fill/back-fill or none. Forward-fill only avoids temporal lookahead if row order is chronological.
- `utils/batch.py` merges uploaded DataFrames and inserts a collision-safe source-file provenance column.
- `utils/dataset_ui.py` maintains active dataset identity/fingerprints and a bounded memory cache (256 MiB); this is important for keeping cached page results tied to the selected dataset.
- `utils/quality.py` computes a simple structural score from completeness and row uniqueness. It does not validate business semantics.
- `utils/markup_parsing.py` handles XML and HTML-table parsing separately from the main delimited/JSON/Parquet parser.
- The current file extension contract is precise: tabular CSV/TSV/XLSX/XLS/JSON/JSONL/NDJSON/Parquet/XML/HTML/HTM; text TXT/LOG/MD/RST/SQL; PDF. SQLite is a separate upload mode. DOCX is not in the central accepted extension set despite a UI description mentioning it.
- Text/PDF pathways are not an active GenAI product feature. Text extraction is bounded; PDFs are limited to 30 pages / 50,000 characters and raise the conversion-required pathway. The current uploader previews extracted text and asks for structured data. Conversion-provenance helpers and manifest code remain but are not wired into a live Gemini conversion workflow.

### 11.2 Verification mismatches worth keeping visible

| Area | Repository evidence | Documentation / claim to avoid |
|---|---|---|
| Tests | 19 test modules in current tree | `docs/TESTING.md` still says 12 modules / 384 passing; do not repeat the old pass count |
| Python support | pyproject >=3.12; Docker Python 3.11; CI 3.11 and 3.12 | Do not claim that all packaging/CI/runtime declarations agree |
| GenAI | Gemini dependencies/pages removed; compatibility remnants remain | GitHub description and old comments still mention Gemini; do not claim a live Gemini capability |
| DOCX | Not present in SUPPORTED_DATASET_EXTENSIONS | Do not list DOCX as a supported input |
| Threshold tuning | UI slider, validation metrics from OOF probabilities | Do not describe this as a fully automated cost optimizer |
| Feature engineering | Fit-transform tools operate on the current full DataFrame | Do not claim pre-training UI transforms are automatically leakage-safe |
| Time-series threshold validation | TimeSeriesSplit path attempts cross_val_predict and catches exceptions | Do not assume OOF threshold probabilities are always available for chronological splits |

The 19 test filenames are listed in PROJECT.md. Test modules were inspected, but the suite was not executed during this review.

## 12. Deeper research: findings that change the roadmap

This pass used the second already-connected Exa account (Gmail-VIT) for further topic-specific research and full-page extraction. The first connected Exa account (Gmail) again returned HTTP 402 for exhausted credits. The tool exposed only these two connected Exa accounts; no third account was available. No account was created and no credit restriction was bypassed.

### 12.1 Tabular dataset shift is often conditional, not just marginal

The WHYSHIFT study (NeurIPS 2023) evaluates five real-world tabular datasets across 86,000+ model configurations. It finds that conditional shifts in the relationship between inputs and outcomes, P(Y|X), account for most of the performance degradation in many studied natural shifts. In one reported analysis, 80% of source-target pairs with more than an eight-percentage-point performance drop were primarily attributed to P(Y|X) shifts. The paper also finds that model rankings vary substantially by shift pattern.

**Implication for TabuLens:** KS/TVD marginal feature comparisons are useful diagnostics, but they cannot detect all consequential changes. The highest-value next step is a labelled monitoring workflow: once actual outcomes arrive, compare performance and calibration over time, and localize losses to interpretable subgroups. Do not label a feature-only score as a complete concept-drift detector.

Source: [On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets (NeurIPS 2023 / WHYSHIFT)](https://proceedings.neurips.cc/paper_files/paper/2023/file/a134eaebd55b7406ff29cd75d5f1a622-Paper-Datasets_and_Benchmarks.pdf).

### 12.2 Use public distribution-shift benchmarks, not only controlled synthetic shifts

TableShift contributes 15 real-world tabular binary-classification tasks across finance, education, public policy, healthcare and civic participation. The paper reports that no model consistently beats the baseline methods across all tasks; domain-robust methods can reduce some shift gaps but may reduce in-distribution accuracy, and label-distribution changes are associated with OOD shift gaps.

**Implication:** keep the synthetic Q1/Q2 benchmark for deterministic drift demonstrations, but add a documented real-world benchmark mode. Record task source/license, domain/time split, label prevalence, ID/OOD scores, and resource cost. Do not claim one algorithm is generally “most robust”.

Source: [Benchmarking Distribution Shift in Tabular Data with TableShift](https://arxiv.org/html/2312.07577).

### 12.3 Monitor meaningful subgroups as well as the global dataset

Research on interpretable subgroup drift shows that global performance can hide deterioration in smaller cohorts. The proposed approach tracks interpretable, sufficiently represented subgroups and explicitly considers sample-support thresholds because estimates for tiny cohorts are unstable. Separate work on statistical inference for fairness auditing formulates subgroup performance gaps as multiple-testing problems and develops simultaneous/false-discovery control.

**Implication:** add opt-in slice analysis over user-selected, non-sensitive business dimensions and, where legally/ethically appropriate, protected attributes. Show group sample sizes, uncertainty intervals, positive/negative rates and group-wise error metrics. Control multiple testing when scanning many candidate slices. Avoid declaring a small-cohort difference meaningful from a raw point estimate.

Sources:
- [Detecting Interpretable Subgroup Drifts (KDD 2025)](https://arxiv.org/html/2408.14682)
- [Statistical Inference for Fairness Auditing (JMLR)](https://jmlr.org/papers/volume25/23-0739/23-0739.pdf)
- [A structured regression approach for evaluating model performance across intersectional subgroups (2024)](https://doi.org/10.1145/3630106.3658908)

### 12.4 More historical training data is not automatically better

A 2025 AIES study examines expanding historical training windows under temporal covariate and concept shifts. It reports that longer training windows may help without shift, but can reduce performance under concept shift; the impact can differ between population subgroups and can affect fairness.

**Implication:** a future retraining assistant should compare rolling and expanding windows, not automatically append all historical records. Preserve temporal backtests and subgroup metrics; explain when old examples are no longer representative of the current decision process.

Source: [When the Past Misleads: Rethinking Training Data Expansion Under Temporal Distribution Shifts (AIES 2025)](https://doi.org/10.1609/aies.v8i3.36757).

### 12.5 Calibration and threshold selection are related, but not interchangeable

The current research confirms a key distinction:
- A decision threshold determines which action is taken from a model score.
- Calibration asks whether predicted probabilities correspond to observed frequencies.
- AUC metrics measure ranking/discrimination, not calibrated risk.
- Proper scores such as Brier loss and log loss combine multiple aspects (reliability/calibration, resolution/discrimination and uncertainty); they should not be treated as pure calibration measures in isolation.

scikit-learn recommends reliability diagrams/calibration curves and documents calibration with cross-validation. It also warns that a calibrator should not be fitted on the same data used to fit the base classifier. Threshold selection should be validated independently and use a meaningful objective.

**Implication:** add calibration curves and probability-bin counts before describing scores as risk probabilities. Keep calibration fitting, threshold tuning and final test evaluation on correctly separated data. Add an expected-cost curve only after users enter realistic error costs.

Sources:
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn decision threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)

### 12.6 Data contracts need semantic rules and controlled evolution

The MLSys paper “Data Validation for Machine Learning” emphasizes that many data errors are generated by upstream code bugs and that alerts must be precise and actionable; noisy alerts train operators to ignore the system. Its schema approach covers type/domain/presence constraints and allows initial schemas to be inferred then reviewed and edited. Schema inference is a starting proposal, not an automatically authoritative contract.

**Implication:** TabuLens should distinguish syntactic checks from semantic rules such as units, allowed ranges, keys, freshness and time windows. An inferred contract should be reviewed and version-controlled. A new category or changed range should prompt a review workflow, not silently redefine the baseline.

Source: [Data Validation for Machine Learning (MLSys 2019)](https://proceedings.mlsys.org/paper_files/paper/2019/file/928f1160e52192e3e0017fb63ab65391-Paper.pdf).

### 12.7 Keep reproducibility lightweight and local-first

DVC's experiment-versioning guidance describes storing parameters, metrics and artifact references in text files that Git can version, while data/model bytes are kept in a separate content-addressed store. This can work without a centralized tracking server. A dataset hash identifies bytes but does not preserve/recover the dataset by itself.

**Implication:** TabuLens can implement a simple local run manifest first: Git commit, dataset fingerprint, retained dataset/snapshot identity, immutable split IDs, preprocessing config, model params, seed, environment lockfile, metric outputs and artifact hash. A hash alone is not enough if the underlying data bytes disappear. Optional MLflow/DVC integration can follow if multi-run or collaboration needs grow.

Sources:
- [DVC — Don't Just Track Your ML Experiments, Version Them](https://dvc.org/blog/ml-experiment-versioning/)
- [DVC User Guide](https://doc.dvc.org/user-guide)
- [MLflow dataset tracking](https://mlflow.org/docs/latest/ml/dataset/)

### 12.8 Additional leakage modes deserve explicit user prompts

A 2023 healthcare ML framework separates label leakage into aggregation-based leakage (e.g., repeated patients/sites crossing splits), timing-based leakage (features created after the intended prediction time) and association-based leakage (a proxy feature effectively reveals the outcome). The framework emphasizes that the task's prediction time and operational use must be specified before choosing predictors.

**Implication:** add a short target-design step before training: What is the prediction origin? When is each feature available? When does the label become known? What entity must be held out? A Pipeline prevents leakage from learned transformations but cannot fix a feature that was already constructed using future/outcome information.

Source: [A framework for understanding label leakage in machine learning for health care (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/).

## 13. Updated roadmap ranking after deeper research

### Highest value
1. **Prediction-time / label-window declaration:** record prediction timestamp, feature-availability assumptions, target horizon and entity key before training.
2. **Group-aware splits:** support new-customer/new-patient/new-device evaluation, not only random and chronological row splits.
3. **Labelled temporal backtesting:** rolling-origin evaluation with explicit train/test windows, label delay and, where needed, purging/embargo.
4. **Subgroup performance dashboard:** group size, precision/recall/FNR/FPR, calibration and uncertainty; control the search across multiple slices.
5. **Calibrated decision analysis:** calibration plot + threshold/cost curve; keep probability reliability and action selection as separate concepts.
6. **Data Contract / Schema Guard:** include semantic meaning, units, keys, freshness and training-vs-inference environments, with reviewed evolution.

### Next tier
7. Public benchmark suite including TableShift/WHYSHIFT-compatible tasks.
8. Local experiment manifest with retained data/split references, code SHA and environment digest.
9. Rolling-vs-expanding training-window comparison for changing populations.
10. Model-aware drift investigation: use feature relevance or prediction sensitivity to help prioritize alerts, then validate against labels when available.

### Avoid
- “More training data is always better.”
- “Drift detected means accuracy has dropped.”
- “No global drift means every subgroup is stable.”
- “Good calibration” inferred solely from ROC-AUC, PR-AUC or a single Brier score.
- “Leakage-free” inferred solely because the final estimator is wrapped in a Pipeline.
- “Reproducible” inferred solely from recording a dataset hash without retaining or being able to retrieve the data snapshot.

## 14. Additional deep-research sources

- [WHYSHIFT / NeurIPS 2023 distribution-shift paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/a134eaebd55b7406ff29cd75d5f1a622-Paper-Datasets_and_Benchmarks.pdf)
- [TableShift real-world tabular distribution-shift benchmark](https://arxiv.org/html/2312.07577)
- [Detecting Interpretable Subgroup Drifts](https://arxiv.org/html/2408.14682)
- [AIES 2025 temporal training-window study](https://doi.org/10.1609/aies.v8i3.36757)
- [Statistical Inference for Fairness Auditing](https://jmlr.org/papers/volume25/23-0739/23-0739.pdf)
- [Intersectional subgroup performance estimation (2024)](https://doi.org/10.1145/3630106.3658908)
- [Data Validation for Machine Learning, MLSys 2019](https://proceedings.mlsys.org/paper_files/paper/2019/file/928f1160e52192e3e0017fb63ab65391-Paper.pdf)
- [Label leakage framework for healthcare ML (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/)
- [DVC experiment versioning](https://dvc.org/blog/ml-experiment-versioning/)
