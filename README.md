# Engineering Data Systems Pipeline: Token Generation Decay Analysis (AIP-04)

## I. Project Overview
This repository contains a production-grade, automated data engineering pipeline designed to evaluate LLM performance behaviors over sustained execution bounds. Specifically, this system isolates and investigates Topic AIP-04: Token Generation Decay using computational inference metrics.

The core research objective is to mathematically model and verify whether model throughput—measured in Tokens Per Second (TPS)—degrades over the course of a continuous generation sequence. To fulfill the strict "No Sharing" policy, this software employs dedicated programmatic filters to separate and evaluate performance metrics specifically targeting the Gemini architectural family, establishing a completely unique telemetry sub-slice for evaluation.

## II. System Architecture & Modular Design
The software pipeline is built using a highly structured, non-nested modular architecture divided into exactly five (5) distinct primary engineering functions to maintain strict decoupling between ingestion, computation, and rendering lifecycles:

* **`ingest_data(file_name)`**: Manages robust file system discovery and data ingestion from localized environments using a programmatic Pandas block.
* **`clean_data(df)`**: Automated data cleaning block that strips column anomalies, drops missing/null records, handles type casting, and programmatically instantiates the cumulative sequence tracking metric (`Sequence_Step = i * 100`).
* **`compute_engineering_statistics(df)`**: Advanced data analysis engine utilizing raw, vectorized NumPy arrays to calculate essential system telemetry metrics (Mean, Median, Standard Deviation, and Variance).
* **`run_decay_model(df)`**: Structural machine learning module that fits an ordinary least squares (OLS) LinearRegression engine to extract token generation decay trendlines (β₁).
* **`update_animation_frames(...)`**: Freestanding time-variant visualization engine using Matplotlib's animation framework to dynamically mutate axes configurations and render real-time execution animations.

## III. Repository Structure
To maintain compliance with the project layout guidelines, the repository matches the following structure exactly:

EDS_[StudentNumber]_[Surname]/
├── data/
│   ├── ai_models_performance.csv          # Original raw telemetry logs
│   └── ai_models_performance_cleaned.csv  # Automatically saved unique clean data slice
├── outputs/
│   ├── static_plots/                      # 6 Mandatory descriptive static figures
│   │   ├── 1_pie.png                      # Model Family Distribution Share
│   │   ├── 2_decay_scatter.png            # Token Decay Linear Regression Line
│   │   ├── 3_bar.png                      # Top 10 Model Throughput Comparisons
│   │   ├── 4_box.png                      # Latency Stability Profiles (Gemini Family)
│   │   ├── 5_hist.png                     # Throughput Performance Density
│   │   └── 6_heatmap.png                  # Inter-variable Correlation Matrix
│   └── animations/                        # 2 Mandatory time-variant simulation GIFs
│       ├── 1_performance_mapping.gif     # Progressive data distribution scan
│       └── 2_latency_pulse.gif           # Real-time coordinate trajectory simulation
├── main.py                                # Master Python pipeline file
├── README.md                              # System documentation and instructions
└── requirements.txt                       # Application dependencies

## IV. Core Dependencies & Requirements
The execution of this pipeline requires Python 3.8+ along with several standard data science and machine learning libraries. Create a virtual environment and install the verified dependencies using:

pip install -r requirements.txt

Your requirements.txt must contain:
pandas
numpy
scikit-learn
matplotlib
seaborn
pillow

## V. Execution & Run Instructions
To initialize the pipeline and automatically generate all analytical outputs, run the master script from the root directory of your repository:

python main.py

### Expected Console Telemetry Output
Upon successful execution, the script automatically catches anomalies, displays runtime logs, and updates your console with the calculations required for your IEEE paper tables:

✅ [FUNCTION 1] Ingested raw dataset: ai_models_performance.csv
✅ [FUNCTION 2] Auto-cleaning and type correction pipeline complete.

=======================================================
📊 [FUNCTION 3] NUMPY CALCULATED SYSTEM TELEMETRY METRICS
=======================================================
Throughput (TPS)     -> Mean: 2.1486 | Median: 1.1300 | StdDev: 1.7980 | Variance: 3.2330
Latency (Seconds)    -> Mean: 12.3043 | Median: 4.2200 | StdDev: 13.2804 | Variance: 176.3684
Sequence Step (Tokens)-> Mean: 4350.0000 | Median: 4350.0000 | StdDev: 2555.2234 | Variance: 6529166.6667
=======================================================

📉 [FUNCTION 4] Regression computed. Coefficient: -0.000386 TPS/token
generate_static_plots execution initiated...

=======================================================
🚀 PERFORMANCE EXTREME OPTIMIZATION REPORT
=======================================================
🥇 FASTEST THROUGHPUT: Gemini 3 Flash
   ↳ Speed Vector:    4.5000 TPS
🥇 FASTEST RESPONSE:   Gemini 2.5 Flash-Lite (Sep)
   ↳ Latency Delay:   4.2200 Seconds
=======================================================

🎬 Rendering dynamic visualization configurations via Function 5...
   ↳ Saved: outputs/animations/1_performance_mapping.gif
   ↳ Saved: outputs/animations/2_latency_pulse.gif

🚀 PIPELINE PROCESS COMPLETE: 5 Main Functions successfully executed.
🏁 PRODUCTION EXECUTION SUCCESSFUL.

## VI. Reporting Standard
All empirical findings, mathematical framework proofs, performance plots, and animation freeze-frames produced by this pipeline are fully compiled in accordance with the official IEEE Two-Column Research Format. 

* **Linear Regression Decay Vector Model:** Ŷ = β₀ + β₁X + ε
* **Empirical Decay Factor:** β₁ = -0.000386 TPS/token
* **Volatile Latency Boundary:** σ² = 176.3684
