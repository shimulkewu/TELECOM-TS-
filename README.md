# TELECOM-TS-

# Mechanistic Interpretability of LoRA-Fine-Tuned LLMs for Telecom Fault Localisation

This repository contains the data serialisation pipeline, diagnostic benchmarks, and mechanistic interpretability infrastructure for auditing internal neural circuits inside Large Language Models (LLMs) trained on telecommunication telemetry. 

This framework explicitly addresses the black-box opacity of automated diagnostics (**Gap 1**) and evaluates circuit resilience against operational out-of-distribution (OOD) telemetry noise (**Gap 2**).

## Research Objectives
_Objective 1:Extract and map monosemantic network diagnostic features from the internal residual streams of a LoRA-fine-tuned LLaMA-3-8B model using domain-confined Sparse Autoencoders (SAEs).
_Objective 2: Discover and causally validate internal neural circuits responsible for fault localisation using activation patching and path ablation, benchmarking against correlational linear probes.
_Objective 3:** Evaluate circuit structural stability under bounded out-of-distribution (OOD) perturbations such as syntax mutations, timestamp jitter, and missing tokens.

## Repository Structure
* `data_serializer.py`: Deterministic data serialisation script that converts raw multi-channel JSON telemetry into schema-compliant strings while preserving absolute scale metrics.
* `requirements.txt`: Minimal dependencies required to execute the preprocessing verification pipeline.

## 🛠️ Getting Started

1. Installation & Environment Setup
Clone this repository and install the baseline evaluation dependencies:
```bash
git clone https://github.com
cd telecom-ts-sae-circuits
pip install -r requirements.txt
```

2. Run Preprocessing Verification Pipeline
Execute the simulation script to verify that your system environment handles the explicit schema serialisation pipeline without dropping high-density 5G KPI metrics:
```bash
python data_serializer.py
```

##Preprocessing Schema Framework
The serialisation layer maps continuous and categorical performance streams into explicit schema delimiters (`[SCHEMA_START]` / `[SCHEMA_END]`). This structure prevents structural telemetry data from degrading into unparsed free-form prose, ensuring the downstream Sparse Autoencoders (SAEs) can isolate clean latent concepts.
