# Data Center Flexibility as a Resource Adequacy Asset

This repository contains the analysis code and data supporting the Nature Energy manuscript "Data Center Flexibility as a Resource Adequacy Asset" by Christopher Dunlap (University of Chicago, Energy Policy Institute at Chicago / Booth / Harris).

## Overview

The analysis quantifies how deeply AI data center fleets can credibly commit as demand response resources, combining a ten-parameter cascade model of spatial migration feasibility with four years of empirical grid stress data across five U.S. regions. Three notebooks reproduce the paper end to end:

- **Notebook 01 — Empirical Evidence.** Cross-region stress decorrelation analysis (§3) and workload characterization (§4).
- **Notebook 02 — Cascade and Simulation.** Ten-parameter cascade framework, variance decomposition, conditional Monte Carlo, and sensitivity analysis (§2, §5).
- **Notebook 03 — System Implications.** ELCC application, avoided capacity cost, and interconnection acceleration NPV (§6).

## Requirements

- Python 3.11+
- Packages: see `requirements.txt`
- Data: see `data/raw/README.md` for source locations. Most data downloads automatically on first notebook run; DA LMP data must be obtained separately from each ISO/RTO market portal.

## How to run

From the repository root:

1. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order: 01 → 02 → 03. Each notebook writes to `outputs/contracts/` and downstream notebooks read from there. Running out of order will fail cleanly with a missing-file error.

## Claim-to-cell mapping

Every numerical claim in the paper is produced by a specific cell. This table is the authoritative reference for reviewer verification.

| Paper claim | Section | Notebook | Cell |
|---|---|---|---|
| Ten-parameter cascade framework | §2 | 02 | Cell 1-1 |
| Cascade central product = 0.0384 | Methods | 02 | Cell 1-2 |
| Commitment depth (cascade) = 20.4% | §5, Methods | 02 | Cell 1-3 |
| Variance shares D2/D4/D5/S2 | §2, Methods | 02 | Cell 2-1 |
| 99.0% dynamic availability (ComEd) | §3 | 01 | Cell 1-3 |
| PJM-DOM 99.5%, ERCOT 100%, CAISO 100% | §3 | 01 | Cell 3-7 |
| Stress overlap heatmap (Fig. 1) | §3 | 01 | Cell 1-4 |
| P99 drain time 4.5–12.1s | §4 | 01 | Cell 2-2 |
| Single facility commit depth 46.9% | §5 | 02 | Cell 3-2 |
| Empirical fleet commit depth 28.9% | §5 | 02 | Cell 3-3 |
| Per-GW sweep (46.4%/42.5%/35.8%/26.9%/23.8%) | §5 | 02 | Cell 3-4 |
| Four-panel conditional MC figure (Fig. 2) | §5 | 02 | Cell 3-5 |
| 2D sensitivity surface (Fig. 3a) | §7 | 02 | Cell 4-1 |
| Tornado chart (Fig. 3b) | §7 | 02 | Cell 4-2 |
| 1 GW: 464 MW curtailable, 427 MW accredited | §6 | 03 | Cell 1-1 |
| 10 GW: 2,689 MW curtailable, 2,474 MW accredited | §6 | 03 | Cell 1-1 |
| $77M avoided capacity cost (1 GW) | §6 | 03 | Cell 2-1 |
| $445M avoided capacity cost (10 GW) | §6 | 03 | Cell 2-1 |
| IX queue acceleration NPV | §6 | 03 | Cell 3-1 |

## Repository structure

See directory tree in `docs/repo_layout.md`.

## Inter-notebook contracts

Notebooks communicate through JSON and parquet files in `outputs/contracts/`. Schema is documented in `docs/contract_schemas.md`. These files are regenerated each time the upstream notebook runs. Do not hand-edit.

## Data and code availability

DA LMP data: publicly available from PJM, ERCOT, CAISO, MISO, NYISO market portals. Microsoft Azure LLM Inference Dataset 2024 (DynamoLLM): available at https://github.com/Azure/AzurePublicDataset. BurstGPT: available at (https://github.com/HPMLL/BurstGPT).

All analysis code in this repository is released under the MIT License.

## Citation

[TODO]

## Contact

Christopher Dunlap — cdunlap@chicagobooth.edu