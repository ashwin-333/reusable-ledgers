# [Write Once, Check Many: Reusable Ledgers for Post-Hoc Policy Compliance Auditing of Agent Trajectories](https://arxiv.org/abs/2610.24483)

Code, per-check scores, and labels for reusing one factual ledger per agent trajectory across many policy checks.

## Overview

When an organization adopts a new natural-language policy, it must recheck the agent trajectories it has already stored. A writer model turns each trajectory once into a ledger, a factual record of what happened, without seeing any policy. A reader model then checks each new policy against the short ledger instead of rereading the full trajectory.

## Repository structure

```
reusable-ledgers/
├── autoresearch/   automated search over writer prompts and evidence tools, with reports and the promoted 31B configuration
├── baselines/      EventRecord identifier extraction, tau2 EventRecord rules, and LLMLingua compression
├── eval/           scoring (eval/metrics.py) and the paper reproducer (eval/reproduce_main.py)
├── export/         exporter that produced results/ and labels/ from stored predictions
├── labels/         per-trajectory policy labels for every policy set
├── ledgers/        ledger writers, policy readers, rendering, and frozen prompts with SHA-256 files
├── policies/       frozen policy texts and the labelers for calendar, SHADE-Arena, tau-bench, tau2-bench, and contrastive pairs
├── results/        per-check scores, row orders and bootstrap strata, token counts, coverage counts, and Figure 2 and Figure 4 data
└── training/       supervised fine-tuning of 12B ledger writers and readers, with run arguments
```

## Setup

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

`eval/` and `export/` need only `numpy`. Writers and readers send requests to an OpenAI-compatible server (for example vLLM) serving `google/gemma-4-12B-it` or `google/gemma-4-31B-it`; the pinned revisions are in `config.py`.

Every setting in `config.py` can be overridden with an environment variable of the same name:

| Variable | Default | Purpose |
|---|---|---|
| `DATA_ROOT` | `./data` | benchmark trajectories, benchmark files, and sampled-ID lists |
| `RESULTS_ROOT` | repository root | directory holding `results/` and `labels/` |
| `OUT_ROOT` | `./outputs` | new ledgers, reads, and labels |
| `WRITER_MODEL` | `google/gemma-4-12B-it` | ledger writer |
| `READER_MODEL` | `google/gemma-4-12B-it` | policy reader |
| `READER_MODEL_LARGE` | `google/gemma-4-31B-it` | larger raw reader |
| `QWEN_READER_MODEL` | `Qwen/Qwen3.5-9B` | robustness reader |
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | model server |

## Quick start

With a model server running at `OPENAI_BASE_URL` and the SHADE-Arena benchmark under `DATA_ROOT`:

```bash
python -m ledgers.write_narrative --role prepare                       # freeze the writer prompt hash and token limit
python -m ledgers.write_narrative --role write                         # write one narrative ledger per trajectory
python -m ledgers.read_controls --body action_summary_8192 --output-dir outputs/reads/narrative   # check every policy against each ledger
```

## Reproducing the paper

```bash
python -m eval.reproduce_main
```

This recomputes the numbers reported in the paper from `results/` and `labels/` and prints each next to the paper value. A value is flagged DIFF when it differs by more than half a unit in the paper's last printed digit. The released scores give 421 matching numbers and no differences.

| Paper item | Function in `eval/reproduce_main.py` | Code that produced the scores |
|---|---|---|
| Figure 2 (forest plot) | `fig2` | `ledgers/write_narrative.py`, `ledgers/read_controls.py`, `ledgers/read_policy.py` |
| Section 4.2.1 text and Takeaway 1 | `text_vs_raw`, `takeaway` | same as Figure 2 |
| Cost figure, table, and text (Figure 4, Appendix K, Table 13) | `costs` | `ledgers/read_policy.py`, `ledgers/write_structured.py` |
| SHADE-Arena compositional table (Table 2) | `tables` | `ledgers/read_controls.py`, `policies/shade_compositional/` |
| Robustness table (Table 3) | `tables` | `ledgers/read_controls.py` with `ledgers/prompts/robust/` |
| Contrastive pairs table (Table 12) | `contrastive` | `policies/contrastive/build_paired_policy.py` |
| Ledger, full log, or both (Table 4) | `evidence` | `ledgers/read_controls.py`, `baselines/eventrecord.py` |
| CUA table (Table 5) | `cua` | `ledgers/read_graded.py`, `ledgers/shade_run.py` |
| Content-judgment table (Table 15) | `tables` | `ledgers/read_controls.py` |
| Writers and search (Section 4.3.2) | `search` | `autoresearch/`, `training/` |

Thirteen reported numbers cannot be recomputed from the released scores, because the reads behind them are not in the score table. The reproducer lists them as NOT COVERED:

- the four section ablations of the fine-tuned ensemble ledger (.901, .685, .914, .907);
- the SHADE-Arena log-first read (.740);
- task-agnostic compression, low and high end (.513, .739);
- Bank-Synth, prompted 12B ledger and 12B reading raw (.940, .521);
- actions named before and after fine-tuning (48%, 73%);
- the recipient-policy delta on the 122 send-tool transcripts (.205);
- the atomic delta within task-policy cells (.089).

## Data

No benchmark transcripts are included. `results/` and `labels/` contain only trajectory identifiers, labels, decisions, graded suspicions, and token counts. To rerun writers or readers, place each benchmark under `DATA_ROOT`:

- SHADE-Arena: the transcripts are gated; request access from the SHADE-Arena authors.
- tau-bench: `policies/taubench/build_taubench.py` downloads the public historical trajectories from the tau-bench repository (sierra-research/tau-bench).
- tau2-bench: telecom trajectories from the tau2-bench repository (sierra-research/tau2-bench), placed under `DATA_ROOT/tau2/tau2_telecom/trajectories`.
- CUA: the computer-use sabotage scenarios are available from their authors.
- Calendar: the calendar trajectories are available from their authors.

The frozen policy texts are in `policies/`, and the 100-trajectory SHADE-Arena subset is read from `DATA_ROOT/e5_w1_subset.json`.

## Hashes

Two files were scrubbed of internal paths for release, so their hashes differ from the versions used in the experiments:

| File | Original SHA-256 | Release SHA-256 |
|---|---|---|
| `baselines/tau2_eventrecord_rules.json` | `50e6e628cf32012f50dfe3c708e82d9ef9c6baf301a9e301b7c5059a5c6255a0` | `c6862a1799fd52b662b32d3772c9f24951f278e992c6ac6eddbb774465b1c211` |
| `autoresearch/candidate_tools.json` | `49f713ba2d93b8fb36e766c53346bf46dd22159e5bd2da507535f7ffa3856c7b` | `976511095e09a233f0065ab3963337c4312a0f8b775cd329e44a0483c518f669` |

## Citation

```bibtex
@misc{nagarajan2026writeoncecheckmany,
      title={Write Once, Check Many: Reusable Ledgers for Post-Hoc Policy Compliance Auditing of Agent Trajectories},
      author={Ashwin Nagarajan and Gursimran Panesar and Alicia Yang and Mona T. Diab and Virginia Smith and Aashiq Muhamed},
      year={2026},
      eprint={2610.24483},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.24483},
}
```

## License

Code is released under the MIT License; see `LICENSE`.
