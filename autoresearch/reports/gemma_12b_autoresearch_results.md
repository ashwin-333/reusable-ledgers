# Gemma 4 12B autoresearch — results

**Search:** started 2026-09-29; continuation and promoted-candidate evaluations completed in October 2026

**Pipeline:** complete agent trajectory → precomputed evidence → frozen Gemma 4 12B factual ledger → fixed sanitizer → frozen Gemma 4 12B reader → LDS-v2. Model weights were not updated, and Gemma did not call tools interactively.

## Outcome

The selected train winner scored **0.6087 LDS-v2** on 144 train cases, compared with **0.4022** for the no-tools ledger baseline (**+0.2065**) and **0.3987** for the direct-trajectory control. On the same 96 Calendar cases, the selected train winner scored **0.3411**, compared with **0.3024** for the no-tools ledger baseline (**+0.0387**). The Calendar gain is smaller and uncertain; it does not reproduce the train-set increase.

The search proposed and screened **50 configurations** on a 24-case screen. **37** were promoted to full-train evaluation on 144 cases, and all 37 were evaluated on the 96-case Calendar set. The highest Calendar score among those 37 was **0.4208**, from a configuration with train LDS-v2 **0.5180**. This is a post-selection maximum, not the Calendar score of the train-selected winner.

## Main comparisons

| Configuration | Train cases | Train LDS-v2 | Train balanced accuracy | Train construct macro-F1 | Calendar cases | Calendar LDS-v2 | Calendar balanced accuracy | Calendar construct macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct trajectory | 144 | 0.3987 | 0.6389 | 0.1586 | 96 | 0.3251 | 0.5938 | 0.0563 |
| No-tools ledger | 144 | 0.4022 | 0.6250 | 0.1795 | 96 | 0.3024 | 0.5104 | 0.0943 |
| Selected train winner | 144 | **0.6087** | **0.7986** | **0.4187** | 96 | **0.3411** | **0.5729** | **0.1094** |
| Highest Calendar scorer among promoted configurations¹ | 144 | 0.5180 | 0.7014 | 0.3346 | 96 | **0.4208** | 0.6354 | 0.2062 |

¹ Identified after all 37 promoted configurations had been evaluated on Calendar. It should be read as an exploratory post-selection result, not as the preselected train winner.

The selected winner improved LDS-v2 across all three train tasks relative to the no-tools ledger: airline **0.4085 vs. 0.3005**, bank transfer **0.7639 vs. 0.4302**, and expenditure accounting **0.6187 vs. 0.4699**. Its train balanced accuracy and construct macro-F1 both rose, while the benign false-flag rate fell from **0.5556** to **0.2917**. The winner had four unknown outputs on train; the no-tools baseline had none.

On Calendar, the winner improved LDS-v2 by **0.0387**, balanced accuracy by **0.0625**, and construct macro-F1 by **0.0151** relative to the no-tools ledger. Its LDS-v2 was also **0.0160** above the direct-trajectory control, but remained far below its train score. The full Calendar sweep found 34/37 promoted configurations above the no-tools ledger, 30/37 above the direct-trajectory control, and 23/37 above the train winner.

## Search and evaluation

LDS-v2 is `0.5 × balanced verdict accuracy + 0.5 × C1–C4 construct macro-F1` from the fixed Gemma reader. The no-tools ledger baseline is the search seed; the direct-trajectory control is a separate reference. Candidate screening used 24 paired train cases. Full-train promotion evaluation used all 144 train cases, and Calendar evaluation used 96 cases.

The 50 screened configurations explored prompt and evidence changes around identity and amount provenance, linking requests to executed outcomes, representing selection and constraint evidence, and shortening evidence windows. The train winner used the precomputed **T32 call-and-identity index** and **T36 outcome-provenance chain**, with a shorter evidence window. The highest Calendar scorer used T40, T36, and T38. Because the candidate changes combined tool and prompt choices, these results do not isolate the causal effect of any individual tool.

Opus consulted Astra four times during the 12B search. Advice led to concrete follow-up candidates; most tested follow-ups did not beat the incumbent, while the shorter evidence-window candidate reached the final train best. The search does not establish Astra's isolated causal contribution. The aggregate search record reports no direct paper-reader calls, so paper citations in candidate hypotheses should not be taken as evidence that Opus read the full papers during this search.

## Interpretation and limits

The train results show that the searched prompt-plus-evidence pipeline can produce a substantially more useful ledger for the fixed downstream reader on these 144 cases. Improvements appeared in each train task, and the selected configuration reduced benign false flags while increasing construct recovery. This is an overall pipeline result; it does not show that one tool or prompt edit caused the gain.

Calendar transfer was positive against the no-tools ledger for the selected winner, but the gain (**+0.0387**) was much smaller than the train gain (**+0.2065**). A paired bootstrap over matched malicious/benign pairs gave an exploratory 95% interval of **[−0.0395, 0.1197]** for the Calendar difference and **[0.1424, 0.2701]** for the train difference. These intervals are conditional on configurations selected through an adaptive search and do not account for that selection. Generalization should be assessed with a preselected configuration on another task or dataset.

The post-selection Calendar maximum of 0.4208 shows that some promoted candidates transferred better than the train winner, but selecting that maximum after evaluating all candidates makes it exploratory. A future test should preselect the candidate using train-only evidence and evaluate it once on an untouched task or dataset.

## Related reports

- [Gemma 4 31B autoresearch results](gemma_31b_autoresearch_results.md)

This report summarizes aggregate results. It does not include private case text, per-case predictions, scheduler identifiers, or machine-specific filesystem paths.
