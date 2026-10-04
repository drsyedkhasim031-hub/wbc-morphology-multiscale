# Reported manuscript summaries

**These values were extracted from the supplied manuscript. They have not been reproduced or independently authenticated.** They must never be presented as results obtained by the code in this repository. No case-level observations, checkpoint files, source splits or seed-level raw outputs accompanied the manuscript.

All 16 tables were extracted verbatim to UTF-8 CSV; source column units, caveats and rounding are retained. A blank cell in a continuation row is not a missing experiment to be invented.

| File | Manuscript table |
|---|---|
| `table_01.csv` | Dataset provenance, selected cohorts and acquisition characteristics |
| `table_02.csv` | Native label counts |
| `table_03.csv` | Common-five harmonization |
| `table_04.csv` | Proposed partitions and transfer protocol |
| `table_05.csv` | Architecture table, including unresolved historical descriptions |
| `table_06.csv` | Training settings |
| `table_07.csv` | Reported backbone-substitution accuracies |
| `table_08.csv` | Reported LODO metrics |
| `table_09.csv` | Reported native AML per-class performance |
| `table_10.csv` | Reported difficult-pair comparisons, unresolved decision-rule provenance |
| `table_11.csv` | Reported architecture/objective ablations |
| `table_12.csv` | Reported crop-scale comparison |
| `table_13.csv` | Reported compartment agreement and N:C errors |
| `table_14.csv` | Reported calibration and selective prediction |
| `table_15.csv` | Reported corruption robustness |
| `table_16.csv` | Reported compute/timing, not verified by current implementation |

The supplied document itself identifies unresolved results provenance, mismatched pairwise/native interpretations, and profiling inconsistencies. The new implementation's specific choices cannot retroactively authenticate those numbers. Raw outputs from actual executed demonstrations are stored separately in [`../executed/`](../executed/).
