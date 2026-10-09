# Non-runnable experiment configs

These YAML files reference oracle/objective modules removed by the OracleResult vectorization (#12 / PR #23):

- `opioid_ki` — deleted with that refactor
- `tdc_tasks` (e.g. `drd2.yaml`) — not restored as a web/CLI preset; see #20

Kept for historical reference only. Do not pass them to `seismo` / `run_experiment` until the modules exist again.
