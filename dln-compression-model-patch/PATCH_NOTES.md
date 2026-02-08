# Patch notes (β₁ stripped / revision-graph framing)

This patch updates Paper 1 to remove algebraic-topology (β₁/Betti/homology/Hatcher) language while keeping the formal account of the Network stage as a structural learning cycle operating over a model space $\mathcal{M}$ and revision graph $\mathcal{R}=(\mathcal{M},\mathcal{T})$.

## Files included

- `paper/main.tex`
  - Abstract: adds model-space / revision-graph framing; adds an analytic crossover condition sentence.
  - Section 1 retitled and language changed from “topology” to “structure.”
  - Section 2.1: adds bridge sentence pointing to Section 2.4.
  - Section 2.2: inserts Proposition 1 (Cost separation and recovery bound) with proof sketches.
  - Section 2.4: new “Model space and revision graph” subsection (definitions of $\mathcal{M}$ and $\mathcal{R}$; stage mapping; two-level distinction; treewidth discussion).
  - Figure 1: caption updated and adds a TikZ panel (b) showing revision graphs (no β₁ labels).
  - Section 4.2: adds a “Formalizing the gap” paragraph tying structured learning to $\mathcal{R}$ and the crossover $K^*$.
  - Section 5.1: expands the structural learning cycle to include contraction and adds `\label{sec:slc}` / `\label{sec:contraction}`.
  - Section 5.3: adds the `Network-NoContract` ablation description.
  - Section 5.4: adds contraction and time-in-model metrics.
  - Results: adds a short subsection on contraction and bounded recovery (return transition).
  - Discussion and Conclusion rewritten to use $G$ + $\mathcal{R}$ framing and to avoid algebraic-topology claims.

- `src/dln_core_variable_cycle.py`
  - Adds an `allow_contract` flag and a `Network-NoContract` agent variant (expand enabled, contraction disabled).
  - Tracks: `contractions`, `time_factor`, and `time_tabular` per episode.
  - Writes these new fields to `results/episode_metrics.csv` and aggregates them into `artifacts/tables/agg_summary.csv`.

- `paper/paper1_new_section_2_4.tex`
  - Standalone drop-in containing the rewritten Section 2.4 plus Proposition 1 (β₁ stripped).

- `paper1_revision_plan.md`
  - Updated implementation plan reflecting the β₁-stripped revision-graph approach.
