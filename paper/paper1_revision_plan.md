# Paper 1 Revision Plan (β₁ Stripped)

## Governing principle

Keep: model space $\mathcal{M}$, revision graph $\mathcal{R}=(\mathcal{M},\mathcal{T})$, two-level distinction (Level 1 vs Level 2), contraction mechanism, code deliverables, and the crossover formula.

Strip: $\beta_1$, Betti numbers, homology groups, algebraic topology language, and any paragraphs whose only role is to justify a topological invariant on $|\mathcal{M}|=2$.

Demote: Theorem 1 → Proposition 1.

Replace:
- Every instance of “$\beta_1(\mathcal{R}) = 1$” with “$\mathcal{R}$ contains a return transition” or “$\mathcal{R}$ contains a cycle.”
- Every instance of “$\beta_1(\mathcal{R}) = 0$” with “$\mathcal{R}$ has no return transition” or “$\mathcal{R}$ is acyclic.”

Log for future: Multigraph precision issue (treat expand/contract as distinct directed 1-cells) matters if/when $\beta_1$ is reintroduced for richer $\mathcal{M}$.

---

## Deliverable map

### 1) Abstract edits

**Action: MODIFY (strip topology).**

Insert a paragraph formalizing the cycle as navigation over $\mathcal{M}$ and $\mathcal{R}$ in plain graph language (no $\beta_1$). Keep the empirical summary and add a sentence stating the analytic crossover condition $K^*$.

### 2) Section 2.1 bridge sentence

**Action: MODIFY.**

Add an explicit bridge that distinguishes:
- belief-dependency graphs $G\in\mathcal{M}$ (Level 1)
- the revision graph $\mathcal{R}$ over $\mathcal{M}$ (Level 2)

and states that the “cycle” is a structural feature of $\mathcal{R}$, not of any individual $G$.

### 3) Section 2.2 Proposition (was Theorem)

**Action: MODIFY.**

Replace Theorem 1 with **Proposition 1**:

- (i) crossover $K^* = F + c_{\mathrm{meta}}/c_{\mathrm{param}}$
- (ii) cross-term / stakes failure mode for Linear
- (iii) recovery bound stated in terms of a **return transition** (not $\beta_1$)

### 4) Section 2.4 (full rewrite)

**Action: REWRITE.**

Add a dedicated subsection defining:

- Model space $\mathcal{M}$
- Revision graph $\mathcal{R}=(\mathcal{M},\mathcal{T})$
- Expansion and contraction as distinct directed transitions

Replace any “topological invariant” paragraphs with a short “Revision capacity by DLN stage” paragraph that uses plain directed-graph language.

### 5) Figure 1 Panel B + caption

**Action: MODIFY.**

Update the figure to include revision graphs (Level 2) and revise the caption to:
- remove $\beta_1$ references
- state the functional consequence (“bounded recovery”) and point to Proposition 1(iii)

### 6) Section 4.2 paragraph

**Action: MODIFY.**

Add a “Formalizing the gap” paragraph that uses $\mathcal{M}$ and $\mathcal{R}$ to connect:
- structure learning as model revision (Level 2)
- inference/learning within a structure (Level 1)
and references the crossover condition.

### 7) Simulation design: contraction mechanism

**Action: MODIFY.**

Extend the “Structural learning cycle” from 3 steps (hypothesis → test → expand) to 4 steps by adding contraction (return transition), matching the implemented `ContractionConfig` and shadow-factor evaluation mechanism.

### 8) Agents

**Action: MODIFY.**

Add Network-NoContract (expansion-only) alongside Network-Full, Network-NoTest, and Network-NoUpdate.

### 9) Metrics

**Action: MODIFY.**

Add contraction counts and time-in-model-class metrics to directly test the return-transition claim.

### 10) Results scaffold (contraction / recovery)

**Action: ADD.**

Add a results subsection that explains how contraction and the NoContract ablation test Proposition 1(iii). (If/when a recovery/shift condition is added, this section can be upgraded from scaffold to full results.)

### 11) Discussion replacement

**Action: MODIFY.**

Replace any lingering “topology” framing with:
- revision graph as the formal deployment layer
- meta-cognition as a property of model space navigation

### 12) Conclusion replacement

**Action: MODIFY.**

Conclude with the “two formal objects” characterization ($G$ for Level 1, $\mathcal{R}$ for Level 2), the crossover condition, and the bounded-recovery consequence of the return transition.

---

## Summary of file changes in this package

- `paper/main.tex`
  - removes algebraic-topology framing and $\beta_1$ language
  - adds Section 2.4 model space/revision graph
  - adds Proposition 1 (crossover + recovery bound)
  - updates Figure 1 to include revision graphs
  - adds contraction mechanism description, NoContract agent, and new metrics
  - adds contraction subsection and updates Discussion/Conclusion accordingly

- `paper/paper1_new_section_2_4.tex`
  - drop-in text for Section 2.4 + Proposition 1 (β₁ stripped)

- `paper/paper1_patch_snippets.tex`
  - updated patch snippets (β₁ stripped; graph language)

---

## Future work registry

| ID | Item | Source | Target | Priority |
|----|------|--------|--------|----------|
| FW-14 | Multigraph precision for $\beta_1$ (when reintroducing $\beta_1$ for richer $|\mathcal{M}|$, compute on a geometric realization treating expand/contract as distinct directed 1-cells, not on the simple undirected graph) | Reviewer-facing critique | Paper with $|\mathcal{M}|>2$ | Conditional |
