# Implementation Plan: Applying Fix with β₁ Stripped

## Governing Principle

Keep: M, R, T, two-level distinction, contraction mechanism, code, crossover formula.  
Strip: β₁, Betti numbers, homology groups, algebraic topology language, Hatcher citation, "Scope of the topological claim" paragraphs.  
Demote: Theorem 1 → Proposition 1.  
Replace: Every instance of "β₁(R) = 1" with "R contains a return transition" or "R contains a cycle." Every instance of "β₁(R) = 0" with "R has no return transition" or "R is acyclic."  
Take from fix: Improved phrasing for Proposition (i) lower-bound argument, Section 4.2 paragraph, code deliverables.  
Log for future: Multigraph precision issue (expand and contract as distinct 1-cells) matters when β₁ is reintroduced for richer M. Record in internal registry.

---

## Deliverable-by-Deliverable Map

### Deliverable 1: Abstract edits

**Action: MODIFY (take from fix, strip β₁)**

First insertion becomes:
```latex
We formalize this cycle as navigation in a model space $\mathcal{M}$ whose
elements are candidate belief-dependency structures.  DLN stages are then
characterized by two formal objects: the belief-dependency graph $G$
(determining Level~1 inference cost) and a revision graph
$\mathcal{R}=(\mathcal{M},\mathcal{T})$ over $\mathcal{M}$ (determining Level~2
meta-cognitive capacity).  Linear cognition is a fixed point in $\mathcal{M}$
with $\mathcal{T}=\emptyset$; Network cognition revises by traversing edges of
$\mathcal{R}$, including a return transition that enables bounded recovery after
model failure.
```

Second insertion: keep crossover sentence (no β₁ content).

---

### Deliverable 2: Section 2.1 bridge sentence

**Action: MODIFY (strip topology reference)**

```latex
Section~\ref{sec:model_space} formalizes this distinction by defining a model
space $\mathcal{M}$ whose elements are candidate belief structures and a
revision graph $\mathcal{R}$ whose edges are transitions between them; the cycle
that defines Network cognition is a structural feature of $\mathcal{R}$, not of
any individual belief graph $G\in\mathcal{M}$.
```

---

### Deliverable 3: Section 2.2 Proposition (was Theorem)

**Action: MODIFY (demote to Proposition, strip β₁)**

- Theorem → Proposition (label updated to `prop:crossover`)
- β₁ language replaced by return-transition language
- Part (i) uses lower-bound framing (Network cannot do better than its cheapest correct-model operating point while still paying Level 2 overhead)

---

### Deliverable 4: Full Section 2.4

**Action: REWRITE**

Keep: definitions of $\mathcal{M}$ and $\mathcal{R}$, contraction paragraph, two-level distinction, treewidth paragraph.  
Strip: β₁ paragraph, scope paragraph.  
Replace with: graph-combinatorial “Revision capacity by DLN stage” paragraph.

---

### Deliverable 5: Figure 1 Panel B + caption

**Action: MODIFY (strip β₁ labels and caption content)**

Replace any β₁ annotation with plain graph language:

```latex
$\mathcal{R}$ contains a cycle
```

---

### Deliverable 6: Section 4.2 paragraph

**Action: MODIFY (strip β₁, retain motivation)**

Add a brief paragraph tying $\mathcal{R}$ to metacognitive deployment and the crossover $K^*$.

---

### Deliverable 7: Section 5.1 contraction mechanism

**Action: ADD**

Add Step 4 (contraction / return transition) and define parameters used in Proposition 1(iii).

---

### Deliverable 8: Section 5.3 agent addition

**Action: ADD**

Add `Network-NoContract` (expand enabled, contraction disabled).

---

### Deliverable 9: Section 5.4 metrics additions

**Action: ADD**

Add contraction events and time-in-model fractions.

---

### Deliverable 10: Section 6.6 scaffold

**Action: ADD**

Add a short subsection describing contraction and bounded recovery, tied to Proposition 1(iii).

---

### Deliverable 11: Discussion 7.1 replacement + new paragraphs

**Action: MODIFY**

- Replace deployment-layer paragraph with $\mathcal{R}$ framing and crossover $K^*$  
- Add “Meta-cognition as a property of model space” paragraph (no β₁)  
- Remove any escalation paragraph that name-drops higher Betti numbers / simplices

---

### Deliverable 12: Conclusion replacement

**Action: MODIFY**

Replace with a two-object characterization (belief graph $G$ and revision graph $\mathcal{R}$), state crossover and bounded recovery in return-transition language.

---

## Summary of Changes to Files

### `paper/main.tex`

- Remove algebraic-topology framing (β₁, Betti numbers, Hatcher, scope paragraphs)
- Add Section 2.4 model space / revision graph
- Add Proposition 1 and references to return transition
- Add contraction mechanism + NoContract agent + metrics
- Update Figure 1 caption and add a small Panel B schematic (no β₁)
- Update Discussion and Conclusion

### `paper1_new_section_2_4.tex`

Full rewrite consistent with the above (β₁ stripped).

---

## New Internal Registry Entry

| ID | Item | Source | Target | Priority |
|---:|------|--------|--------|----------|
| FW-14 | Multigraph precision for β₁ (treat expand/contract as distinct 1-cells when reintroducing β₁ for richer $\mathcal{M}$) | Fix document note | Paper with $|\mathcal{M}|>2$ | Conditional |
