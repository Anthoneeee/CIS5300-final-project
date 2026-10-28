# Data-stage audit

Generated from the real pinned XML; no classifier was evaluated at this stage.

- Parsed questions: 47457
- Retained questions: 17845
- Disease groups: 6238
- Split SHA-256: `6e67ba6b510efeea0a567fd0963842ca03297c19ba6d5fd17de959061b69fe66`

| Split | Questions | Groups |
|---|---:|---:|
| train | 14158 | 4990 |
| dev | 1880 | 624 |
| test | 1807 | 624 |

| Class | Questions |
|---|---:|
| Basic information | 5302 |
| Symptoms | 4039 |
| Causes | 2217 |
| Diagnosis and tests | 1891 |
| Treatment | 3221 |
| Prevention | 1175 |

Checks: unique IDs, six-class coverage in every split, disjoint disease groups and normalized questions passed.

Grouping uses normalized focus names and unambiguous explicitly annotated synonym-to-focus links, rejecting disjoint annotated CUI sets. Identical-question links couple split allocation before deduplication. Shared CUIs alone do not establish disease equivalence.

Unresolved CUI IDs spanning splits: 471. See unresolved_cross_split_cuis.csv; complete concept isolation is not claimed. MedQuAD Disease annotations are accepted as supplied and may include broad health topics. Clinical scope and synonym audit remain limitations.

Same-text conflicting labels are quarantined pending human review. See label_conflicts.csv and row_disposition.csv. Missing answers do not cause exclusion. Template replacement is used only for auditing.

Pending user confirmation: instructor dataset approval, cross-course reuse permission and permitted generative-AI assistance. No approval is inferred. Human-written/reviewed paraphrases have not been completed.
