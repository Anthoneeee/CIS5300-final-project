"""Parse, audit and freeze disease-grouped splits before model development.

Source: abachaa/MedQuAD XML and readme.txt (CC BY 4.0). No answers are saved
or used as model features. Grouping uses explicit names/synonyms, not shared
arbitrary CUIs. All unresolved concept overlap is reported as a limitation.
"""
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .common import ROOT, config, digest, labels, name_key, normalize, save_json, text_key
from .download_data import download

MIXED = {"4_MPlus_Health_Topics_QA", "10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"}


class Groups:
    """Deterministic connected components with a stable lexical representative."""
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        if self.parent[k] != k:
            self.parent[k] = self.find(self.parent[k])
        return self.parent[k]

    def union(self, a, b):
        x, y = sorted([self.find(a), self.find(b)])
        self.parent[y] = x


def node_text(node):
    return "" if node is None else "".join(node.itertext())


def parse_xml(raw):
    rows, issues = [], []
    for path in sorted(raw.glob("*/*.xml")):
        relative = path.relative_to(raw).as_posix()
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as exc:
            issues.append({"source_file": relative, "issue": "parse_failed", "detail": str(exc)})
            continue
        # Four legacy NINDS documents use lower-case tags; one CDC file uses DiseaseFile.
        focus = root.find("Focus")
        if focus is None:
            focus = root.find("doctitle-focus")
        category = root.findtext("FocusAnnotations/Category", "")
        qs = [n for n in root.iter() if n.tag.casefold() == "question"]
        if root.tag != "Document":
            issues.append({"source_file": relative, "issue": "legacy_schema_supported", "detail": root.tag})
        if not qs:
            issues.append({"source_file": relative, "issue": "no_questions", "detail": root.tag})
        synonyms = sorted(set(normalize(node_text(n)) for n in root.iter() if n.tag.casefold() == "synonym" and normalize(node_text(n))))
        cuis = sorted(set(normalize(node_text(n)) for n in root.iter() if n.tag.casefold() == "cui" and normalize(node_text(n))))
        semantic_types = sorted(set(normalize(node_text(n)) for n in root.iter() if n.tag.casefold() == "semantictype" and normalize(node_text(n))))
        for index, q in enumerate(qs, 1):
            qid = q.attrib.get("qid", "")
            if not qid:
                issues.append({"source_file": relative, "issue": "missing_qid", "detail": str(index)})
            text = node_text(q)
            rows.append({
                "question_id": relative + "::" + (qid or "index-{}".format(index)),
                "original_qid": qid, "question_text_original": text,
                "question_text": normalize(text), "qtype": normalize(q.attrib.get("qtype", "")).casefold(),
                "focus": normalize(node_text(focus)), "category": category,
                "scope_basis": "explicit_category" if path.parent.name in MIXED else "upstream_disease_collection",
                "source": root.attrib.get("source", root.attrib.get("corpus", path.parent.name)),
                "source_file": relative, "source_url": root.attrib.get("url", ""),
                "synonyms": json.dumps(synonyms, ensure_ascii=False),
                "concept_ids": json.dumps(cuis), "semantic_types": json.dumps(semantic_types)
            })
    frame = pd.DataFrame(rows)
    if frame.question_id.duplicated().any():
        raise ValueError("Question IDs are not unique; inspect raw files before proceeding.")
    return frame, pd.DataFrame(issues, columns=["source_file", "issue", "detail"])


def assign_groups(frame):
    """Merge focus aliases only when explicit and unambiguous; log every edge.

    Exact repeated questions couple split groups even if their medical synonymy
    is unclear. This is a leakage safeguard, not a clinical equivalence claim.
    """
    names = {name_key(x) for x in frame.focus}
    groups = Groups(names)
    owners, concept_sets = defaultdict(set), defaultdict(set)
    docs = frame.drop_duplicates("source_file")
    for row in docs.itertuples():
        key = name_key(row.focus)
        concept_sets[key].update(json.loads(row.concept_ids))
        for alias in json.loads(row.synonyms):
            owners[name_key(alias)].add(key)
    edges, unresolved = [], []
    for alias, candidates in sorted(owners.items()):
        if alias not in names:
            continue
        other = candidates - {alias}
        if len(other) != 1 or len(alias) < 3:
            if other:
                unresolved.append({"alias": alias, "candidate_focuses": json.dumps(sorted(other)), "reason": "ambiguous_or_short_alias"})
            continue
        target = next(iter(other))
        a, b = concept_sets[alias], concept_sets[target]
        if a and b and not a.intersection(b):
            unresolved.append({"alias": alias, "candidate_focuses": target, "reason": "disjoint_annotated_cuis"})
            continue
        groups.union(alias, target)
        edges.append({"left": alias, "right": target, "basis": "explicit_unambiguous_synonym"})
    for _, repeated in frame.groupby("text_key", sort=True):
        keys = sorted({name_key(x) for x in repeated.focus})
        for other in keys[1:]:
            groups.union(keys[0], other)
            edges.append({"left": keys[0], "right": other, "basis": "identical_question_split_coupling"})
    result = frame.copy()
    result["disease_group"] = [groups.find(name_key(x)) for x in result.focus]
    return result, pd.DataFrame(edges, columns=["left", "right", "basis"]), pd.DataFrame(unresolved, columns=["alias", "candidate_focuses", "reason"])


def split_data(frame, seed):
    """One predeclared seed; 80/10/10 group allocation, never optimize on scores."""
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, hold = next(first.split(frame, groups=frame.disease_group))
    held = frame.iloc[hold]
    second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    dv, te = next(second.split(held, groups=held.disease_group))
    result = frame.copy()
    result["split"] = ""
    result.loc[result.index[tr], "split"] = "train"
    result.loc[held.index[dv], "split"] = "dev"
    result.loc[held.index[te], "split"] = "test"
    validate_splits(result)
    return result


def validate_splits(frame):
    if frame.disease_group.isin([""]).any() or frame.question_id.duplicated().any():
        raise ValueError("Missing groups or duplicate IDs.")
    for key in ["disease_group", "text_key"]:
        if (frame.groupby(key).split.nunique() > 1).any():
            raise ValueError("Cross-split overlap in " + key)
    for split in ["train", "dev", "test"]:
        if set(frame.loc[frame.split == split, "label"]) != set(labels()):
            raise ValueError("Split lacks a class; revise documented split design, not test-score-based seeds.")


def prepare():
    raw = download()
    out, results, splits = ROOT / "data/processed", ROOT / "results", ROOT / "data/splits"
    for p in [out, results, splits]:
        p.mkdir(parents=True, exist_ok=True)
    parsed, issues = parse_xml(raw)
    parsed.to_csv(out / "questions_parsed.csv", index=False)
    issues.to_csv(results / "parse_issues.csv", index=False)
    cfg = config()
    frame = parsed.copy()
    frame["label"] = frame.qtype.map(cfg["label_map"]).fillna("")
    frame["text_key"] = frame.question_text.map(text_key)
    frame["exclusion_reason"] = ""
    mixed = frame.source_file.str.split("/").str[0].isin(MIXED)
    frame.loc[mixed & (frame.category != "Disease"), "exclusion_reason"] = "non_disease_or_unknown_category"
    for condition, reason in [
        (frame.label == "", "outside_six_intents"), (frame.question_text == "", "missing_question"),
        (frame.qtype == "", "missing_qtype"), (frame.focus.map(name_key) == "", "missing_focus")
    ]:
        frame.loc[(frame.exclusion_reason == "") & condition, "exclusion_reason"] = reason
    eligible = frame[frame.exclusion_reason == ""].copy()
    conflicts = eligible.groupby("text_key").label.nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    # Quarantine all instances, rather than arbitrarily picking a target label.
    frame.loc[(frame.exclusion_reason == "") & frame.text_key.isin(conflict_keys), "exclusion_reason"] = "conflicting_labels_pending_review"
    frame[frame.exclusion_reason == "conflicting_labels_pending_review"].to_csv(results / "label_conflicts.csv", index=False)
    grouped, edges, unresolved = assign_groups(frame[frame.exclusion_reason == ""].copy())
    edges.to_csv(results / "grouping_edges.csv", index=False)
    unresolved.to_csv(results / "ambiguous_aliases.csv", index=False)
    representative = grouped.groupby("text_key").question_id.first().to_dict()
    duplicates = grouped[grouped.duplicated("text_key", keep="first")].copy()
    duplicates["retained_question_id"] = duplicates.text_key.map(representative)
    duplicates.to_csv(results / "duplicates.csv", index=False)
    frame.loc[frame.question_id.isin(duplicates.question_id), "exclusion_reason"] = "same_text_same_label_duplicate"
    clean = split_data(grouped.drop_duplicates("text_key", keep="first"), cfg["seed"])
    clean["question_length_words"] = clean.question_text.str.split().str.len()
    # Template analysis only: no masking in the actual classifier inputs.
    clean["analysis_template"] = [re.sub(re.escape(row.focus), "<FOCUS>", row.question_text, flags=re.I) for row in clean.itertuples()]
    clean.to_csv(out / "questions_clean.csv", index=False)
    frame[["question_id", "source_file", "qtype", "category", "exclusion_reason"]].to_csv(results / "row_disposition.csv", index=False)
    clean.groupby(["label", "source"]).size().rename("questions").to_csv(results / "label_source_counts.csv")
    clean.groupby("analysis_template").size().sort_values(ascending=False).rename("questions").to_csv(results / "template_counts.csv")
    clean.groupby("label").agg(questions=("question_id", "size"), disease_groups=("disease_group", "nunique"), median_words=("question_length_words", "median")).to_csv(results / "class_counts.csv")
    clean.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(columns=labels()).to_csv(results / "split_class_counts.csv")
    manifest = clean[["question_id", "disease_group", "split"]].sort_values("question_id")
    frozen = splits / "split_ids.csv"
    if frozen.exists() and frozen.read_text(encoding="utf-8") != manifest.to_csv(index=False):
        raise ValueError("Frozen split would change; preserve existing split and investigate explicitly.")
    manifest.to_csv(frozen, index=False)
    cui_rows = []
    for row in clean.itertuples():
        for cui in json.loads(row.concept_ids):
            cui_rows.append({"cui": cui, "disease_group": row.disease_group, "split": row.split})
    cui_frame = pd.DataFrame(cui_rows, columns=["cui", "disease_group", "split"])
    cui_overlap = cui_frame.groupby("cui").agg(groups=("disease_group", "nunique"), splits=("split", "nunique"))
    cui_overlap[cui_overlap.splits > 1].to_csv(results / "unresolved_cross_split_cuis.csv")
    split_summary = {s: {"questions": int((clean.split == s).sum()), "groups": int(clean[clean.split == s].disease_group.nunique())} for s in ["train", "dev", "test"]}
    audit = {
        "raw_questions": len(parsed), "retained_questions": len(clean),
        "xml_files": len(list(raw.glob("*/*.xml"))), "parsed_issues": issues.issue.value_counts().to_dict(),
        "dispositions": frame.exclusion_reason.replace("", "retained").value_counts().to_dict(),
        "classes": clean.label.value_counts().to_dict(), "disease_groups": int(clean.disease_group.nunique()),
        "splits": split_summary, "cross_split_disease_groups": 0, "cross_split_normalized_duplicate_questions": 0,
        "unresolved_cross_split_cui_count": int((cui_overlap.splits > 1).sum()),
        "grouping_claim": "available annotated focus/synonym grouping; not full medical concept isolation",
        "split_sha256": digest(frozen), "dataset_approval": "pending_user_confirmation",
        "six_classes_fixed_before_modeling": True
    }
    save_json(results / "data_audit.json", audit)
    lines = ["# Data-stage audit", "", "Generated from the real pinned XML; no classifier was evaluated at this stage.", "",
             "- Parsed questions: {}".format(len(parsed)), "- Retained questions: {}".format(len(clean)),
             "- Disease groups: {}".format(audit["disease_groups"]), "- Split SHA-256: `{}`".format(audit["split_sha256"]), "",
             "| Split | Questions | Groups |", "|---|---:|---:|"]
    for s, values in split_summary.items():
        lines.append("| {} | {} | {} |".format(s, values["questions"], values["groups"]))
    lines += ["", "| Class | Questions |", "|---|---:|"]
    for label in labels():
        lines.append("| {} | {} |".format(label, audit["classes"][label]))
    lines += ["", "Checks: unique IDs, six-class coverage in every split, disjoint disease groups and normalized questions passed.", "",
              "Grouping uses normalized focus names and unambiguous explicitly annotated synonym-to-focus links, rejecting disjoint annotated CUI sets. Identical-question links couple split allocation before deduplication. Shared CUIs alone do not establish disease equivalence.", "",
              "Unresolved CUI IDs spanning splits: {}. See unresolved_cross_split_cuis.csv; complete concept isolation is not claimed. MedQuAD Disease annotations are accepted as supplied and may include broad health topics. Clinical scope and synonym audit remain limitations.".format(audit["unresolved_cross_split_cui_count"]), "",
              "Same-text conflicting labels are quarantined pending human review. See label_conflicts.csv and row_disposition.csv. Missing answers do not cause exclusion. Template replacement is used only for auditing.", "",
              "Pending user confirmation: instructor dataset approval, cross-course reuse permission and permitted generative-AI assistance. No approval is inferred. Human-written/reviewed paraphrases have not been completed."]
    (results / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return clean


if __name__ == "__main__":
    prepare()
