"""Create 180 explicitly unreviewed AI-origin training paraphrases.

The templates and source-specific overrides below were authored with Codex
assistance. Their realization is deterministic. They are not human-written
or human-reviewed materials. The user explicitly authorized their direct use
for an exploratory augmentation run; course permission remains unconfirmed.
"""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.common import config, digest, normalize, read_questions, text_key
from scripts.prepare_training_review import AUDIT, METADATA, WORKSHEET, check_protection, validate_worksheet

OUTPUT = ROOT / "data/paraphrases/train_paraphrases_ai_generated.csv"
RECORD = ROOT / "results/ai_augmentation_generation.json"

TEMPLATES = {
    "Basic information": [
        "Could you explain what {focus} means?", "How would you describe {focus}?",
        "What is meant by the term {focus}?", "Can you give an overview of {focus}?",
        "Could you provide basic information about {focus}?", "What does {focus} refer to?",
        "Can you explain the meaning of {focus}?", "What is a general description of {focus}?"
    ],
    "Symptoms": [
        "What signs or symptoms can occur with {focus}?", "How does {focus} usually present?",
        "Which symptoms are associated with {focus}?", "What symptoms might someone with {focus} experience?",
        "What are the signs of {focus}?", "How can the symptoms of {focus} be described?",
        "What symptoms can {focus} produce?", "Which signs and symptoms are seen in {focus}?"
    ],
    "Causes": [
        "What can cause {focus}?", "Why does {focus} occur?",
        "What leads to {focus}?", "What is responsible for causing {focus}?",
        "How does {focus} develop?", "What are the underlying causes of {focus}?",
        "Why might someone develop {focus}?", "What factors cause {focus}?"
    ],
    "Diagnosis and tests": [
        "How is {focus} diagnosed?", "What examinations or tests are used to diagnose {focus}?",
        "How do clinicians establish a diagnosis of {focus}?", "What is involved in diagnosing {focus}?",
        "How can a diagnosis of {focus} be confirmed?", "What diagnostic methods are used for {focus}?",
        "Which tests can help identify {focus}?", "How do doctors check for {focus}?"
    ],
    "Treatment": [
        "What treatment options are available for {focus}?", "How can {focus} be treated?",
        "Which treatments are used for {focus}?", "What can be done to manage {focus}?",
        "What are the available approaches to treating {focus}?", "How is {focus} managed?",
        "What therapies are available for {focus}?", "What does treatment for {focus} involve?"
    ],
    "Prevention": [
        "What can be done to prevent {focus}?", "How can the risk of {focus} be reduced?",
        "What steps help prevent {focus}?", "How can someone avoid developing {focus}?",
        "What measures can help prevent {focus}?", "How can {focus} be prevented from occurring?",
        "What are the ways to reduce the chance of {focus}?", "What actions can help prevent {focus}?"
    ]
}

# Preserve the particular subject of rare source templates rather than turning
# history, government action, or mixed diagnosis/treatment questions into a
# generic definition or treatment question.
OVERRIDES = {
    "9_CDC_QA/0000212.xml::0000212-5": "What is known about the history of HPS in relation to {focus}?",
    "9_CDC_QA/0000054.xml::0000054-10": "Could you explain what {focus} is?",
    "9_CDC_QA/0000439.xml::0000439-11": "Could you explain yersiniosis as it relates to {focus}?",
    "9_CDC_QA/0000273.xml::0000273-19": "What action is the government taking on diseases associated with {focus}?",
    "6_NINDS_QA/0000007.xml::0000007-1": "What is meant by {focus} syndrome?",
    "6_NINDS_QA/0000018.xml::0000018-1": "What is meant by amyotrophic lateral sclerosis, also called {focus}?",
    "6_NINDS_QA/0000182.xml::0000182-1": "Could you give an overview of {focus}?",
    "10_MPlus_ADAM_QA/0000048.xml::0000048-2": "Could you provide general information about the topic {focus}?",
    "5_NIDDK_QA/0000048.xml::0000048-2": "Could you provide general information about the topic {focus}?",
    "10_MPlus_ADAM_QA/0004301.xml::0004301-1": "Could you explain the topic {focus}?",
    "9_CDC_QA/0000054.xml::0000054-14": "How do clinicians diagnose {focus}?",
    "9_CDC_QA/0000273.xml::0000273-14": "How are diseases associated with {focus} diagnosed?",
    "9_CDC_QA/0000439.xml::0000439-15": "How are Y. enterocolitica infections related to {focus} diagnosed?",
    "9_CDC_QA/0000054.xml::0000054-15": "Which treatment options are available for {focus}?",
    "9_CDC_QA/0000439.xml::0000439-16": "How are Y. enterocolitica infections related to {focus} treated?",
    "9_CDC_QA/0000212.xml::0000212-2": "How is HPS related to {focus} diagnosed, and how is it treated?",
    "9_CDC_QA/0000273.xml::0000273-15": "What treatments are used for diseases associated with {focus}?",
    "6_NINDS_QA/0000018.xml::0000018-2": "Are there any treatment options for {focus}?",
    "6_NINDS_QA/0000007.xml::0000007-2": "Is any treatment available for {focus}?",
    "6_NINDS_QA/0000182.xml::0000182-2": "Are there treatments available for {focus}?",
    "9_CDC_QA/0000273.xml::0000273-18": "What can I do to avoid poisoning from {focus}?",
    "9_CDC_QA/0000054.xml::0000054-17": "What measures can prevent {focus}?",
    "9_CDC_QA/0000212.xml::0000212-4": "What can be done to prevent HPS related to {focus}?",
    "9_CDC_QA/0000439.xml::0000439-19": "What are public health agencies doing to prevent or control yersiniosis related to {focus}?",
    "9_CDC_QA/0000273.xml::0000273-20": "What additional measures can prevent diseases associated with {focus}?",
    "9_CDC_QA/0000054.xml::0000054-18": "What actions are public health agencies taking to prevent or control {focus}?"
}


def generate():
    if OUTPUT.exists() or RECORD.exists():
        raise ValueError("AI generation output already exists; preserve the existing version.")
    frame = read_questions()
    source = pd.read_csv(WORKSHEET, keep_default_na=False)
    validate_worksheet(source, frame)
    selection = json.loads((AUDIT / "selection.json").read_text())
    if source.parent_question_id.tolist() != selection["selected_parent_ids"] or digest(METADATA) != selection["metadata_sha256"]:
        raise ValueError("Frozen candidate IDs or provenance changed.")
    protection = json.loads((AUDIT / "baseline_snapshot.json").read_text())
    check_protection(protection)
    original_hash = digest(WORKSHEET)
    parents = frame.set_index("question_id")
    rows, counts, seen = [], {}, set(frame.question_text.map(text_key))
    for row in source.itertuples():
        parent = parents.loc[row.parent_question_id]
        count = counts.get(row.label, 0)
        counts[row.label] = count + 1
        template = OVERRIDES.get(row.parent_question_id, TEMPLATES[row.label][count % len(TEMPLATES[row.label])])
        text = normalize(template.format(focus=parent.focus))
        # In case a template matches an original question exactly, use a fixed
        # request prefix and repeat the same duplicate check. No Test scores or
        # Test error examples influence wording or template choice.
        if text_key(text) in seen:
            text = "Could you clarify: " + text[0].lower() + text[1:]
        if text_key(text) in seen:
            raise ValueError("Unresolved exact duplicate: " + row.parent_question_id)
        if text_key(parent.focus) not in text_key(text):
            raise ValueError("Disease focus was not retained: " + row.parent_question_id)
        seen.add(text_key(text))
        record = row._asdict()
        record.update(paraphrase_text=text, review_status="pending", review_notes="",
                      author="OpenAI Codex: templates and overrides", reviewer="", reviewed_at_utc="",
                      origin="ai_generated", human_review_status="not_reviewed",
                      generation_method="AI-authored templates with deterministic realization and source-specific overrides",
                      generation_notes="Preserves the annotated source intent; requires human semantic review.",
                      parent_intent_ambiguity_flag="diagnosis_and_treatment_in_original" if row.parent_question_id == "9_CDC_QA/0000212.xml::0000212-2" else "")
        rows.append(record)
    result = pd.DataFrame(rows)
    checks = validate_worksheet(result, frame)
    assert len(result) == 180 and result.groupby("label").size().eq(30).all()
    assert result.review_status.eq("pending").all() and result.reviewer.eq("").all()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    generation = {"rows": len(result), "per_class": counts, "checks": checks,
                  "origin": "ai_generated", "human_reviewed": False,
                  "method": "AI-authored templates with deterministic realization and source-specific overrides",
                  "authorization": "User explicitly requested generation of all 180 candidates and their direct use for training in this chat.",
                  "permitted_use": "exploratory_ai_unreviewed_augmentation",
                  "course_permission": "pending_user_confirmation",
                  "human_reviewed_protocol_satisfied": False,
                  "original_worksheet_sha256": original_hash, "generated_csv_sha256": digest(OUTPUT),
                  "parent_metadata_sha256": digest(METADATA), "generator_sha256": digest(Path(__file__)),
                  "seed": config()["seed"], "ambiguous_source_parent_ids": ["9_CDC_QA/0000212.xml::0000212-2"]}
    with RECORD.open("x", encoding="utf-8") as stream:
        json.dump(generation, stream, indent=2)
        stream.write("\n")
    check_protection(protection)
    assert digest(WORKSHEET) == original_hash
    print(json.dumps(generation, indent=2))
    return result


if __name__ == "__main__":
    generate()
