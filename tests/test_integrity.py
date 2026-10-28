"""Scientific safeguards: train-only fitting, legacy XML and review boundaries."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.common import labels
from src.models import RuleClassifier, build_model
from src.prepare_data import parse_xml, validate_splits
from src.variants import FIELDS, reviewed_variants, spelling_variant


class IntegrityTests(unittest.TestCase):
    def test_legacy_schema_and_answers_are_not_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "6_NINDS_QA"
            folder.mkdir()
            (folder / "x.xml").write_text('<doc corpus="NINDS"><doctitle-focus>Asthma</doctitle-focus><qaPairs><pair><question qid="1" qtype="information">What is asthma?</question><answer>ANSWER_LEAK</answer></pair></qaPairs></doc>')
            frame, issues = parse_xml(Path(tmp))
            self.assertEqual(frame.question_text.tolist(), ["What is asthma?"])
            self.assertNotIn("ANSWER_LEAK", frame.to_csv())
            self.assertEqual(issues.issue.tolist(), ["legacy_schema_supported"])

    def test_test_vocabulary_never_fitted(self):
        for model_id in ["M2", "M3"]:
            model = build_model(model_id)
            model.fit(["what is asthma", "what causes asthma", "what is flu", "what causes flu"], ["basic", "causes", "basic", "causes"])
            features = model.named_steps["features"]
            vectorizer = features if model_id == "M2" else features.transformer_list[0][1]
            before = vectorizer.vocabulary_.copy()
            model.predict(["unseenheldoutcanary test word"])
            self.assertEqual(vectorizer.vocabulary_, before)
            self.assertNotIn("unseenheldoutcanary", vectorizer.vocabulary_)

    def test_rules_fallback_uses_training_majority(self):
        model = RuleClassifier().fit(["a", "b", "c"], ["Symptoms", "Symptoms", "Causes"])
        self.assertEqual(model.predict(["unrecognized cue"])[0], "Symptoms")

    def test_cross_split_duplicate_is_rejected(self):
        rows = []
        for split in ["train", "dev", "test"]:
            for i, label in enumerate(labels()):
                rows.append({"question_id": split + str(i), "disease_group": split + str(i), "text_key": split + str(i), "label": label, "split": split})
        frame = pd.DataFrame(rows)
        validate_splits(frame)
        frame.loc[6, "text_key"] = frame.loc[0, "text_key"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_splits(frame)

    def test_pending_variants_and_wrong_split(self):
        frame = pd.DataFrame([{"question_id": "p", "split": "test", "label": "Symptoms", "disease_group": "asthma", "question_text": "What are asthma symptoms?", "text_key": "what are asthma symptoms?"}])
        row = dict.fromkeys(FIELDS, "")
        row.update(variant_id="v", parent_question_id="p", variant_type="paraphrase", original_text="What are asthma symptoms?", text="Which signs indicate asthma?", label="Symptoms", disease_group="asthma", split="train", review_status="pending")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "variants.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertTrue(reviewed_variants(path, "train", frame).empty)
            row.update(review_status="approved", author="test fixture", reviewer="test fixture", review_notes="synthetic fixture only")
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "split"):
                reviewed_variants(path, "train", frame)

    def test_spelling_preserves_disease_and_negation(self):
        text, change = spelling_variant("What are the symptoms of symptoms syndrome, not flu?", "symptoms syndrome")
        self.assertEqual(text, "")  # The cue is part of the protected focus.
        text, change = spelling_variant("What are the symptoms of asthma, not flu?", "asthma")
        self.assertIn("asthma, not flu?", text)
        self.assertIn("transposition", change)


if __name__ == "__main__":
    unittest.main()
