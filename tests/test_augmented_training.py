"""Check provenance, review policy and architecture for M4/M5 training."""
import unittest
import pandas as pd
from scripts.prepare_training_review import FIELDS
from scripts.train_augmented_models import augmented_table, eligible_rows, signature
from src.common import config, text_key
from src.models import build_model


class AugmentedTrainingTests(unittest.TestCase):
    def fixture(self):
        parents, variants = [], []
        for i, (qtype, label) in enumerate(config()["label_map"].items()):
            question = "Original fixture question " + str(i)
            parents.append({"question_id": "p" + str(i), "question_text": question, "text_key": text_key(question),
                            "question_text_original": question, "label": label, "qtype": qtype,
                            "disease_group": "group" + str(i), "split": "train", "source_file": "fixture" + str(i) + ".xml",
                            "source_url": "https://example.invalid/fixture/" + str(i), "analysis_template": "fixture"})
            row = dict.fromkeys(FIELDS, "")
            row.update(parent_question_id="p" + str(i), original_text=question,
                       paraphrase_text="Reworded fixture question " + str(i), label=label,
                       disease_group="group" + str(i), review_status="pending", author="synthetic AI fixture",
                       origin="ai_generated", human_review_status="not_reviewed", generation_method="synthetic test only")
            variants.append(row)
        parents.append({**parents[0], "question_id": "dev-parent", "question_text": "Held-out fixture question", "text_key": "held-out fixture question", "disease_group": "held-out", "split": "dev"})
        return pd.DataFrame(parents), pd.DataFrame(variants)

    def test_default_policy_excludes_unreviewed_ai(self):
        frame, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, "Eligible paraphrases"):
            eligible_rows(rows, frame, "human_reviewed")

    def test_explicit_ai_policy_preserves_pending_status(self):
        frame, rows = self.fixture()
        selected, origin = eligible_rows(rows, frame, "ai_exploratory")
        self.assertEqual(origin, "ai_generated_unreviewed")
        self.assertTrue(selected.review_status.eq("pending").all())
        self.assertTrue(selected.reviewer.eq("").all())

    def test_ai_policy_requires_origin_records(self):
        frame, rows = self.fixture()
        rows.loc[0, "origin"] = "unknown"
        with self.assertRaisesRegex(ValueError, "origin/review"):
            eligible_rows(rows, frame, "ai_exploratory")

    def test_cross_split_text_is_rejected(self):
        frame, rows = self.fixture()
        rows.loc[0, "paraphrase_text"] = "HELD-OUT fixture question"
        with self.assertRaisesRegex(ValueError, "Dev/Test"):
            eligible_rows(rows, frame, "ai_exploratory")

    def test_original_ids_and_source_provenance_are_retained(self):
        frame, rows = self.fixture()
        selected, origin = eligible_rows(rows, frame, "ai_exploratory")
        augmented = augmented_table(frame, selected, origin)
        self.assertEqual(len(augmented), 12)
        self.assertEqual(set(augmented.split), {"train"})
        original = augmented[augmented.row_origin == "original"]
        self.assertEqual(original.question_id.tolist(), ["p" + str(i) for i in range(6)])
        added = augmented[augmented.row_origin != "original"]
        parents = frame.set_index("question_id").loc[added.parent_question_id]
        self.assertEqual(added.source_file.tolist(), parents.source_file.tolist())
        self.assertEqual(added.label.tolist(), parents.label.tolist())
        self.assertFalse(augmented.question_id.duplicated().any())
        self.assertNotIn("dev-parent", augmented.question_id.tolist())

    def test_architecture_matches_each_baseline(self):
        self.assertEqual(signature(build_model("M2", 1.0)), signature(build_model("M4", 10.0)))
        self.assertEqual(signature(build_model("M3", 0.1)), signature(build_model("M5", 1.0)))


if __name__ == "__main__":
    unittest.main()
