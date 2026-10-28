"""Test Train-only candidate selection and human-review validation boundaries."""
import unittest
import pandas as pd
from scripts.prepare_training_review import FIELDS, select_candidates, validate_worksheet
from src.common import labels


class TrainingReviewTests(unittest.TestCase):
    def test_selection_is_reproducible_and_covers_templates(self):
        rows = []
        for label in labels():
            for i in range(6):
                rows.append({"question_id": label + str(i), "label": label, "split": "train",
                             "disease_group": "group-" + str(i),
                             "analysis_template": "template-" + str(i % 3)})
        frame = pd.DataFrame(rows)
        one = select_candidates(frame, per_class=3, seed=5220)
        two = select_candidates(frame.sample(frac=1, random_state=17), per_class=3, seed=5220)
        self.assertEqual(one.question_id.tolist(), two.question_id.tolist())
        self.assertTrue(one.groupby("label").size().eq(3).all())
        self.assertTrue(one.groupby("label").template_key.nunique().eq(3).all())
        self.assertFalse(one.question_id.duplicated().any())

    def test_selection_rejects_dev_or_test_inputs(self):
        for split in ["dev", "test"]:
            frame = pd.DataFrame([{"question_id": "held-out", "split": split}])
            with self.assertRaisesRegex(ValueError, "Train-only"):
                select_candidates(frame)

    def fixture(self):
        parents = pd.DataFrame([
            {"question_id": "train-parent", "split": "train", "question_text": "What causes asthma?", "label": "Causes", "disease_group": "asthma"},
            {"question_id": "dev-parent", "split": "dev", "question_text": "Why does flu occur?", "label": "Causes", "disease_group": "flu"}
        ])
        row = dict.fromkeys(FIELDS, "")
        row.update(parent_question_id="train-parent", original_text="What causes asthma?", label="Causes",
                   disease_group="asthma", review_status="pending")
        return parents, row

    def test_blank_pending_is_valid_and_never_approved(self):
        parents, row = self.fixture()
        checks = validate_worksheet(pd.DataFrame([row]), parents)
        self.assertEqual(checks["review_status"], {"pending": 1})
        self.assertEqual(checks["nonblank_paraphrases"], 0)

    def test_pending_dev_duplicate_is_rejected(self):
        parents, row = self.fixture()
        row["paraphrase_text"] = "  WHY does flu occur?  "
        with self.assertRaisesRegex(ValueError, "Dev/Test"):
            validate_worksheet(pd.DataFrame([row]), parents)

    def test_wrong_parent_label_or_split_is_rejected(self):
        parents, row = self.fixture()
        row["label"] = "Treatment"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            validate_worksheet(pd.DataFrame([row]), parents)
        row.update(parent_question_id="dev-parent", original_text="Why does flu occur?", label="Causes", disease_group="flu")
        with self.assertRaisesRegex(ValueError, "frozen Train"):
            validate_worksheet(pd.DataFrame([row]), parents)

    def test_approval_requires_actual_review_record_fields(self):
        parents, row = self.fixture()
        row.update(review_status="approved", paraphrase_text="Why does asthma develop?")
        with self.assertRaisesRegex(ValueError, "Approved rows require"):
            validate_worksheet(pd.DataFrame([row]), parents)


if __name__ == "__main__":
    unittest.main()
