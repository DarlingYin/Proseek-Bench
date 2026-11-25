"""Contract tests for the only active Need evaluator: Score Twice."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matching import maximum_weight_matching, route_global_top1
from whole_file_judge import METHODS, WholeFilePromptBundle
from judge_client import normalize_score_twice_transport


PROMPTS = Path(__file__).resolve().parents[2] / "prompts/evaluate"


def matrix_payload() -> dict:
    return {
        "prediction": {"predictions": [{
            "prediction_id": "p0001", "need": "Correct the existing destination.",
            "claims": "A destination conflict is visible.", "committed_after": "o002",
        }]},
        "ground_truth": {"needs": [{
            "reference_id": "r0001", "gt_need_id": "gt-001", "alternative_id": None,
            "need": "Correct the existing destination and preserve the current reservation.",
            "need_claims": "The destination is stale.", "commit_time_sets": ["o002"],
        }]},
        "pairs": [{"prediction_id": "p0001", "reference_id": "r0001"}],
    }


class ActiveScoreTwiceTests(unittest.TestCase):
    def test_only_score_twice_is_active(self):
        self.assertEqual(METHODS, ("score-twice",))
        versions = sorted(path.name for path in (PROMPTS / "score-twice").iterdir()
                          if path.is_dir() and not path.name.startswith("._"))
        self.assertEqual(versions, ["v1"])

    def test_frozen_global_router_filters_collapses_and_matches_one_to_one(self):
        def edge(prediction, option, gt, level, weight, alternative=None):
            return {
                "prediction_index": prediction, "option_index": option,
                "gt_need_id": gt, "alternative_id": alternative,
                "level": level, "route_weight": weight,
            }

        entries = [
            edge(1, 1, "g1", 2, 2, "a"),
            edge(1, 2, "g1", 5, 5, "b"),
            edge(1, 3, "g2", 5, 5),
            edge(2, 1, "g1", 4, 4, "a"),
            edge(2, 3, "g2", 1, 1),
        ]
        selected, statuses = route_global_top1(entries)
        self.assertEqual(
            {(item["prediction_index"], item["gt_need_id"]) for item in selected},
            {(2, "g1"), (1, "g2")},
        )
        self.assertEqual(statuses[(1, 1)], "filtered_score_0_2")
        self.assertEqual(statuses[(2, 3)], "filtered_score_0_2")

    def test_prompts_use_generic_semantic_slots(self):
        prefilter = (PROMPTS / "score-twice/v1/prefilter.md").read_text()
        isolated = (PROMPTS / "score-twice/v1/isolated.md").read_text()
        self.assertIn("primary object", prefilter)
        self.assertIn("repair direction", prefilter)
        self.assertIn("explicit authority", prefilter)
        self.assertIn("intended help outcome", isolated)
        self.assertIn("primary target", isolated)
        self.assertIn("desired change", isolated)
        for text in (prefilter, isolated):
            for scenario_token in ("Evening Folk", "Harbor", "G7210", "QR-B60", "Taodian"):
                self.assertNotIn(scenario_token, text)
        self.assertIn("high-recall router", prefilter)
        self.assertIn("MUST be at least 3", prefilter)

    def test_prefilter_rejection_requires_two_sided_exact_evidence(self):
        bundle = WholeFilePromptBundle(PROMPTS, "score-twice")
        payload = matrix_payload()
        result = {
            "prediction_id": "p0001", "reference_id": "r0001", "score": 2,
            "reason": "The preserved state is explicitly replaced.",
            "material_deviation": {
                "conflict_slot": "repair_direction",
                "gt_requirement": "preserve the current reservation",
                "prediction_evidence": "Correct the existing destination",
                "consequence": "The destination may change while the required reservation state is unresolved.",
            },
        }
        bundle.validate("need_prefilter", payload, {"results": [result]})
        bad = {**result, "material_deviation": {**result["material_deviation"],
               "prediction_evidence": "text not present in prediction"}}
        with self.assertRaisesRegex(ValueError, "Prediction-Need substring"):
            bundle.validate("need_prefilter", payload, {"results": [bad]})

    def test_prefilter_acceptance_has_no_deviation(self):
        bundle = WholeFilePromptBundle(PROMPTS, "score-twice")
        payload = matrix_payload()
        for score in (3, 4, 5):
            result = {"prediction_id": "p0001", "reference_id": "r0001",
                      "score": score, "reason": "Same normalized task.",
                      "material_deviation": None}
            bundle.validate("need_prefilter", payload, {"results": [result]})
            self.assertTrue(bundle.normalize("need_prefilter", result)["pass"])

    def test_unique_curly_quote_witness_is_repaired_to_exact_substring(self):
        bundle = WholeFilePromptBundle(PROMPTS, "score-twice")
        payload = {
            "prediction": {"predictions": [{
                "prediction_id": "p0001", "need": "Open Song Qi’s current post.",
                "claims": "Visible post.", "committed_after": "o001",
            }]},
            "ground_truth": {"needs": [{
                "reference_id": "r0001", "gt_need_id": "gt-001", "alternative_id": None,
                "need": "Open Song Qi’s archived post.", "need_claims": "Archived post.",
                "commit_time_sets": ["o001"],
            }]},
            "pairs": [{"prediction_id": "p0001", "reference_id": "r0001"}],
        }
        verdict = {"results": [{
            "prediction_id": "p0001", "reference_id": "r0001", "score": 2,
            "reason": "Different object state.",
            "material_deviation": {
                "conflict_slot": "object", "gt_requirement": "Song Qi's archived post",
                "prediction_evidence": "Song Qi's current post",
                "consequence": "The archived post may remain unopened.",
            },
        }]}
        changes = bundle.repair_substring_witnesses("need_prefilter", payload, verdict)
        self.assertEqual(verdict["results"][0]["material_deviation"]["gt_requirement"],
                         "Song Qi’s archived post")
        self.assertEqual(verdict["results"][0]["material_deviation"]["prediction_evidence"],
                         "Song Qi’s current post")
        self.assertEqual(len(changes), 2)
        bundle.validate("need_prefilter", payload, verdict)

    def test_unique_ellipsis_witness_is_expanded_to_exact_span(self):
        bundle = WholeFilePromptBundle(PROMPTS, "score-twice")
        payload = matrix_payload()
        payload["prediction"]["predictions"][0]["need"] = (
            "Advise the courier at the pickup step whether the delivery can still be completed."
        )
        verdict = {"results": [{
            "prediction_id": "p0001", "reference_id": "r0001", "score": 2,
            "reason": "Different action.",
            "material_deviation": {
                "conflict_slot": "action",
                "gt_requirement": "Correct the existing destination",
                "prediction_evidence": "Advise the courier ... whether the delivery",
                "consequence": "The destination correction may remain undone.",
            },
        }]}
        changes = bundle.repair_substring_witnesses("need_prefilter", payload, verdict)
        self.assertEqual(
            verdict["results"][0]["material_deviation"]["prediction_evidence"],
            "Advise the courier at the pickup step whether the delivery",
        )
        self.assertTrue(changes)
        bundle.validate("need_prefilter", payload, verdict)

    def test_isolated_scale_accepts_only_three_and_four(self):
        bundle = WholeFilePromptBundle(PROMPTS, "score-twice")
        payload = {
            "prediction": {"prediction_id": "p0001", "need": "Correct the destination."},
            "ground_truth": {"reference_id": "r0001", "need": "Correct the destination."},
        }
        for score in (1, 2, 3, 4):
            result = {"prediction_id": "p0001", "reference_id": "r0001",
                      "score": score, "reason": "Fixture verdict."}
            bundle.validate("need_isolated", payload, {"results": [result]})
            self.assertEqual(bundle.normalize("need_isolated", result)["pass"], score >= 3)

    def test_global_assignment_is_one_to_one(self):
        selected = maximum_weight_matching([
            {"prediction_index": 1, "gt_need_id": "a", "alternative_id": None, "match_score": 4},
            {"prediction_index": 1, "gt_need_id": "b", "alternative_id": None, "match_score": 3},
            {"prediction_index": 2, "gt_need_id": "a", "alternative_id": None, "match_score": 3},
        ])
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({edge["prediction_index"] for edge in selected}), 2)
        self.assertEqual(len({edge["gt_need_id"] for edge in selected}), 2)

    def test_transport_repair_is_format_only(self):
        verdict, changes = normalize_score_twice_transport("need_prefilter", [{
            "pair_id": "p0001-r0001", "score": 2,
            "material_deviation": {
                "conflict_slot": "assistance action",
                "gt_requirement": "Correct the existing destination",
                "prediction_evidence": "Correct the existing destination",
                "consequence": "A different action would leave the repair undone.",
            },
        }])
        self.assertEqual(verdict["results"][0]["prediction_id"], "p0001")
        self.assertEqual(verdict["results"][0]["reference_id"], "r0001")
        self.assertEqual(verdict["results"][0]["score"], 2)
        self.assertEqual(verdict["results"][0]["material_deviation"]["conflict_slot"], "action")
        self.assertIn("wrap_bare_result_array", changes)
        direct, direct_changes = normalize_score_twice_transport("need_isolated", {
            "prediction_id": "p0001", "reference_id": "r0001",
            "score": 3, "reason": "Same task.",
        })
        self.assertEqual(direct["results"][0]["score"], 3)
        self.assertIn("wrap_direct_result_object", direct_changes)


if __name__ == "__main__":
    unittest.main()
