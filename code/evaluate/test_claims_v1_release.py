"""Offline checks for the released Claims contract."""

from pathlib import Path
import unittest

from claims_judge import IsolatedClaimsPromptBundle


class ClaimsReleaseTests(unittest.TestCase):
    def test_decisive_unsupported_claim_caps_score(self):
        prompts = Path(__file__).resolve().parents[2] / "prompts/evaluate"
        bundle = IsolatedClaimsPromptBundle(prompts, "v1")
        result = {
            "prediction_id": "p0001",
            "reference_id": "r0001",
            "reference_resolution_type": "information_action",
            "completion_checks": [{
                "requirement": "Identify the current destination",
                "materiality": "decisive",
                "completion_role": "target",
                "prediction_status": "resolved",
                "evidence_status": "unsupported",
                "evidence_source": "none",
                "reason": "The destination was not established.",
            }],
            "completion_status": "complete",
            "score": 4,
            "reason": "The answer offers a destination.",
        }
        bundle.validator.validate({"results": [result]})
        verdict = bundle.normalize("claims_isolated", result)
        self.assertEqual(verdict["score"], 2)
        self.assertFalse(verdict["pass"])
