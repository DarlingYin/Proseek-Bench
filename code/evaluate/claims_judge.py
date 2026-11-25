"""Isolated 1--4 Claims prompt contract."""

from __future__ import annotations

from judge_client import file_hash, read_json


class IsolatedClaimsPromptBundle:
    method = "claims-1-4"

    def __init__(self, root, version="v1"):
        from jsonschema import Draft202012Validator

        if version != "v1":
            raise ValueError("This release supports Claims v1")
        self.version = version
        base_version = version
        base = root / "claims" / base_version
        prompt_name = ("isolated.md" if version == "v1" else
                       "completion.md" if version == "v4-completion" else
                       "fixed_completion.md" if version == "v4-fixed" else "isolated.md")
        schema_name = ("isolated.output.schema.json" if version == "v1" else
                       "completion.output.schema.json" if version == "v4-completion" else
                       "fixed_completion.output.schema.json" if version == "v4-fixed" else
                       "isolated.output.schema.json")
        self.paths = {
            "claims_isolated.md": base / prompt_name,
            "claims_isolated.schema.json": base / schema_name,
        }
        self.hashes = {name: file_hash(path) for name, path in self.paths.items()}
        self.text = self.paths["claims_isolated.md"].read_text(encoding="utf-8")
        schema = read_json(self.paths["claims_isolated.schema.json"])
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema)

    def system(self, stage):
        if stage != "claims_isolated":
            raise ValueError("Unknown Claims stage")
        return self.text

    def validate_input(self, stage, inputs):
        if stage != "claims_isolated" or set(inputs) != {"prediction", "ground_truth", "evidence_ledger"}:
            raise ValueError("Unexpected isolated Claims input fields")
        prediction, reference = inputs["prediction"], inputs["ground_truth"]
        if set(prediction) != {"prediction_id", "need", "claims"}:
            raise ValueError("Invalid Prediction Claims envelope")
        if set(reference) != {"reference_id", "need", "claims"}:
            raise ValueError("Invalid Reference Claims envelope")
        if prediction["prediction_id"] != "p0001" or reference["reference_id"] != "r0001":
            raise ValueError("Isolated Claims IDs must be canonical")
        for owner, item in (("Prediction", prediction), ("Reference", reference)):
            for field in ("need", "claims"):
                if not isinstance(item[field], str) or not item[field].strip():
                    raise ValueError(f"{owner} {field} must be nonempty")
        ledger = inputs["evidence_ledger"]
        expected_ledger = {
            "evidence_contract", "commit_context", "source_policy",
            "query_evidence_before_commit", "temporal_policy",
            "reference_causal_target",
        }
        if self.version in {"v1", "v2", "v3", "v4", "v4-completion", "v4-fixed"}:
            expected_ledger.add("passive_evidence_before_commit")
        if self.version in {"v1", "v3", "v4", "v4-completion", "v4-fixed"}:
            expected_ledger.add("passive_visual_verification")
        if not isinstance(ledger, dict) or set(ledger) != expected_ledger:
            raise ValueError("Invalid text Evidence Ledger")
        source_policy = ledger["source_policy"]
        if (not isinstance(source_policy, dict)
                or set(source_policy) != {
                    "sampler_capability", "source_tier", "reference_role",
                    "thread_linkage_policy",
                }
                or source_policy["sampler_capability"] not in {
                    "GUI-Only", "Active", "Long-Term",
                }):
            raise ValueError("Invalid Claims source policy")
        if True:
            reference = ledger["reference_causal_target"]
            if (not isinstance(reference, dict)
                    or set(reference) != {"claims", "valid_for_commit_window", "admissibility"}
                    or not isinstance(reference["valid_for_commit_window"], list)):
                raise ValueError("Invalid reference causal target")
            passive = ledger["passive_evidence_before_commit"]
            if (not isinstance(passive, dict)
                    or set(passive) != {"extraction", "observations"}
                    or not isinstance(passive["observations"], list)
                    or not passive["observations"]):
                raise ValueError("Invalid Passive evidence ledger")
            if self.version in {"v1", "v3", "v4", "v4-completion", "v4-fixed"}:
                visual = ledger["passive_visual_verification"]
                if not isinstance(visual, list):
                    raise ValueError("Invalid Passive visual verification list")
                for item in visual:
                    if (not isinstance(item, dict)
                            or item.get("status") not in {"visible", "not_visible", "unclear"}
                            or not isinstance(item.get("premise"), str)
                            or not isinstance(item.get("supporting_observation_ids"), list)):
                        raise ValueError("Invalid Passive visual verification receipt")
        query_evidence = ledger["query_evidence_before_commit"]
        if (not isinstance(query_evidence, dict)
                or set(query_evidence) != {"thread_linked", "other_visible"}
                or not all(isinstance(query_evidence[key], list) for key in query_evidence)):
            raise ValueError("Query evidence must be partitioned lists")

    def validate(self, stage, inputs, output):
        self.validate_input(stage, inputs)
        self.validator.validate(output)
        result = output["results"][0]
        if result["prediction_id"] != "p0001" or result["reference_id"] != "r0001":
            raise ValueError("Claims verdict IDs do not match the isolated pair")
        if self.version in {"v1", "v4-completion", "v4-fixed"}:
            return
        if result["score"] != min(result["grounding_score"], result["sufficiency_score"]):
            raise ValueError("Final Claims score must be the minimum component score")
        decisive_premise_failure = any(
            check["materiality"] == "decisive" and check["status"] != "supported"
            for check in result["premise_checks"]
        )
        failed_gate = (
                decisive_premise_failure
                or any(check["status"] != "supported" for check in result["action_checks"])
                or result["temporal_consistency_status"] != "supported"
                or result["unresolved_gap_status"] != "established"
                or result["applicability_status"] not in {"established", "conditional_preserved"}
                or result["intervention_strength_status"] != "supported"
        )
        # v4 accepts schema-valid raw scores and applies deterministic semantic
        # caps in normalize(). This avoids transport retries when the Judge's
        # aggregate score disagrees with its own structured requirement fields.
        if self.version != "v4" and failed_gate and result["score"] > 2:
            raise ValueError("A failed Claims gate cannot receive a passing score")

    def repair_substring_witnesses(self, stage, inputs, output):
        """Claims v1 has no exact-substring transport fields to repair."""
        return []

    def normalize(self, stage, result):
        if stage != "claims_isolated":
            raise ValueError("Unknown Claims stage")
        if self.version == "v4-fixed":
            score = result["score"]
            normalization = []
            contradicted = (
                result["action_basis_evidence"] == "contradicted"
                or result["gap_evidence"] == "contradicted"
                or result["boundary_coverage"] == "violated"
                or result["target_coverage"] == "different"
                or result["completion_status"] in {"different", "contradicted"}
            )
            incomplete = (
                result["target_coverage"] != "complete"
                or result["action_basis_evidence"] not in {"supported", "not_required"}
                or result["gap_evidence"] not in {"supported", "not_required"}
                or result["boundary_coverage"] not in {"preserved", "not_required"}
                or result["completion_status"] != "complete"
            )
            if contradicted and score > 1:
                score = 1
                normalization.append("fixed_completion_contradiction_caps_1")
            elif incomplete and score > 2:
                score = 2
                normalization.append("fixed_completion_incomplete_caps_2")
            return {"prompt_version": self.version, "method": "claims-fixed-completion-1-4",
                    "stage": stage, "status": "evaluated", "pass": score >= 3,
                    "score": score, "normalization": normalization,
                    "completion_status": result["completion_status"],
                    "reference_resolution_type": result["reference_resolution_type"],
                    "reason": result["reason"], "raw_result": result}
        if self.version in {"v1", "v4-completion"}:
            score = result["score"]
            normalization = []
            decisive = [check for check in result["completion_checks"]
                        if check["materiality"] == "decisive"]
            if any(check["evidence_status"] == "contradicted" for check in decisive):
                if score > 1:
                    score = 1
                    normalization.append("decisive_completion_contradiction_caps_1")
            elif (any(check["prediction_status"] != "resolved" for check in decisive)
                  or any(check["evidence_status"] == "unsupported" for check in decisive)
                  or result["completion_status"] != "complete"):
                if score > 2:
                    score = 2
                    normalization.append("incomplete_decisive_completion_caps_2")
            return {
                "prompt_version": self.version,
                "method": "claims-completion-1-4",
                "stage": stage,
                "status": "evaluated",
                "pass": score >= 3,
                "score": score,
                "normalization": normalization,
                "completion_status": result["completion_status"],
                "reference_resolution_type": result["reference_resolution_type"],
                "reason": result["reason"],
                "raw_result": result,
            }
        grounding_score = result["grounding_score"]
        sufficiency_score = result["sufficiency_score"]
        normalization = []
        if self.version in {"v3", "v4"}:
            decisive = [check for check in result["premise_checks"]
                        if check["materiality"] == "decisive"]
            if any(check["status"] == "contradicted" for check in decisive):
                if grounding_score > 1:
                    grounding_score = 1
                    normalization.append("decisive_contradiction_caps_grounding_1")
            elif any(check["status"] == "unsupported" for check in decisive):
                if grounding_score > 2:
                    grounding_score = 2
                    normalization.append("decisive_unsupported_caps_grounding_2")
            elif any(check["status"] != "supported" for check in result["premise_checks"]):
                if grounding_score > 3:
                    grounding_score = 3
                    normalization.append("nondecisive_failure_caps_grounding_3")
            failed_gate = (
                any(check["status"] != "supported" for check in result["action_checks"])
                or result["temporal_consistency_status"] != "supported"
                or result["unresolved_gap_status"] != "established"
                or result["applicability_status"] not in {"established", "conditional_preserved"}
                or result["intervention_strength_status"] != "supported"
            )
            if failed_gate and sufficiency_score > 2:
                sufficiency_score = 2
                normalization.append("failed_causal_gate_caps_sufficiency_2")
        if self.version == "v4":
            decisive_requirements = [
                check for check in result["reference_requirements"]
                if check["materiality"] == "decisive"
            ]
            if any(check["evidence_status"] == "contradicted"
                   for check in decisive_requirements):
                if grounding_score > 1:
                    grounding_score = 1
                    normalization.append("decisive_reference_contradiction_caps_grounding_1")
            elif any(check["evidence_status"] == "unsupported"
                     for check in decisive_requirements):
                if grounding_score > 2:
                    grounding_score = 2
                    normalization.append("unsupported_decisive_evidence_caps_grounding_2")

            derived_coverage = "complete"
            if any(check["prediction_coverage"] == "contradictory"
                   for check in decisive_requirements):
                derived_coverage = "different"
            elif any(check["prediction_coverage"] in {"partial", "absent"}
                     for check in decisive_requirements):
                derived_coverage = "partial"
            raw_coverage = result["reference_action_coverage"]
            if "different" in {derived_coverage, raw_coverage}:
                effective_coverage = "different"
            elif "partial" in {derived_coverage, raw_coverage}:
                effective_coverage = "partial"
            else:
                effective_coverage = "complete"

            if (result["evidence_completion_status"] == "contradicted"
                    or any(check["evidence_status"] == "contradicted"
                           for check in decisive_requirements)):
                effective_evidence = "contradicted"
            elif (result["evidence_completion_status"] == "incomplete"
                  or any(check["evidence_status"] == "unsupported"
                         for check in decisive_requirements)):
                effective_evidence = "incomplete"
            elif (result["evidence_completion_status"] == "not_required"
                  and all(check["evidence_status"] == "not_required"
                          for check in decisive_requirements)):
                effective_evidence = "not_required"
            else:
                effective_evidence = "complete"

            resolution_failure = (
                effective_coverage != "complete"
                or effective_evidence not in {"complete", "not_required"}
                or bool(result["missing_decisive_requirements"])
            )
            if resolution_failure and sufficiency_score > 2:
                sufficiency_score = 2
                normalization.append("incomplete_reference_resolution_caps_sufficiency_2")
        score = min(grounding_score, sufficiency_score)
        normalized = {
            "prompt_version": self.version,
            "method": self.method,
            "stage": stage,
            "status": "evaluated",
            "pass": score >= 3,
            "grounding_score": grounding_score,
            "sufficiency_score": sufficiency_score,
            "score": score,
            "normalization": normalization,
            "reason": result["reason"],
            "raw_result": result,
        }
        if self.version == "v4":
            normalized["reference_action_coverage"] = effective_coverage
            normalized["evidence_completion_status"] = effective_evidence
        return normalized
