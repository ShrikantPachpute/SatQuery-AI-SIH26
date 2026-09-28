"""Automated checks for the workflow router used by the Streamlit app."""

import unittest

from core.routing import (
    WORKFLOW_CHANGE,
    WORKFLOW_GROUNDING,
    WORKFLOW_SAR,
    WORKFLOW_UNSUPPORTED,
    WORKFLOW_VQA,
    route_query,
)


class RoutingTests(unittest.TestCase):
    def test_change_query_requires_both_images(self):
        result = route_query("What changed between these dates?", object(), None)
        self.assertEqual(result["workflow"], WORKFLOW_UNSUPPORTED)
        self.assertIn("later image", result["reason"])

    def test_change_query_routes_with_pair(self):
        result = route_query("What changed between these dates?", object(), object())
        self.assertEqual(result["workflow"], WORKFLOW_CHANGE)
        self.assertEqual(result["tool_name"], "Change Detection Specialist")

    def test_single_image_description_routes_but_does_not_claim_analysis(self):
        result = route_query("Describe the scene", object(), None)
        self.assertEqual(result["workflow"], WORKFLOW_VQA)
        self.assertIn("No description has been generated", result["reason"])

    def test_grounding_routes_but_does_not_claim_analysis(self):
        result = route_query("Highlight the water body", object(), None)
        self.assertEqual(result["workflow"], WORKFLOW_GROUNDING)
        self.assertIn("has not been run", result["reason"])

    def test_sar_keyword_with_two_images_routes_without_claiming_analysis(self):
        result = route_query("Use SAR with optical", object(), object())
        self.assertEqual(result["workflow"], WORKFLOW_SAR)
        self.assertIn("No image analysis has been performed", result["reason"])

    def test_unmatched_query_is_unsupported(self):
        result = route_query("Hello there", None, None)
        self.assertEqual(result["workflow"], WORKFLOW_UNSUPPORTED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
