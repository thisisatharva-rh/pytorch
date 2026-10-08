import copy
import json
from pathlib import Path

from torch.distributed._pysymmem.experimental.benchmarks.benchmark import (
    summarize,
    validate_manifest,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestHarness(TestCase):
    def setUp(self):
        super().setUp()
        self.manifest = json.loads(
            (
                Path(__file__).resolve().parents[1] / "benchmarks/cases/smoke.json"
            ).read_text()
        )

    def test_valid_manifest(self):
        self.assertEqual(validate_manifest(self.manifest), self.manifest)

    @parametrize(
        "failure",
        [
            "world_size",
            "negative_count",
            "boolean_count",
            "duplicate_name",
            "unequal_allreduce",
            "invalid_dtype",
            "invalid_mode",
            "zero_trials",
        ],
    )
    def test_invalid_manifest(self, failure):
        manifest = copy.deepcopy(self.manifest)
        if failure == "world_size":
            manifest["world_size"] = 0
        elif failure == "negative_count":
            manifest["cases"][0]["counts"][0] = -1
        elif failure == "boolean_count":
            manifest["cases"][0]["counts"][0] = True
        elif failure == "duplicate_name":
            manifest["cases"][1]["name"] = "ar"
        elif failure == "unequal_allreduce":
            manifest["cases"][0]["counts"][0] = 1
        elif failure == "invalid_dtype":
            manifest["cases"][0]["dtype"] = "int64"
        elif failure == "invalid_mode":
            manifest["timing"]["modes"] = ["unknown"]
        else:
            manifest["timing"]["trials"] = 0
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_fp32_wrapper_requires_allreduce(self):
        self.manifest["baseline"]["all_reduce_accumulation"] = "float32"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)
        self.manifest["cases"] = [self.manifest["cases"][0]]
        self.assertEqual(validate_manifest(self.manifest), self.manifest)
        self.manifest["baseline"]["all_reduce_accumulation"] = "float64"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)

    def test_paired_summary(self):
        result = summarize({"baseline": [4.0, 2.0, 6.0], "candidate": [2.0, 2.0, 3.0]})
        self.assertEqual(result["paired_speedup_samples"], [2.0, 1.0, 2.0])
        self.assertEqual(result["median_paired_speedup"], 2.0)
        self.assertEqual(result["baseline"]["p50_ms"], 4.0)


instantiate_parametrized_tests(TestHarness)

if __name__ == "__main__":
    run_tests()
