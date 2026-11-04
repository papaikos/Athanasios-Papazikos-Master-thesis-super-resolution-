import json
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.credit_guard import calculate_allowance


NOW = "2026-09-17T10:00:00+00:00"


def request(**changes):
    data = dict(
        available_credit_eur=200,
        reserved_eur=50,
        committed_eur=10,
        unbilled_eur=20,
        other_anticipated_eur=5,
        whole_vm_upper_eur_per_hour=10,
        snapshot_at=NOW,
        credit_expires_at="2026-09-20T10:00:00+00:00",
        stage_cap_seconds=100000,
        user_time_cap_seconds=100000,
    )
    data.update(changes)
    return data


class CreditGuardTests(unittest.TestCase):
    def evaluate(self, **changes):
        return calculate_allowance(request(**changes), now=NOW)

    def assert_denied(self, result):
        self.assertEqual(result["allowed_seconds"], 0)
        self.assertEqual(result["spend_allowance_eur"], 0)
        self.assertTrue(result["reasons"])

    def test_deducts_all_costs_and_reserve(self):
        result = self.evaluate()
        self.assertEqual(result["allowed_seconds"], 41400)
        self.assertEqual(result["spend_allowance_eur"], 115)
        self.assertIn("credit_budget", result["reasons"])

    def test_stage_user_and_expiry_caps(self):
        for changes, expected, reason in [
            ({"stage_cap_seconds": 1200}, 1200, "stage_cap"),
            ({"user_time_cap_seconds": 300}, 300, "user_time_cap"),
            ({"credit_expires_at": "2026-09-17T10:00:45.900+00:00"}, 45, "credit_expiry"),
        ]:
            with self.subTest(changes=changes):
                result = self.evaluate(**changes)
                self.assertEqual(result["allowed_seconds"], expected)
                self.assertLessEqual(result["spend_allowance_eur"], expected * 10 / 3600)
                self.assertIn(reason, result["reasons"])

    def test_floor_prevents_rounding_up_runtime(self):
        result = self.evaluate(available_credit_eur=30.01, reserved_eur=30,
                               committed_eur=0, unbilled_eur=0, other_anticipated_eur=0)
        self.assertEqual(result["allowed_seconds"], 3)
        self.assertLessEqual(result["spend_allowance_eur"], .01)

    def test_all_numeric_fields_reject_invalid_values(self):
        for field, value in request().items():
            if isinstance(value, (int, float)):
                for invalid in (-1, float("nan"), float("inf"), None, True, "10", [], {}):
                    with self.subTest(field=field, invalid=invalid):
                        self.assert_denied(self.evaluate(**{field: invalid}))

    def test_missing_fields_and_wrong_root_fail_closed(self):
        for field in request():
            data = request()
            del data[field]
            with self.subTest(field=field):
                self.assert_denied(calculate_allowance(data, now=NOW))
        for data in (None, [], "hello", 1):
            self.assert_denied(calculate_allowance(data, now=NOW))

    def test_snapshot_freshness_boundary_and_future(self):
        self.assertGreater(self.evaluate(snapshot_at="2026-09-17T04:00:00Z")["allowed_seconds"], 0)
        for value in ("2026-09-17T03:59:59Z", "2026-09-17T10:00:01Z"):
            self.assert_denied(self.evaluate(snapshot_at=value))

    def test_invalid_or_naive_times(self):
        for field in ("snapshot_at", "credit_expires_at"):
            for value in (None, 0, "bad", "2026-09-17T10:00:00"):
                self.assert_denied(self.evaluate(**{field: value}))
        self.assert_denied(calculate_allowance(request(), now="bad"))

    def test_reserve_policy_and_unavailable_budget(self):
        for changes in (
            {"reserved_eur": 49.99},
            {"available_credit_eur": 100, "reserved_eur": 29.99},
            {"available_credit_eur": 0},
            {"reserved_eur": 200},
            {"committed_eur": 150},
            {"whole_vm_upper_eur_per_hour": 0},
            {"stage_cap_seconds": 0},
            {"user_time_cap_seconds": .9},
            {"credit_expires_at": NOW},
            {"credit_expires_at": "2026-09-16T10:00:00Z"},
        ):
            with self.subTest(changes=changes):
                self.assert_denied(self.evaluate(**changes))

    def test_cli_json_and_malformed_input(self):
        script = Path(__file__).resolve().parents[1] / "src" / "credit_guard.py"
        fresh = datetime.now(timezone.utc)
        valid = json.dumps(request(snapshot_at=fresh.isoformat(),
                                   credit_expires_at=(fresh + timedelta(days=3)).isoformat()))
        for payload in (valid, "{broken", "[]", "{\"available_credit_eur\": NaN}"):
            process = subprocess.run([sys.executable, str(script)], input=payload,
                                     text=True, capture_output=True, check=False)
            result = json.loads(process.stdout)
            self.assertIn("allowed_seconds", result)
            self.assertTrue(result["reasons"])
            self.assertNotIn("Traceback", process.stderr)
            if payload == valid:
                self.assertEqual(result["allowed_seconds"], 41400)
                self.assertEqual(process.returncode, 0)
            else:
                self.assert_denied(result)
                self.assertEqual(process.returncode, 2)

    def test_offsets_and_very_large_values(self):
        result = self.evaluate(snapshot_at="2026-09-17T13:00:00+03:00")
        self.assertEqual(result["allowed_seconds"], 41400)
        self.assert_denied(self.evaluate(available_credit_eur=10 ** 400))
        result = self.evaluate(available_credit_eur=1e308, reserved_eur=2.5e307,
                               whole_vm_upper_eur_per_hour=1e308)
        self.assertEqual(result["allowed_seconds"], 2699)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
