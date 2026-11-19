"""Tests for the GTA VI trophy rarity forecaster (model.py + main.py).

Written first, per test-driven development: each test names the behaviour
the production code must have. Run with: python3 -m unittest test_model -v
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model  # noqa: E402  (imported after the sys.path line, like main.py does)


# ---------------------------------------------------------------- helpers

def make_case(trophies, history, at, target, horizon_minutes, levels=None):
    return {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "t00",
        "at": at,
        "horizon_minutes": horizon_minutes,
        "target": target,
        "levels": levels or [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies,
        "history": history,
        "notes": "test case",
    }


def trophy(tid, name, description, tier="bronze", points=15, secret=False):
    return {"id": tid, "name": name, "description": description,
            "tier": tier, "points": points, "secret": secret}


def read(slot, at, players, earned, rates):
    return {"slot": slot, "at": at, "players_tracked": players,
            "trophies_earned": earned, "rates": rates}


# ---------------------------------------------------------------- rounding

class TestRound4(unittest.TestCase):
    def test_matches_starter_practice_row(self):
        # Spec example: trophy last at 1.10%, 90-minute horizon -> the house
        # baseline's row [0.8974, 1.012, 1.1, 1.1955, 1.3477].
        self.assertEqual(model.baseline_row(1.1, 90),
                         [0.8974, 1.012, 1.1, 1.1955, 1.3477])

    def test_matches_starter_final_row(self):
        # Spec example: trophy last at 1.10%, final horizon (2026-11-21T11:30Z
        # to 2026-11-26T00:00Z = 6510 minutes) -> [0.1931, 0.5403, 1.1,
        # 2.2264, 6.0088].
        self.assertEqual(model.baseline_row(1.1, 6510),
                         [0.1931, 0.5403, 1.1, 2.2264, 6.0088])

    def test_round4_half_up(self):
        self.assertEqual(model.round4(1.23445), 1.2345)
        self.assertEqual(model.round4(1.23444), 1.2344)
        self.assertEqual(model.round4(100.0), 100)
        self.assertIsInstance(model.round4(100.0), int)


class TestLogit(unittest.TestCase):
    def test_round_trip(self):
        for pct in (0.01, 0.5, 1.1, 25.0, 50.0, 99.99):
            self.assertAlmostEqual(model.inv_logit(model.logit(pct)), pct, places=6)

    def test_clamps(self):
        # 0% and 100% have no logit: they clamp to the scorer's [0.01, 99.99].
        self.assertAlmostEqual(model.logit(0), model.logit(0.01))
        self.assertAlmostEqual(model.logit(100), model.logit(99.99))
        self.assertAlmostEqual(model.logit(150), model.logit(99.99))


class TestPinball(unittest.TestCase):
    def test_perfect_forecast_scores_zero(self):
        # Every quantile equal to the truth: loss 0.
        self.assertAlmostEqual(model.pinball_loss(1.1, 1.1, 0.5), 0.0)
        self.assertAlmostEqual(model.pinball_loss(50.0, 50.0, 0.05), 0.0)

    def test_equal_quantiles_score_mean_abs_logit_error(self):
        # Spec: a forecast whose five quantiles are equal scores exactly its
        # mean absolute logit error -- the mean over the five levels, not
        # each level alone (a low level pays little for being under).
        y, q = 2.0, 1.1
        expected = abs(model.logit(y) - model.logit(q))
        mean_loss = sum(model.pinball_loss(y, q, tau) for tau in model.LEVELS) / 5
        self.assertAlmostEqual(mean_loss, expected, places=9)

    def test_asymmetry(self):
        # A quantile below the truth costs 2*tau per logit unit; above it
        # 2*(1-tau). So at tau=0.05 being below is cheap, being above dear.
        y = 1.1
        below = model.pinball_loss(y, 0.5, 0.05)   # q below y
        above = model.pinball_loss(y, 2.0, 0.05)   # q above y
        self.assertLess(below, above)

    def test_score_case_uses_spec_formula(self):
        # Mean pinball over every trophy and quantile, 2x the standard pinball.
        truth = [1.1, 2.0]
        fc = {"trophies": [{"id": 1, "q": [1.1] * 5},
                           {"id": 2, "q": [2.0] * 5}]}
        self.assertAlmostEqual(model.score_case(truth, fc), 0.0)
        fc2 = {"trophies": [{"id": 1, "q": [0.5] * 5},
                            {"id": 2, "q": [1.0] * 5}]}
        per = [abs(model.logit(1.1) - model.logit(0.5)),
               abs(model.logit(2.0) - model.logit(1.0))]
        self.assertAlmostEqual(model.score_case(truth, fc2), sum(per) / 2, places=9)


class TestNormal(unittest.TestCase):
    def test_ppf_known_values(self):
        self.assertAlmostEqual(model.norm_ppf(0.5), 0.0, places=6)
        self.assertAlmostEqual(model.norm_ppf(0.95), 1.6448536269514722, places=5)
        self.assertAlmostEqual(model.norm_ppf(0.05), -1.6448536269514722, places=5)
        self.assertAlmostEqual(model.norm_ppf(0.975), 1.959963984540054, places=5)

    def test_cdf_round_trip(self):
        for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
            self.assertAlmostEqual(model.norm_cdf(model.norm_ppf(p)), p, places=6)


# ---------------------------------------------------------------- priors

class TestCategoryPrior(unittest.TestCase):
    def test_platinum_is_tiny(self):
        cap, prior = model.category_cap_prior(
            trophy(1, "Vice Got a Price!", "Congratulations! You're a Leonida Legend.",
                   tier="platinum", points=300))
        self.assertLessEqual(cap, 1.0)
        self.assertLessEqual(prior, 0.5)

    def test_chapter_order(self):
        # Earlier chapters saturate higher than later ones.
        caps = []
        for n, name in ((1, "Paradise Lost"), (2, "T**s Ass Drug Money"),
                        (3, "Vice City, Baby!"), (4, "Top Of The World, Ma")):
            cap, _ = model.category_cap_prior(
                trophy(2 + n, name, f"Complete Chapter {n}."))
            caps.append(cap)
        self.assertEqual(caps, sorted(caps, reverse=True))

    def test_gold_final_mission_below_chapter_one(self):
        cap_final, _ = model.category_cap_prior(
            trophy(7, "You Gotta Know When to Get Off", "Complete the final mission.",
                   tier="gold", points=90))
        cap_ch1, _ = model.category_cap_prior(
            trophy(3, "Paradise Lost", "Complete Chapter 1."))
        self.assertLess(cap_final, cap_ch1)

    def test_hundred_percent_gold_is_tiny(self):
        cap, _ = model.category_cap_prior(
            trophy(14, "A Long Road", "Attain 100% Completion.", tier="gold", points=90))
        self.assertLessEqual(cap, 2.0)


# ---------------------------------------------------------------- series

class TestDedupeReads(unittest.TestCase):
    def test_collapses_identical_consecutive_rates(self):
        hist = [
            read("2026-11-19T00:00:00Z", "2026-11-19T00:00:00.100Z", 638, 943, [1.0]),
            read("2026-11-19T00:30:00Z", "2026-11-19T00:30:00.100Z", 638, 943, [1.0]),
            read("2026-11-19T01:00:00Z", "2026-11-19T01:00:00.100Z", 700, 1000, [1.5]),
            read("2026-11-19T01:30:00Z", "2026-11-19T01:30:00.100Z", 700, 1000, [1.5]),
            read("2026-11-19T02:00:00Z", "2026-11-19T02:00:00.100Z", 800, 1200, [2.0]),
        ]
        pts = model.distinct_series(hist, 0)
        self.assertEqual([p[1] for p in pts], [1.0, 1.5, 2.0])
        # First-appearance times are kept.
        self.assertEqual(pts[0][0], 0.0)
        self.assertAlmostEqual(pts[1][0], 1.0, places=6)
        self.assertAlmostEqual(pts[2][0], 2.0, places=6)


# ---------------------------------------------------------------- forecast

class TestForecastFlatHistory(unittest.TestCase):
    def setUp(self):
        self.trophies = [
            trophy(1, "Vice Got a Price!", "Congratulations! You're a Leonida Legend.",
                   tier="platinum", points=300),
            trophy(3, "Paradise Lost", "Complete Chapter 1."),
            trophy(4, "T**s Ass Drug Money", "Complete Chapter 2."),
        ]
        hist = [
            read("2026-11-18T23:50:00Z", "2026-11-18T23:50:49Z", 638, 943, [0.63, 1.25, 1.1]),
            read("2026-11-19T00:00:00Z", "2026-11-19T00:00:00Z", 638, 943, [0.63, 1.25, 1.1]),
            read("2026-11-19T02:30:00Z", "2026-11-19T02:30:00Z", 638, 943, [0.63, 1.25, 1.1]),
        ]
        self.case = make_case(self.trophies, hist,
                              "2026-11-19T03:00:00Z", "2026-11-19T04:30:00Z", 90)

    def test_median_stays_at_last_rate_when_history_is_flat(self):
        fc = model.forecast_case(self.case)
        for row, last in zip(fc["trophies"], [0.63, 1.25, 1.1]):
            self.assertAlmostEqual(row["q"][2], last, places=4)

    def test_spread_is_wider_than_the_random_walk(self):
        # A flat history hides refresh jumps, so the spread must exceed the
        # starter's narrow 90-minute random-walk spread (0.125 logit).
        fc = model.forecast_case(self.case)
        for row in fc["trophies"]:
            lo = model.logit(row["q"][0])
            hi = model.logit(row["q"][4])
            self.assertGreater((hi - lo) / (2 * 1.645), 0.3)

    def test_output_contract(self):
        fc = model.forecast_case(self.case)
        self.assertEqual([r["id"] for r in fc["trophies"]], [1, 3, 4])
        for row in fc["trophies"]:
            q = row["q"]
            self.assertEqual(len(q), 5)
            self.assertTrue(all(0 <= v <= 100 for v in q))
            self.assertEqual(q, sorted(q))
            self.assertTrue(all(v == v and abs(v) != float("inf") for v in q))


class TestForecastRisingHistory(unittest.TestCase):
    def setUp(self):
        # A chapter-1 trophy whose rate rises steeply in log-time.
        self.trophies = [trophy(3, "Paradise Lost", "Complete Chapter 1."),
                         trophy(4, "T**s Ass Drug Money", "Complete Chapter 2.")]
        hist = []
        for h, rate1, rate2 in ((0, 0.5, 0.4), (6, 2.0, 1.2), (12, 5.0, 2.5),
                                (24, 12.0, 5.0), (36, 20.0, 8.0), (48, 28.0, 11.0)):
            slot = f"2026-11-{19 + h // 24:02d}T{h % 24:02d}:00:00Z"
            hist.append(read(slot, slot, 1000 + 100 * h, 100, [rate1, rate2]))
        self.case = make_case(self.trophies, hist,
                              "2026-11-21T09:30:00Z", "2026-11-26T00:00:00Z",
                              int(4.52 * 1440))

    def test_median_rises_above_last_rate(self):
        fc = model.forecast_case(self.case)
        self.assertGreater(fc["trophies"][0]["q"][2], 28.0)
        self.assertGreater(fc["trophies"][1]["q"][2], 11.0)

    def test_earlier_chapter_forecasts_higher_than_later(self):
        fc = model.forecast_case(self.case)
        self.assertGreater(fc["trophies"][0]["q"][2], fc["trophies"][1]["q"][2])

    def test_median_stays_below_the_category_cap(self):
        fc = model.forecast_case(self.case)
        cap1, _ = model.category_cap_prior(self.trophies[0])
        self.assertLessEqual(fc["trophies"][0]["q"][2], cap1)


class TestFlatTrophyBorrowing(unittest.TestCase):
    def test_flat_platinum_does_not_borrow_the_field_growth(self):
        # A platinum whose history never moved sits in a category of its
        # own, so it must not inherit the story chapters' rise: its median
        # stays at its last rate even while other trophies climb steeply.
        trophies = [
            trophy(1, "Vice Got a Price!", "Congratulations! You're a Leonida Legend.",
                   tier="platinum", points=300),
            trophy(3, "Paradise Lost", "Complete Chapter 1."),
            trophy(4, "T**s Ass Drug Money", "Complete Chapter 2."),
        ]
        hist = []
        for h, r3, r4 in ((0, 0.5, 0.4), (12, 5.0, 2.5), (24, 12.0, 5.0),
                          (36, 20.0, 8.0), (48, 28.0, 11.0)):
            slot = f"2026-11-{19 + h // 24:02d}T{h % 24:02d}:00:00Z"
            hist.append(read(slot, slot, 1000, 100, [0.0, r3, r4]))
        case = make_case(trophies, hist, "2026-11-21T09:30:00Z",
                         "2026-11-26T00:00:00Z", int(4.52 * 1440))
        fc = model.forecast_case(case)
        # The platinum stays near zero; the chapters rise.
        self.assertLess(fc["trophies"][0]["q"][2], 0.5)
        self.assertGreater(fc["trophies"][1]["q"][2], 28.0)

    def test_flat_late_chapter_borrows_its_category(self):
        # A chapter whose history never moved, in a category whose other
        # chapters are rising, does borrow their growth.
        trophies = [
            trophy(3, "Paradise Lost", "Complete Chapter 1."),
            trophy(6, "Top Of The World, Ma", "Complete Chapter 4."),
        ]
        hist = []
        for h, r3 in ((0, 0.5), (12, 5.0), (24, 12.0), (36, 20.0), (48, 28.0)):
            slot = f"2026-11-{19 + h // 24:02d}T{h % 24:02d}:00:00Z"
            hist.append(read(slot, slot, 1000, 100, [r3, 1.1]))
        case = make_case(trophies, hist, "2026-11-21T09:30:00Z",
                         "2026-11-26T00:00:00Z", int(4.52 * 1440))
        fc = model.forecast_case(case)
        # Chapter 4 was flat at 1.1 but its category (chapters) is rising.
        self.assertGreater(fc["trophies"][1]["q"][2], 1.1)


class TestForecastEdgeCases(unittest.TestCase):
    def test_trophy_at_100_percent_stays_near_100(self):
        trophies = [trophy(2, "Just Like Paradise", "Complete Gone Fishing.")]
        hist = [read("2026-11-19T00:00:00Z", "2026-11-19T00:00:00Z", 638, 943, [100.0]),
                read("2026-11-19T02:00:00Z", "2026-11-19T02:00:00Z", 638, 943, [100.0])]
        case = make_case(trophies, hist, "2026-11-19T03:00:00Z",
                         "2026-11-19T04:30:00Z", 90)
        fc = model.forecast_case(case)
        self.assertGreater(fc["trophies"][0]["q"][2], 99.0)

    def test_trophy_at_zero_gets_a_small_positive_forecast(self):
        trophies = [trophy(1, "Vice Got a Price!",
                           "Congratulations! You're a Leonida Legend.",
                           tier="platinum", points=300)]
        hist = [read("2026-11-19T00:00:00Z", "2026-11-19T00:00:00Z", 638, 943, [0.0]),
                read("2026-11-19T02:00:00Z", "2026-11-19T02:00:00Z", 638, 943, [0.0])]
        case = make_case(trophies, hist, "2026-11-19T03:00:00Z",
                         "2026-11-19T04:30:00Z", 90)
        fc = model.forecast_case(case)
        q = fc["trophies"][0]["q"]
        self.assertGreaterEqual(q[0], 0.0)
        self.assertLessEqual(q[4], 1.0)  # platinum cap

    def test_empty_history_does_not_crash(self):
        trophies = [trophy(3, "Paradise Lost", "Complete Chapter 1.")]
        case = make_case(trophies, [], "2026-11-19T03:00:00Z",
                         "2026-11-19T04:30:00Z", 90)
        fc = model.forecast_case(case)
        self.assertEqual(len(fc["trophies"]), 1)
        self.assertEqual(len(fc["trophies"][0]["q"]), 5)


class TestLognormalRecovery(unittest.TestCase):
    def test_recovers_a_lognormal_trajectory(self):
        # Synthetic truth: r(t) = R * Phi((ln t - mu) / sigma), R=60%, and the
        # case's history samples it every 6 hours to day 2.5. The forecast at
        # day 7 must land within its own 90% interval.
        R, mu, sigma = 60.0, math.log(30.0), 0.8
        trophies = [trophy(3, "Paradise Lost", "Complete Chapter 1.")]
        hist = []
        for h in range(1, 61, 6):
            r = R * model.norm_cdf((math.log(h) - mu) / sigma)
            slot = f"2026-11-{19 + h // 24:02d}T{h % 24:02d}:00:00Z"
            hist.append(read(slot, slot, 1000, 100, [round(r, 2)]))
        case = make_case(trophies, hist, "2026-11-21T12:00:00Z",
                         "2026-11-26T00:00:00Z", int(4.5 * 1440))
        fc = model.forecast_case(case)
        truth = R * model.norm_cdf((math.log(7 * 24) - mu) / sigma)
        q = fc["trophies"][0]["q"]
        self.assertLessEqual(q[0], truth)
        self.assertLessEqual(truth, q[4])
        # And the median is not absurdly far from the truth.
        self.assertLess(abs(model.logit(q[2]) - model.logit(truth)), 1.0)


# ---------------------------------------------------------------- main.py I/O

class TestValidateOutputs(unittest.TestCase):
    def test_valid_outputs_pass_and_malformed_ones_fail(self):
        import main as entry

        good = {"trophies": [{"id": k, "q": [0.1, 0.5, 1.0, 2.0, 5.0]}
                             for k in range(1, 56)]}
        bad = {"trophies": [{"id": k, "q": [0.1, 0.5, 1.0, 2.0, 5.0]}
                            for k in range(1, 55)]}
        with tempfile.TemporaryDirectory() as tmp:
            entry.OUTPUT_DIR = Path(tmp)
            Path(tmp, "p00.out").write_text(json.dumps(good))
            entry.validate_outputs()  # must not raise
            Path(tmp, "p01.out").write_text(json.dumps(bad))
            with self.assertRaises(SystemExit):
                entry.validate_outputs()


class TestMainIO(unittest.TestCase):
    def test_runs_on_the_kit_practice_inputs(self):
        kit_in = HERE.parent / "gta6-trophy-rarity-forecast-01" / \
            "gta6-trophy-rarity-forecast-01-kit" / "input"
        self.assertTrue(kit_in.exists(), "kit input dir missing")
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, MAKEFUN_INPUT_DIR=str(kit_in),
                       MAKEFUN_OUTPUT_DIR=tmp)
            cp = subprocess.run([sys.executable, str(HERE / "main.py")],
                                env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            for name in ("p00.out", "p01.out"):
                out = json.loads((Path(tmp) / name).read_text())
                self.assertEqual(len(out["trophies"]), 55)
                self.assertEqual(set(out.keys()), {"trophies"})
                for row in out["trophies"]:
                    self.assertEqual(set(row.keys()), {"id", "q"})
                    self.assertEqual(len(row["q"]), 5)
                    self.assertEqual(row["q"], sorted(row["q"]))
                    self.assertTrue(all(0 <= v <= 100 for v in row["q"]))
                self.assertLess((Path(tmp) / name).stat().st_size, 65536)

    def test_two_runs_are_byte_identical(self):
        kit_in = HERE.parent / "gta6-trophy-rarity-forecast-01" / \
            "gta6-trophy-rarity-forecast-01-kit" / "input"
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for outdir in (a, b):
                env = dict(os.environ, MAKEFUN_INPUT_DIR=str(kit_in),
                           MAKEFUN_OUTPUT_DIR=outdir)
                cp = subprocess.run([sys.executable, str(HERE / "main.py")],
                                    env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(cp.returncode, 0, cp.stderr)
            for name in ("p00.out", "p01.out"):
                self.assertEqual((Path(a) / name).read_bytes(),
                                 (Path(b) / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
