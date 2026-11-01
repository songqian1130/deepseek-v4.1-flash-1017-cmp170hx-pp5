"""Recompute the three controlled PP5 measurements from saved raw evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = {"date": "2026-11-01", "timezone": "Asia/Shanghai", "profiles": {}}
    request_reference = None
    for name, folder in (("v2_piecewise", "live-python"),
                         ("v1_piecewise", "live-v1"),
                         ("v1_full", "live-v1-full")):
        root = args.directory / folder
        data = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        baselines = [x for x in data["requests"] if x["label"].startswith("baseline-")]
        assert len(baselines) == 2
        for row in baselines:
            assert row["usage"]["completion_tokens"] == 256
            assert row["usage"]["prompt_tokens"] == 55
            req = json.loads((root / (row["label"] + ".json")).read_text(encoding="utf-8"))["request"]
            if request_reference is None:
                request_reference = req
            assert req == request_reference, "Request differs between controlled baselines"
        io = {}
        for label, bounds in data["boundaries"].items():
            io[label] = []
            for rank, before in bounds["before"]["workers"].items():
                after = bounds["after"]["workers"][rank]
                io[label].append({"rank": int(rank),
                                  "physical_read_bytes": after["io"]["read_bytes"] - before["io"]["read_bytes"],
                                  "major_faults": after["major_faults"] - before["major_faults"]})
        rates = [x["decode_tok_s"] for x in baselines]
        mean = sum(rates) / len(rates)
        out["profiles"][name] = {"baseline_tok_s": rates, "mean_tok_s": mean,
                                 "ms_per_token": 1000 / mean, "io": io}
    profiles = out["profiles"]
    out["v1_gain_percent"] = (profiles["v1_piecewise"]["mean_tok_s"] / profiles["v2_piecewise"]["mean_tok_s"] - 1) * 100
    out["full_incremental_gain_percent"] = (profiles["v1_full"]["mean_tok_s"] / profiles["v1_piecewise"]["mean_tok_s"] - 1) * 100
    out["total_gain_percent"] = (profiles["v1_full"]["mean_tok_s"] / profiles["v2_piecewise"]["mean_tok_s"] - 1) * 100
    out["identical_request"] = request_reference
    out["tokens_per_second_definition"] = "completion_tokens / (last_content_time - first_content_time); includes reasoning"
    out["reference_41_or_target_50_reached"] = False
    out["limits"] = ["Hot-cache short single requests only", "No matched request on reference machine",
                     "Torch profiler measurements are not used as baseline speed or a wall-time percentage breakdown"]
    (args.directory / "final-performance-summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profiles": {k: v["mean_tok_s"] for k, v in profiles.items()},
                      "v1_gain_percent": out["v1_gain_percent"],
                      "full_incremental_gain_percent": out["full_incremental_gain_percent"],
                      "total_gain_percent": out["total_gain_percent"]}))
    print("PP5_PERFORMANCE_SUMMARY_DONE")


if __name__ == "__main__":
    main()
