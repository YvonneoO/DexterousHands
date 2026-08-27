#!/usr/bin/env python3
"""Validate per-success tactile and trajectory NPZ artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np


def pressure_report(path: Path):
    with np.load(str(path), allow_pickle=False) as data:
        keys = list(data.files)
        pressure_keys = [key for key in keys if "pressure" in key.lower()]
        arrays = []
        for key in pressure_keys:
            value = np.asarray(data[key])
            if np.issubdtype(value.dtype, np.number):
                arrays.append((key, value))
        if not arrays:
            raise ValueError("no numeric pressure array")
        finite_values = [(key, value, np.isfinite(value)) for key, value in arrays]
        no_inf = all(not bool(np.isinf(value).any()) for _, value, _ in finite_values)
        valid_counts = {
            key: sorted(set(mask.reshape(mask.shape[0], -1).sum(axis=1).astype(int).tolist()))
            for key, value, mask in finite_values
            if value.ndim == 3
        }
        masks_consistent = all(
            bool((mask == mask[0:1]).all()) for _, value, mask in finite_values if value.ndim == 3
        )
        valid_layout = bool(valid_counts) and all(counts == [217] for counts in valid_counts.values())
        nonzero = sum(
            int(np.count_nonzero(value[mask])) for _, value, mask in finite_values
        )
        peak = max(float(np.nanmax(value)) for _, value, _ in finite_values)
        return {
            "keys": pressure_keys,
            "shapes": {key: list(value.shape) for key, value in arrays},
            "no_inf": no_inf,
            "valid_taxels_per_frame": valid_counts,
            "valid_mask_consistent": masks_consistent,
            "valid_layout": valid_layout,
            "nonzero_values": nonzero,
            "peak": peak,
        }


def trajectory_report(path: Path):
    with np.load(str(path), allow_pickle=False) as data:
        keys = list(data.files)
        lower = {key: key.lower() for key in keys}
        groups = {
            "hand_dof": [key for key, name in lower.items() if "dof" in name and "object" not in name],
            "object_state": [key for key, name in lower.items() if "object" in name],
            "camera": [key for key, name in lower.items() if "camera" in name],
        }
        finite = True
        for key in keys:
            value = np.asarray(data[key])
            if np.issubdtype(value.dtype, np.number):
                finite = finite and bool(np.isfinite(value).all())
        return {
            "array_count": len(keys),
            "groups": groups,
            "finite": finite,
            "has_hand_dof": bool(groups["hand_dof"]),
            "has_object_state": bool(groups["object_state"]),
            "has_camera": bool(groups["camera"]),
        }


def coverage_report(path: Path, min_mapped_force_fraction: float):
    with np.load(str(path), allow_pickle=False) as data:
        required_suffixes = (
            "coverage_available",
            "total_hand_object_contact_count",
            "mapped_hand_object_contact_count",
            "unmapped_hand_object_contact_count",
            "total_hand_object_normal_force_n",
            "mapped_hand_object_normal_force_n",
            "unmapped_hand_object_normal_force_n",
        )
        missing = [
            side + "_" + suffix
            for side in ("left", "right")
            for suffix in required_suffixes
            if side + "_" + suffix not in data.files
        ]
        if missing:
            raise ValueError("missing contact coverage arrays: {}".format(", ".join(missing)))

        sides = {}
        combined = {
            "total_contact_count": 0,
            "mapped_contact_count": 0,
            "unmapped_contact_count": 0,
            "total_normal_force_n": 0.0,
            "mapped_normal_force_n": 0.0,
            "unmapped_normal_force_n": 0.0,
        }
        available = True
        invariants_ok = True
        for side in ("left", "right"):
            prefix = side + "_"
            side_available = bool(np.asarray(data[prefix + "coverage_available"], dtype=bool).all())
            values = {
                "total_contact_count": int(np.asarray(
                    data[prefix + "total_hand_object_contact_count"]
                ).sum()),
                "mapped_contact_count": int(np.asarray(
                    data[prefix + "mapped_hand_object_contact_count"]
                ).sum()),
                "unmapped_contact_count": int(np.asarray(
                    data[prefix + "unmapped_hand_object_contact_count"]
                ).sum()),
                "total_normal_force_n": float(np.asarray(
                    data[prefix + "total_hand_object_normal_force_n"]
                ).sum()),
                "mapped_normal_force_n": float(np.asarray(
                    data[prefix + "mapped_hand_object_normal_force_n"]
                ).sum()),
                "unmapped_normal_force_n": float(np.asarray(
                    data[prefix + "unmapped_hand_object_normal_force_n"]
                ).sum()),
            }
            count_ok = values["total_contact_count"] == (
                values["mapped_contact_count"] + values["unmapped_contact_count"]
            )
            force_error_n = values["total_normal_force_n"] - (
                values["mapped_normal_force_n"] + values["unmapped_normal_force_n"]
            )
            force_ok = abs(force_error_n) <= max(1.0e-5, 1.0e-6 * values["total_normal_force_n"])
            total_force = values["total_normal_force_n"]
            total_count = values["total_contact_count"]
            values.update({
                "coverage_available": side_available,
                "mapped_force_fraction": (
                    values["mapped_normal_force_n"] / total_force if total_force > 0.0 else None
                ),
                "mapped_contact_fraction": (
                    float(values["mapped_contact_count"]) / float(total_count)
                    if total_count > 0 else None
                ),
                "count_conservation_ok": count_ok,
                "force_conservation_error_n": force_error_n,
                "force_conservation_ok": force_ok,
            })
            sides[side] = values
            available = available and side_available
            invariants_ok = invariants_ok and count_ok and force_ok
            for key in combined:
                combined[key] += values[key]

        total_force = combined["total_normal_force_n"]
        total_count = combined["total_contact_count"]
        combined["mapped_force_fraction"] = (
            combined["mapped_normal_force_n"] / total_force if total_force > 0.0 else None
        )
        combined["mapped_contact_fraction"] = (
            float(combined["mapped_contact_count"]) / float(total_count)
            if total_count > 0 else None
        )
        passed = bool(
            available
            and invariants_ok
            and total_force > 0.0
            and combined["mapped_force_fraction"] >= min_mapped_force_fraction
        )
        return {
            "coverage_available": available,
            "minimum_mapped_force_fraction": min_mapped_force_fraction,
            "invariants_ok": invariants_ok,
            "sides": sides,
            "combined": combined,
            "pass": passed,
        }


def _body_matches_tokens(body_name: str, tokens):
    lower = body_name.lower()
    return any(token.lower() in lower for token in tokens)


def saved_coverage_relevant_report(saved_coverage, ignore_unmapped_body_tokens, min_mapped_force_fraction):
    """Return task-relevant coverage pass while still recording ignored unmapped contacts.

    The dense tactile layout intentionally covers fingertip/finger taxels, not forearm/wrist/knuckle
    rigid bodies. Contacts on ignored bodies are kept in the report but removed from the denominator
    when deciding whether mapped taxel force coverage is acceptable.
    """

    ignore_unmapped_body_tokens = list(ignore_unmapped_body_tokens or [])
    sides = saved_coverage.get("sides") or {}
    ignored = {}
    nonignored = {}
    ignored_force_n = 0.0
    ignored_contact_count = 0
    nonignored_force_n = 0.0
    nonignored_contact_count = 0

    for side, side_report in sides.items():
        body_counts = side_report.get("unmapped_body_contact_count") or {}
        body_forces = side_report.get("unmapped_body_force_n") or {}
        for body, count in body_counts.items():
            force = float(body_forces.get(body, 0.0))
            count = int(count)
            bucket = ignored if _body_matches_tokens(body, ignore_unmapped_body_tokens) else nonignored
            bucket.setdefault(side, {})[body] = {"count": count, "force_n": force}
            if bucket is ignored:
                ignored_force_n += force
                ignored_contact_count += count
            else:
                nonignored_force_n += force
                nonignored_contact_count += count

    combined = dict(saved_coverage.get("combined") or {})
    total_force = float(combined.get("total_normal_force_n") or 0.0)
    mapped_force = float(combined.get("mapped_normal_force_n") or 0.0)
    total_count = int(combined.get("total_contact_count") or 0)
    mapped_count = int(combined.get("mapped_contact_count") or 0)
    relevant_total_force = max(0.0, total_force - ignored_force_n)
    relevant_total_count = max(0, total_count - ignored_contact_count)
    relevant_mapped_force_fraction = (
        mapped_force / relevant_total_force if relevant_total_force > 0.0 else None
    )
    relevant_mapped_contact_fraction = (
        float(mapped_count) / float(relevant_total_count) if relevant_total_count > 0 else None
    )
    saved_available = bool(saved_coverage.get("combined", {}).get("coverage_available", True))
    relevant_pass = bool(
        saved_available
        and nonignored_contact_count == 0
        and relevant_total_force > 0.0
        and relevant_mapped_force_fraction is not None
        and relevant_mapped_force_fraction >= min_mapped_force_fraction
    )
    return {
        "strict_saved_report_pass": bool(saved_coverage.get("pass")),
        "ignore_unmapped_body_tokens": ignore_unmapped_body_tokens,
        "ignored_unmapped_bodies": ignored,
        "ignored_unmapped_contact_count": ignored_contact_count,
        "ignored_unmapped_force_n": ignored_force_n,
        "nonignored_unmapped_bodies": nonignored,
        "nonignored_unmapped_contact_count": nonignored_contact_count,
        "nonignored_unmapped_force_n": nonignored_force_n,
        "relevant_total_contact_count": relevant_total_count,
        "relevant_total_normal_force_n": relevant_total_force,
        "relevant_mapped_force_fraction": relevant_mapped_force_fraction,
        "relevant_mapped_contact_fraction": relevant_mapped_contact_fraction,
        "minimum_mapped_force_fraction": min_mapped_force_fraction,
        "pass": relevant_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--require-coverage", action="store_true")
    parser.add_argument("--min-mapped-force-fraction", type=float, default=0.95)
    parser.add_argument(
        "--ignore-unmapped-body-token",
        action="append",
        default=[],
        help=(
            "Treat unmapped contacts whose rigid-body name contains this token as recorded "
            "but not invalid for task-relevant coverage QA. May be repeated, e.g. "
            "--ignore-unmapped-body-token forearm --ignore-unmapped-body-token wrist."
        ),
    )
    args = parser.parse_args()
    reports = []
    failed = False
    for root in args.roots:
        for episode in sorted((root / "successful_episodes").glob("episode_*")):
            item = {"episode": str(episode)}
            try:
                item["pressure"] = pressure_report(episode / "pressure_grids.npz")
                item["trajectory"] = trajectory_report(episode / "trajectory_env0.npz")
                if args.require_coverage:
                    item["coverage"] = coverage_report(
                        episode / "pressure_grids.npz", args.min_mapped_force_fraction
                    )
                    coverage_json = episode / "contact_coverage.json"
                    if not coverage_json.is_file():
                        raise ValueError("missing contact_coverage.json")
                    with coverage_json.open() as handle:
                        saved_coverage = json.load(handle)
                    item["coverage"]["saved_report_pass"] = bool(saved_coverage.get("pass"))
                    item["coverage"]["task_relevant_saved_report"] = saved_coverage_relevant_report(
                        saved_coverage,
                        args.ignore_unmapped_body_token,
                        args.min_mapped_force_fraction,
                    )
                item["pass"] = (
                    item["pressure"]["no_inf"]
                    and item["pressure"]["valid_layout"]
                    and item["pressure"]["valid_mask_consistent"]
                    and item["pressure"]["nonzero_values"] > 0
                    and item["trajectory"]["finite"]
                    and item["trajectory"]["has_hand_dof"]
                    and item["trajectory"]["has_object_state"]
                    and item["trajectory"]["has_camera"]
                    and (
                        not args.require_coverage
                        or (
                            (
                                item["coverage"]["pass"]
                                and item["coverage"]["saved_report_pass"]
                            )
                            or item["coverage"]["task_relevant_saved_report"]["pass"]
                        )
                    )
                )
            except Exception as exc:  # report every episode before failing
                item["pass"] = False
                item["error"] = str(exc)
            failed = failed or not item["pass"]
            reports.append(item)
    print(json.dumps({"episodes": reports, "all_pass": not failed}, indent=2, sort_keys=True))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
