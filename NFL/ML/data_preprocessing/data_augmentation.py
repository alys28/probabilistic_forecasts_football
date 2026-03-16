from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

FEATURES_TO_SYNTHESIZE = [
    "game_completed",
    "relative_strength",
    "score_difference",
    "home_has_possession",
    "end.down",
    "end.distance",
    "end.yardsToEndzone",
    "home_timeouts_left",
    "away_timeouts_left",
]

REQUIRED_COLUMNS = set(FEATURES_TO_SYNTHESIZE + ["timestep", "model"])


def parse_years_arg(years_arg: str | None) -> List[int] | None:
    """
    Parse years from a comma/range string.

    Supported examples:
    - "2021,2022,2024"
    - "2021-2024"
    - "2021,2023-2025"
    """
    if years_arg is None:
        return None

    raw = years_arg.strip()
    if not raw:
        return None

    years = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            parts = [part.strip() for part in token.split("-")]
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"Invalid year range token: '{token}'")
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                raise ValueError(f"Invalid descending year range: '{token}'")
            years.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid year token: '{token}'")
            years.add(int(token))

    return sorted(years) if years else None


def _stable_seed(*parts: object) -> int:
    joined = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _clip_int(value: int, lower: int, upper: int) -> int:
    return int(min(max(value, lower), upper))


def _as_float(value: object, default: float) -> float:
    try:
        if pd.isna(value):
            return float(default)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: object, default: int) -> int:
    return int(round(_as_float(value, float(default))))


def _as_bool(value: object, default: bool) -> bool:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "t", "yes", "y"}:
        return True
    if text in {"false", "0", "f", "no", "n"}:
        return False
    return default


def _build_timestep_index(df: pd.DataFrame) -> Dict[float, List[int]]:
    index: Dict[float, List[int]] = {}
    if "timestep" not in df.columns:
        return index

    for idx in range(1, len(df)):
        try:
            timestep = round(float(df.at[idx, "timestep"]), 3)
        except Exception:
            continue
        index.setdefault(timestep, []).append(idx)
    return index


def _choose_donor_row_index(
    base_timestep: float | None,
    donor_index: Dict[float, List[int]],
    donor_keys: List[float],
    rng: np.random.Generator,
    donor_len: int,
) -> int:
    if donor_len <= 1:
        return 0
    if not donor_index or base_timestep is None:
        return int(rng.integers(1, donor_len))

    rounded = round(base_timestep, 3)
    if rounded in donor_index:
        return int(rng.choice(donor_index[rounded]))

    nearest_key = min(donor_keys, key=lambda key: abs(key - rounded))
    return int(rng.choice(donor_index[nearest_key]))


def _compute_relative_strength(
    base_df: pd.DataFrame,
    donor_df: pd.DataFrame,
    rng: np.random.Generator,
) -> float:
    base_default = 0.5
    donor_default = base_default

    if len(base_df) > 1 and "relative_strength" in base_df.columns:
        base_default = _as_float(base_df.at[1, "relative_strength"], base_default)
    elif len(base_df) > 0 and "homeWinProbability" in base_df.columns:
        base_default = _as_float(base_df.at[0, "homeWinProbability"], base_default)

    if len(donor_df) > 1 and "relative_strength" in donor_df.columns:
        donor_default = _as_float(donor_df.at[1, "relative_strength"], donor_default)
    elif len(donor_df) > 0 and "homeWinProbability" in donor_df.columns:
        donor_default = _as_float(donor_df.at[0, "homeWinProbability"], donor_default)

    alpha_game = rng.uniform(0.6, 0.9)
    relative_strength = (
        alpha_game * base_default
        + (1.0 - alpha_game) * donor_default
        + rng.normal(0.0, 0.01)
    )
    return _clip(relative_strength, 0.0, 1.0)


def _synthesize_row_features(
    out_df: pd.DataFrame,
    base_row: pd.Series,
    donor_row: pd.Series,
    row_index: int,
    relative_strength: float,
    rng: np.random.Generator,
) -> None:
    alpha_row = rng.uniform(0.65, 0.85)

    if "game_completed" in out_df.columns:
        base_gc = _as_float(base_row.get("game_completed"), 0.0)
        donor_gc = _as_float(donor_row.get("game_completed"), base_gc)
        timestep = _as_float(base_row.get("timestep"), base_gc)
        gc = alpha_row * base_gc + (1.0 - alpha_row) * donor_gc + rng.normal(0.0, 0.0015)
        gc = _clip(gc, max(0.0, timestep - 0.07), min(1.0, timestep + 0.07))
        out_df.at[row_index, "game_completed"] = round(_clip(gc, 0.0, 1.0), 6)

    if "relative_strength" in out_df.columns:
        out_df.at[row_index, "relative_strength"] = round(relative_strength, 6)

    if "score_difference" in out_df.columns:
        base_sd = _as_int(base_row.get("score_difference"), 0)
        donor_sd = _as_int(donor_row.get("score_difference"), base_sd)
        discrete_noise = int(rng.choice([-1, 0, 1], p=[0.15, 0.70, 0.15]))
        sd = int(round(alpha_row * base_sd + (1.0 - alpha_row) * donor_sd + discrete_noise))
        local_low = min(base_sd, donor_sd) - 3
        local_high = max(base_sd, donor_sd) + 3
        sd = _clip_int(sd, local_low, local_high)
        sd = _clip_int(sd, -70, 70)
        out_df.at[row_index, "score_difference"] = int(sd)

    if "home_has_possession" in out_df.columns:
        base_possession = _as_bool(base_row.get("home_has_possession"), default=False)
        donor_possession = _as_bool(
            donor_row.get("home_has_possession"),
            default=base_possession,
        )
        possession = base_possession if rng.random() < 0.75 else donor_possession
        out_df.at[row_index, "home_has_possession"] = bool(possession)

    if "end.down" in out_df.columns:
        base_down = _as_int(base_row.get("end.down"), 1)
        donor_down = _as_int(donor_row.get("end.down"), base_down)
        down = base_down if rng.random() < 0.5 else donor_down

        if down != -1:
            down = _clip_int(down, 1, 4)
            if rng.random() < 0.05:
                down += int(rng.choice([-1, 1]))
                down = _clip_int(down, 1, 4)
        out_df.at[row_index, "end.down"] = int(down)

    if "end.distance" in out_df.columns:
        base_distance = _as_int(base_row.get("end.distance"), 10)
        donor_distance = _as_int(donor_row.get("end.distance"), base_distance)
        distance = int(
            round(
                alpha_row * base_distance
                + (1.0 - alpha_row) * donor_distance
                + rng.normal(0.0, 1.5)
            )
        )
        out_df.at[row_index, "end.distance"] = int(_clip_int(distance, 0, 99))

    if "end.yardsToEndzone" in out_df.columns:
        base_yards = _as_int(base_row.get("end.yardsToEndzone"), 50)
        donor_yards = _as_int(donor_row.get("end.yardsToEndzone"), base_yards)
        yards = int(
            round(
                alpha_row * base_yards
                + (1.0 - alpha_row) * donor_yards
                + rng.normal(0.0, 2.5)
            )
        )
        out_df.at[row_index, "end.yardsToEndzone"] = int(_clip_int(yards, 0, 100))

    if "home_timeouts_left" in out_df.columns:
        base_timeouts = _as_int(base_row.get("home_timeouts_left"), 3)
        donor_timeouts = _as_int(donor_row.get("home_timeouts_left"), base_timeouts)
        timeouts = int(round(alpha_row * base_timeouts + (1.0 - alpha_row) * donor_timeouts))
        out_df.at[row_index, "home_timeouts_left"] = int(_clip_int(timeouts, 0, 3))

    if "away_timeouts_left" in out_df.columns:
        base_timeouts = _as_int(base_row.get("away_timeouts_left"), 3)
        donor_timeouts = _as_int(donor_row.get("away_timeouts_left"), base_timeouts)
        timeouts = int(round(alpha_row * base_timeouts + (1.0 - alpha_row) * donor_timeouts))
        out_df.at[row_index, "away_timeouts_left"] = int(_clip_int(timeouts, 0, 3))


def augment_single_game(
    base_file: Path,
    donor_file: Path,
    output_file: Path,
    rng_seed: int,
    dry_run: bool = False,
) -> dict:
    """
    Create one augmented synthetic game file from a base and donor game.
    """
    rng = np.random.default_rng(rng_seed)
    base_df = pd.read_csv(base_file)
    donor_df = pd.read_csv(donor_file)

    missing = sorted(REQUIRED_COLUMNS - set(base_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {base_file.name}: {missing}")

    out_df = base_df.copy(deep=True)
    if len(out_df) <= 1:
        if not dry_run:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(output_file, index=False)
        return {
            "base_file": str(base_file),
            "donor_file": str(donor_file),
            "output_file": str(output_file),
            "created": True,
            "rows": int(len(out_df)),
            "note": "Base file has no play rows beyond metadata row.",
        }

    if len(donor_df) <= 1:
        donor_df = base_df

    relative_strength = _compute_relative_strength(base_df, donor_df, rng)
    if "homeWinProbability" in out_df.columns and len(out_df) > 0:
        out_df.at[0, "homeWinProbability"] = round(relative_strength, 6)

    donor_index = _build_timestep_index(donor_df)
    donor_keys = sorted(donor_index.keys())

    for idx in range(1, len(out_df)):
        base_row = base_df.iloc[idx]
        try:
            base_timestep = float(base_row["timestep"])
        except Exception:
            base_timestep = None

        donor_idx = _choose_donor_row_index(
            base_timestep=base_timestep,
            donor_index=donor_index,
            donor_keys=donor_keys,
            rng=rng,
            donor_len=len(donor_df),
        )
        donor_row = donor_df.iloc[donor_idx]

        _synthesize_row_features(
            out_df=out_df,
            base_row=base_row,
            donor_row=donor_row,
            row_index=idx,
            relative_strength=relative_strength,
            rng=rng,
        )

    if not dry_run:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_file, index=False)

    return {
        "base_file": str(base_file),
        "donor_file": str(donor_file),
        "output_file": str(output_file),
        "created": True,
        "rows": int(len(out_df)),
    }


def _collect_year_dirs(root_path: Path) -> Dict[int, Path]:
    year_dirs: Dict[int, Path] = {}
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        if child.name.isdigit():
            year_dirs[int(child.name)] = child
    return year_dirs


def _reserve_output_path(base_file: Path, tag: str, taken_names: set[str]) -> Path:
    index = 1
    while True:
        candidate_name = f"{base_file.stem}_{tag}_{index:02d}.csv"
        if candidate_name not in taken_names:
            return base_file.with_name(candidate_name)
        index += 1


def _growth_counts(num_files: int, growth_mode: str, rng: np.random.Generator) -> List[int]:
    if growth_mode == "100":
        return [1] * num_files
    if growth_mode == "200":
        return [2] * num_files
    if growth_mode == "50":
        counts = [0] * num_files
        selected_size = int(round(num_files * 0.5))
        if selected_size > 0:
            selected_indices = rng.choice(num_files, size=selected_size, replace=False)
            for index in selected_indices:
                counts[int(index)] = 1
        return counts
    raise ValueError(f"Unsupported growth mode '{growth_mode}'. Expected one of 50, 100, 200.")


def _augment_task(task: dict) -> dict:
    year = task["year"]
    try:
        result = augment_single_game(
            base_file=task["base_file"],
            donor_file=task["donor_file"],
            output_file=task["output_file"],
            rng_seed=task["task_seed"],
            dry_run=task["dry_run"],
        )
        result["year"] = year
        return result
    except Exception as exc:
        return {
            "year": year,
            "created": False,
            "base_file": str(task["base_file"]),
            "donor_file": str(task["donor_file"]),
            "output_file": str(task["output_file"]),
            "error": str(exc),
        }


def augment_dataset(
    root_dir: str,
    years: List[int] | None = None,
    growth_mode: str = "100",
    tag: str = "AUGMENTED",
    seed: int = 42,
    max_workers: int | None = None,
    dry_run: bool = False,
) -> dict:
    root_path = Path(root_dir).expanduser()
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")

    all_year_dirs = _collect_year_dirs(root_path)
    warnings: List[str] = []

    requested_years = sorted(years) if years is not None else None
    if requested_years is None:
        selected_years = sorted(all_year_dirs.keys())
    else:
        available = set(all_year_dirs.keys())
        missing_years = sorted(set(requested_years) - available)
        if missing_years:
            warnings.append(
                f"Requested years not found under root and skipped: {missing_years}"
            )
        selected_years = sorted([year for year in requested_years if year in available])

    if not selected_years:
        summary = {
            "root_dir": str(root_path),
            "requested_years": requested_years,
            "selected_years": [],
            "growth_mode": growth_mode,
            "tag": tag,
            "dry_run": dry_run,
            "total_base_files": 0,
            "planned_outputs": 0,
            "created_outputs": 0,
            "failed_outputs": 0,
            "per_year": {},
            "warnings": warnings + ["No valid year folders selected for augmentation."],
        }
        print("No valid year folders selected for augmentation.")
        for warning in summary["warnings"]:
            print(f"[WARN] {warning}")
        return summary

    tasks = []
    per_year_summary: Dict[int, dict] = {}

    for year in selected_years:
        year_dir = all_year_dirs[year]
        all_csv_files = sorted(year_dir.glob("*.csv"))
        filtered_files = [
            file
            for file in all_csv_files
            if f"_{tag.lower()}_" not in file.stem.lower()
        ]
        year_rng = np.random.default_rng(_stable_seed(seed, year, "growth"))
        counts = _growth_counts(len(filtered_files), growth_mode, year_rng)

        taken_names = {file.name for file in all_csv_files}
        files_with_augments = 0
        planned_outputs = 0

        for index, base_file in enumerate(filtered_files):
            augment_count = counts[index]
            if augment_count <= 0:
                continue

            files_with_augments += 1
            donor_pool = [candidate for candidate in filtered_files if candidate != base_file]
            if not donor_pool:
                donor_pool = [base_file]

            for augment_idx in range(1, augment_count + 1):
                output_file = _reserve_output_path(base_file, tag, taken_names)
                taken_names.add(output_file.name)

                task_seed = _stable_seed(seed, year, base_file.name, augment_idx, output_file.name)
                task_rng = np.random.default_rng(task_seed)
                donor_file = donor_pool[int(task_rng.integers(0, len(donor_pool)))]

                tasks.append(
                    {
                        "year": year,
                        "base_file": base_file,
                        "donor_file": donor_file,
                        "output_file": output_file,
                        "task_seed": task_seed,
                        "dry_run": dry_run,
                    }
                )
                planned_outputs += 1

        per_year_summary[year] = {
            "base_files": len(filtered_files),
            "files_with_augments": files_with_augments,
            "skipped_files": len(filtered_files) - files_with_augments,
            "planned_outputs": planned_outputs,
            "created_outputs": 0,
            "failed_outputs": 0,
        }

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)
    max_workers = max(1, max_workers)

    results: List[dict] = []
    if tasks:
        if max_workers == 1:
            for task in tasks:
                results.append(_augment_task(task))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_augment_task, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())

    created_outputs = 0
    failed_outputs = 0
    for result in results:
        year = int(result["year"])
        if result.get("created"):
            created_outputs += 1
            per_year_summary[year]["created_outputs"] += 1
        else:
            failed_outputs += 1
            per_year_summary[year]["failed_outputs"] += 1
            warnings.append(
                f"Failed augmentation for {Path(result['base_file']).name}: {result.get('error', 'unknown error')}"
            )

    summary = {
        "root_dir": str(root_path),
        "requested_years": requested_years,
        "selected_years": selected_years,
        "growth_mode": growth_mode,
        "tag": tag,
        "dry_run": dry_run,
        "total_base_files": sum(year_stats["base_files"] for year_stats in per_year_summary.values()),
        "planned_outputs": sum(year_stats["planned_outputs"] for year_stats in per_year_summary.values()),
        "created_outputs": created_outputs,
        "failed_outputs": failed_outputs,
        "per_year": per_year_summary,
        "warnings": warnings,
    }

    print(
        f"Augmentation complete | years={selected_years} | growth={growth_mode}% | "
        f"dry_run={dry_run} | planned={summary['planned_outputs']} | "
        f"created={summary['created_outputs']} | failed={summary['failed_outputs']}"
    )
    for year in selected_years:
        stats = per_year_summary[year]
        print(
            f"Year {year}: base_files={stats['base_files']}, "
            f"files_with_augments={stats['files_with_augments']}, "
            f"skipped_files={stats['skipped_files']}, "
            f"planned_outputs={stats['planned_outputs']}, "
            f"created_outputs={stats['created_outputs']}, "
            f"failed_outputs={stats['failed_outputs']}"
        )
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"[WARN] {warning}")

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create synthetic NFL game CSVs with row-mix augmentation."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="NFL/ML/dataset_interpolated_fixed",
        help="Root directory containing year folders of CSV game files.",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Optional years filter, e.g. '2022' or '2021,2023-2025'.",
    )
    parser.add_argument(
        "--growth",
        type=str,
        default="100",
        choices=["50", "100", "200"],
        help="Dataset growth percentage relative to original count.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="AUGMENTED",
        help="Tag inserted into generated filenames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for deterministic augmentation.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum worker threads for file processing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and execute augmentation logic without writing output files.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        parsed_years = parse_years_arg(args.years)
    except ValueError as exc:
        parser.error(str(exc))

    augment_dataset(
        root_dir=args.root,
        years=parsed_years,
        growth_mode=args.growth,
        tag=args.tag,
        seed=args.seed,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
