                      

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

METADATA_COLUMNS: List[str] = [
    "source_relpath",
    "source_filename",
    "source_modality",
    "source_group",
    "source_split_folder",
    "assigned_split",
    "split_strategy",
    "sample_name",
    "sample_core_name",
    "protocol_scope",
    "protocol_hint",
    "device",
    "scenario",
    "attack_name",
    "attack_family",
    "attack_variant",
    "is_attack",
    "is_benign",
    "label",
    "source_row_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CICIoMT CSV files with derived metadata into train/test CSV outputs.",
    )
    parser.add_argument(
        "--root",
        default="/home/capstone15/data/ciciomt2024",
        help="Dataset root folder.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/capstone15/data/ciciomt2024/merged",
        help="Output directory for merged CSV files.",
    )
    parser.add_argument(
        "--profiling-test-percent",
        type=int,
        default=20,
        help="Percent of profiling files assigned to test via deterministic hash.",
    )
    parser.add_argument(
        "--train-out-name",
        default="metadata_train.csv",
        help="Output train filename.",
    )
    parser.add_argument(
        "--test-out-name",
        default="metadata_test.csv",
        help="Output test filename.",
    )
    return parser.parse_args()


def hash_bucket(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def strip_known_extensions(filename: str) -> str:
    name = filename
    if name.lower().endswith(".csv"):
        name = name[:-4]
    if name.lower().endswith(".pcap"):
        name = name[:-5]
    return name


def trim_split_suffix(sample_name: str) -> str:
    lower = sample_name.lower()
    if lower.endswith("_train"):
        return sample_name[:-6]
    if lower.endswith("_test"):
        return sample_name[:-5]
    return sample_name


def detect_split_folder(parts_lower: List[str]) -> str:
    if "train" in parts_lower:
        return "train"
    if "test" in parts_lower:
        return "test"
    return "unspecified"


def parse_attack_family(attack_name: str) -> Tuple[str, str]:
    if not attack_name:
        return ("Benign", "")
    if attack_name.startswith("TCP_IP-DDoS-"):
        return ("DDoS", attack_name[len("TCP_IP-DDoS-") :])
    if attack_name.startswith("TCP_IP-DoS-"):
        return ("DoS", attack_name[len("TCP_IP-DoS-") :])
    if attack_name.startswith("MQTT-DDoS-"):
        return ("DDoS", attack_name[len("MQTT-DDoS-") :])
    if attack_name.startswith("MQTT-DoS-"):
        return ("DoS", attack_name[len("MQTT-DoS-") :])
    if attack_name.startswith("Recon-"):
        return ("Recon", attack_name[len("Recon-") :])
    if "Spoof" in attack_name or "spoof" in attack_name:
        return ("Spoofing", attack_name)
    if "Malformed" in attack_name:
        return ("Malformed", attack_name)
    if "Benign" in attack_name or "benign" in attack_name:
        return ("Benign", attack_name)
    return ("Other", attack_name)


def parse_profiling_device_and_scenario(sample_core: str) -> Tuple[str, str]:
    if sample_core in {"Active", "Idle", "ActiveBroker"}:
        return (sample_core, sample_core)
    if "_LAN_" in sample_core:
        device, action = sample_core.split("_LAN_", 1)
        return (device, f"LAN_{action}")
    if "_WAN_" in sample_core:
        device, action = sample_core.split("_WAN_", 1)
        return (device, f"WAN_{action}")
    if sample_core.endswith("_Power"):
        return (sample_core[: -len("_Power")], "Power")
    return (sample_core, "Profiling")


def modality_from_path(first_part: str) -> str:
    lower = first_part.lower()
    if "bluetooth" in lower:
        return "bluetooth"
    if "wifi" in lower or "mqtt" in lower:
        return "wifi_mqtt"
    return first_part


def protocol_hint(modality: str, sample_core: str) -> str:
    if modality == "bluetooth":
        return "bluetooth"
    if "mqtt" in sample_core.lower() or "broker" in sample_core.lower():
        return "mqtt"
    if modality == "wifi_mqtt":
        return "wifi"
    return "unknown"


def derive_metadata(
    csv_path: Path,
    dataset_root: Path,
    profiling_test_percent: int,
) -> Dict[str, str]:
    rel = csv_path.relative_to(dataset_root)
    parts = list(rel.parts)
    parts_lower = [p.lower() for p in parts]

    source_modality = modality_from_path(parts[0]) if parts else "unknown"
    source_group = "attacks" if "attacks" in parts_lower else "profiling" if "profiling" in parts_lower else "other"
    source_split_folder = detect_split_folder(parts_lower)

    sample_name = strip_known_extensions(csv_path.name)
    sample_core = trim_split_suffix(sample_name)
    lower_core = sample_core.lower()
    benign_named = "benign" in lower_core

    if source_group == "attacks":
        is_attack = 0 if benign_named else 1
    else:
        is_attack = 0

    if source_split_folder in {"train", "test"}:
        assigned_split = source_split_folder
        split_strategy = "folder"
    else:
        bucket = hash_bucket(str(rel))
        assigned_split = "test" if bucket < profiling_test_percent else "train"
        split_strategy = f"profile_hash_{profiling_test_percent}"

    if source_group == "attacks":
        attack_name = sample_core
        attack_family, attack_variant = parse_attack_family(attack_name)
        if source_modality == "bluetooth":
            device = "Bluetooth_Device"
        elif "mqtt" in lower_core:
            device = "MQTT_Device"
        else:
            device = "WiFi_IoMT_Device"
        scenario = "Attack"
    else:
        attack_name = "Benign"
        attack_family = "Benign"
        attack_variant = ""
        device, scenario = parse_profiling_device_and_scenario(sample_core)

    out = {
        "source_relpath": str(rel),
        "source_filename": csv_path.name,
        "source_modality": source_modality,
        "source_group": source_group,
        "source_split_folder": source_split_folder,
        "assigned_split": assigned_split,
        "split_strategy": split_strategy,
        "sample_name": sample_name,
        "sample_core_name": sample_core,
        "protocol_scope": source_modality,
        "protocol_hint": protocol_hint(source_modality, sample_core),
        "device": device,
        "scenario": scenario,
        "attack_name": attack_name,
        "attack_family": attack_family,
        "attack_variant": attack_variant,
        "is_attack": str(is_attack),
        "is_benign": str(1 - is_attack),
        "label": str(is_attack),
        "source_row_index": "0",
    }
    return out


def discover_csv_files(dataset_root: Path, out_dir: Path) -> List[Path]:
    files: List[Path] = []
    out_dir_resolved = out_dir.resolve()
    for p in sorted(dataset_root.rglob("*.csv")):
        p_resolved = p.resolve()
        if out_dir_resolved in p_resolved.parents:
            continue
        parts_lower = [x.lower() for x in p.relative_to(dataset_root).parts]
        if ("attacks" in parts_lower and "csv" in parts_lower) or ("profiling" in parts_lower and "csv" in parts_lower):
            files.append(p)
    return files


def write_merged(
    csv_files: Iterable[Path],
    dataset_root: Path,
    out_train: Path,
    out_test: Path,
    profiling_test_percent: int,
) -> Dict[str, int]:
    stats: Dict[str, int] = {
        "files": 0,
        "rows_total": 0,
        "rows_train": 0,
        "rows_test": 0,
        "attack_train": 0,
        "attack_test": 0,
        "benign_train": 0,
        "benign_test": 0,
    }

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    with out_train.open("w", newline="", encoding="utf-8") as train_fh, out_test.open(
        "w", newline="", encoding="utf-8"
    ) as test_fh:
        train_writer = csv.writer(train_fh)
        test_writer = csv.writer(test_fh)

        feature_columns: List[str] = []
        wrote_header = False

        for csv_path in csv_files:
            metadata = derive_metadata(csv_path, dataset_root, profiling_test_percent)
            split = metadata["assigned_split"]
            writer = train_writer if split == "train" else test_writer
            stats["files"] += 1

            with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as in_fh:
                reader = csv.DictReader(in_fh)
                if not reader.fieldnames:
                    continue

                if not feature_columns:
                    feature_columns = list(reader.fieldnames)
                else:
                    missing = [c for c in feature_columns if c not in reader.fieldnames]
                    if missing:
                        raise RuntimeError(f"Header mismatch in {csv_path}: missing {missing}")

                if not wrote_header:
                    header = METADATA_COLUMNS + feature_columns
                    train_writer.writerow(header)
                    test_writer.writerow(header)
                    wrote_header = True

                for idx, row in enumerate(reader, start=1):
                    metadata["source_row_index"] = str(idx)
                    out_row = [metadata[c] for c in METADATA_COLUMNS] + [row.get(c, "") for c in feature_columns]
                    writer.writerow(out_row)

                    stats["rows_total"] += 1
                    if split == "train":
                        stats["rows_train"] += 1
                        if metadata["is_attack"] == "1":
                            stats["attack_train"] += 1
                        else:
                            stats["benign_train"] += 1
                    else:
                        stats["rows_test"] += 1
                        if metadata["is_attack"] == "1":
                            stats["attack_test"] += 1
                        else:
                            stats["benign_test"] += 1

            if stats["files"] % 10 == 0:
                print(
                    f"progress files={stats['files']} rows={stats['rows_total']} "
                    f"train={stats['rows_train']} test={stats['rows_test']}",
                    flush=True,
                )

    return stats


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_train = out_dir / args.train_out_name
    out_test = out_dir / args.test_out_name

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    if args.profiling_test_percent < 0 or args.profiling_test_percent > 100:
        raise SystemExit("--profiling-test-percent must be between 0 and 100")

    csv_files = discover_csv_files(dataset_root, out_dir)
    print(f"discovered_csv_files={len(csv_files)}", flush=True)
    print(f"output_train={out_train}", flush=True)
    print(f"output_test={out_test}", flush=True)

    stats = write_merged(
        csv_files=csv_files,
        dataset_root=dataset_root,
        out_train=out_train,
        out_test=out_test,
        profiling_test_percent=args.profiling_test_percent,
    )

    print("merge_complete", flush=True)
    for k in sorted(stats):
        print(f"{k}={stats[k]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
