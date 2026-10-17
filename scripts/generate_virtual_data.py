"""가상 배치 데이터를 만들어 data/virtual/ 에 저장한다.

사용법:  python scripts/generate_virtual_data.py [--seed 0] [--out data/virtual]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.generate import default_design, run_design, save_dataset  # noqa: E402
from twin.plant import PlantParams  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "virtual")
    args = ap.parse_args()

    params = PlantParams()
    specs = default_design(seed=args.seed)
    measured, truth, design = run_design(specs, params, seed=args.seed)
    save_dataset(args.out, measured, truth, design, params, args.seed)

    final = measured.groupby("batch_id").last()
    summary = design.set_index("batch_id")[["strategy"]].join(final[["X", "G", "L", "P", "V"]])
    print(f"배치 {len(specs)}개 생성 → {args.out}")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
