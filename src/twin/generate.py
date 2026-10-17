"""가상 배치 데이터 생성기.

여러 공급 조건으로 가상 공정을 돌리고, 오프라인 샘플링(기본 24시간 간격)과
측정 노이즈를 흉내 내어 '실측처럼 보이는' 데이터를 만든다.
정답 곡선(노이즈 없음)도 함께 저장해 모델 학습이 제대로 됐는지 비교할 수 있다.

표준 데이터 형식 (data/<source>/batches.csv) — 공개 데이터 로더도 이 형식으로 변환한다.
    batch_id, time_h, X, G, L, P, V
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .plant import FeedPlan, PlantParams, STATE_NAMES, simulate

# 측정 노이즈: 변수별 상대 표준편차(CV)와 최소 절대 오차
NOISE_CV = {"X": 0.06, "G": 0.04, "L": 0.05, "P": 0.05, "V": 0.0}
NOISE_ABS = {"X": 0.05, "G": 0.2, "L": 0.2, "P": 5.0, "V": 0.0}


@dataclass
class BatchSpec:
    batch_id: str
    strategy: str          # "batch" | "continuous" | "bolus"
    x0: dict
    feed: FeedPlan


def default_design(seed: int = 0) -> list[BatchSpec]:
    """공급 전략 3종 × 강도 여러 단계, 초기 조건은 약간씩 흔들어 배치 간 차이를 만든다."""
    rng = np.random.default_rng(seed)
    specs: list[BatchSpec] = []

    def x0():
        return {
            "X": float(rng.normal(0.4, 0.04)),
            "G": float(rng.normal(35.0, 2.0)),
            "L": 0.0,
            "P": 0.0,
            "V": 1.0,
        }

    n = 0
    for _ in range(2):  # 회분식 기준 배치
        n += 1
        specs.append(BatchSpec(f"B{n:02d}", "batch", x0(), FeedPlan(rate_L_per_h=0.0)))
    for rate in [0.0006, 0.0009, 0.0012, 0.0015, 0.0018]:
        n += 1
        specs.append(BatchSpec(f"B{n:02d}", "continuous", x0(),
                               FeedPlan(rate_L_per_h=rate, max_volume_L=1.5)))
    for vol in [0.02, 0.03, 0.04, 0.045]:  # 11회 × 0.045 L → 최대 약 1.5 L
        n += 1
        boluses = [(float(t), vol) for t in range(72, 336, 24)]
        specs.append(BatchSpec(f"B{n:02d}", "bolus", x0(),
                               FeedPlan(bolus=boluses, max_volume_L=1.5)))
    return specs


def add_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    noisy = df.copy()
    for k in STATE_NAMES:
        sd = NOISE_CV[k] * noisy[k].abs() + NOISE_ABS[k]
        if NOISE_CV[k] == 0 and NOISE_ABS[k] == 0:
            continue
        noisy[k] = np.clip(noisy[k] + rng.normal(0.0, sd), 0.0, None)
    return noisy


def run_design(
    specs: list[BatchSpec],
    params: PlantParams | None = None,
    t_end_h: float = 336.0,
    sample_every_h: float = 24.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """반환: (측정 데이터, 정답 곡선(1시간 간격), 배치 설계표)"""
    params = params or PlantParams()
    rng = np.random.default_rng(seed + 1)
    measured, truth, design = [], [], []
    t_dense = np.arange(0.0, t_end_h + 1e-9, 1.0)
    t_sample = np.arange(0.0, t_end_h + 1e-9, sample_every_h)

    for s in specs:
        sim = simulate(s.x0, s.feed, params, t_end_h, t_dense)
        df_true = pd.DataFrame({"time_h": sim["t"], **{k: sim[k] for k in STATE_NAMES}})
        df_true.insert(0, "batch_id", s.batch_id)
        truth.append(df_true)

        df_s = df_true[df_true["time_h"].isin(t_sample)].reset_index(drop=True)
        measured.append(add_noise(df_s, rng))

        total_bolus = sum(v for _, v in s.feed.bolus)
        design.append({
            "batch_id": s.batch_id,
            "strategy": s.strategy,
            "X0": s.x0["X"], "G0": s.x0["G"],
            "G_feed_mM": s.feed.G_feed,
            "feed_start_h": s.feed.start_h,
            "rate_L_per_h": s.feed.rate_L_per_h,
            "bolus_volume_L_each": s.feed.bolus[0][1] if s.feed.bolus else 0.0,
            "bolus_count": len(s.feed.bolus),
            "bolus_total_L": total_bolus,
            "feed_json": json.dumps(s.feed.to_dict()),
        })

    return (pd.concat(measured, ignore_index=True),
            pd.concat(truth, ignore_index=True),
            pd.DataFrame(design))


def save_dataset(out_dir: Path, measured, truth, design, params: PlantParams, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    measured.round(4).to_csv(out_dir / "batches.csv", index=False)
    truth.round(5).to_csv(out_dir / "truth.csv", index=False)
    design.to_csv(out_dir / "design.csv", index=False)
    meta = {
        "source": "virtual_plant",
        "seed": seed,
        "true_params": params.to_dict(),
        "noise_cv": NOISE_CV,
        "noise_abs": NOISE_ABS,
        "units": {"time_h": "h", "X": "1e6 cells/mL", "G": "mM", "L": "mM",
                  "P": "mg/L", "V": "L"},
        "note": "계수는 예시값이며 실제 공정 데이터가 아님",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
