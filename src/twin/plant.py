"""가상 공정(virtual plant): CHO 유가식 배양 기계론 모델.

상태 변수 (모든 배치 데이터가 공유하는 표준 단위)
    X : 생세포 밀도 (VCD)      [1e6 cells/mL]
    G : 포도당                 [mM]
    L : 젖산                   [mM]
    P : 항체 역가 (titer)      [mg/L]
    V : 배양 부피              [L]

속도식
    비성장속도   mu = mu_max * G/(Ks+G) * KiL/(KiL+L)          (Monod + 젖산 저해)
    사멸속도     kd = kd0 + kd_L * L/(Kd_L+L)
    포도당 소모  qG = mu/Y_XG + m_G                             (Pirt 식: 성장 + 유지)
    젖산 생성    qL = Y_LG * qG * G/(K_LG+G)                    (포도당이 높을수록 과잉 대사)
    항체 생산    qP = alpha*mu + beta                           (Luedeking-Piret 식)

물질수지 (연속 공급 F [L/h], 공급액 포도당 농도 G_feed [mM])
    dV/dt = F
    dX/dt = (mu - kd) X - (F/V) X
    dG/dt = -qG X + (F/V)(G_feed - G)
    dL/dt =  qL X - (F/V) L
    dP/dt =  qP X - (F/V) P

볼루스 공급은 해당 시점에 부피를 더하고 농도를 혼합 계산한다.

주의: 기본 계수값은 "그럴듯한 범위"로 직접 정한 예시값이며, 특정 논문이나
실제 공정에서 가져온 값이 아니다. 이 모델의 목적은 시스템 전체 흐름을
만들기 위한 정답(ground truth) 생성기이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.integrate import solve_ivp

STATE_NAMES = ["X", "G", "L", "P", "V"]


@dataclass
class PlantParams:
    mu_max: float = 0.045   # 1/h
    Ks: float = 1.0         # mM
    KiL: float = 60.0       # mM
    kd0: float = 0.002      # 1/h
    kd_L: float = 0.010     # 1/h
    Kd_L: float = 80.0      # mM
    Y_XG: float = 0.30      # (1e6 cells/mL) / mM
    m_G: float = 0.020      # mM / (1e6 cells/mL) / h
    Y_LG: float = 0.9       # mol 젖산 / mol 포도당
    K_LG: float = 15.0      # mM
    alpha: float = 8.0      # mg/L / (1e6 cells/mL)
    beta: float = 0.55      # mg/L / (1e6 cells/mL) / h

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedPlan:
    """공급 조건.

    continuous: start_h 이후 rate_L_per_h 로 연속 공급
    bolus:      (시간 h, 부피 L) 목록
    """
    G_feed: float = 400.0                       # 공급액 포도당 농도 mM
    start_h: float = 72.0
    rate_L_per_h: float = 0.0
    bolus: list[tuple[float, float]] = field(default_factory=list)
    max_volume_L: float | None = None           # 이 부피에 도달하면 연속 공급 중단

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bolus"] = [list(b) for b in self.bolus]
        return d


def rates(state: np.ndarray, p: PlantParams) -> dict[str, float]:
    """세포당 비속도 계산 (학습 대상이 되는 부분)."""
    X, G, L, P, V = state
    G = max(G, 0.0)
    L = max(L, 0.0)
    mu = p.mu_max * G / (p.Ks + G) * p.KiL / (p.KiL + L)
    kd = p.kd0 + p.kd_L * L / (p.Kd_L + L)
    qG = mu / p.Y_XG + p.m_G * G / (0.1 + G)   # 포도당이 0이면 유지 소모도 멈춤
    qL = p.Y_LG * qG * G / (p.K_LG + G)
    qP = p.alpha * mu + p.beta
    return {"mu": mu, "kd": kd, "qG": qG, "qL": qL, "qP": qP}


def _rhs(t: float, y: np.ndarray, p: PlantParams, feed: FeedPlan) -> np.ndarray:
    X, G, L, P, V = y
    r = rates(y, p)
    F = 0.0
    if t >= feed.start_h and feed.rate_L_per_h > 0:
        if feed.max_volume_L is None or V < feed.max_volume_L:
            F = feed.rate_L_per_h
    D = F / V
    return np.array([
        (r["mu"] - r["kd"]) * X - D * X,
        -r["qG"] * X + D * (feed.G_feed - G),
        r["qL"] * X - D * L,
        r["qP"] * X - D * P,
        F,
    ])


def simulate(
    x0: dict[str, float],
    feed: FeedPlan,
    params: PlantParams | None = None,
    t_end_h: float = 336.0,
    t_eval_h: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """배치 하나를 시뮬레이션한다.

    볼루스 시점마다 적분을 끊고 혼합을 적용한 뒤 이어서 적분한다.
    반환값: {"t": 시간 배열, "X": ..., "G": ..., "L": ..., "P": ..., "V": ...}
    """
    p = params or PlantParams()
    if t_eval_h is None:
        t_eval_h = np.arange(0.0, t_end_h + 1e-9, 1.0)
    t_eval_h = np.asarray(t_eval_h, dtype=float)

    y = np.array([x0[k] for k in STATE_NAMES], dtype=float)
    boluses = sorted((t, v) for t, v in feed.bolus if 0 < t < t_end_h)
    edges = [0.0] + [t for t, _ in boluses] + [t_end_h]

    out_t: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    for i in range(len(edges) - 1):
        t0, t1 = edges[i], edges[i + 1]
        if i > 0:
            vol = boluses[i - 1][1]
            y = _apply_bolus(y, vol, feed.G_feed)
        # 볼루스 시점과 겹치는 평가점은 '샘플링 후 공급' 관행에 맞춰 공급 직전 값을 쓴다
        lower_ok = (t_eval_h >= t0) if i == 0 else (t_eval_h > t0)
        mask = lower_ok & (t_eval_h <= t1)
        sol = solve_ivp(
            _rhs, (t0, t1), y, args=(p, feed),
            t_eval=t_eval_h[mask] if mask.any() else None,
            method="LSODA", rtol=1e-7, atol=1e-9, max_step=1.0,
        )
        if not sol.success:
            raise RuntimeError(f"적분 실패: {sol.message}")
        if mask.any():
            out_t.append(sol.t)
            out_y.append(sol.y)
        y = _solve_endpoint(t0, t1, y, p, feed)

    t = np.concatenate(out_t)
    Y = np.concatenate(out_y, axis=1)
    Y[:4] = np.clip(Y[:4], 0.0, None)
    result = {"t": t}
    result.update({name: Y[i] for i, name in enumerate(STATE_NAMES)})
    return result


def _solve_endpoint(t0, t1, y, p, feed):
    sol = solve_ivp(_rhs, (t0, t1), y, args=(p, feed), method="LSODA",
                    rtol=1e-7, atol=1e-9, max_step=1.0)
    return sol.y[:, -1]


def _apply_bolus(y: np.ndarray, vol_L: float, G_feed: float) -> np.ndarray:
    """볼루스 혼합: 부피 증가, 세포·젖산·항체는 희석, 포도당은 공급액과 혼합."""
    X, G, L, P, V = y
    Vn = V + vol_L
    return np.array([
        X * V / Vn,
        (G * V + G_feed * vol_L) / Vn,
        L * V / Vn,
        P * V / Vn,
        Vn,
    ])
