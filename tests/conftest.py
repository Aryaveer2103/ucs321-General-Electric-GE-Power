"""Shared fixtures. All data here is SYNTHETIC toy data used only to test code behaviour.

No performance claim in the project is derived from it - real metrics come from the Paderborn
motor measurements only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motor_thermal.config import load_config


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


def make_synthetic(
    n_profiles: int = 12, n_rows: int = 3000, seed: int = 0, hz: float = 2.0
) -> pd.DataFrame:
    """SYNTHETIC motor-like data: winding temperature = first-order lag of coolant + k * i^2."""
    rng = np.random.default_rng(seed)
    frames = []
    for pid in range(1, n_profiles + 1):
        steps = rng.integers(200, 600, size=n_rows // 200 + 5)
        speed = np.repeat(rng.uniform(0, 4000, size=len(steps)), steps)[:n_rows]
        torque = np.repeat(rng.uniform(-100, 200, size=len(steps)), steps)[:n_rows] * 0.6
        i_q = torque * 1.4 + rng.normal(0, 1, n_rows)
        i_d = -0.5 * np.abs(i_q) + rng.normal(0, 1, n_rows)
        u_q = speed * 0.04 + rng.normal(0, 1, n_rows)
        u_d = -np.abs(i_q) * 0.2 + rng.normal(0, 1, n_rows)
        coolant = 20 + 15 * np.sin(np.linspace(0, 3, n_rows) + pid) ** 2
        ambient = np.full(n_rows, 22.0) + rng.normal(0, 0.1, n_rows)
        drive = coolant + 0.0008 * (i_d**2 + i_q**2)
        tau = 300 * hz
        winding = np.empty(n_rows)
        winding[0] = coolant[0]
        for k in range(1, n_rows):
            winding[k] = winding[k - 1] + (drive[k] - winding[k - 1]) / tau
        frames.append(
            pd.DataFrame(
                {
                    "u_q": u_q,
                    "coolant": coolant,
                    "stator_winding": winding,
                    "u_d": u_d,
                    "stator_tooth": winding - 3,
                    "motor_speed": speed,
                    "i_d": i_d,
                    "i_q": i_q,
                    "pm": winding - 8,
                    "stator_yoke": winding - 5,
                    "ambient": ambient,
                    "torque": torque,
                    "profile_id": pid,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        df[c] = df[c].astype("int16" if c == "profile_id" else "float32")
    return df


@pytest.fixture(scope="session")
def toy() -> pd.DataFrame:
    return make_synthetic()
