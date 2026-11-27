import numpy as np
import pytest

from src.engine import SimulationEngine


def _run_short(planet_key: str, seed: int, ticks: int = 300) -> SimulationEngine:
    """Helper to run a fast, deterministic simulation for unit testing."""
    engine = SimulationEngine(planet_key=planet_key, seed=seed)
    engine.ticks_per_run = ticks
    engine.run()
    return engine


@pytest.mark.parametrize("planet_key", ["earth", "mars", "venus"])
def test_seed_determinism(planet_key: str):
    """Two runs with the same seed must produce bit-for-bit identical states."""
    eng_a = _run_short(planet_key, seed=123, ticks=250)
    eng_b = _run_short(planet_key, seed=123, ticks=250)

    assert np.array_equal(eng_a.pop.alive, eng_b.pop.alive)
    assert np.array_equal(eng_a.pop.energy, eng_b.pop.energy)
    assert np.array_equal(eng_a.pop.age, eng_b.pop.age)
    assert np.array_equal(eng_a.pop.is_dormant, eng_b.pop.is_dormant)
    assert np.array_equal(eng_a.pop.habitat_layer, eng_b.pop.habitat_layer)

    for key in eng_a.pop.FLOAT_GENOME_KEYS:
        arr_a = getattr(eng_a.pop, key)
        arr_b = getattr(eng_b.pop, key)
        assert np.array_equal(arr_a, arr_b), f"Mismatch in genome array '{key}'"


def test_different_seeds_diverge():
    """Different random seeds should produce distinct evolutionary trajectories."""
    eng_a = _run_short("earth", seed=1, ticks=200)
    eng_b = _run_short("earth", seed=2, ticks=200)

    assert not np.array_equal(eng_a.pop.energy, eng_b.pop.energy)


@pytest.mark.parametrize("planet_key", ["earth", "mars", "venus"])
def test_zero_nans_and_infs(planet_key: str):
    """No state or genome array may contain NaN or Inf values after stepping."""
    eng = _run_short(planet_key, seed=42, ticks=300)

    assert np.all(np.isfinite(eng.pop.energy)), "Non-finite values found in energy"

    for key in eng.pop.FLOAT_GENOME_KEYS:
        arr = getattr(eng.pop, key)
        assert np.all(np.isfinite(arr)), f"Non-finite values found in '{key}'"


@pytest.mark.parametrize("planet_key", ["earth", "mars", "venus"])
def test_energy_and_trait_bounds(planet_key: str):
    """
    Living organisms must have strictly positive energy, age below max_age_ticks,
    continuous traits clipped to [0.0, 1.0], and valid habitat layers {0, 1, 2}.
    """
    eng = _run_short(planet_key, seed=99, ticks=300)
    alive_idx = np.flatnonzero(eng.pop.alive)

    if alive_idx.size > 0:
        assert np.all(eng.pop.energy[alive_idx] > 0.0), "Living organism has <= 0 energy"
        assert np.all(eng.pop.age[alive_idx] < eng.max_age_ticks), "Living organism exceeded max age"

        layers = eng.pop.habitat_layer[alive_idx]
        assert np.all((layers >= 0) & (layers <= 2)), "Invalid habitat layer ID"

        for key in eng.pop.FLOAT_GENOME_KEYS:
            vals = getattr(eng.pop, key)[alive_idx]
            assert np.all((vals >= 0.0) & (vals <= 1.0)), f"Trait '{key}' out of [0, 1] bounds"