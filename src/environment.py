import json
from pathlib import Path
import numpy as np


class PlanetEnvironment:
    """Represents a planet's physical constants and 3-layer habitat conditions."""

    def __init__(self, planet_key: str, config_path: str | Path = "configs/planets.json"):
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            all_planets = json.load(f)

        if planet_key not in all_planets:
            valid = ", ".join(all_planets.keys())
            raise ValueError(f"Unknown planet '{planet_key}'. Valid options: {valid}")

        data = all_planets[planet_key]
        self.key: str = planet_key
        self.name: str = data["name"]
        self.reference: str = data["reference"]

        # Global physical parameters
        phys = data["physical"]
        self.gravity_ms2: float = float(phys["gravity_ms2"])
        self.solar_irradiance_wm2: float = float(phys["solar_irradiance_wm2"])
        self.day_length_hours: float = float(phys["day_length_hours"])
        self.surface_pressure_bar: float = float(phys["surface_pressure_bar"])

        # Ensure layers are strictly ordered [0: subsurface, 1: surface, 2: airborne]
        layers = sorted(data["layers"], key=lambda x: x["id"])
        if [layer["id"] for layer in layers] != [0, 1, 2]:
            raise ValueError(f"Planet '{planet_key}' must define layer IDs 0, 1, and 2.")

        self.layer_names: list[str] = [layer["name"] for layer in layers]

        # Pre-packed 3-element float32 arrays for O(1) vectorized indexing
        self.temp_mean_c = np.array([l["temp_mean_c"] for l in layers], dtype=np.float32)
        self.temp_diurnal_amp_c = np.array([l["temp_diurnal_amp_c"] for l in layers], dtype=np.float32)
        self.solar_flux_base = np.array([l["solar_flux_norm"] for l in layers], dtype=np.float32)
        self.chem_energy = np.array([l["chem_energy_norm"] for l in layers], dtype=np.float32)
        self.radiation = np.array([l["radiation_norm"] for l in layers], dtype=np.float32)
        self.acidity = np.array([l["acidity_norm"] for l in layers], dtype=np.float32)
        self.pressure_hazard = np.array([l["pressure_hazard_norm"] for l in layers], dtype=np.float32)
        self.buoyancy_cost = np.array([l["buoyancy_cost"] for l in layers], dtype=np.float32)

    def get_tick_conditions(self, tick: int, hours_per_tick: float = 1.0) -> dict[str, np.ndarray]:
        """
        Computes instantaneous environmental conditions across all 3 layers for a given tick.
        Uses a sinusoidal diurnal cycle based on the planet's real day length.
        """
        elapsed_hours = tick * hours_per_tick
        phase = 2.0 * np.pi * (elapsed_hours / self.day_length_hours)
        diurnal_wave = np.float32(np.sin(phase))

        # Temperature oscillates smoothly around the layer mean
        current_temp_c = self.temp_mean_c + (self.temp_diurnal_amp_c * diurnal_wave)

        # Sunlight is active only during the daytime half of the sine wave
        daylight_factor = np.float32(max(0.0, float(diurnal_wave)))
        current_solar_flux = self.solar_flux_base * daylight_factor

        return {
            "temp_c": current_temp_c,
            "solar_flux": current_solar_flux,
            "chem_energy": self.chem_energy,
            "radiation": self.radiation,
            "acidity": self.acidity,
            "pressure_hazard": self.pressure_hazard,
            "buoyancy_cost": self.buoyancy_cost,
        }