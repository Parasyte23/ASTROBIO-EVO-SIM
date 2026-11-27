import csv
from pathlib import Path


class TelemetryLogger:
    """Buffered CSV writer for high-speed simulation telemetry."""

    FIELDNAMES = [
        "planet",
        "seed",
        "tick",
        "births",
        "deaths",
        "pop_size",
        "layer_0_subsurface",
        "layer_1_surface",
        "layer_2_airborne",
        "dormant_frac",
        "mean_age",
        "mean_energy",
        "mean_metabolism_mode",
        "mean_rad_shield",
        "mean_acid_resist",
        "mean_thermal_tol",
        "mean_dormancy_thresh",
    ]

    def __init__(
        self,
        output_path: str | Path,
        planet_key: str,
        seed: int,
        buffer_size: int = 100,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.planet_key = planet_key
        self.seed = seed
        self.buffer_size = buffer_size
        self._buffer: list[dict] = []

        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def log(self, tick: int, stats: dict, births: int, deaths: int) -> None:
        row = {
            "planet": self.planet_key,
            "seed": self.seed,
            "tick": tick,
            "births": births,
            "deaths": deaths,
            **stats,
        }
        self._buffer.append(row)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if self._buffer:
            self._writer.writerows(self._buffer)
            self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()