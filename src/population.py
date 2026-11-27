import numpy as np


class Population:
    """
    Pre-allocated Structure of Arrays (SoA) population pool.
    Uses a fixed capacity and boolean `alive` mask to avoid dynamic array resizing.
    """

    FLOAT_GENOME_KEYS = (
        "metabolism_mode",
        "rad_shield",
        "acid_resist",
        "thermal_tol",
        "dormancy_thresh",
    )

    def __init__(self, max_pop: int, seed: int):
        self.max_pop: int = max_pop
        self.seed: int = seed
        self.rng = np.random.default_rng(seed)

        # Mutable state arrays
        self.alive = np.zeros(max_pop, dtype=bool)
        self.energy = np.zeros(max_pop, dtype=np.float32)
        self.age = np.zeros(max_pop, dtype=np.int32)
        self.is_dormant = np.zeros(max_pop, dtype=bool)

        # Inherited genome arrays
        self.metabolism_mode = np.zeros(max_pop, dtype=np.float32)  # 0=chemo, 1=photo
        self.habitat_layer = np.zeros(max_pop, dtype=np.int8)       # 0=sub, 1=surf, 2=air
        self.rad_shield = np.zeros(max_pop, dtype=np.float32)
        self.acid_resist = np.zeros(max_pop, dtype=np.float32)
        self.thermal_tol = np.zeros(max_pop, dtype=np.float32)
        self.dormancy_thresh = np.zeros(max_pop, dtype=np.float32)

    def seed_initial_population(self, initial_pop: int, initial_energy: float) -> None:
        """
        Spawns an unbiased starting population uniformly distributed across all
        3 habitat layers and [0.0, 1.0] trait space so natural selection alone
        drives planetary specialization.
        """
        if initial_pop > self.max_pop:
            raise ValueError(f"initial_pop ({initial_pop}) exceeds max_pop ({self.max_pop}).")

        self.alive[:initial_pop] = True
        self.energy[:initial_pop] = np.float32(initial_energy)
        self.age[:initial_pop] = 0
        self.is_dormant[:initial_pop] = False

        # Uniform random starting traits
        for key in self.FLOAT_GENOME_KEYS:
            arr = getattr(self, key)
            arr[:initial_pop] = self.rng.random(initial_pop, dtype=np.float32)

        # Equal starting representation across subsurface (0), surface (1), airborne (2)
        self.habitat_layer[:initial_pop] = self.rng.integers(0, 3, size=initial_pop, dtype=np.int8)

    def reproduce(
        self,
        parent_indices: np.ndarray,
        repro_cost_ratio: float,
        mutation_sigma: float,
        layer_mutation_prob: float,
    ) -> int:
        """
        Vectorized asexual reproduction with Gaussian trait mutation and discrete
        habitat-layer switching. Returns the number of offspring created.
        """
        num_parents = parent_indices.size
        if num_parents == 0:
            return 0

        free_slots = np.flatnonzero(~self.alive)
        num_births = min(num_parents, free_slots.size)
        if num_births == 0:
            return 0

        # If more parents are ready than free slots exist, pick a random subset
        if num_parents > num_births:
            parent_indices = self.rng.choice(parent_indices, size=num_births, replace=False)

        child_indices = free_slots[:num_births]

        # Split energy between parent and offspring
        transfer_energy = self.energy[parent_indices] * np.float32(repro_cost_ratio)
        self.energy[parent_indices] -= transfer_energy
        self.energy[child_indices] = transfer_energy

        # Initialize child state
        self.alive[child_indices] = True
        self.age[child_indices] = 0
        self.is_dormant[child_indices] = False

        # Inherit and mutate continuous genome traits
        for key in self.FLOAT_GENOME_KEYS:
            arr = getattr(self, key)
            noise = self.rng.normal(0.0, mutation_sigma, size=num_births).astype(np.float32)
            arr[child_indices] = np.clip(arr[parent_indices] + noise, 0.0, 1.0)

        # Inherit habitat layer with a small probability of migrating to another layer
        child_layers = self.habitat_layer[parent_indices].copy()
        migrate_mask = self.rng.random(num_births) < layer_mutation_prob
        num_migrants = int(np.count_nonzero(migrate_mask))
        if num_migrants > 0:
            child_layers[migrate_mask] = self.rng.integers(0, 3, size=num_migrants, dtype=np.int8)
        self.habitat_layer[child_indices] = child_layers

        return num_births

    def get_stats(self) -> dict[str, float | int]:
        """Returns aggregated population metrics for CSV telemetry."""
        alive_idx = np.flatnonzero(self.alive)
        pop_size = int(alive_idx.size)

        if pop_size == 0:
            return {
                "pop_size": 0,
                "layer_0_subsurface": 0,
                "layer_1_surface": 0,
                "layer_2_airborne": 0,
                "dormant_frac": 0.0,
                "mean_age": 0.0,
                "mean_energy": 0.0,
                "mean_metabolism_mode": 0.0,
                "mean_rad_shield": 0.0,
                "mean_acid_resist": 0.0,
                "mean_thermal_tol": 0.0,
                "mean_dormancy_thresh": 0.0,
            }

        layers = self.habitat_layer[alive_idx]
        layer_counts = np.bincount(layers, minlength=3)

        return {
            "pop_size": pop_size,
            "layer_0_subsurface": int(layer_counts[0]),
            "layer_1_surface": int(layer_counts[1]),
            "layer_2_airborne": int(layer_counts[2]),
            "dormant_frac": float(np.mean(self.is_dormant[alive_idx])),
            "mean_age": float(np.mean(self.age[alive_idx])),
            "mean_energy": float(np.mean(self.energy[alive_idx])),
            "mean_metabolism_mode": float(np.mean(self.metabolism_mode[alive_idx])),
            "mean_rad_shield": float(np.mean(self.rad_shield[alive_idx])),
            "mean_acid_resist": float(np.mean(self.acid_resist[alive_idx])),
            "mean_thermal_tol": float(np.mean(self.thermal_tol[alive_idx])),
            "mean_dormancy_thresh": float(np.mean(self.dormancy_thresh[alive_idx])),
        }