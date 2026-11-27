import json
from pathlib import Path
import numpy as np

from src.environment import PlanetEnvironment
from src.population import Population
from src.logger import TelemetryLogger


class SimulationEngine:
    """Headless, vectorized evolutionary simulation engine."""

    def __init__(
        self,
        planet_key: str,
        seed: int = 42,
        planets_config: str | Path = "configs/planets.json",
        sim_config: str | Path = "configs/sim_defaults.json",
    ):
        with Path(sim_config).open("r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.planet_key = planet_key
        self.seed = seed
        self.env = PlanetEnvironment(planet_key, planets_config)

        pop_cfg = self.cfg["population"]
        self.pop = Population(max_pop=int(pop_cfg["max_pop"]), seed=seed)
        self.pop.seed_initial_population(
            initial_pop=int(pop_cfg["initial_pop"]),
            initial_energy=float(self.cfg["energy"]["initial_energy"]),
        )

        # Unpack hot-loop constants as float32 to avoid dict lookups per tick
        self.carrying_capacity = np.float32(pop_cfg["carrying_capacity_per_layer"])
        self.ticks_per_run = int(self.cfg["time"]["ticks_per_run"])
        self.hours_per_tick = float(self.cfg["time"]["hours_per_tick"])
        self.log_interval = int(self.cfg["time"]["log_interval"])

        e_cfg = self.cfg["energy"]
        self.repro_thresh = np.float32(e_cfg["reproduction_threshold"])
        self.repro_cost_ratio = float(e_cfg["reproduction_cost_ratio"])
        self.basal_cost = np.float32(e_cfg["basal_cost"])
        self.dormant_basal_mult = np.float32(e_cfg["dormant_basal_multiplier"])
        self.max_solar_harvest = np.float32(e_cfg["max_solar_harvest"])
        self.max_chem_harvest = np.float32(e_cfg["max_chem_harvest"])
        self.optimal_temp_c = np.float32(e_cfg["optimal_temp_c"])
        self.base_temp_band_c = np.float32(e_cfg["base_temp_band_c"])
        self.max_extra_temp_band_c = np.float32(e_cfg["max_extra_temp_band_c"])

        u_cfg = e_cfg["upkeep_weights"]
        self.w_rad_upkeep = np.float32(u_cfg["rad_shield"])
        self.w_acid_upkeep = np.float32(u_cfg["acid_resist"])
        self.w_therm_upkeep = np.float32(u_cfg["thermal_tol"])

        d_cfg = e_cfg["damage_weights"]
        self.w_rad_dmg = np.float32(d_cfg["radiation"])
        self.w_acid_dmg = np.float32(d_cfg["acidity"])
        self.w_therm_dmg = np.float32(d_cfg["thermal_per_c"])
        self.w_pressure_dmg = np.float32(d_cfg["pressure"])

        evo_cfg = self.cfg["evolution"]
        self.mutation_sigma = float(evo_cfg["mutation_sigma"])
        self.layer_mutation_prob = float(evo_cfg["layer_mutation_prob"])
        self.max_age_ticks = int(evo_cfg["max_age_ticks"])

    def step(self, tick: int) -> tuple[int, int, bool]:
        """
        Advances the simulation by one tick.
        Returns (births, deaths, is_extinct).
        """
        alive_idx = np.flatnonzero(self.pop.alive)
        if alive_idx.size == 0:
            return 0, 0, True

        cond = self.env.get_tick_conditions(tick, self.hours_per_tick)
        layers = self.pop.habitat_layer[alive_idx]

        # 1. Density-dependent resource competition per habitat layer
        layer_counts = np.bincount(layers, minlength=3).astype(np.float32)
        density_factor = np.clip(1.0 - (layer_counts / self.carrying_capacity), 0.0, 1.0)
        local_density = density_factor[layers]

        # 2. Compute Potential Harvest & Active Costs first
        metab = self.pop.metabolism_mode[alive_idx]
        photo_gain = metab * cond["solar_flux"][layers] * self.max_solar_harvest
        chemo_gain = (1.0 - metab) * cond["chem_energy"][layers] * self.max_chem_harvest
        potential_harvest = (photo_gain + chemo_gain) * local_density

        trait_upkeep = (
            self.pop.rad_shield[alive_idx] * self.w_rad_upkeep
            + self.pop.acid_resist[alive_idx] * self.w_acid_upkeep
            + self.pop.thermal_tol[alive_idx] * self.w_therm_upkeep
        )
        buoyancy = cond["buoyancy_cost"][layers]

        # 3. Environmental Damage
        rad_dmg = cond["radiation"][layers] * (1.0 - self.pop.rad_shield[alive_idx]) * self.w_rad_dmg
        acid_dmg = cond["acidity"][layers] * (1.0 - self.pop.acid_resist[alive_idx]) * self.w_acid_dmg

        temp_diff = np.abs(cond["temp_c"][layers] - self.optimal_temp_c)
        allowed_band = self.base_temp_band_c + (self.pop.thermal_tol[alive_idx] * self.max_extra_temp_band_c)
        thermal_dmg = np.maximum(np.float32(0.0), temp_diff - allowed_band) * self.w_therm_dmg

        pressure_dmg = cond["pressure_hazard"][layers] * self.w_pressure_dmg
        total_damage = rad_dmg + acid_dmg + thermal_dmg + pressure_dmg

        # 4. Smart Dormancy: sleep when low on energy AND running a net deficit; wake up at sunrise
        active_cost = self.basal_cost + trait_upkeep + buoyancy + total_damage
        dormancy_level = self.pop.dormancy_thresh[alive_idx] * self.repro_thresh
        is_dormant = (self.pop.energy[alive_idx] < dormancy_level) & (potential_harvest < active_cost)
        self.pop.is_dormant[alive_idx] = is_dormant

        # 5. Apply Actual Harvest & Costs based on Dormancy State
        harvest = np.where(is_dormant, np.float32(0.0), potential_harvest)
        basal = np.where(is_dormant, self.basal_cost * self.dormant_basal_mult, self.basal_cost)
        upkeep = np.where(is_dormant, trait_upkeep * np.float32(0.25), trait_upkeep) + buoyancy

        # 6. Apply Energy Conservation & Aging
        self.pop.energy[alive_idx] += harvest - (basal + upkeep + total_damage)
        self.pop.age[alive_idx] += 1

        # 7. Process Deaths (Starvation or Old Age)
        dead_local = (self.pop.energy[alive_idx] <= 0.0) | (self.pop.age[alive_idx] >= self.max_age_ticks)
        dead_idx = alive_idx[dead_local]
        deaths = int(dead_idx.size)
        if deaths > 0:
            self.pop.alive[dead_idx] = False

        # 8. Process Reproduction (Active survivors above energy threshold)
        survivor_idx = alive_idx[~dead_local]
        if survivor_idx.size == 0:
            return 0, deaths, True

        ready_mask = (~self.pop.is_dormant[survivor_idx]) & (self.pop.energy[survivor_idx] >= self.repro_thresh)
        parent_idx = survivor_idx[ready_mask]
        births = self.pop.reproduce(
            parent_indices=parent_idx,
            repro_cost_ratio=self.repro_cost_ratio,
            mutation_sigma=self.mutation_sigma,
            layer_mutation_prob=self.layer_mutation_prob,
        )

        return births, deaths, False

    def run(self, logger: TelemetryLogger | None = None) -> dict:
        """Runs the full simulation loop and returns final population stats."""
        if logger is not None:
            logger.log(tick=0, stats=self.pop.get_stats(), births=0, deaths=0)

        extinct_tick = None
        for tick in range(1, self.ticks_per_run + 1):
            births, deaths, is_extinct = self.step(tick)

            if is_extinct:
                extinct_tick = tick
                if logger is not None:
                    logger.log(tick=tick, stats=self.pop.get_stats(), births=births, deaths=deaths)
                break

            if logger is not None and (tick % self.log_interval == 0 or tick == self.ticks_per_run):
                logger.log(tick=tick, stats=self.pop.get_stats(), births=births, deaths=deaths)

        final_stats = self.pop.get_stats()
        final_stats["extinct_tick"] = extinct_tick
        return final_stats