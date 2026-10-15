
from pathlib import Path

import numpy as np
import pandas as pd

# 1. Prepare the random number generator
rng = np.random.default_rng(42)

# 2. Decide how many production runs to simulate
n = 3000

# 3. Create timestamps covering roughly one year
dates = pd.date_range(
    start="2025-01-01",
    periods=n,
    freq="3h"
)

# 4. Create basic factory information
line = rng.choice(
    ["L1", "L2", "L3"],
    size=n
)

product = rng.choice(
    ["PCB-A", "PCB-B", "PCB-C"],
    size=n
)

hour = dates.hour

shift = np.select(
    [
        (hour >= 6) & (hour < 14),
        (hour >= 14) & (hour < 22)
    ],
    ["Day", "Evening"],
    default="Night"
)

# 5. Simulate process measurements
pressure = rng.normal(55, 4, n)
paste_temp = rng.normal(24, 2, n)
ambient_temp = rng.normal(25, 2.5, n)

line_bias = np.select(
    [line == "L2", line == "L3"],
    [-4, 2],
    default=0
)

spi_height = (
    120
    + (pressure - 55) * 1.25
    + (paste_temp - 24) * 0.8
    + line_bias
    + rng.normal(0, 6, n)
)

paste_volume = (
    100
    + (pressure - 55) * 1.1
    + rng.normal(0, 4.5, n)
)

machine_temp = (
    35
    + 0.5 * (ambient_temp - 25)
    + (line == "L3") * 2
    + rng.normal(0, 3, n)
)

# 6. Simulate production time and output
planned_minutes = np.full(n, 60.0)

downtime_minutes = np.clip(
    rng.gamma(2.0, 2.2, n)
    + (line == "L3") * 1.5,
    0,
    20
).round(1)

runtime_minutes = planned_minutes - downtime_minutes

ideal_cycle_seconds = rng.uniform(2.0, 3.2, n)
performance_factor = rng.uniform(0.65, 0.93, n)

units = np.floor(
    runtime_minutes * 60
    / ideal_cycle_seconds
    * performance_factor
).astype(int)

# 7. Create simulated quality outcomes
failure_probability = (
    0.025
    + 0.0018 * np.abs(spi_height - 120)
    + 0.0007 * np.abs(paste_volume - 100)
    + (line == "L2") * 0.006
    + (shift == "Night") * 0.004
    + (machine_temp > 40) * 0.004
)

failure_probability = np.clip(
    failure_probability, 0.015, 0.18
)

first_pass_failures = rng.binomial(
    units, failure_probability
)

passed_first_pass = units - first_pass_failures

# Allocate every first-pass failure to one primary category
category_counts = np.array([
    rng.multinomial(
        int(failures),
        [0.35, 0.45, 0.20]
    )
    for failures in first_pass_failures
])

scrapped = rng.binomial(first_pass_failures, 0.18)
reworked = first_pass_failures - scrapped

# Assume reworked units pass after repair
good_final = units - scrapped

# 8. Assign illustrative product costs in rupees
unit_cost = np.select(
    [
        product == "PCB-A",
        product == "PCB-B",
        product == "PCB-C"
    ],
    [650.0, 950.0, 1400.0],
    default=800.0
)

scrap_cost = scrapped * unit_cost
rework_cost = reworked * rng.uniform(100, 250, n)

# 9. Combine everything into one table
df = pd.DataFrame({
    "run_id": np.arange(1, n + 1),
    "run_timestamp": dates,
    "line_id": line,
    "product_id": product,
    "shift": shift,
    "units_produced": units,
    "units_passed_first_pass": passed_first_pass,
    "units_reworked": reworked,
    "units_scrapped": scrapped,
    "units_good_final": good_final,
    "planned_time_min": planned_minutes,
    "downtime_min": downtime_minutes,
    "ideal_cycle_time_sec": ideal_cycle_seconds,
    "stencil_pressure_kpa": pressure,
    "paste_temp_c": paste_temp,
    "spi_height_um": spi_height,
    "paste_volume_pct": paste_volume,
    "ambient_temp_c": ambient_temp,
    "machine_temp_c": machine_temp,
    "spi_defects": category_counts[:, 0],
    "aoi_defects": category_counts[:, 1],
    "ict_failures": category_counts[:, 2],
    "scrap_cost_inr": scrap_cost,
    "rework_cost_inr": rework_cost
})

df["fpy_pct"] = (
    df["units_passed_first_pass"]
    / df["units_produced"] * 100
)

# Training label for the simulated demonstration
df["high_quality_risk"] = df["fpy_pct"] < 95

df["data_origin"] = "synthetic_demo"

# 10. Save the data to a CSV file
output = Path("data/raw/production_runs.csv")
output.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output, index=False)

print("Dataset created successfully!")
print("Rows and columns:", df.shape)
print(df.head())
print("Saved to:", output)
