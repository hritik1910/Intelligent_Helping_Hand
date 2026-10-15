
from pathlib import Path
import pandas as pd

source = Path("data/raw/production_runs.csv")
destination = Path("data/processed/production_runs.csv")

df = pd.read_csv(source)

print("Original row count:", len(df))
print("Missing values by column:")
print(df.isna().sum())

# Production run IDs must be unique.
if df["run_id"].duplicated().any():
    raise ValueError("Duplicate run IDs found. Investigate before cleaning.")

# These columns are necessary for basic validation.
required_columns = [
    "run_id",
    "line_id",
    "units_produced",
    "units_passed_first_pass",
    "units_reworked",
    "units_scrapped",
    "downtime_min"
]

if df[required_columns].isna().any().any():
    raise ValueError(
        "Required fields contain missing data. Investigate the source."
    )

# Production counts must be sensible.
if (df["units_produced"] <= 0).any():
    raise ValueError("Some runs have zero or negative production.")

if (df["downtime_min"] < 0).any():
    raise ValueError("Negative downtime detected.")

# Every unit must be accounted for.
accounted_units = (
    df["units_passed_first_pass"]
    + df["units_reworked"]
    + df["units_scrapped"]
)

if not (accounted_units == df["units_produced"]).all():
    raise ValueError("Production counts do not reconcile.")

# Check that the primary defect categories reconcile too.
classified_failures = (
    df["spi_defects"]
    + df["aoi_defects"]
    + df["ict_failures"]
)

actual_failures = (
    df["units_reworked"] + df["units_scrapped"]
)

if not (classified_failures == actual_failures).all():
    raise ValueError("Defect counts do not reconcile.")

# Convert the date field to a proper date/time
df["run_timestamp"] = pd.to_datetime(
    df["run_timestamp"],
    format="%d-%m-%Y %H:%M",
    errors="raise"
)

destination.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(destination, index=False)

print("Validation passed.")
print("Clean data saved to:", destination)
