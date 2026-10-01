import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# Ordered list of validation scripts to execute
script_order = [
    "validate_initial_local_dose.py",
    "validate_specific_energy.py",
    "validate_specific_energy_saturation.py",
    "validate_mk_table_classic.py",
    "validate_mk_table_classic_dose.py",
    "validate_mk_table_stochastic.py",
    "validate_sf_table_classic.py",
    "validate_sf_table_stochastic.py",
    "validate_sf_table_stochastic_pO2.py",
    "validate_sf_table_stochastic_OER.py",
    "validate_mk_table_mcf_RBE.py",
    "validate_mk_table_mcf_weight.py",
    "validate_sf_table_mcf.py",
]

validation_dir = Path(__file__).resolve().parent
log_messages = []

print("Starting execution of validation scripts:\n")

with tqdm(
    total=len(script_order),
    unit="script",
    bar_format="Validating |{bar}| {n_fmt}/{total_fmt} {percentage:3.0f}% • ⏱ {elapsed}  | {postfix}"
) as pbar:
    for script_name in script_order:
        pbar.set_postfix_str(script_name)
        script_path = validation_dir / script_name

        if not script_path.exists():
            log_messages.append(f"⚠️  Script not found: {script_name}")
            pbar.update(1)
            continue

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            log_messages.append(
                f"❌ Error in {script_name}:\n{result.stderr.strip()}\n"
            )
        else:
            log_messages.append(f"✅ Completed: {script_name}")

        pbar.update(1)

print("\nValidation summary:\n" + "-" * 60)
for line in log_messages:
    print(line)
print("-" * 60)
print("All validation scripts have been processed.")
