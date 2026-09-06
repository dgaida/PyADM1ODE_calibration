# ============================================================================
# examples/calibration_workflow_complete.py
# ============================================================================
"""
Complete Calibration Workflow Example

This example demonstrates the full calibration workflow including:
1. Plant setup
2. Measurement data preparation
3. Parameter calibration
4. Sensitivity analysis
5. Identifiability assessment
6. Validation
7. Application of calibrated parameters

Author: PyADM1 Team
Date: 2025
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import pandas as pd

# PyADM1 imports
from pyadm1 import BiogasPlant, Feedstock
from pyadm1.components.biological import Digester
from pyadm1.components.energy import CHP
from pyadm1.core.adm1 import get_state_zero_from_csv

from pyadm1ode_calibration import InitialCalibrator, MeasurementData
from pyadm1ode_calibration.calibration.analysis.identifiability import MAX_COLLINEARITY_INDEX
from pyadm1ode_calibration.calibration.core.simulator import PlantSimulator


def create_example_plant(feedstock: Feedstock) -> BiogasPlant:
    """
    Create an example biogas plant for calibration.

    Args:
        feedstock: Feedstock object for substrate management.

    Returns:
        BiogasPlant: Configured plant ready for calibration.
    """
    print("Creating example biogas plant...")

    plant = BiogasPlant("Calibration Example Plant")

    # Add main digester
    digester = Digester(
        component_id="main_digester",
        feedstock=feedstock,
        V_liq=2000.0,  # 2000 m³
        V_gas=300.0,  # 300 m³
        T_ad=308.15,  # 35°C
        name="Main Fermenter",
    )

    # Load initial state
    data_path = Path("data/initial_states")
    initial_state_file = data_path / "digester_initial8.csv"

    if initial_state_file.exists():
        adm1_state = get_state_zero_from_csv(str(initial_state_file))
        Q_substrates = [15.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0]
        digester.initialize({"adm1_state": adm1_state, "Q_substrates": Q_substrates})
    else:
        print("Warning: Initial state file not found, using defaults")
        digester.initialize()

    plant.add_component(digester)

    # Add CHP unit
    chp = CHP(component_id="chp_main", P_el_nom=500.0, eta_el=0.40, eta_th=0.45, name="CHP Unit")  # 500 kW
    plant.add_component(chp)

    # Connect components
    from pyadm1.configurator.connection_manager import Connection

    plant.add_connection(Connection("main_digester", "chp_main", "gas"))

    # Initialize plant
    plant.initialize()

    print(f"Plant created with {len(plant.components)} components")
    return plant


#: Parameter values the twin record is generated with. They differ from the ADM1
#: defaults (k_m_ac 8.0, Y_su 0.10, k_hyd_ch 4.0), so an uncalibrated model visibly
#: misses the record and the calibration has a known answer to be judged against.
TRUE_PARAMETERS = {"k_m_ac": 6.0, "Y_su": 0.12, "k_hyd_ch": 2.5}

#: Feed profile of the record, one phase per entry: (days, Q_maize, Q_manure) in m3/d.
#: Kinetics only show up in the response to a change. A constant feed drives the
#: digester to a steady state in which several parameters produce the same gas curve,
#: so the record steps the organic load up and down and shifts the substrate mix.
FEED_PHASES = [
    (7, 15.0, 10.0),  # base load, lets the start transient decay
    (6, 22.5, 10.0),  # +50 % maize: organic load step up
    (6, 9.0, 18.0),  # mix shifted to manure at a similar volume
    (6, 10.0, 7.0),  # load step down
    (5, 15.0, 10.0),  # back to base
]


def feed_profile(duration_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Hourly substrate feeds [m3/d] for the phases in :data:`FEED_PHASES`."""
    q1, q2 = [], []
    for days, maize, manure in FEED_PHASES:
        q1 += [maize] * (days * 24)
        q2 += [manure] * (days * 24)
    n = duration_days * 24
    if len(q1) < n:  # repeat the last phase if a longer record was asked for
        q1 += [q1[-1]] * (n - len(q1))
        q2 += [q2[-1]] * (n - len(q2))
    return np.array(q1[:n]), np.array(q2[:n])


def create_twin_measurements(plant: BiogasPlant, duration_days: int = 30, noise: float = 0.02) -> pd.DataFrame:
    """Simulate the plant with known parameters and return the result as measurements.

    A calibration example needs a known answer, otherwise "it converged" is the only
    thing it can report. The record here is produced by this very plant running with
    :data:`TRUE_PARAMETERS`, plus measurement noise, so the calibration can be judged
    on whether it recovers those values.

    Args:
        plant: The plant to simulate. Its state is restored afterwards.
        duration_days: Length of the record.
        noise: Relative Gaussian noise added to every measured channel.

    Returns:
        pd.DataFrame: Feed columns and noisy measured channels, hourly.
    """
    print(f"Generating a {duration_days} day twin record from the plant itself...")
    print(f"  true parameters: {TRUE_PARAMETERS}")

    n_hours = duration_days * 24
    timestamps = pd.date_range(start="2024-01-01", periods=n_hours, freq="h")
    q_maize, q_manure = feed_profile(duration_days)

    feeds = pd.DataFrame({"timestamp": timestamps, "Q_sub1": q_maize, "Q_sub2": q_manure})
    simulated = PlantSimulator(plant, verbose=False, time_varying_feed=True).simulate_with_parameters(
        TRUE_PARAMETERS, MeasurementData(feeds)
    )

    rng = np.random.default_rng(42)
    data = feeds.copy()
    for channel in ("Q_ch4", "Q_gas", "pH", "VFA", "TAC"):
        values = np.asarray(simulated[channel], dtype=float)
        data[channel] = values * (1.0 + rng.normal(0.0, noise, values.size))

    # The digester temperature is controlled, not simulated. Record it as measured.
    data["T_digester"] = 308.15 + rng.normal(0.0, 0.5, n_hours)

    for col in ("Q_sub1", "Q_sub2", "Q_ch4", "Q_gas", "VFA", "TAC"):
        data[col] = data[col].clip(lower=0)
    data["pH"] = data["pH"].clip(6.5, 8.0)

    print(
        f"Created {len(data)} measurement points, "
        f"Q_ch4 {data['Q_ch4'].mean():.0f} m3/d, VFA {data['VFA'].mean():.2f} kg/m3"
    )
    return data


def main():
    """Main calibration workflow."""

    print("=" * 70)
    print("PyADM1 Calibration Workflow Example")
    print("=" * 70)

    # ========================================================================
    # 1. Setup
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. SETUP")
    print("=" * 70)

    # Create feedstock
    feedstock = Feedstock(feeding_freq=48)  # Feed every 48 hours

    # Create plant
    plant = create_example_plant(feedstock)

    # Create measurement data by running the plant itself, see create_twin_measurements
    measurements_df = create_twin_measurements(plant, duration_days=30)

    # Save to CSV for reference
    measurements_df.to_csv("calibration_measurements.csv", index=False)
    print("Saved twin measurements to 'calibration_measurements.csv'")

    # ========================================================================
    # 2. Load and Validate Measurement Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. LOAD AND VALIDATE MEASUREMENT DATA")
    print("=" * 70)

    measurements = MeasurementData.from_csv("calibration_measurements.csv", timestamp_column="timestamp", resample="1H")

    # Validate data quality
    validation = measurements.validate(
        expected_ranges={
            "pH": (6.0, 8.5),
            "VFA": (0.0, 10.0),
            "Q_ch4": (0.0, 2000.0),
        }
    )

    print("\nData validation:")
    print(f"  Valid: {validation.is_valid}")
    print(f"  Quality score: {validation.quality_score:.2f}")
    # ValidationResult.statistics only carries the aggregate; the row count comes
    # straight from the frame.
    print(f"  Number of samples: {len(measurements.data)}")
    print(f"  Missing data: {validation.statistics['missing_pct_avg']:.1f}%")

    if not validation.is_valid:
        validation.print_report()

    # Clean data
    print("\nCleaning measurement data...")
    n_outliers = measurements.remove_outliers(method="zscore", threshold=3.0)
    print(f"  Removed {n_outliers} outliers")

    measurements.fill_gaps(method="interpolate", limit=3)
    print("  Filled gaps with interpolation")

    # ========================================================================
    # 3. Parameter Calibration
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. PARAMETER CALIBRATION")
    print("=" * 70)

    # Create calibrator
    # time_varying_feed is what lets the model see the load changes in the record.
    # Without it PlantSimulator averages the whole feed series into one constant and
    # the phases of the record become invisible.
    calibrator = InitialCalibrator(plant, verbose=True, time_varying_feed=True)

    # Parameters this plant actually reveals. A +20 % change moves VFA by 29 %
    # (k_m_ac), 8.7 % (Y_su) and 0.6 % (k_hyd_ch), so the set spans a strong, a
    # moderate and a weak case and step 6 has something to distinguish.
    # k_dis is deliberately absent: the substrates enter as hydrolysable fractions,
    # never as composites, so disintegration has no substrate and changing k_dis
    # moves no output at all.
    parameters_to_calibrate = [
        "k_m_ac",  # Max. acetate uptake rate, the strongest lever on VFA
        "Y_su",  # Sugar degrader yield
        "k_hyd_ch",  # Carbohydrate hydrolysis rate, weakly observable here
    ]

    # Narrower than the defaults, but wide enough to contain the true values.
    custom_bounds = {
        "k_m_ac": (4.0, 12.0),
        "Y_su": (0.05, 0.15),
        "k_hyd_ch": (1.0, 8.0),
    }

    # Run calibration
    print("\nStarting calibration...")
    result = calibrator.calibrate(
        measurements=measurements,
        parameters=parameters_to_calibrate,
        bounds=custom_bounds,
        objectives=["Q_ch4", "VFA", "pH"],
        weights={"Q_ch4": 0.4, "VFA": 0.5, "pH": 0.1},
        method="differential_evolution",
        validation_split=0.2,
        max_iterations=50,  # Reduced for example
        population_size=10,
        sensitivity_analysis=True,
    )

    # ========================================================================
    # 4. Analyze Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. CALIBRATION RESULTS")
    print("=" * 70)

    if result.success:
        print("\n✓ Calibration successful!")
        print(f"\nObjective value: {result.objective_value:.6f}")
        print(f"Number of iterations: {result.n_iterations}")
        print(f"Execution time: {result.execution_time:.1f} seconds")

        print("\n" + "-" * 70)
        print("Calibrated Parameters:")
        print("-" * 70)
        print(f"  {'parameter':15s}  {'start':>8s}  {'fitted':>8s}  {'true':>8s}  {'error':>8s}")
        for param, value in result.parameters.items():
            initial = result.initial_parameters[param]
            true = TRUE_PARAMETERS.get(param)
            if true is None:
                print(f"  {param:15s}  {initial:8.4f}  {value:8.4f}  {'-':>8s}  {'-':>8s}")
            else:
                print(f"  {param:15s}  {initial:8.4f}  {value:8.4f}  {true:8.4f}  {(value - true) / true * 100:+7.1f}%")

        if result.validation_metrics:
            print("\n" + "-" * 70)
            print("Validation Metrics:")
            print("-" * 70)
            for metric, value in result.validation_metrics.items():
                print(f"  {metric:20s}: {value:8.4f}")

        if result.sensitivity:
            print("\n" + "-" * 70)
            print("Parameter Sensitivities:")
            print("-" * 70)
            for param, sensitivity in result.sensitivity.items():
                print(f"  {param:15s}: {sensitivity:8.4e}")

        # Save results
        result.to_json("calibration_result.json")
        print("\n✓ Results saved to 'calibration_result.json'")

    else:
        print(f"\n✗ Calibration failed: {result.message}")
        return

    # ========================================================================
    # 5. Sensitivity Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. DETAILED SENSITIVITY ANALYSIS")
    print("=" * 70)

    sensitivity_results = calibrator.sensitivity_analysis(
        parameters=result.parameters, measurements=measurements, objectives=["Q_ch4", "VFA", "pH"]
    )

    print("\nParameter Sensitivity Indices:")
    print("-" * 70)
    for param, sens_result in sensitivity_results.items():
        print(f"\n{param} (value: {sens_result.base_value:.4f}):")
        print("  Sensitivity indices:")
        for obj, sens in sens_result.sensitivity_indices.items():
            print(f"    {obj:8s}: {sens:10.4e}")
        print(f"  Variance contribution: {sens_result.variance_contribution:.4e}")

    # ========================================================================
    # 6. Identifiability Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("6. PARAMETER IDENTIFIABILITY ANALYSIS")
    print("=" * 70)

    identifiability_results = calibrator.identifiability_analysis(
        parameters=result.parameters, measurements=measurements, correlation_threshold=0.8
    )

    print("\nParameter Identifiability:")
    print("-" * 70)
    for param, ident_result in identifiability_results.items():
        status = "✓ Identifiable" if ident_result.is_identifiable else "✗ Not identifiable"
        print(f"\n{param}: {status}")
        print(f"  Reason: {ident_result.reason}")
        print(f"  Confidence interval: [{ident_result.confidence_interval[0]:.4f}, {ident_result.confidence_interval[1]:.4f}]")
        print(f"  Objective sensitivity: {ident_result.objective_sensitivity:.4e}")

        if ident_result.correlation_with:
            print("  Correlations:")
            for other_param, corr in ident_result.correlation_with.items():
                if abs(corr) > 0.5:
                    print(f"    {other_param}: {corr:6.3f}")

    # Individual verdicts are only half the answer: parameters that each pass can
    # still be unidentifiable as a set when one can undo what another did.
    subset = calibrator.identifiability_analyzer.analyze_subset(result.parameters, measurements)
    print()
    print("-" * 70)
    print("The set as a whole:")
    print("-" * 70)
    print(f"  Collinearity index: {subset.collinearity_index:.2f} (limit {MAX_COLLINEARITY_INDEX:.0f})")
    print(f"  {'OK' if subset.is_identifiable else 'Rejected'}: {subset.reason}")

    # ========================================================================
    # 7. Apply Calibrated Parameters
    # ========================================================================
    print("\n" + "=" * 70)
    print("7. APPLY CALIBRATED PARAMETERS")
    print("=" * 70)

    # Apply to plant
    digester = plant.components["main_digester"]
    digester.apply_calibration_parameters(result.parameters)

    print("\n✓ Applied calibrated parameters to plant")
    print(f"  Parameters applied: {list(result.parameters.keys())}")

    # Verify application
    applied_params = digester.get_calibration_parameters()
    print("\n  Verification:")
    for param, value in applied_params.items():
        print(f"    {param}: {value:.4f}")

    # ========================================================================
    # 8. Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("8. SUMMARY")
    print("=" * 70)

    print("\nCalibration workflow completed successfully!")
    print("\nKey achievements:")
    print(f"  • Calibrated {len(result.parameters)} parameters")
    print(f"  • Achieved objective value: {result.objective_value:.6f}")
    print(f"  • Validation R² (Q_ch4): {result.validation_metrics.get('Q_ch4_r2', 0):.3f}")
    print(f"  • All parameters identifiable: {all(r.is_identifiable for r in identifiability_results.values())}")
    print("  • Parameters applied to plant model")

    print("\nOutput files generated:")
    print("  • calibration_measurements.csv - Measurement data")
    print("  • calibration_result.json - Calibration results")

    print("\nNext steps:")
    print("  1. Validate calibrated model with independent data")
    print("  2. Use calibrated plant for process optimization")
    print("  3. Monitor parameter drift over time")
    print("  4. Re-calibrate periodically with new measurements")

    print("\n" + "=" * 70)
    print("Calibration workflow complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
