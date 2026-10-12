from __future__ import annotations

import numpy as np

from .model import PhysicsSystem, Sensor, SensorGroup


def coupled_process_benchmark() -> tuple[PhysicsSystem, list[SensorGroup]]:
    """Hydraulic-style coupled process with redundant flow sensing."""
    A = np.array(
        [
            [0.92, 0.08, 0.00, 0.00],
            [0.05, 0.90, 0.10, 0.00],
            [0.00, 0.08, 0.88, 0.10],
            [0.00, 0.00, 0.07, 0.93],
        ],
        dtype=float,
    )
    coupling = np.abs(A - np.diag(np.diag(A)))
    sensors = {
        "pressure_in": Sensor("pressure_in", 0, cost=1.0, energy=0.8, reliability=0.98),
        "flow": Sensor("flow", 1, cost=1.2, energy=1.0, reliability=0.96),
        "level": Sensor("level", 2, cost=1.0, energy=0.7, reliability=0.99),
        "pressure_out": Sensor("pressure_out", 3, cost=1.1, energy=0.9, reliability=0.97),
        "flow_backup": Sensor("flow_backup", 1, cost=0.9, energy=0.7, reliability=0.94),
    }
    groups = [
        SensorGroup("hydraulic-core", ("pressure_in", "flow", "level")),
        SensorGroup("downstream-state", ("flow", "level", "pressure_out")),
        SensorGroup("redundant-flow", ("flow", "flow_backup")),
        SensorGroup("boundary-pressure", ("pressure_in", "pressure_out")),
        SensorGroup("low-power-state", ("flow_backup", "level")),
    ]
    return PhysicsSystem(A=A, sensors=sensors, coupling=coupling), groups


def thermal_building_benchmark() -> tuple[PhysicsSystem, list[SensorGroup]]:
    """Reduced-order zone/wall/HVAC thermal benchmark."""
    A = np.array(
        [
            [0.91, 0.06, 0.02, 0.00],
            [0.05, 0.92, 0.00, 0.02],
            [0.03, 0.00, 0.90, 0.05],
            [0.00, 0.03, 0.06, 0.89],
        ],
        dtype=float,
    )
    coupling = np.abs(A - np.diag(np.diag(A)))
    sensors = {
        "zone_a_temp": Sensor("zone_a_temp", 0, 0.8, 0.4, 0.99),
        "zone_b_temp": Sensor("zone_b_temp", 1, 0.8, 0.4, 0.99),
        "supply_temp": Sensor("supply_temp", 2, 1.0, 0.6, 0.97),
        "return_temp": Sensor("return_temp", 3, 1.0, 0.6, 0.97),
        "zone_a_backup": Sensor("zone_a_backup", 0, 0.7, 0.35, 0.95),
    }
    groups = [
        SensorGroup("zone-comfort", ("zone_a_temp", "zone_b_temp")),
        SensorGroup("hvac-loop", ("supply_temp", "return_temp")),
        SensorGroup("zone-a-resilient", ("zone_a_temp", "zone_a_backup", "supply_temp")),
        SensorGroup("thermal-chain", ("zone_a_temp", "supply_temp", "return_temp")),
        SensorGroup("low-power-thermal", ("zone_a_backup", "zone_b_temp")),
    ]
    return PhysicsSystem(A=A, sensors=sensors, coupling=coupling), groups


def battery_pack_benchmark() -> tuple[PhysicsSystem, list[SensorGroup]]:
    """Reduced-order battery electro-thermal benchmark."""
    A = np.array(
        [
            [0.98, 0.02, 0.00, 0.00],
            [0.03, 0.94, 0.02, 0.00],
            [0.00, 0.04, 0.92, 0.03],
            [0.01, 0.00, 0.05, 0.93],
        ],
        dtype=float,
    )
    coupling = np.abs(A - np.diag(np.diag(A)))
    sensors = {
        "soc": Sensor("soc", 0, 1.2, 0.5, 0.97),
        "voltage": Sensor("voltage", 1, 0.8, 0.4, 0.99),
        "current": Sensor("current", 2, 0.8, 0.45, 0.99),
        "temperature": Sensor("temperature", 3, 0.9, 0.35, 0.98),
        "voltage_backup": Sensor("voltage_backup", 1, 0.7, 0.35, 0.95),
    }
    groups = [
        SensorGroup("electrical-core", ("soc", "voltage", "current")),
        SensorGroup("electro-thermal", ("voltage", "current", "temperature")),
        SensorGroup("voltage-redundancy", ("voltage", "voltage_backup")),
        SensorGroup("safety-monitor", ("current", "temperature")),
        SensorGroup("low-cost-electrical", ("voltage_backup", "current")),
    ]
    return PhysicsSystem(A=A, sensors=sensors, coupling=coupling), groups


def benchmark_suite() -> dict[str, tuple[PhysicsSystem, list[SensorGroup]]]:
    return {
        "coupled-process": coupled_process_benchmark(),
        "thermal-building": thermal_building_benchmark(),
        "battery-pack": battery_pack_benchmark(),
    }
