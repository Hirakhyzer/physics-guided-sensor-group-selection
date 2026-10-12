from pgsgs import (
    SelectionWeights,
    coupled_process_benchmark,
    evaluate_sensor_dropout,
    select_groups,
)


def main() -> None:
    system, groups = coupled_process_benchmark()
    result = select_groups(
        system,
        groups,
        max_cost=3.0,
        max_energy=2.5,
        max_groups=2,
        weights=SelectionWeights(),
    )
    dropout = evaluate_sensor_dropout(system, result.selected_sensors, failures=1)

    print("selected groups:", result.selected_groups)
    print("selected sensors:", result.selected_sensors)
    print("total cost:", round(result.total_cost, 3))
    print("total energy:", round(result.total_energy, 3))
    print("utility:", round(result.utility, 3))
    print("baseline observability:", round(dropout.baseline_observability, 3))
    print("worst single-dropout observability:", round(dropout.worst_case_observability, 3))


if __name__ == "__main__":
    main()
