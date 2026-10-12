from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .adaptive import reconfigure_selection
from .benchmarks import benchmark_suite
from .selection import select_groups


def _system(name: str):
    suite = benchmark_suite()
    if name not in suite:
        raise ValueError(f"unknown domain {name!r}; choose from {', '.join(suite)}")
    return suite[name]


def main() -> None:
    parser = argparse.ArgumentParser(prog="pgsgs", description="Physics-guided sensor group selection")
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="select sensor groups under resource budgets")
    select.add_argument("--domain", default="coupled-process")
    select.add_argument("--max-cost", type=float, default=3.2)
    select.add_argument("--max-energy", type=float, default=2.6)

    adapt = sub.add_parser("adapt", help="reconfigure after sensor unavailability")
    adapt.add_argument("--domain", default="coupled-process")
    adapt.add_argument("--max-cost", type=float, default=3.2)
    adapt.add_argument("--max-energy", type=float, default=2.6)
    adapt.add_argument("--unavailable", default="")
    adapt.add_argument("--minimum-observability", type=float, default=0.75)

    args = parser.parse_args()
    system, groups = _system(args.domain)

    if args.command == "select":
        result = select_groups(system, groups, max_cost=args.max_cost, max_energy=args.max_energy)
        print(json.dumps(asdict(result), indent=2))
        return

    unavailable = {name.strip() for name in args.unavailable.split(",") if name.strip()}
    result = reconfigure_selection(
        system,
        groups,
        max_cost=args.max_cost,
        max_energy=args.max_energy,
        unavailable_sensors=unavailable,
        minimum_observability=args.minimum_observability,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
