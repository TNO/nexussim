import logging

import altair as alt
import polars as pl

from nexussim.scene.scene_simulation import (make_simulation_report,
                                             nexussim_simulation_from_yaml)

logging.basicConfig(level=logging.INFO)


def main():
    """This example shows how to load a Nexus scene from a YAML file."""

    simulated_scene = nexussim_simulation_from_yaml(
        file="examples/example-control-segments.yaml",
        max_simulation_time=60
        # file="examples/example-uc-traffic.yaml",
        # max_simulation_time=1
    )
    report = make_simulation_report(simulated_scene)

    for node_id, data in report["sim_data"].items():
        if data:
            data = pl.from_dict(data)
            alt.Chart(data).mark_area(
                color="lightblue", interpolate="step-after", line=True
            ).encode(x="timestamp", y="cpu_used").save(f"total_usage_{node_id}.html")

    for node_id, kpis in report["node_kpis"].items():
        print(f"{node_id}:")
        print(f"    mean_cpu_usage: {kpis['mean_cpu_usage']}")
        print(f"    max_cpu_usage: {kpis['max_cpu_usage']}")
        print(f"    min_cpu_usage: {kpis['min_cpu_usage']}")


if __name__ == "__main__":
    main()
