import logging
from pathlib import Path

import numpy as np
import polars as pl
import pydynaa as pd
from numpy.typing import ArrayLike

from nexussim.loggers import to_polars_dataframes
from nexussim.scene.scene import NexusScene

logger = logging.getLogger("♾️.simulation")


class NexusSimulationError(Exception):
    """Exception raised when a simulation fails."""


def nexussim_simulation_from_yaml(
    file: Path, max_simulation_time: float = 1.0
) -> NexusScene:
    """Simulates a Nexus scene from a YAML file.

    Args:
        file (Path):
        The path to the YAML file containing the scene specification.

    Returns:
        NexusScene:
        A NexusScene object representing the simulated scene.
    """
    logger.info("Preparing simulation engine...")
    pd.DynAASim().reset()

    logger.info("Loading Nexus scene...")
    try:
        scene = NexusScene.from_yaml(
            file=file,
            validation_schema="examples/mirto-dynaa-v1-openapi.yaml",
        )
    except Exception as e:
        logger.error("Could not load Nexus scene: %s", e)
        raise NexusSimulationError(f"♾️⛓️‍💥 Scene creation failed.\n Cause {e}") from e

    logger.info("Starting simulation...")
    try:
        pd.DynAASim().run(max_simulation_time)
    except Exception as e:
        logger.error("Simulation failed with error: %s", e)
        raise NexusSimulationError(f"♾️⛓️‍💥 Simulation failed.\n Cause {e}") from e

    logger.info("Simulation completed.")
    simulated_scene = scene
    logger.info("Collecting results and reporting...")

    return simulated_scene


def nexussim_simulation_from_dict(
    desc: dict, max_simulation_time: float = 1.0
) -> NexusScene:
    """Simulates a Nexus scene from a YAML file.

    Args:
        file (Path):
        The path to the YAML file containing the scene specification.

    Returns:
        NexusScene:
        A NexusScene object representing the simulated scene.
    """
    logger.info("Preparing simulation engine...")
    pd.DynAASim().reset()

    logger.info("Loading Nexus scene...")
    try:
        scene = NexusScene.from_dict(scene_specification=desc)
    except Exception as e:
        logger.error(f"Could not load Nexus scene: {e}")
        raise NexusSimulationError(f"♾️⛓️‍💥 Scene creation failed.\n Cause {e}")

    logger.info("Starting simulation...")
    try:
        pd.DynAASim().run(max_simulation_time)
    except Exception as e:
        logger.error(f"Simulation failed with error: {e}")
        raise NexusSimulationError(f"♾️⛓️‍💥 Simulation failed.\n Cause {e}")

    logger.info("Simulation completed.")
    simulated_scene = scene
    logger.info("Collecting results and reporting...")

    return simulated_scene


def make_simulation_report(scene: NexusScene):
    """Generates a report out of the simulated scene.

    The report is in form of a dictionary with the following fields:

    - scene_description: A string description of the scene.
    """
    return dict(scene_description=scene.as_dict,
                node_kpis=_node_kpis(scene),
                sim_data=_sim_data(scene)
                )


def _node_kpis(scene: NexusScene):
    """Generates KPIs reports per compute node in the scene.

    Args:
        scene: The scene to generate KPIs for.
    """
    node_kpis = dict()
    # Collect results and report
    for node_id, compute in scene.computes.items():
        if not compute.context.compute.cpu._observers:
            continue

        cpu_usage_buffer = compute.context.compute.cpu._observers[0]
        total_usage, _ = to_polars_dataframes(cpu_usage_buffer)
        total_usage = total_usage.with_columns(
            timestamp=pl.col("timestamp").dt.epoch(time_unit="ns") / 1000000000.0
        )
        cpu_used_measurements = total_usage["cpu_used"].to_list()
        timestamps = total_usage["timestamp"].to_list()
        kpi_mean_cpu_usage = _mean_cpu_usage(
            timestamps, cpu_used_measurements
        )
        kpi_max_cpu_usage = _max_cpu_usage(cpu_used_measurements)
        kpi_min_cpu_usage = _min_cpu_usage(cpu_used_measurements)
        node_kpis[node_id] = dict(
            mean_cpu_usage=kpi_mean_cpu_usage,
            max_cpu_usage=kpi_max_cpu_usage,
            min_cpu_usage=kpi_min_cpu_usage,
        )

    return node_kpis


def _sim_data(scene: NexusScene):
    simulation_data = dict()
    for node_id, compute in scene.computes.items():
        if not compute.context.compute.cpu._observers:
            continue

        cpu_usage_buffer = compute.context.compute.cpu._observers[0]
        total_usage, _ = to_polars_dataframes(cpu_usage_buffer)
        total_usage = total_usage.with_columns(
            timestamp=pl.col("timestamp").dt.epoch(time_unit="ns") / 1000000000.0
        )
        simulation_data[node_id] = total_usage.to_dict(as_series=False)
    return simulation_data


# KPIS

def _mean_cpu_usage(timestamps: ArrayLike, cpu_usage: ArrayLike) -> float:
    """Calculate the mean CPU usage over the given timestamps and CPU usage values.

    Parameters
    ----------
    timestamps : array-like
        The timestamps of the CPU usage values, sorted in ascending order.
    cpu_usage : array-like
        The CPU usage values equivalent to the timestamps.

    Returns
    -------
    float
        The mean CPU usage over the given timestamps and CPU usage values.
    """
    return np.dot(np.diff(timestamps), cpu_usage[:-1]) / (
        timestamps[-1] - timestamps[0]
    )


def _max_cpu_usage(cpu_usage: ArrayLike) -> float:
    """Calculate the maximum CPU usage over the given timestamps and CPU usage values.

    Parameters
    ----------
    cpu_usage : array-like
        The CPU usage values equivalent to the timestamps.

    Returns
    -------
    float
        The maximum CPU usage over the given timestamps and CPU usage values.
    """
    return max(cpu_usage)


def _min_cpu_usage(cpu_usage: ArrayLike) -> float:
    """Calculate the minimum CPU usage over the given timestamps and CPU usage values.

    Parameters
    ----------
    cpu_usage : array-like
        The CPU usage values equivalent to the timestamps.

    Returns
    -------
    float
        The minimum CPU usage over the given timestamps and CPU usage values.
    """
    return min(cpu_usage)
