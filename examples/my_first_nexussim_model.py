import pydynaa as pd

from nexussim.compute import Compute
from nexussim.cpus import UnboundedCPU
from nexussim.loggers import CPUUsageBuffer, to_polars_dataframes
from nexussim.network import BaseNetwork
from nexussim.segments.behaviour import ConstantLoadSegment


def sim_time():
    return pd.DynAASim().current_time * 10**6  # in seconds


def main():
    """
    This function:
     (1) creates an instance of ConstantLoadSegment,

     (2)executes it with a compute context containing an UnboundedCPU with 10 CPUs, and

     (3) runs the simulation until completion.
    """
    # Setup a program or segment to execute
    segment = ConstantLoadSegment(cpu_load=3.0, cycles=1000.0)

    # Setup a computing node
    compute = Compute(
        memory=None,
        cpu=UnboundedCPU(10.0, cpu_speed=1000.0, observers=[CPUUsageBuffer(timestampfunc=sim_time)]),
        networks={"lo": BaseNetwork(max_bandwidth=100000000, max_latency=0.010)},
    )

    # Start the execution of the program in the computing node
    compute.execute(segment)

    # Run dynaa simulator
    pd.DynAASim().run()

    # Collect results and report: find the CPUUsageBuffer observer safely
    cpu_usage_buffer = next(
        (o for o in compute.cpu._observers if isinstance(o, CPUUsageBuffer)),
        None,
    )
    if cpu_usage_buffer is None:
        raise RuntimeError("CPUUsageBuffer observer not found on compute.cpu")
    total_usage, usage = to_polars_dataframes(cpu_usage_buffer)
    print(total_usage)
    print(usage)


if __name__ == "__main__":
    main()
