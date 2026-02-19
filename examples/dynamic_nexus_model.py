import pydynaa as pd

from nexussim.compute import Compute
from nexussim.cpus import UnboundedCPU
from nexussim.loggers import CPUUsageBuffer, to_polars_dataframes
from nexussim.network import BaseNetwork
from nexussim.samplers import exponential_load_sampler, gaussian_load_sampler
from nexussim.segments.behaviour import SampledLoadSegment
from nexussim.segments.control import SequenceSegment, WhileSegment, forever


def sim_time():
    return pd.DynAASim().current_time * 10**6  # in seconds


def main():
    """
    This function:
     (1) creates an instance of SampledLoadSegment,

     (2) executes it with a compute context containing sequence of two segments, and

     (3) runs the simulation until completion.
    """
    # Setup a program or segment to execute (segments takes one hour to complete)
    segment1 = SampledLoadSegment(
        cpu_load_sampler=gaussian_load_sampler(min_load=0, max_load=2.0, mean=0.8, std=0.01),
        freq=1.0,
        duration=200.0,
    )
    segment2 = SampledLoadSegment(
        cpu_load_sampler=exponential_load_sampler(max_load=4.0, mean=1.2),
        freq=1.0,
        duration=200.0,
    )

    # Setup a computing node
    compute = Compute(
        memory=None,
        cpu=UnboundedCPU(10.0, cpu_speed=1000, observers=[CPUUsageBuffer(timestampfunc=sim_time)]),
        networks={"lo": BaseNetwork(max_bandwidth=100000000, max_latency=0.010)},
    )

    # Start the execution of the program in the computing node
    # The program is a while loop that executes a of segment1, segment2, segment1,
    # segment2.
    # It never stops
    compute.execute(
        WhileSegment(
            segment=SequenceSegment([segment1, segment2, segment1, segment2]),
            condition=forever,
        )
    )

    # Run dynaa simulator
    pd.DynAASim().run(3600)  # runs for one hour

    # Collect results and report: find the CPUUsageBuffer observer safely
    cpu_usage_buffer = next(
        (o for o in compute.cpu._observers if isinstance(o, CPUUsageBuffer)),
        None,
    )
    if cpu_usage_buffer is None:
        raise RuntimeError("CPUUsageBuffer observer not found on compute.cpu")
    total_usage, usage = to_polars_dataframes(cpu_usage_buffer)
    print(total_usage)
    print(total_usage.describe())
    total_usage.plot.line(x="timestamp", y="cpu_used").save("total_usage.html")
    print(usage)


if __name__ == "__main__":
    main()
