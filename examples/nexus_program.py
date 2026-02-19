from nexussim.compute import Compute
from nexussim.cpus import UnboundedCPU
from nexussim.samplers import gaussian_load_sampler
from nexussim.loggers import CPUUsageBuffer, to_polars_dataframes
import pydynaa as pd
from nexussim.program import (
    Program,
    constant_load_segment,
    repeat,
    sampled_load_segment,
    sequence,
)


def sim_time():
    return pd.DynAASim().current_time * 10**6  # in seconds


def main():
    """
    This function:
     (1) creates an instance of ConstantLoadSegment,

     (2)executes it with a compute context containing an UnboundedCPU with 10 CPUs, and

     (3) runs the simulation until completion.
    """
    ## Setup a program or segment to execute
    program = Program(
        name="example",
        body=repeat(
            sequence(
                constant_load_segment(cpu_load=3.0, duration_sec=7.0),
                sampled_load_segment(
                    cpu_load_sampler=gaussian_load_sampler(
                        min_load=0, max_load=2.0, mean=0.8, std=0.01
                    ),
                    freq=1.0,
                    duration_sec=7.0,
                ),
            ),
            times=2,
        ),
    )

    ## Setup a computing node
    compute = Compute(
        memory=None,
        cpu=UnboundedCPU(10.0, observers=[CPUUsageBuffer(timestampfunc=sim_time)]),
    )

    ## Start the execution of the program in the computing node
    compute.execute(program)

    ## Run dynaa simulator
    pd.DynAASim().run()

    ## Collect results and report
    cpu_usage_buffer = compute.context.compute.cpu._observers[0]
    total_usage, usage = to_polars_dataframes(cpu_usage_buffer)
    print(total_usage)
    print(usage)


if __name__ == "__main__":
    main()
