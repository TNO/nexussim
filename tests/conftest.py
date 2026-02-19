import pydynaa as pd
import pytest

from nexussim.compute import Compute
from nexussim.context import Context
from nexussim.cpus import UnboundedCPU
from nexussim.memory import UnboundedMemory
from nexussim.network import BaseNetwork


@pytest.fixture(scope="function")
def dynaa_sim():
    sim = pd.DynAASim()
    sim.reset()
    yield sim
    sim.reset()


@pytest.fixture(scope="function")
def context():
    return Context(
        compute=Compute(
            memory=UnboundedMemory(max_memory=10),
            cpu=UnboundedCPU(max_cpus=10.0, cpu_speed=1000),
            networks={"lo": BaseNetwork()},
        )
    )
