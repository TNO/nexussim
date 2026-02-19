import pytest
from nexussim.loggers import CPUUsageBuffer, to_polars_dataframes
from nexussim.cpus import UnboundedCPU


class DummyObserver:
    def update(self, cpu_provider):
        self.total = cpu_provider.used_cpus()
        self.usage = cpu_provider.cpu_usage()


def test_unbounded_cpu_initialization():
    max_cpus = 10.0
    cpu = UnboundedCPU(max_cpus)
    assert cpu.max_cpus() == max_cpus
    assert cpu.used_cpus() == 0
    assert cpu.free_cpus() == max_cpus
    assert cpu.cpu_usage() == {}


def test_unbounded_cpu_request_cpu():
    cpu = UnboundedCPU(10.0)
    assert cpu.request_cpu("process1", 2.5) is True
    assert cpu.used_cpus() == 2.5
    assert cpu.free_cpus() == 7.5
    assert cpu.cpu_usage() == {"process1": 2.5}


def test_unbounded_cpu_release_cpu():
    cpu = UnboundedCPU(10.0)
    cpu.request_cpu("process1", 2.5)
    assert cpu.release_cpu("process1") is True
    assert cpu.used_cpus() == 0
    assert cpu.free_cpus() == 10.0
    assert cpu.cpu_usage() == {"process1": 0.0}


def test_unbounded_cpu_flush():
    cpu = UnboundedCPU(10.0)
    cpu.request_cpu("process1", 2.5)
    cpu.flush()
    assert cpu.used_cpus() == 0
    assert cpu.free_cpus() == 10.0
    assert cpu.cpu_usage() == {}


def test_dummy_observer():
    dummy_observer = DummyObserver()
    cpu = UnboundedCPU(10.0, observers=[dummy_observer])
    cpu.request_cpu("process1", 2.5)
    assert dummy_observer.total == 2.5
    assert dummy_observer.usage == {"process1": 2.5}


def test_CPUDataFrameLogger():
    observer = CPUUsageBuffer()
    cpu = UnboundedCPU(10.0, observers=[observer])
    cpu.request_cpu("process1", 2.5)
    cpu.request_cpu("process2", 1.0)
    cpu.release_cpu("process1")
    total_usage, usage = to_polars_dataframes(observer)
    assert "timestamp" in total_usage.columns
    assert "cpu_used" in total_usage.columns
    assert "timestamp" in usage.columns
    assert "requester" in usage.columns
    assert "uses" in usage.columns
