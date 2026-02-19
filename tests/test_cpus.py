import importlib
import os
from datetime import datetime
from math import isclose
from unittest.mock import MagicMock, patch

import pydynaa as pd
import pytest
from pint import UnitRegistry

import nexussim
import nexussim.loggers
from nexussim.cpus import BoundedCPU, UnboundedCPU
from nexussim.loggers import CPUUsageBuffer, CPUUsageInflux, to_polars_dataframes

unit = UnitRegistry()


@pytest.fixture(autouse=True)
def isolate_influx_env(monkeypatch):
    """Clear INFLUX-related env vars for each test and restore afterwards."""
    original_influx_ini = nexussim.loggers.INFLUX_INI
    keys = (
        "INFLUX_INI",
        "INFLUXDB_V2_URL",
        "INFLUXDB_V2_ORG",
        "INFLUXDB_V2_TOKEN",
        "INFLUXDB_V2_BUCKET",
    )
    original_env = {k: os.environ.get(k) for k in keys}

    # ensure env vars for influx are not set at test start
    for k in keys:
        monkeypatch.delenv(k, raising=False)

    yield

    # restore env vars and module-level INFLUX_INI
    for k, v in original_env.items():
        monkeypatch.delenv(k, raising=False)
        if v is not None:
            monkeypatch.setenv(k, v)
    nexussim.loggers.INFLUX_INI = original_influx_ini


class DummyObserver:
    total = 0.0
    usage = {}

    def update(self, cpu_provider):
        self.total = cpu_provider.used_cpus()
        self.usage = cpu_provider.cpu_usage()


@pytest.fixture
def mock_influx_client():
    with patch("nexussim.loggers.InfluxDBClient") as mock_client:
        mock_write_api_instance = MagicMock()
        mock_client_instance = MagicMock()

        mock_client.from_config_file.return_value = mock_client_instance
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.write_api.return_value = mock_write_api_instance
        yield mock_client


def test_unbounded_cpu_initialization():
    max_cpus = 10.0
    cpu_speed = unit("1 GHz").to("Hz")
    cpu = UnboundedCPU(max_cpus, cpu_speed)
    assert isclose(cpu.max_cpus(), max_cpus)
    assert isclose(cpu.used_cpus(), 0.0)
    assert cpu.cpu_speed() == cpu_speed
    assert isclose(cpu.free_cpus(), max_cpus)
    assert not cpu.cpu_usage()


def test_unbounded_cpu_request_cpu():
    cpu = UnboundedCPU(max_cpus=10.0, cpu_speed=1000)
    cpu.request_cpu(requester="process1", cpus_frac=2.5, cpu_cycles=1000)
    assert isclose(cpu.used_cpus(), 2.5)
    assert isclose(cpu.free_cpus(), 7.5)
    assert cpu.cpu_usage() == {"process1": 2.5}
    cpu.request_cpu(requester="process2", cpus_frac=10, cpu_cycles=1000)
    assert isclose(cpu.used_cpus(), 12.5)
    assert isclose(cpu.free_cpus(), -2.5)
    assert cpu.cpu_usage() == {"process1": 2.5, "process2": 10.0}


def test_unbounded_cpu_release_cpu():
    cpu = UnboundedCPU(max_cpus=10.0, cpu_speed=1000)
    cpu.request_cpu(requester="process1", cpus_frac=2.5, cpu_cycles=1000)
    assert cpu.release_cpu("process1") is True
    assert cpu.used_cpus() == 0
    assert isclose(cpu.free_cpus(), 10.0)
    assert cpu.cpu_usage() == {"process1": 0.0}


def test_unbounded_cpu_flush():
    cpu = UnboundedCPU(max_cpus=10.0, cpu_speed=1000)
    cpu.request_cpu(requester="process1", cpus_frac=2.5, cpu_cycles=1000)
    cpu.flush()
    assert cpu.used_cpus() == 0
    assert isclose(cpu.free_cpus(), 10.0)
    assert not cpu.cpu_usage()


def test_dummy_observer():
    dummy_observer = DummyObserver()
    cpu = UnboundedCPU(max_cpus=10.0, cpu_speed=1000, observers=[dummy_observer])
    cpu.request_cpu(requester="process1", cpus_frac=2.5, cpu_cycles=1000)
    assert dummy_observer.total == 2.5
    assert dummy_observer.usage == {"process1": 2.5}


# BOUNDED CPU TESTS


def test_bounded_cpu_initialization():
    max_cpus = 10.0
    cpu_speed = unit("1 GHz").to("Hz")
    cpu = BoundedCPU(max_cpus, cpu_speed)
    assert isclose(cpu.max_cpus(), max_cpus)
    assert isclose(cpu.used_cpus(), 0.0)
    assert cpu.cpu_speed() == cpu_speed
    assert isclose(cpu.free_cpus(), max_cpus)
    assert not cpu.cpu_usage()


def test_bounded_cpu_request_cpu():
    cpu = BoundedCPU(10.0, 1000.0)
    cpu.request_cpu(requester="process1", cpus_frac=2.5, cpu_cycles=1000)
    assert isclose(cpu.used_cpus(), 2.5)
    assert isclose(cpu.free_cpus(), 7.5)
    assert cpu.cpu_usage() == {"process1": 2.5}
    cpu.request_cpu(requester="process2", cpus_frac=10, cpu_cycles=1000)
    # should not exceed max cpus
    assert isclose(cpu.used_cpus(), 10.0)
    assert isclose(cpu.free_cpus(), 0.0)
    assert cpu.cpu_usage() == {
        "process1": 2.5 * (10.0 / 12.5),
        "process2": 10 * (10.0 / 12.5),
    }


def test_bounded_cpu_overfull_time_dilation(dynaa_sim):
    # pylint: disable=W0212
    cpu = BoundedCPU(4.0, 1000.0)

    context = {}

    def trigger(*args, **kwargs):
        context["triggered"] = context.get("triggered", 0) + 1

    ee = cpu.request_cpu(requester="process1", cpus_frac=2, cpu_cycles=6000)
    pd.Entity()._wait(pd.EventHandler(trigger), expression=ee)

    dynaa_sim.run(1)
    assert dynaa_sim.current_time == 1

    cpu._update_allocations()
    assert cpu.used_cpus() == 2
    assert cpu.free_cpus() == 2
    assert cpu._used_cpus["process1"].cycles_remaining == 5000
    assert cpu._used_cpus["process1"].cpus_requested == 2
    assert cpu._used_cpus["process1"].cpus_allocated == 2

    dynaa_sim.run(2)
    ee2 = cpu.request_cpu(requester="process2", cpus_frac=4, cpu_cycles=4000)
    pd.Entity()._wait(pd.EventHandler(trigger), expression=ee2)
    assert cpu.used_cpus() == 4
    assert cpu.free_cpus() == 0

    assert cpu._used_cpus["process1"].cycles_remaining == 4000
    assert cpu._used_cpus["process1"].cpus_requested == 2
    assert cpu._used_cpus["process1"].cpus_allocated == 2 * (4 / 6)

    assert cpu._used_cpus["process2"].cycles_remaining == 4000
    assert cpu._used_cpus["process2"].cpus_requested == 4
    assert cpu._used_cpus["process2"].cpus_allocated == 4 * (4 / 6)

    assert cpu.cpu_usage() == {"process1": 2 * (4 / 6), "process2": 4 * (4 / 6)}

    dynaa_sim.run(4)
    cpu._update_allocations()
    assert cpu._used_cpus["process1"].cpus_requested == 2
    assert cpu._used_cpus["process1"].cpus_allocated == 2 * (4 / 6)
    assert isclose(cpu._used_cpus["process1"].cycles_remaining, 2666.666666)

    assert cpu._used_cpus["process2"].cpus_requested == 4
    assert cpu._used_cpus["process2"].cpus_allocated == 4 * (4 / 6)
    assert isclose(cpu._used_cpus["process2"].cycles_remaining, 2666.666666)

    dynaa_sim.run()
    assert context["triggered"] == 2
    assert dynaa_sim.current_time


def test_CPUDataFrameLogger():
    observer = CPUUsageBuffer()
    cpu = UnboundedCPU(10.0, cpu_speed=1000, observers=[observer])
    cpu.request_cpu("process1", 2.5, cpu_cycles=1000)
    cpu.request_cpu("process2", 1.0, cpu_cycles=1000)
    cpu.release_cpu("process1")
    total_usage, usage = to_polars_dataframes(observer)
    assert "timestamp" in total_usage.columns
    assert "cpu_used" in total_usage.columns
    assert "timestamp" in usage.columns
    assert "requester" in usage.columns
    assert "uses" in usage.columns

    assert [(process, val) for ts, process, val in observer.usage_timeseries() if isinstance(ts, datetime)] == [
        ("process1", 2.5),
        ("process1", 2.5),
        ("process2", 1.0),
        ("process1", 0),
        ("process2", 1.0),
    ]
    assert [val for ts, val in observer.total_usage_timeseries() if isinstance(ts, datetime)] == [2.5, 3.5, 1.0]


def test_ini_env():
    os.environ["INFLUX_INI"] = "tst"
    ns = importlib.reload(nexussim.loggers)
    assert ns.INFLUX_INI == "tst"


def test_influx_error_no_ini(mock_influx_client):
    # ini does not exist
    nexussim.loggers.INFLUX_INI = "bla"

    cpu_buf = CPUUsageInflux("my_compute")
    assert not cpu_buf.client.influx_connected
    cpu = UnboundedCPU(10.0, cpu_speed=1000, observers=[cpu_buf])
    cpu.request_cpu("process1", 2.5, cpu_cycles=1000)

    # check we try to create client but never write
    mock_influx_client.from_config_file.assert_called_once_with(nexussim.loggers.INFLUX_INI)
    mock_influx_client.from_config_file().write_api.assert_not_called()
    mock_influx_client.from_config_file().write_api().write.assert_not_called()


def test_influx_error_ini_no_bucket(mock_influx_client):
    # ini wrongly formatted no bucket
    nexussim.loggers.INFLUX_INI = os.path.join("tests", "invalid_test.ini")

    cpu_buf = CPUUsageInflux("my_compute")
    assert not cpu_buf.client.influx_connected
    cpu = UnboundedCPU(10.0, cpu_speed=1000, observers=[cpu_buf])
    cpu.request_cpu("process1", 2.5, cpu_cycles=1000)

    # check we try to create client but never write
    mock_influx_client.from_config_file.assert_called_once_with(nexussim.loggers.INFLUX_INI)
    mock_influx_client.from_config_file().write_api.assert_not_called()
    mock_influx_client.from_config_file().write_api().write.assert_not_called()


def test_influx_error_ini_no_influx2_sec(mock_influx_client):
    # ini wrongly formatted no [influx2]
    nexussim.loggers.INFLUX_INI = os.path.join("tests", "invalid_test2.ini")

    cpu_buf = CPUUsageInflux("my_compute")
    assert not cpu_buf.client.influx_connected
    cpu = UnboundedCPU(10.0, cpu_speed=1000, observers=[cpu_buf])
    cpu.request_cpu("process1", 2.5, cpu_cycles=1000)

    # check we try to create client but never write
    mock_influx_client.from_config_file.assert_called_once_with(nexussim.loggers.INFLUX_INI)
    mock_influx_client.from_config_file().write_api.assert_not_called()
    mock_influx_client.from_config_file().write_api().write.assert_not_called()


def test_influx_via_env_vars():
    nexussim.loggers.INFLUX_INI = "influx.ini.default"

    cpu_buf = CPUUsageInflux("my_compute")
    assert cpu_buf.client._client.url == "http://localhost:8086"
    assert cpu_buf.client._client.org == "TNO"
    assert cpu_buf.client._client.token == "my-super-secret-token"

    # ini does not exist
    nexussim.loggers.INFLUX_INI = "bla"
    os.environ["INFLUXDB_V2_URL"] = "http://localhost:1111"
    os.environ["INFLUXDB_V2_ORG"] = "tstorg"
    os.environ["INFLUXDB_V2_TOKEN"] = "testtoken"
    os.environ["INFLUXDB_V2_BUCKET"] = "my_bucket"

    cpu_buf = CPUUsageInflux("my_compute")
    assert cpu_buf.client._client.url == "http://localhost:1111"
    assert cpu_buf.client._client.org == "tstorg"
    assert cpu_buf.client._client.token == "testtoken"
    assert cpu_buf.client.bucket == "my_bucket"


def test_influx2(mock_influx_client):
    assert nexussim.loggers.INFLUX_INI == "influx.ini"
    # use default ini as we can be sure that will be there
    nexussim.loggers.INFLUX_INI = "influx.ini.default"

    cpu_buf = CPUUsageInflux("my_compute")
    assert cpu_buf.client.influx_connected
    cpu = UnboundedCPU(10.0, cpu_speed=1000, observers=[cpu_buf])
    cpu.request_cpu("process1", 2.5, cpu_cycles=1000)

    # check we create client and write
    mock_influx_client.from_config_file.assert_called_once_with("influx.ini.default")
    mock_influx_client.from_config_file().write_api.assert_called_once()
    assert mock_influx_client.from_config_file().write_api().write.call_count == 2
