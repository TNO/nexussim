import configparser
import logging
import os
from collections import defaultdict
from datetime import datetime

import polars as pl
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS
from influxdb_client.rest import ApiException
from urllib3.exceptions import NewConnectionError

from nexussim.cpus import CPUProvider

LOGGER = logging.getLogger(__name__)

INFLUX_INI = os.getenv('INFLUX_INI', 'influx.ini')


class CPUUsageBuffer:
    __BUFFER_SIZE = 4096

    def __init__(self, timestampfunc: callable = datetime.now):
        self._total_usage_series = dict()
        self._usage_series = dict()
        self.__timestampfunc = timestampfunc

    def update(self, cpu_provider: CPUProvider):
        now = self.__timestampfunc()
        self._total_usage_series[now] = cpu_provider.used_cpus()
        self._usage_series[now] = cpu_provider.cpu_usage()

    def total_usage_timeseries(self):
        """
        Returns a generator yielding timestamp and total used CPUs.

        This method provides a timeseries of total CPU usage, returning a generator that
        yields tuples containing timestamps and the corresponding total number of used
        CPUs.

        If you need a list instead, you can use `list(self.total_usage_timeseries())`.

        Yields:
            tuple[datetime, float]: A tuple where the first element is the timestamp and
            the second element is the total used CPUs at that time.
        """
        return (
            (timestamp, used_cpus)
            for timestamp, used_cpus in self._total_usage_series.items()
        )

    def usage_timeseries(self):
        """
        Returns a generator yielding timestamp, requester, and CPU usage.

        This method provides a timeseries of per-requester CPU usage, returning a
        generator that yields tuples containing timestamps, requesters, and the
        corresponding CPU usage.

        If you need a list instead, you can use `list(self.usage_timeseries())`.

        Yields:
            tuple[datetime, str, float]: A tuple where the first element is the
            timestamp, the second element is the requester, and the third element is the
            CPU usage for that requester at that time.
        """
        return (
            (timestamp, requester, uses)
            for timestamp, use_dict in self._usage_series.items()
            for requester, uses in use_dict.items()
        )


def to_polars_dataframes(cpu_buffer: CPUUsageBuffer):
    """
    Converts CPU usage data from a CPUUsageBuffer into Polars DataFrames.

    This function takes a CPUUsageBuffer instance and transforms its CPU usage
    timeseries data into two separate Polars DataFrames. The first DataFrame contains
    total CPU usage over time, while the second DataFrame provides per-requester CPU
    usage over time.

    Args:
        cpu_buffer (CPUUsageBuffer): An instance of CPUUsageBuffer containing
            the timeseries CPU usage data.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing two Polars DataFrames.
            The first DataFrame has columns "timestamp" and "cpu_used", representing the
            total CPU usage at each timestamp. The second DataFrame has columns
            "timestamp", "requester", and "uses", representing the CPU usage by each
            requester at each timestamp.
    """
    _pl_total_usage = pl.DataFrame(
        cpu_buffer.total_usage_timeseries(),
        schema=(("timestamp", pl.Datetime), ("cpu_used", pl.Float32)),
    )
    _pl_usage = pl.DataFrame(
        cpu_buffer.usage_timeseries(),
        schema=(
            ("timestamp", pl.Datetime),
            ("requester", pl.Utf8),
            ("uses", pl.Float32),
        ),
    )
    return _pl_total_usage, _pl_usage


class _InfluxClientWrapper:
    def __enter__(self):
        self.bucket = None
        self._client = None
        try:
            self.bucket = os.environ['INFLUXDB_V2_BUCKET']
            self._client = InfluxDBClient.from_env_properties()
        except KeyError:
            LOGGER.warning('INFLUX config via environment variables failed trying ini')
            try:
                self._client = InfluxDBClient.from_config_file(INFLUX_INI)
                config = configparser.ConfigParser()
                config.read(INFLUX_INI)
                self.bucket = config['influx2']['bucket']
            except KeyError:
                LOGGER.warning(
                    'INFLUX config invalid via env and ini: %s',
                    INFLUX_INI
                )
        finally:
            self.influx_connected = self.bucket is not None and self._client.ping()
            self.influx_write_api = \
                self._client.write_api(write_options=ASYNCHRONOUS)\
                if self.influx_connected else None
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.close()


class CPUUsageInflux:
    def __init__(self, compute_id):
        self.compute_id = compute_id
        with _InfluxClientWrapper() as client:
            self.client = client

    def update(self, cpu_provider: CPUProvider):
        if not self.client.influx_connected:
            return

        try:
            # Write the total CPU usage for this compute
            self.client.influx_write_api.write(
                bucket=self.client.bucket,
                record=Point("total_usage")
                       .tag('compute', self.compute_id)
                       .field('cpu_count', cpu_provider.used_cpus())
            )

            # keys in cpu_provider.cpu_usage() are segments
            # the part before any :: is the process name
            # for every main program we want the sum of the cpu usage for its's segments
            # Calculate sums
            cpu_usages_process = defaultdict(float)
            for segment, usage in cpu_provider.cpu_usage().items():
                cpu_usages_process[segment.split("::")[0]] += usage

            # Write sums to influx
            for process, usage in cpu_usages_process.items():
                self.client.influx_write_api.write(
                    bucket=self.client.bucket,
                    record=Point('cpu_usage')
                    .tag('process', process)
                    .tag('compute', self.compute_id)
                    .field('cpu_count', usage)
                 )

            LOGGER.debug("Wrote to influx")
        except (NewConnectionError, ApiException):
            # We managed before to connect but cannot write now.
            LOGGER.warning("Influx write failed", exc_info=True)
