from nexussim.cpus import CPUProvider
import polars as pl
from datetime import datetime


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
