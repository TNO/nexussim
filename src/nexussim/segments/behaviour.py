from typing import Generator

from nexussim.samplers import gaussian_load_sampler
from nexussim.segments.base import EventProvider, Segment


class SampledLoadSegment(Segment):
    def __init__(
        self,
        cpu_load_sampler: Generator = gaussian_load_sampler(),
        freq: float = 1.0,
        duration: float = 10.0,
    ):
        """
        Initializes the SampledLoadSegment with a CPU load sampler, frequency, and
        duration.

        Args:
            cpu_load_sampler (Generator, optional):
                A generator function that samples CPU load values. Sampler __must__ take
                care of maximal load and minimal load constraints.Defaults to a Gaussian
                load sampler.

            freq (float, optional):
                The frequency at which the CPU load is sampled, in Hz. Must be greater
                than 0. Defaults to 1.0.

            duration (float, optional):
                The duration of the segment in seconds. Must be greater than 0. Defaults
                to 1.0.

        Raises:
            ValueError:
                If the freq parameter is less than or equal to 0. ValueError: If the
            duration parameter is less than or equal to 0.
        """

        super().__init__()
        self._cpu_load_sampler = cpu_load_sampler
        if freq <= 0:
            raise ValueError(
                "freq parameter (frequency, in Hz) must be greater than 0."
            )
        self._freq = freq
        if duration <= 0:
            raise ValueError("duration parameter must be greater than 0.")
        self._duration = duration

    def _segment_body(self, context):
        """
        This method implements the behavior of SampledLoadSegment.

        It samples from the cpu_load_sampler and requests the sampled CPU load
        from the context's CPU for the specified duration, at the specified
        frequency. The sampler is called every 1/freq seconds. The segment
        transitions to the RUNNING state and keeps requesting CPU resources until
        the duration is over.

        At the end of the duration, the segment transitions to the COMPLETED
        state and releases the previously requested CPU resources.

        Args:
            context (dict): A context object providing access to the segment's
                context.

        Yields:
            EventExpression: An event expression that represents a condition to
                wait for during the segment's execution.
        """
        sample_time = 0.0
        dt = 1.0 / self._freq
        while sample_time < self._duration:
            cpu_load = self._cpu_load_sampler()
            context.compute.cpu.request_cpu(self.id, cpu_load)
            yield self._schedule_after(dt, self.RESUME_SEGMENT)
            sample_time += dt
        # Cleaning resources
        context.compute.cpu.release_cpu(self.id)


class ConstantLoadSegment(Segment):
    def __init__(
        self,
        cpu_load: float = 1.0,
        mem_load: float = 0.0,
        duration: float = 1.0,
        cycles: int = 0,
    ):
        """
        Initializes the ConstantLoadSegment with a CPU load and duration.

        Args:
            cpu_load (float, optional): The CPU load as a fraction of total CPU
            capacity.
                Defaults to 1.0.
            mem_load (float, optional): The memory load in bytes.
            duration (float, optional): The duration of the segment in seconds.
                Defaults to 1.0.
            cycles (int, optional): The number of cycles to run. Defaults to 0.

        Remarks:
            _duration_ and _cycles_ add up.

        """
        self._cpu_load = cpu_load
        self._duration = duration if duration is not None else 0.0
        self._cycles = cycles
        super().__init__()

    @property
    def cpu_load(self):
        """
        Returns the CPU load for the segment.

        This property provides the CPU load that is requested by this segment during
        its execution.

        Returns:
            float: The CPU load as a fraction of total CPU capacity.
        """

        return self._cpu_load

    @property
    def duration(self):
        """
        Returns the duration for which the segment runs.

        Returns:
            float: The duration in seconds.
        """

        return self._duration

    @property
    def cycles(self):
        """
        Returns the number of cycles for the segment.

        This property provides the number of cycles this segment is supposed to run.

        Returns:
            int: The number of cycles.
        """
        return self._cycles

    def _segment_body(self, context):
        """
        This method requests constant CPU and Memory resources for the segment, then
        yields an event expression to wait for the specified duration. After the
        duration has elapsed, the segment is resumed.

        Once the segment completes its execution, it releases the CPU resources.

        Args:
            context (dict): A context object providing access to the compute resources.

        Yields:
            EventExpression: An event expression that represents the wait condition for
            the duration of the segment's execution.
        """

        context.compute.cpu.request_cpu(self.id, self.cpu_load)
        cpu_speed = context.compute.cpu.cpu_speed()
        cycles_duration = 0.0
        if cpu_speed > 0.0:  # Only if cpu_speed is not known
            cycles_duration = self.cycles / cpu_speed
        yield self._schedule_after(self.duration + cycles_duration, self.RESUME_SEGMENT)

        # Cleaning resources
        context.compute.cpu.release_cpu(self.id)


class ConstantLoadWaitSegment(Segment):
    def __init__(
        self,
        wait_on: EventProvider = None,
        cpu_load: float = 1.0,
        mem_load: float = 0.0,
    ):
        """
        Initializes the ConstantLoadWaitSegment with a CPU and memory load and event to
        wait upon.

        Args:
            wait_on (EventProvider):
                The event provider to wait upon.
            cpu_load (float, optional):
                The CPU load as a fraction of total CPU capacity.Defaults to 1.0.
            mem_load (float, optional):
                The memory load in bytes.
        Remarks:
            _duration_ and _cycles_ add up.

        """
        self._event_provider = wait_on
        self._cpu_load = cpu_load
        super().__init__()

    @property
    def cpu_load(self):
        """
        Returns the CPU load for the segment.

        This property provides the CPU load that is requested by this segment during
        its execution.

        Returns:
            float: The CPU load as a fraction of total CPU capacity.
        """

        return self._cpu_load

    def _segment_body(self, context):
        """
        This method requests constant CPU and Memory resources for the segment, then
        collects an event to be observed (from an event provider). The segment is
        resumed after the event is produced by the provider.

        Only waits for the first event to be thrown by the event provider.

        Once the segment completes its execution, it releases the CPU resources.

        Args:
            context (dict): A context object providing access to the compute resources.

        Yields:
            EventExpression: An event expression that represents the wait condition that
            resumes the segment's execution.
        """

        context.compute.cpu.request_cpu(self.id, self.cpu_load)
        yield self._event_provider.scheduled_event()

        # Cleaning resources
        context.compute.cpu.release_cpu(self.id)
