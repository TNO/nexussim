import pydynaa as pd
import pytest

from nexussim.exceptions import NexussimException
from nexussim.segments.base import Segment, SegmentState
from nexussim.segments.behaviour import (
    ConstantLoadSegment,
    ConstantLoadWaitSegment,
    SampledLoadSegment,
    SleepSegment,
    WaitSegment,
)


class TestSegment(Segment):
    __test__ = False

    def _segment_body(self, context):
        """
        Creates a dummy segment behaviour for tests.

        This segment runs for 10 seconds and stops"""

        yield self._schedule_after(10, self.RESUME_SEGMENT)


class SegmentNoGeneratorBody(Segment):
    def _segment_body(self, context):
        pass


class OneShotEventProducer(pd.Entity):
    OneShotProducerEventType = pd.EventType("OneShotProducerEventType", "An example of an externally generated event.")
    event_time = 10

    def scheduled_event(self):
        return self._schedule_after(OneShotEventProducer.event_time, self.OneShotProducerEventType)


def test_segment_is_abstract():
    with pytest.raises(TypeError):
        Segment()


def test_segment_creation():
    segment = TestSegment()
    assert segment.state == SegmentState.IDLE
    assert segment.parent == "<root>"


def test_segment_initial_execution(dynaa_sim):
    segment = TestSegment()
    segment.parent = "parent"
    segment.execute({"a": 1})
    assert segment.state == SegmentState.RUNNING
    assert dynaa_sim.state == 0  # idle state of the simulator, nothing ran
    assert dynaa_sim.current_time == 0  # time of the simulator, nothing ran
    assert len(pd.Diagnostics().get_future_events()) == 1
    assert segment.parent == "parent"
    assert segment.id.startswith("parent::")
    segment.reset()
    assert segment.state == SegmentState.IDLE


def test_segment_full_run(dynaa_sim):
    segment = TestSegment()
    segment.execute({"a": 1})
    assert dynaa_sim.state == 0  # idle state of the simulator, nothing ran
    assert dynaa_sim.current_time == 0  # time of the simulator, nothing ran
    dynaa_sim.run()
    assert dynaa_sim.state == 0  # paused
    assert dynaa_sim.current_time == 10
    assert segment.state == SegmentState.COMPLETED
    with pytest.raises(NexussimException):
        segment.execute({"a": 1})  # Should raise an exception


def test_segment_with_no_generator_body():
    segment = SegmentNoGeneratorBody()
    segment.execute({"a": 1})
    assert segment.state == SegmentState.COMPLETED
    with pytest.raises(NexussimException):
        segment.execute({"a": 1})  # Should raise an exception


def test_constant_load_segment():
    ConstantLoadSegment(cpu_load=3.0, cycles=7000.0)
    with pytest.raises(ValueError):
        ConstantLoadSegment(cpu_load=0.0)
    with pytest.raises(ValueError):
        ConstantLoadSegment(cpu_load=0.0, cycles=0.0, duration=0.0)
    with pytest.raises(ValueError):
        ConstantLoadSegment(cpu_load=0.0, cycles=100.0, duration=10.0)


def test_constant_load_segment2(dynaa_sim, context):
    segment = ConstantLoadSegment(cpu_load=3.0, cycles=9000)
    segment.execute(context)
    assert segment.cycles == 9000
    dynaa_sim.run(2)  # runs for 2 seconds
    assert dynaa_sim.current_time == 2.0
    assert context.compute.cpu.used_cpus() == 3.0
    dynaa_sim.run()
    assert dynaa_sim.current_time == 9.0
    assert context.compute.cpu.used_cpus() == 0


def test_constant_load_segment3(dynaa_sim, context):
    segment = ConstantLoadSegment(cpu_load=3.0, duration=3)
    segment.execute(context)
    assert segment.duration == 3
    dynaa_sim.run(2)  # runs for 2 seconds
    assert dynaa_sim.current_time == 2.0
    assert context.compute.cpu.used_cpus() == 3.0
    dynaa_sim.run()
    assert dynaa_sim.current_time == 3.0
    assert context.compute.cpu.used_cpus() == 0


def test_depreaction_wait_segment(recwarn):
    assert len(recwarn.list) == 0
    WaitSegment(duration=7.0)
    assert len(recwarn.list) == 1


def test_depreaction_wait_segment2(recwarn):
    assert len(recwarn.list) == 0
    SleepSegment(duration=7.0)
    assert len(recwarn.list) == 0


def test_wait_segment_memory_free(dynaa_sim, context):
    """SleepSegment should allocate memory on start and free it when the wait ends."""
    segment = SleepSegment(duration=1.0, mem_load=2)
    segment.execute(context)
    # memory should be allocated immediately
    assert context.compute.memory.used_memory() == 2

    # run until the wait completes
    dynaa_sim.run()
    assert dynaa_sim.current_time == 1.0
    # memory should have been freed by the scheduled handler
    assert context.compute.memory.used_memory() == 0
    assert segment.state == SegmentState.COMPLETED


@pytest.mark.skip(reason="Reconsider sampled segments with the new CPU model")
def test_sampled_load_segment(dynaa_sim, context):
    segment = SampledLoadSegment(freq=1.0, duration=3600.0)
    segment.execute(context)
    dynaa_sim.run(2)  # runs for 2 seconds
    assert dynaa_sim.current_time == 2.0
    assert context.compute.cpu.used_cpus() >= 0.0
    dynaa_sim.run()
    assert dynaa_sim.current_time == 3600.0
    assert context.compute.cpu.used_cpus() == 0

    with pytest.raises(ValueError):
        segment = SampledLoadSegment(freq=0.0, duration=7.0)

    with pytest.raises(ValueError):
        segment = SampledLoadSegment(freq=1.0, duration=0.0)


@pytest.mark.skip(reason="Reconsider wait-with-load segments with the new CPU model")
def test_constant_load_wait_segment(dynaa_sim, context):
    event_producer = OneShotEventProducer()
    segment = ConstantLoadWaitSegment(wait_on=event_producer, cpu_load=3.0)
    segment.execute(context)
    assert segment.cpu_load == 3.0
    dynaa_sim.run(2)  # runs for 2 seconds
    assert dynaa_sim.current_time == 2.0
    assert context.compute.cpu.used_cpus() == 3.0
    dynaa_sim.run()
    assert dynaa_sim.current_time == event_producer.event_time
    assert context.compute.cpu.used_cpus() == 0
