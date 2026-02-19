import pydynaa as pd
import pytest

from nexussim.compute import Compute
from nexussim.context import Context
from nexussim.cpus import UnboundedCPU
from nexussim.memory import UnboundedMemory
from nexussim.segments.base import Segment, SegmentState
from nexussim.segments.behaviour import (ConstantLoadSegment,
                                         ConstantLoadWaitSegment,
                                         SampledLoadSegment)


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
    OneShotProducerEventType = pd.EventType(
        "OneShotProducerEventType", "An example of an externally generated event."
    )
    event_time = 10

    def scheduled_event(self):
        return self._schedule_after(
            OneShotEventProducer.event_time, self.OneShotProducerEventType
        )


@pytest.fixture
def dynaa_sim():
    pd.DynAASim().reset()
    return pd.DynAASim()


@pytest.fixture
def context():
    return Context(
        compute=Compute(
            memory=UnboundedMemory(max_memory=10),
            cpu=UnboundedCPU(max_cpus=10.0, cpu_speed=3000),
        )
    )


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
    with pytest.raises(Exception):
        segment.execute({"a": 1})  # Should raise an exception


def test_segment_with_no_generator_body():
    segment = SegmentNoGeneratorBody()
    segment.execute({"a": 1})
    assert segment.state == SegmentState.COMPLETED
    with pytest.raises(Exception):
        segment.execute({"a": 1})  # Should raise an exception


def test_constant_load_segment(dynaa_sim, context):
    segment = ConstantLoadSegment(cpu_load=3.0, duration=7.0, cycles=3000)
    segment.execute(context)
    assert segment.cycles == 3000
    dynaa_sim.run(2)  # runs for 2 seconds
    assert dynaa_sim.current_time == 2.0
    assert context.compute.cpu.used_cpus() == 3.0
    dynaa_sim.run()
    assert dynaa_sim.current_time == 8.0
    assert context.compute.cpu.used_cpus() == 0


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
