from nexussim.segments.control import ChoiceSegment, ConcurrentSegment, RepeatSegment, SequenceSegment
import pytest
from nexussim.compute import Compute
from nexussim.context import Context
from nexussim.cpus import UnboundedCPU
from nexussim.memory import UnboundedMemory
from nexussim.segments.base import SegmentState
from nexussim.segments.behaviour import SampledLoadSegment, ConstantLoadSegment
from nexussim.segments.control import WhileSegment, forever
import pydynaa as pd


@pytest.fixture
def dynaa_sim():
    pd.DynAASim().reset()
    return pd.DynAASim()


@pytest.fixture
def context():
    return Context(
        compute=Compute(
            memory=UnboundedMemory(max_memory=10), cpu=UnboundedCPU(max_cpus=10.0)
        )
    )


def test_sequence_segment(dynaa_sim, context):
    segment_c = ConstantLoadSegment(cpu_load=3.0, duration_sec=7.0)
    segment_v = SampledLoadSegment(freq=1.0, duration_sec=7.0)
    sequence = SequenceSegment(segments=[segment_c, segment_v, segment_c, segment_v])
    assert sequence.segments == [segment_c, segment_v, segment_c, segment_v]
    for segment in sequence.segments:
        segment.parent == sequence.id
    sequence.execute(context)
    dynaa_sim.run(8)  # runs until 8 seconds
    assert dynaa_sim.current_time == 8.0
    assert segment_c.state == SegmentState.COMPLETED
    assert segment_v.state == SegmentState.RUNNING
    dynaa_sim.run(16)  # run until 16 seconds
    assert dynaa_sim.current_time == 16.0
    assert segment_v.state == SegmentState.COMPLETED
    assert segment_c.state == SegmentState.RUNNING
    assert context.compute.cpu.used_cpus() == 3.0
    dynaa_sim.run()
    assert dynaa_sim.current_time == 28.0
    assert context.compute.cpu.used_cpus() == 0
    assert segment_v.state == SegmentState.COMPLETED
    assert segment_c.state == SegmentState.COMPLETED
    sequence.reset()
    assert sequence.state == SegmentState.IDLE


def test_while_segment(dynaa_sim, context):
    segment_c = ConstantLoadSegment(cpu_load=2.0, duration_sec=5.0)
    segment_v = SampledLoadSegment(freq=1.0, duration_sec=5.0)
    sequence = SequenceSegment(segments=[segment_c, segment_v])
    while_segment = WhileSegment(segment=sequence, condition=forever)

    while_segment.execute(context)
    dynaa_sim.run(6)  # runs until 6 seconds
    assert dynaa_sim.current_time == 6.0
    assert segment_c.state == SegmentState.COMPLETED
    assert segment_v.state == SegmentState.RUNNING
    dynaa_sim.run(11)  # runs until 11 seconds
    assert dynaa_sim.current_time == 11.0
    assert segment_v.state == SegmentState.COMPLETED
    assert segment_c.state == SegmentState.RUNNING
    assert context.compute.cpu.used_cpus() == 2.0
    dynaa_sim.run(16)  # runs until 16 seconds
    assert dynaa_sim.current_time == 16.0
    assert segment_c.state == SegmentState.COMPLETED
    assert segment_v.state == SegmentState.RUNNING
    dynaa_sim.run(21)  # runs until 20 seconds
    assert dynaa_sim.current_time == 21.0
    assert segment_v.state == SegmentState.COMPLETED
    assert segment_c.state == SegmentState.RUNNING
    assert context.compute.cpu.used_cpus() == 2.0


def test_repeat_segment(dynaa_sim, context):
    segment_c = ConstantLoadSegment(cpu_load=2.0, duration_sec=5.0)
    segment_v = SampledLoadSegment(freq=1.0, duration_sec=5.0)
    sequence = SequenceSegment(segments=[segment_c, segment_v])
    repeat_segment = RepeatSegment(segment=sequence, times=3)
    assert (
        sequence.parent == repeat_segment.id
    )  # after insertion of sequence into repeat segment, the parent of sequence should be repeat segment
    repeat_segment.execute(context)
    dynaa_sim.run(11)  # runs until 11 seconds
    assert segment_v.state == SegmentState.COMPLETED
    assert segment_c.state == SegmentState.RUNNING
    assert context.compute.cpu.used_cpus() == 2.0
    dynaa_sim.run(16)  # runs until 16 seconds
    assert segment_c.state == SegmentState.COMPLETED
    assert segment_v.state == SegmentState.RUNNING
    dynaa_sim.run(21)  # runs until 20 seconds
    assert segment_v.state == SegmentState.COMPLETED
    assert segment_c.state == SegmentState.RUNNING
    assert context.compute.cpu.used_cpus() == 2.0
    dynaa_sim.run(26)  # runs until 26 seconds
    assert segment_c.state == SegmentState.COMPLETED
    assert segment_v.state == SegmentState.RUNNING
    dynaa_sim.run(31)  # runs until 31 seconds
    assert dynaa_sim.current_time == 31.0
    assert segment_v.state == SegmentState.COMPLETED
    assert segment_c.state == SegmentState.COMPLETED
    assert context.compute.cpu.used_cpus() == 0.0

def test_repeat_segment_accepts_no_negative_times():
    with pytest.raises(ValueError):
        repeat_segment = RepeatSegment(segment=ConstantLoadSegment(), times=-1)

def test_concurrent_segment(dynaa_sim, context):
    segment_c = ConstantLoadSegment(cpu_load=2.0, duration_sec=5.0)
    segment_v = SampledLoadSegment(freq=1.0, duration_sec=3.0)
    concurrent_segment = ConcurrentSegment(segments=[segment_c, segment_v])
    assert (segment_c.parent == concurrent_segment.id)  # after insertion of segment into concurrent segment, the parent of segment should be concurrent segment
    assert (segment_v.parent == concurrent_segment.id)
    concurrent_segment.execute(context)
    dynaa_sim.run(2)  # runs for 2 seconds
    assert segment_c.state == SegmentState.RUNNING
    assert segment_v.state == SegmentState.RUNNING
    assert concurrent_segment.state == SegmentState.RUNNING
    dynaa_sim.run(4)  # runs for 3 seconds
    assert segment_c.state == SegmentState.RUNNING
    assert segment_v.state == SegmentState.COMPLETED
    assert concurrent_segment.state == SegmentState.RUNNING
    dynaa_sim.run(6)  # runs for 3 seconds
    assert segment_c.state == SegmentState.COMPLETED
    assert segment_v.state == SegmentState.COMPLETED
    assert concurrent_segment.state == SegmentState.COMPLETED

def test_concurrent_segment_accepts_no_empty_list():
    with pytest.raises(ValueError):
        concurrent_segment = ConcurrentSegment(segments=[])

def test_choice_segment(dynaa_sim, context):
    segment_c = ConstantLoadSegment(cpu_load=2.0, duration_sec=5.0)
    segment_v = SampledLoadSegment(freq=1.0, duration_sec=3.0)
    choice_segment = ChoiceSegment(segments=[segment_c, segment_v], weights=[0.5, 0.5])
    assert (segment_c.parent == choice_segment.id)  # after insertion of segment into choice segment, the parent of segment should be choice segment
    assert (segment_v.parent == choice_segment.id)
    choice_segment.execute(context)
    dynaa_sim.run(2)  # runs for 2 seconds
    assert (segment_c.state == SegmentState.RUNNING) != (segment_v.state == SegmentState.RUNNING)
    assert choice_segment.state == SegmentState.RUNNING
    dynaa_sim.run(6)  # runs for 6 seconds
    assert segment_c.state == SegmentState.COMPLETED or SegmentState.IDLE
    assert segment_v.state == SegmentState.COMPLETED or SegmentState.IDLE
    assert choice_segment.state == SegmentState.COMPLETED

def test_choice_segment_accepts_no_empty_list():
    with pytest.raises(ValueError):
        choice_segment = ChoiceSegment(segments=[], weights=[0.5, 0.5])