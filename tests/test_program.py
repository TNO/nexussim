import pytest

from nexussim.compute import Compute
from nexussim.context import Context
from nexussim.cpus import UnboundedCPU
from nexussim.memory import UnboundedMemory
from nexussim.network import BaseNetwork
from nexussim.program import (
    Program,
    choice,
    concurrent,
    constant_load_segment,
    repeat,
    sampled_load_segment,
    sequence,
    while_do,
)
from nexussim.segments.base import Segment
from nexussim.segments.behaviour import ConstantLoadSegment, SampledLoadSegment
from nexussim.segments.control import (
    ChoiceSegment,
    ConcurrentSegment,
    RepeatSegment,
    SequenceSegment,
    WhileSegment,
    forever,
)


class MockSegment(Segment):
    __test__ = False

    def _segment_body(self, context):
        """
        Creates a dummy segment behaviour for tests.

        This segment runs for 10 seconds and stops"""
        yield self._schedule_after(10, self.RESUME_SEGMENT)
        context.store(executed=True)


def test_program_initialization():
    segment = MockSegment()
    program = Program(name="Test Program", body=segment, cpu_load=1.0, mem_load=2.0)
    assert program.name == "Test Program"
    assert program.body == segment
    assert program.base_cpu_load == 1.0
    assert program.base_mem_load == 2.0


def test_program_execution(dynaa_sim, context):
    segment = MockSegment()
    program = Program(name="Test Program", body=segment)
    # Start the execution of the program in the computing node
    program.execute(context)
    # Run dynaa simulator
    dynaa_sim.run(5)
    assert context.retrieve("executed") is None  # not yet executed to end.
    dynaa_sim.run()
    assert context.retrieve("executed")


def test_program_stop(dynaa_sim, context):
    segment = MockSegment()
    program = Program(name="Test Program", body=segment)
    # Start the execution of the program in the computing node
    program.execute(context)
    # Run dynaa simulator
    dynaa_sim.run(5)
    assert context.retrieve("executed") is None
    program.stop()
    dynaa_sim.run()
    assert context.retrieve("executed") is None


def test_program_stop2(dynaa_sim, context):
    # try with more complex segment
    constant1 = constant_load_segment(cpu_load=1.0, duration=5.0)
    constant2 = constant_load_segment(cpu_load=1.0, duration=5.0)
    segment = choice(constant1, constant2)
    program = Program(name="Test Program", body=segment)
    # Start the execution of the program in the computing node
    program.execute(context)
    # Run dynaa simulator
    dynaa_sim.run(5)
    assert context.retrieve("executed") is None
    program.stop()
    dynaa_sim.run()
    assert context.retrieve("executed") is None


def test_program_sequence_creation(dynaa_sim, context):
    seq = sequence(MockSegment(), MockSegment())
    assert isinstance(seq, SequenceSegment)
    assert len(seq.segments) == 2


def test_program_choice_creation(dynaa_sim, context):
    choice_ = choice(MockSegment(), MockSegment())
    assert isinstance(choice_, ChoiceSegment)
    assert len(choice_.segments) == 2


def test_program_concurrent_creation(dynaa_sim, context):
    concurrent_ = concurrent(MockSegment(), MockSegment())
    assert isinstance(concurrent_, ConcurrentSegment)
    assert len(concurrent_.segments) == 2


def test_program_repeat_creation(dynaa_sim, context):
    repeat_ = repeat(MockSegment(), times=2)
    assert isinstance(repeat_, RepeatSegment)
    assert repeat_.segment is not None


def test_program_while_creation(dynaa_sim, context):
    while_segment = while_do(segment=MockSegment(), condition=forever)
    assert isinstance(while_segment, WhileSegment)
    assert while_segment.segment is not None


def test_program_constant_creation(dynaa_sim, context):
    constant_ = constant_load_segment(cpu_load=1.0, duration=5.0)
    assert isinstance(constant_, ConstantLoadSegment)


@pytest.mark.skip("Sampled load segments are undefined with new CPU model")
def test_program_sampled_constant_creation(dynaa_sim, context):
    sampled_ = sampled_load_segment(freq=1.0, duration=5.0, cpu_load_sampler=None)
    assert isinstance(sampled_, SampledLoadSegment)


class MockMemorySegment(Segment):
    __test__ = False

    def __init__(self, size: int = 128):
        super().__init__()
        self._size = size

    def _segment_body(self, context):
        # allocate memory, wait a bit, then free it
        addr = context.compute.memory.malloc(self._size, requester=self.id)
        # remain active for 1 second
        yield self._schedule_after(1, self.RESUME_SEGMENT)
        context.compute.memory.free(addr)
        context.store(executed=True)


def test_program_memory_allocation(dynaa_sim):
    # create own compute with more generous memory for the test
    ctx = Context(
        compute=Compute(
            memory=UnboundedMemory(max_memory=10000),
            cpu=UnboundedCPU(max_cpus=10.0, cpu_speed=1000),
            networks={"lo": BaseNetwork(max_bandwidth=100000000, max_latency=0.010)},
        )
    )

    seg = MockMemorySegment(size=512)
    program = Program(name="MemProgram", body=seg)
    program.execute(ctx)

    # run until the allocation is in place
    dynaa_sim.run(0.5)
    assert ctx.compute.memory.used_memory() == 512

    # finish the simulation and ensure memory was freed
    dynaa_sim.run()
    assert ctx.compute.memory.used_memory() == 0
