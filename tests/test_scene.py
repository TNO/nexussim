import pytest
from spy_network import SpyNetwork

from nexussim.program import Program
from nexussim.scene.scene import NexusScene
from nexussim.segments.base import SegmentState
from nexussim.segments.behaviour import ConstantLoadSegment
from nexussim.segments.control import RepeatSegment


def test_scene_initialization():
    scene = NexusScene.from_yaml(file="tests/test_scene.yaml", auto_execute=False)
    assert len(scene.computes) == 1
    assert len(scene.software_components) == 1
    # scene defines a single network with expected properties
    assert len(scene.networks) == 1
    net = list(scene.networks.values())[0]
    # bandwidth parsed from '1 Gbps' -> 1 * 10**9 (in baud)
    assert net.max_bandwidth == 1 * 10**9
    # latency parsed from '6 ms' -> 0.0 seconds
    assert net.latency == pytest.approx(0.006)

    compute = scene.computes["a_computer"].context.compute
    assert compute.memory.max_memory() == 256 * 1024**2  # from YAML
    assert compute.memory.free_memory() == 256 * 1024**2  # from YAML
    assert compute.memory.used_memory() == 0  # from YAML


def test_scene_representation():
    scene = NexusScene.from_yaml(file="tests/test_scene.yaml")
    assert str(scene)
    # Verify the software component structure matches the YAML
    assert "test_task" in scene.software_components
    program = scene.software_components["test_task"]
    assert isinstance(program, Program)

    # program body should be a RepeatSegment repeating a ConstantLoadSegment
    assert isinstance(program.body, RepeatSegment)
    assert program.body._times == 4

    inner = program.body.segment
    assert isinstance(inner, ConstantLoadSegment)
    # YAML specified id 'test_task_1', cpu_load 1.5 and cycles 10000
    assert inner._id_suffix == "test_task_1"
    assert inner.cpu_load == 1.5
    assert inner.mem_load == 1 * 1024**2
    assert inner.cycles == 10000


def test_scene_execution(dynaa_sim):
    scene = NexusScene.from_yaml(file="tests/test_scene.yaml", auto_execute=False)
    # Wrap existing networks with spies to observe any sends during execution
    spies = {}
    for name, net in list(scene.networks.items()):
        spy = SpyNetwork(net)
        spies[name] = spy
        # replace scene and compute references with spy
        scene._networks[name] = spy
        for compute in scene.computes.values():
            if name in compute._networks:
                compute._networks[name] = spy

    # Verify that the program executed and completed as expected
    program = scene.software_components["test_task"]
    assert program.body._times == 4  # should have repeated 4 times

    context = scene.computes["a_computer"].context
    compute = context.compute

    # Start the execution of the program in the computing node and measure
    # simulated time elapsed and check used.
    assert compute.memory.free_memory() == 256 * 1024**2  # from YAML
    assert compute.memory.used_memory() == 0  # from YAML
    start_time = dynaa_sim.current_time
    program.execute(context)
    # Start dynae sim
    dynaa_sim.run(0.00001)
    assert compute.memory.free_memory() == (256 * 1024**2) - (1024**2)  # from YAML
    assert compute.memory.used_memory() == 1024**2  # from YAML
    # Run dynaa simulator until completion
    dynaa_sim.run()
    elapsed = dynaa_sim.current_time - start_time
    # after the run eevrything should be released
    assert compute.memory.free_memory() == 256 * 1024**2  # from YAML
    assert compute.memory.used_memory() == 0  # from YAML

    # Compute expected simulated duration from cycles and cpu speed
    inner = program.body.segment
    cycles = inner.cycles
    assert cycles == 10000  # from YAML

    cpu_speed = compute.cpu.cpu_speed()
    assert cpu_speed == 800000000.0  # from YAML
    expected_cpu = program.body._times * cycles / cpu_speed

    assert expected_cpu == 0.00005  # 40000 cycles at 800MHz should take 0.00005 seconds
    # With large emits (128 MB), elapsed time is CPU execution plus the total
    # transfer time for all emitted messages.
    network = next(iter(spies.values()))
    emit_size = 128 * 1024**2
    expected = expected_cpu + (program.body._times * emit_size / network.max_bandwidth) + network.latency
    assert elapsed == pytest.approx(expected), f"Expected elapsed time to be {expected}, but got {elapsed}"
    # YAML-derived segments don't set 'executed' in context; verify completion
    assert program.body.state == SegmentState.COMPLETED

    # The segment in this scene now emits a message on each repetition.
    for spy in spies.values():
        # we expect one send per repetition
        assert len(spy.sent) == program.body._times
        for _, dest, descriptor in spy.sent:
            assert dest == "test_task::repeat4::test_task_1"
            assert descriptor.get("size") == 128 * 1024**2


def test_scene_auto_execution(dynaa_sim):
    # if run is not set to false in from_yaml, the program should start executing immediately after scene initialization
    scene = NexusScene.from_yaml(file="tests/test_scene.yaml")
    program = scene.software_components["test_task"]
    assert program.body.state != SegmentState.COMPLETED
    dynaa_sim.run()
    assert program.body.state == SegmentState.COMPLETED


# Tests for second scene with two inner segments
def test_scene2_initialization():
    scene = NexusScene.from_yaml(file="tests/test_scene2.yaml", auto_execute=False)
    assert len(scene.computes) == 1
    assert len(scene.software_components) == 1

    # scene defines a single network with expected properties
    assert len(scene.networks) == 1
    net = list(scene.networks.values())[0]
    # bandwidth parsed from '10 Gbps' -> 10 * 10**9 (in baud)
    assert net.max_bandwidth == 10 * 10**9
    # latency parsed from '6 ms' -> 0.006 seconds
    assert net.latency == pytest.approx(0.006)

    compute = scene.computes["a_computer"].context.compute
    assert compute.memory.max_memory() == 256 * 1024**2  # from YAML
    assert compute.memory.free_memory() == 256 * 1024**2  # from YAML
    assert compute.memory.used_memory() == 0  # from YAML

    # Verify the software component structure matches the YAML
    assert "test_task" in scene.software_components
    program = scene.software_components["test_task"]
    assert isinstance(program, Program)

    # program body should be a RepeatSegment
    # repeating a Sequence of two ConstantLoadSegments
    assert isinstance(program.body, RepeatSegment)
    assert program.body._times == 4

    inner = program.body.segment
    # For a Repeat of multiple segments, the body is a SequenceSegment
    # which holds the two inner segments
    # We check the first inner segment
    assert len(inner.segments) == 2
    first = inner.segments[0]
    second = inner.segments[1]

    assert first.mem_load == 1024**2
    assert second.mem_load == 512 * 1024

    assert isinstance(first, ConstantLoadSegment)
    assert isinstance(second, ConstantLoadSegment)

    # YAML specified ids and loads
    assert first._id_suffix == "test_task_1"
    assert first.cpu_load == 1.5
    assert first.cycles == 10000

    assert second._id_suffix == "test_task_2"
    assert second.cpu_load == 0.5
    assert second.cycles == 5000


def test_scene2_execution(dynaa_sim):
    scene = NexusScene.from_yaml(file="tests/test_scene2.yaml", auto_execute=False)
    # Wrap existing networks with spies to observe any sends during execution
    spies = {}
    for name, net in list(scene.networks.items()):
        spy = SpyNetwork(net)
        spies[name] = spy
        # replace scene and compute references with spy
        scene._networks[name] = spy
        for compute in scene.computes.values():
            if name in compute._networks:
                compute._networks[name] = spy

    program = scene.software_components["test_task"]
    assert program.body._times == 4

    context = scene.computes["a_computer"].context
    compute = context.compute

    # Start the execution of the program in the computing node and measure
    # simulated time elapsed and check used.
    assert compute.memory.free_memory() == 256 * 1024**2  # from YAML
    assert compute.memory.used_memory() == 0  # from YAML
    start_time = dynaa_sim.current_time
    program.execute(context)
    dynaa_sim.run(0.00001)
    assert compute.memory.free_memory() == (256 * 1024**2) - (1024**2)  # from YAML
    assert compute.memory.used_memory() == 1024**2  # from YAML
    dynaa_sim.run(0.000015)
    assert compute.memory.free_memory() == (255.5 * 1024**2)  # from YAML
    assert compute.memory.used_memory() == 0.5 * 1024**2  # from YAML

    dynaa_sim.run()
    elapsed = dynaa_sim.current_time - start_time
    # after the run eevrything should be released
    assert compute.memory.free_memory() == 256 * 1024**2  # from YAML
    assert compute.memory.used_memory() == 0  # from YAML

    cpu_speed = compute.cpu.cpu_speed()
    assert cpu_speed == 1000000000.0  # from YAML
    inner = program.body.segment
    assert len(inner.segments) == 2
    cycles = sum([s.cycles for s in inner.segments])
    assert cycles == 15000  # from YAML
    expected = program.body._times * cycles / cpu_speed

    assert expected == 0.00006  # 4 * 15000 cycles @1GHz
    assert elapsed == pytest.approx(expected)
    assert program.body.state == SegmentState.COMPLETED

    # There are no 'emits' in this test scene, so no sends should have occurred
    for spy in spies.values():
        assert spy.sent == []


# Tests for second scene with two inner segments
def test_scene3_initialization():
    scene = NexusScene.from_yaml(file="tests/test_scene3.yaml")
    assert len(scene.computes) == 1
    assert len(scene.software_components) == 1

    # scene3 does not define networks
    assert len(scene.networks) == 0

    compute = scene.computes["a_computer"].context.compute
    assert compute.memory.max_memory() == 1024**3  # from YAML
    assert compute.memory.free_memory() == 1024**3  # from YAML
    assert compute.memory.used_memory() == 0  # from YAML

    # Verify the software component structure matches the YAML
    assert "test_task" in scene.software_components
    program = scene.software_components["test_task"]
    assert isinstance(program, Program)

    # program body should be a RepeatSegment
    # repeating a Sequence of two ConstantLoadSegments
    assert isinstance(program.body, RepeatSegment)
    assert program.body._times == 4

    inner = program.body.segment
    # For a Repeat of multiple segments, the body is a SequenceSegment
    # which holds the two inner segments
    # We check the first inner segment
    assert len(inner.segments) == 2
    first = inner.segments[0]
    second = inner.segments[1]
    assert first.mem_load == 1 * 1024**2
    assert second.mem_load == 512 * 1024

    assert isinstance(first, ConstantLoadSegment)
    assert isinstance(second, ConstantLoadSegment)

    # YAML specified ids and loads
    assert first._id_suffix == "test_task_1"
    assert first.cpu_load == 1.5
    assert first.cycles == 10000

    assert second._id_suffix == "test_task_2"
    assert second.cpu_load == 0.5
    assert second.duration == 1.0  # from YAML


def test_scene3_execution(dynaa_sim):
    scene = NexusScene.from_yaml(file="tests/test_scene3.yaml")
    program = scene.software_components["test_task"]
    assert program.body._times == 4

    context = scene.computes["a_computer"].context
    compute = context.compute

    # Start the execution of the program in the computing node and measure
    # simulated time elapsed and check used.
    assert compute.memory.free_memory() == 1024**3  # from YAML
    assert compute.memory.used_memory() == 0  # from YAML
    start_time = dynaa_sim.current_time
    program.execute(context)
    dynaa_sim.run(1)
    assert compute.memory.free_memory() == 1024**3 - (512 * 1024)  # from YAML
    assert compute.memory.used_memory() == (512) * 1024  # from YAML
    dynaa_sim.run()
    elapsed = dynaa_sim.current_time - start_time
    # after the run eevrything should be released
    assert compute.memory.free_memory() == 1024**3  # from YAML
    assert compute.memory.used_memory() == 0  # from YAML

    cpu_speed = compute.cpu.cpu_speed()
    assert cpu_speed == 800000000.0  # from YAML

    assert isinstance(program.body, RepeatSegment)
    expected = 0.0
    inner = program.body.segment
    for segment in inner.segments:
        if segment.cycles != 0.0:
            expected += program.body._times * segment.cycles / cpu_speed
        else:
            assert segment.duration != 0.0
            expected += program.body._times * segment.duration

    assert expected == 4.00005  # 4 * 10000 cycles @800 MHz + 4 * 1 second duration
    assert elapsed == pytest.approx(expected)
    assert program.body.state == SegmentState.COMPLETED


def test_scene4_execution(dynaa_sim):
    # no allocations present
    scene = NexusScene.from_yaml(file="tests/test_scene4.yaml", auto_execute=False)
    assert scene.allocations == {}
    # Verify that the program executed and completed as expected
    program = scene.software_components["test_task"]
    context = scene.computes["a_computer"].context

    program.execute(context)
    with pytest.raises(RuntimeError, match=r"No network found by router to send message"):
        dynaa_sim.run()


def test_scene5_execution(dynaa_sim):
    # Invalid allocations
    scene = NexusScene.from_yaml(file="tests/test_scene5.yaml", auto_execute=False)
    assert scene.allocations == {"some_other_task": "some_other_computer"}
    # Verify that the program executed and completed as expected
    program = scene.software_components["test_task"]
    context = scene.computes["a_computer"].context

    program.execute(context)
    with pytest.raises(RuntimeError, match=r"No network found by router to send message"):
        dynaa_sim.run()


def test_scen6_execution(dynaa_sim):
    # Invalid target
    scene = NexusScene.from_yaml(file="tests/test_scene6.yaml", auto_execute=False)
    assert scene.allocations == {"test_task": "a_computer"}
    # Verify that the program executed and completed as expected
    program = scene.software_components["test_task"]
    context = scene.computes["a_computer"].context

    program.execute(context)
    with pytest.raises(RuntimeError, match=r"No network found by router to send message"):
        dynaa_sim.run()
