import pytest

from nexussim.compute import Compute
from nexussim.cpus import UnboundedCPU
from nexussim.memory import UnboundedMemory
from nexussim.segments.base import Segment, SegmentState


class DummySegment(Segment):
    def _segment_body(self, context):
        yield self._schedule_after(10, self.RESUME_SEGMENT)


@pytest.fixture
def compute():
    return Compute(UnboundedCPU(10), UnboundedMemory(1000))


def test_compute_execute(compute):
    segment = DummySegment()

    compute.execute(segment)
    assert segment.state == SegmentState.RUNNING


def test_compute_execute_invalid(compute):
    with pytest.raises(AttributeError):
        compute.execute("not a segment")
