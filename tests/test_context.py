import pytest
from nexussim.context import Context


def test_empty_context():
    context = Context()
    assert context.compute is None
