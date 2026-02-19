import pytest

from nexussim.scene.scene import NexusScene


@pytest.fixture
def scene():
    return NexusScene.from_yaml(file="tests/test_scene.yaml")


def test_scene_initialization(scene):
    assert len(scene.computes) == 1
    assert len(scene.software_components) == 1
