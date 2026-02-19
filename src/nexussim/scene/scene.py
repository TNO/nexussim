from enum import Enum
from pathlib import Path

import pydynaa as pd
import yaml
from jsonschema import ValidationError, validate
from pint import UnitRegistry

from nexussim.compute import Compute
from nexussim.cpus import UnboundedCPU
from nexussim.loggers import CPUUsageBuffer, CPUUsageInflux
from nexussim.memory import UnboundedMemory
from nexussim.program import (Program, choice, concurrent,
                              constant_load_segment, repeat, while_do)
from nexussim.segments.control import SequenceSegment, forever

COMPUTES_TAG = "computes"
SOFTWARE_COMPONENTS_TAG = "software components"
DESCRIPTION_TAG = "description"
ALLOCATIONS_TAG = "allocations"
ID_TAG = "id"
CPUS_TAG = "cpus"
MEM_TAG = "mem"
BASE_LOAD_TAG = "base_load"
CPU_LOAD_TAG = "cpu_load"
CPU_SPEED_TAG = "cpu_speed"
MEM_LOAD_TAG = "memory"
SEGMENTS_TAG = "segments"
SEGMENT_LOAD_TAG = "load"
SEGMENT_DURATION_TAG = "duration"
TIME_LOAD_TAG = "time"
CYCLES_LOAD_TAG = "cycles"
STOPS_AFTER_TAG = "stops_after"
EMITS_TAG = "emits"
WHEN_TAG = "when"
N_TIMES_TAG = "n_times"
WEIGHTS_TAG = "weights"
CONCURRENT_TAG = "concurrent"
REPEAT_TAG = "repeat"
CHOICE_TAG = "choice"


class SegmentType(Enum):
    """Enum for the different types of segments in a program."""

    SEQUENCE = 1
    CHOICE = 2
    REPEAT = 3
    CONCURRENT = 4
    WHILE = 5
    CONSTANT_LOAD = 6
    SAMPLED_LOAD = 7


def yaml_reader(file: Path, validation_schema: Path = None) -> dict:
    """
    Reads a YAML file and returns the contents as a Python dictionary.

    If `validation_schema` is given, validates the contents of the YAML file
    against the given schema before returning the contents.

    Args:
        file: The YAML file to read.
        validation_schema: The JSON schema to validate the contents of
            the YAML file against, if given.

    Returns:
        The contents of the YAML file as a Python dictionary.

    Raises:
        ValueError: If the YAML file is not valid according to the
            given schema.
    """
    with open(file, "r", encoding="utf-8") as scene_file:
        scene = yaml.safe_load(scene_file)

    # Makes a specific validation against only the service chain part of the input file.
    if validation_schema is not None:
        with open(validation_schema, "r", encoding="utf-8") as schema_file:
            validation_schema = yaml.safe_load(schema_file)
        if not validation_schema:
            raise ValueError(f"Schema file {validation_schema} empty or not found.")

        try:
            validate(instance=scene, schema=validation_schema)
        except ValidationError as e:
            raise ValueError(f"Scene {file} is not valid: {e}") from e

    return scene


# Later, we may think about re-factoring this construction class to something more
# sofisticated.  The actual implementation translates pragmatically the yaml description
# into nexus components.


class NexusScene:
    """Object model for a nexussim scene"""

    def __init__(self, scene_specification: dict):
        """
        Create a new NexusScene object

        Give preference to use the static method `from_dict`.  This constructor may
        change in the future.

        Args:
            scene_specification (dict): Specification of the scene.
        """
        self._specs = scene_specification
        self._computes = None
        self._sw_components = None
        self._allocations = None

    @staticmethod
    def from_yaml(file: Path, validation_schema: Path = None):
        """Helper funtion to create a nexus scene directly from the scene spec file"""
        scene = NexusScene(scene_specification=yaml_reader(file, validation_schema))
        scene.build()
        return scene

    @staticmethod
    def from_dict(scene_specification: dict):
        """Helper funtion to create a nexus scene directly from the scene spec dict"""
        scene = NexusScene(scene_specification=scene_specification)
        scene.build()
        return scene

    @property
    def computes(self):
        """
        A dictionary of Compute objects representing the computes in the scene.

        The key for each compute is the id given in the scene specification.
        The value is the Compute object itself.

        The Compute objects are created lazily, only when retrieved the first time.
        """
        if self._computes is None:
            cpu_clock = UnitRegistry()
            computes = dict()
            for compute in self._specs.get(COMPUTES_TAG, []):
                _id, description = compute[ID_TAG], compute.get(DESCRIPTION_TAG, "")
                compute_memory = compute.get(MEM_TAG, 0)
                compute_cpus = compute.get(CPUS_TAG, 1)
                compute_cpu_speed = (
                    cpu_clock(compute.get(CPU_SPEED_TAG, "0.0 Hz")).to("Hz").magnitude
                )
                computes[_id] = Compute(
                    memory=UnboundedMemory(max_memory=compute_memory),
                    cpu=UnboundedCPU(
                        max_cpus=compute_cpus, cpu_speed=compute_cpu_speed
                    ),
                    description=description,
                )
            self._computes = computes

        return self._computes

    @property
    def software_components(self):
        """
        A dictionary of Program objects representing the software components in the
        scene.

        The key for each component is the name of the program.  The value is the Program
        object itself.

        The Program objects are created lazily, only when retrieved the first time.
        """
        if self._sw_components is None:
            sw_components = dict()
            for component in self._specs.get(SOFTWARE_COMPONENTS_TAG, []):
                a_program = self._as_nexus_program(component)
                sw_components[a_program.name] = a_program
            self._sw_components = sw_components
        return self._sw_components

    @property
    def allocations(self):
        """
        A dictionary of allocations between software components and the computes where
        they run.

        The keys of the dictionary are the names of the software components.  The values
        are the computes to which they are assigned.
        """
        if self._allocations is None:
            self._allocations = {}
            self._allocations.update(self._specs.get(ALLOCATIONS_TAG, {}))
        return self._allocations

    def _as_nexus_program(self, sw_component):
        """
        Translate a software component specified in the scene into a Program object.

        Args:
            component: The specification of the software component as a dictionary.

        Returns:
            A Program object created from the given specification.

        Raises:
            ValueError: If the software component specification is invalid.
        """
        program_name = sw_component[ID_TAG]
        program_base_load = _program_base_load(sw_component)
        try:
            body = _program_body(sw_component.get(SEGMENTS_TAG, []))
        except ValueError as e:
            raise ValueError(f"Invalid software component {program_name}: {e}") from e
        a_program = Program(
            name=program_name,
            body=body,
            cpu_load=program_base_load[CPU_LOAD_TAG],
            mem_load=program_base_load[MEM_LOAD_TAG],
        )

        return a_program

    def build(self):
        """Builds the full scene as a Nexus scene setup, making it ready for
        simulation.

        A scene is make of Nexus components (program, computes, segments, etc.), which
        in turn, are DynAA entities.  When a scene is built, new elements are created
        and connected, preparing the scene for simulation.  In this process, old
        elements and their references are __invalidated__.

        The methods `build()` can also be used as a way to reset the scene for a new
        simulation round.
        """
        for sw_component_id, compute_id in self.allocations.items():
            if (
                compute_id in self.computes
                and sw_component_id in self.software_components
            ):
                compute = self.computes[compute_id]
                # only add observers if we have not already added them
                if not compute.context.compute.cpu._observers:
                    compute.context.compute.cpu._observers = [
                        CPUUsageBuffer(timestampfunc=sim_time),
                        CPUUsageInflux(compute_id=compute_id)
                    ]
                compute.execute(self.software_components[sw_component_id])

    @property
    def as_dict(self):
        return {"NexusScene": {"Computes": [f"{compute}" for compute in self.computes],
                               "Software Components": [f"{software_component}"
                                                       for software_component
                                                       in self.software_components],
                               "Allocations": {f"{allocation}":
                                               f"{self.allocations[allocation]}"
                                               for allocation in self.allocations}
                               }
                }

    def __str__(self):
        """String representation of the NexusScene object."""
        _desc = "NexusScene:\n"
        _desc += "  Computes:\n"
        for compute in self.computes:
            _desc += f"    {compute}\n"
        _desc += "  Software Components:\n"
        for software_component in self.software_components:
            _desc += f"    {software_component}\n"
        _desc += "  Allocations:\n"
        for allocation in self.allocations:
            _desc += f"    {allocation}: {self.allocations[allocation]}\n"
        return _desc


def _program_base_load(component):
    base = component.get(BASE_LOAD_TAG, {CPU_LOAD_TAG: 0.0, MEM_LOAD_TAG: 0.0})
    return base


def _program_body(segments):
    if not segments:
        raise ValueError("No segments in program body")

    if len(segments) == 1:
        return _extract_segment(segments[0])

    # If there are multiple segments, we need to create a sequence
    # segment to hold them all.
    segs = [_extract_segment(segment) for segment in segments]
    return SequenceSegment(segments=segs)


def _extract_segment(segment):
    """
    Extracts and constructs a segment from a given specification.

    This function interprets the provided segment dictionary and constructs
    the corresponding segment object based on its type. Supported segment
    types include REPEAT, CONSTANT_LOAD, CONCURRENT, and CHOICE.

    Args:
        segment (dict): A dictionary containing the segment specification.

    Returns:
        Segment: An instance of the appropriate segment type based on the
        provided specification.

    Raises:
        KeyError: If required keys for the segment type are missing in the
        specification.
    """
    match _segment_type(segment):
        case SegmentType.REPEAT:
            segments = segment.get(REPEAT_TAG).get(SEGMENTS_TAG)
            n_times = segment.get(REPEAT_TAG).get(N_TIMES_TAG, -1)
            if n_times >= 0:
                return repeat(segment=_program_body(segments), times=n_times)
            return while_do(_program_body(segments), condition=forever)
        case SegmentType.CONSTANT_LOAD:
            seg_load = {
                CPU_LOAD_TAG: 0.0,
                MEM_LOAD_TAG: 0.0,
                TIME_LOAD_TAG: 0.0,
                CYCLES_LOAD_TAG: 0,
            }
            if SEGMENT_LOAD_TAG in segment:
                seg_load.update(segment[SEGMENT_LOAD_TAG])
            if STOPS_AFTER_TAG in segment:
                duration = UnitRegistry()
                time = segment[STOPS_AFTER_TAG].get(TIME_LOAD_TAG, "0.0s")
                time = duration(time).to("s").magnitude
                cycles = segment[STOPS_AFTER_TAG].get(CYCLES_LOAD_TAG, 0.0)
                seg_load[TIME_LOAD_TAG] = time
                seg_load[CYCLES_LOAD_TAG] = cycles

            cseg = constant_load_segment(
                cpu_load=seg_load[CPU_LOAD_TAG],
                mem_load=seg_load[MEM_LOAD_TAG],
                duration=seg_load[TIME_LOAD_TAG],
                cycles=seg_load[CYCLES_LOAD_TAG],
            )
            cseg.suffix(segment.get(ID_TAG))
            return cseg

        case SegmentType.CONCURRENT:
            segments = [
                _extract_segment(segment)
                for segment in segment.get(CONCURRENT_TAG).get(SEGMENTS_TAG)
            ]
            return concurrent(*segments)
        case SegmentType.CHOICE:
            segments = [
                _extract_segment(segment)
                for segment in segment.get(CHOICE_TAG).get(SEGMENTS_TAG)
            ]
            weights = segment.get(CHOICE_TAG).get(WEIGHTS_TAG, None)
            return choice(*segments, weights=weights)


def _segment_type(segment: dict) -> SegmentType:
    """
    Identifies the type of a segment.

    A segment can be identified as either a repeat segment, a concurrent segment or a
    constant load segment.

    Args:
        segment (dict): The segment to be identified.

    Returns:
        SegmentType: The type of the segment.
    """
    if len(segment) == 1 and REPEAT_TAG in segment.keys():
        return SegmentType.REPEAT
    if len(segment) == 1 and CONCURRENT_TAG in segment.keys():
        return SegmentType.CONCURRENT
    if len(segment) == 1 and CHOICE_TAG in segment.keys():
        return SegmentType.CHOICE
    return SegmentType.CONSTANT_LOAD


def sim_time():
    """
    Return the current time in seconds of the dynaa simulator.
    """
    return pd.DynAASim().current_time * 10**6  # in seconds
