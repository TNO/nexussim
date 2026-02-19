import re


class Router:
    """
    Router for software components, segments, and computes, with precomputed best network lookup.

    This class calculates and manages a routing table that maps each pair of software components or segments
    to the best network (lowest latency, highest bandwidth) connecting their assigned computes.
    The routing table is built at initialization and used for fast lookups during simulation.
    """

    def __init__(self, allocations: dict, networks: dict, software_components: dict):
        """
        Initialize the Router and build the routing table for all segments and software components.

        Args:
            allocations (dict): Mapping of software component names to compute objects
            networks (dict): Mapping of network names to network objects (must have .latency and .bandwidth).
            software_components (dict): Mapping of software component names to Program objects
            (with segment trees).

        Notes:
            The Router stores references to the provided dicts (not copies).
            The routing table is built at initialization
            and will not update if allocations or networks change after init.
        """
        assert allocations is not None, "Allocations dict must be provided."
        assert networks is not None, "Networks dict must be provided."
        assert software_components is not None, "Software components dict must be provided."
        self.allocations = allocations
        self.networks = networks
        self.software_components = software_components

        # Map each segment (including nested) to the set of network names available to its compute
        segment_network_allocations = {}
        # First, map top-level software components to their networks
        for sw_id, compute_id in self.allocations.items():
            nets = [nw_key for nw_key, nw in self.networks.items() if compute_id in nw.peers]
            segment_network_allocations[sw_id] = nets
        # Now, propagate to all nested segments
        for sw_id, sw_comp in self.software_components.items():
            if not hasattr(sw_comp, "body"):
                raise ValueError(
                    f"Software component {sw_id} must have a 'body' attribute representing its main segment."
                )
            stack = [sw_comp.body]
            while stack:
                segment = stack.pop()
                segment_network_allocations[segment.id] = segment_network_allocations.get(sw_id, [])
                if hasattr(segment, "segment"):
                    stack.extend([segment.segment])
                if hasattr(segment, "segments"):
                    stack.extend(segment.segments)

        def net_sort_key(nw_key):
            """Key function for sorting networks by latency then bandwidth.

            This is the key function used to sort candidate networks: lower
            latency is preferred first, and higher bandwidth is preferred next
            by negating bandwidth in the returned tuple.
            """
            nw = self.networks[nw_key]
            return (getattr(nw, "latency", float("inf")), -getattr(nw, "bandwidth", 0))

        # Build routing table for all pairs of segments
        self._routing_table = {}
        segment_names = tuple(segment_network_allocations.keys())
        for i, sw1 in enumerate(segment_names):
            nw1 = set(segment_network_allocations.get(sw1, []))
            for sw2 in segment_names[i:]:
                nw2 = set(segment_network_allocations.get(sw2, []))
                common = nw1 & nw2
                if not common:
                    continue
                best = min(common, key=net_sort_key)
                self._routing_table[Router._key(sw1, sw2)] = best

    def _key(sw1: str, sw2: str) -> tuple:
        """
        Helper to create a consistent key for the routing table, regardless of argument order.
        """
        return tuple(sorted((sw1, sw2)))

    def get_network(self, sw1: str, sw2: str) -> str:
        """
        Look up the best network connecting the computes assigned to two software components or segments.

        Args:
            sw1 (str): Name or ID of the first software component or segment.
            sw2 (str): Name or ID of the second software component or segment.

        Returns:
            str: The name of the best network connecting both computes, or None if none found.
        """
        return self._routing_table.get(Router._key(sw1, sw2))

    def emit_message(self, segment_id: str, segment_parent: str, message_descriptor: dict, compute):
        """
        Emit a message by routing it through the network.

        Takes a segment's message descriptor and uses the router to find the best
        network path to the target destination. Handles normalization of target IDs
        (converting single ':' to '::') before routing.

        Args:
            segment_id (str): The ID of the sending segment.
            segment_parent (str): The parent ID of the segment (for context in error messages).
            message_descriptor (dict): Message metadata with keys:
                - "msg_size": size in bytes
                - "msg_destination": target segment ID (string or list)
            compute: The compute object with network access.

        Raises:
            RuntimeError: If no network route is found.
        """
        if not message_descriptor:
            return

        size = message_descriptor.get("msg_size", 0)
        dest = message_descriptor.get("msg_destination", None)
        if not dest or size <= 0:
            return

        # allow single destination string or a list
        targets = dest if isinstance(dest, (list, tuple)) else [dest]

        data_descriptor = {"size": size, "segment": segment_id}

        for t in targets:
            # Router keys are built from segment/program ids that use "::"
            # hierarchy separator (e.g. parent::child::segment).
            # Normalize user/YAML targets that may use single ":" so lookups are
            # consistent.
            t = re.sub(r"(?<!:):(?!:)", "::", t)
            net_name = self.get_network(segment_parent, t)
            if not net_name:
                raise RuntimeError(f"No network found by router to send message from {segment_parent} to {t}.")
            net = compute.network(net_name)
            assert net is not None, f"Compute must have have network '{net_name}'."
            net.send(segment_parent, t, data_descriptor)

    def __str__(self):
        """
        Return a string representation of the Router, showing the routing table.
        """
        return f"Router(routing_table={self._routing_table})"
