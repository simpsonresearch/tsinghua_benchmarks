"""
Bellman-Ford shortest path algorithm implementation.
"""

from typing import List, Tuple, Optional, Set
from .utils import (
    validate_shortest_paths,
    detect_negative_cycle,
    bellman_ford_relaxation,
)
from .graph import Graph


class BellmanFordAlgorithm:
    """
    Implementation of the Bellman-Ford shortest path algorithm.
    Time complexity: O(VE)
    Space complexity: O(V)

    Can handle negative edge weights and detect negative cycles.
    """

    def __init__(self):
        self.name = "Bellman-Ford Algorithm"

    def shortest_paths(
        self, graph: Graph, source: int
    ) -> Tuple[List[float], List[Optional[int]]]:
        """
        Find shortest paths from source to all other nodes.

        Args:
            graph: Input graph
            source: Source node

        Returns:
            Tuple of (distances, predecessors) where:
            - distances[i] is shortest distance from source to node i
            - predecessors[i] is the predecessor of node i in shortest path tree

        Raises:
            ValueError: If source is invalid
            RuntimeError: If negative cycle is detected
        """
        if source < 0 or source >= graph.num_nodes:
            raise ValueError(f"Invalid source node: {source}")

        # Initialize distances and predecessors
        distances = [float("inf")] * graph.num_nodes
        predecessors = [None] * graph.num_nodes
        distances[source] = 0.0

        # Optimize: track which nodes can still be relaxed
        active_nodes = {source}

        # Relax edges repeatedly
        for iteration in range(graph.num_nodes - 1):
            if not active_nodes:  # Early termination if no active nodes
                break

            updated = False
            next_active = set()

            # Process only active nodes for better performance
            for u in active_nodes:
                if distances[u] == float("inf"):
                    continue

                for v, weight in graph.get_neighbors_fast(u):
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        predecessors[v] = u
                        next_active.add(v)
                        updated = True

            active_nodes = next_active

            # Early termination if no updates in this iteration
            if not updated:
                break

        # Check for negative cycles
        if detect_negative_cycle(graph, distances):
            raise RuntimeError("Negative cycle detected in the graph")

        # Validate the result
        if not validate_shortest_paths(graph, source, distances, predecessors):
            raise RuntimeError("Algorithm produced invalid results")

        return distances, predecessors

    def shortest_paths_with_negative_cycle_detection(
        self, graph: Graph, source: int
    ) -> Tuple[List[float], List[Optional[int]], bool]:
        """
        Find shortest paths and return whether a negative cycle was detected.

        Args:
            graph: Input graph
            source: Source node

        Returns:
            Tuple of (distances, predecessors, has_negative_cycle)
        """
        if source < 0 or source >= graph.num_nodes:
            raise ValueError(f"Invalid source node: {source}")

        # Initialize distances and predecessors
        distances = [float("inf")] * graph.num_nodes
        predecessors = [None] * graph.num_nodes
        distances[source] = 0.0

        # Relax edges repeatedly
        for _ in range(graph.num_nodes - 1):
            updated = False

            # Process all edges
            for u, v, weight in graph.edges:
                if (
                    distances[u] != float("inf")
                    and distances[u] + weight < distances[v]
                ):
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
                    updated = True

            # Early termination if no updates in this iteration
            if not updated:
                break

        # Check for negative cycles
        has_negative_cycle = detect_negative_cycle(graph, distances)

        return distances, predecessors, has_negative_cycle

    def get_algorithm_info(self) -> dict:
        """Get information about this algorithm implementation."""
        return {
            "name": self.name,
            "time_complexity": "O(VE)",
            "space_complexity": "O(V)",
            "handles_negative_weights": True,
            "optimal": True,
            "description": "Classic algorithm that can handle negative weights and detect negative cycles",
        }
