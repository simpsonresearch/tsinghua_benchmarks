"""
Dijkstra's shortest path algorithm implementation.
"""

from typing import List, Tuple, Optional
from .utils import PriorityQueue, validate_shortest_paths
from .graph import Graph


class DijkstraAlgorithm:
    """
    Implementation of Dijkstra's shortest path algorithm.
    Time complexity: O((V + E) log V) with binary heap
    Space complexity: O(V)
    """

    def __init__(self, use_fibonacci_heap: bool = False):
        """
        Initialize Dijkstra's algorithm.

        Args:
            use_fibonacci_heap: Whether to use Fibonacci heap (experimental)
        """
        self.use_fibonacci_heap = use_fibonacci_heap
        self.name = "Dijkstra's Algorithm"

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
            ValueError: If graph has negative weights or invalid source
        """
        if source < 0 or source >= graph.num_nodes:
            raise ValueError(f"Invalid source node: {source}")

        if graph.has_negative_weights():
            raise ValueError("Dijkstra's algorithm cannot handle negative weights")

        # Initialize distances and predecessors
        distances = [float("inf")] * graph.num_nodes
        predecessors = [None] * graph.num_nodes
        distances[source] = 0.0

        # Initialize priority queue
        pq = PriorityQueue()
        pq.push(source, 0.0)

        # Track visited nodes - use set for O(1) lookup
        visited = set()
        nodes_processed = 0

        # Optimize: precompute node degrees for better processing order
        node_degrees = [
            (i, len(graph.get_neighbors_fast(i))) for i in range(graph.num_nodes)
        ]
        node_degrees.sort(
            key=lambda x: x[1], reverse=True
        )  # Process high-degree nodes first

        while not pq.is_empty():
            current, current_dist = pq.pop()

            # Skip if already processed (can happen with decrease-key operations)
            if current in visited:
                continue

            visited.add(current)
            nodes_processed += 1

            # Early termination if all nodes are processed
            if nodes_processed == graph.num_nodes:
                break

            # Relax all neighbors using optimized neighbor lookup
            for neighbor, weight in graph.get_neighbors_fast(current):
                if neighbor in visited:
                    continue

                new_distance = current_dist + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current
                    pq.push(neighbor, new_distance)

        # Validate the result
        if not validate_shortest_paths(graph, source, distances, predecessors):
            raise RuntimeError("Algorithm produced invalid results")

        return distances, predecessors

    def shortest_path_to_target(
        self, graph: Graph, source: int, target: int
    ) -> Tuple[float, List[int]]:
        """
        Find shortest path from source to a specific target.
        Optimized version that stops when target is reached.

        Args:
            graph: Input graph
            source: Source node
            target: Target node

        Returns:
            Tuple of (distance, path) where:
            - distance is the shortest distance from source to target
            - path is the list of nodes in the shortest path
        """
        if source < 0 or source >= graph.num_nodes:
            raise ValueError(f"Invalid source node: {source}")
        if target < 0 or target >= graph.num_nodes:
            raise ValueError(f"Invalid target node: {target}")

        if graph.has_negative_weights():
            raise ValueError("Dijkstra's algorithm cannot handle negative weights")

        if source == target:
            return 0.0, [source]

        # Initialize distances and predecessors
        distances = [float("inf")] * graph.num_nodes
        predecessors = [None] * graph.num_nodes
        distances[source] = 0.0

        # Initialize priority queue
        pq = PriorityQueue()
        pq.push(source, 0.0)

        # Track visited nodes
        visited = set()

        while not pq.is_empty():
            current, current_dist = pq.pop()

            # Early termination if target is reached
            if current == target:
                break

            # Skip if already processed
            if current in visited:
                continue

            visited.add(current)

            # Relax all neighbors
            for neighbor, weight in graph.get_neighbors(current):
                if neighbor in visited:
                    continue

                new_distance = current_dist + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current
                    pq.push(neighbor, new_distance)

        # Reconstruct path
        if distances[target] == float("inf"):
            return float("inf"), []

        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()

        return distances[target], path

    def get_algorithm_info(self) -> dict:
        """Get information about this algorithm implementation."""
        return {
            "name": self.name,
            "time_complexity": "O((V + E) log V)",
            "space_complexity": "O(V)",
            "handles_negative_weights": False,
            "optimal": True,
            "description": "Classic greedy algorithm that always processes the closest unvisited node",
        }


class DijkstraWithFibonacciHeap(DijkstraAlgorithm):
    """
    Dijkstra's algorithm using Fibonacci heap for better theoretical complexity.
    Time complexity: O(E + V log V)

    Note: In practice, binary heap often performs better due to lower constant factors.
    """

    def __init__(self):
        super().__init__(use_fibonacci_heap=True)
        self.name = "Dijkstra with Fibonacci Heap"

    def shortest_paths(
        self, graph: Graph, source: int
    ) -> Tuple[List[float], List[Optional[int]]]:
        """
        Implementation using Fibonacci heap for decrease-key operations.
        """
        from .utils import FibonacciHeap

        if source < 0 or source >= graph.num_nodes:
            raise ValueError(f"Invalid source node: {source}")

        if graph.has_negative_weights():
            raise ValueError("Dijkstra's algorithm cannot handle negative weights")

        # Initialize distances and predecessors
        distances = [float("inf")] * graph.num_nodes
        predecessors = [None] * graph.num_nodes
        distances[source] = 0.0

        # Initialize Fibonacci heap
        fib_heap = FibonacciHeap()

        # Insert all nodes into the heap
        for i in range(graph.num_nodes):
            fib_heap.insert(distances[i], i)

        # Track visited nodes
        visited = set()

        while not fib_heap.is_empty():
            current_dist, current = fib_heap.extract_min()

            # Skip if already processed
            if current in visited:
                continue

            visited.add(current)

            # Early termination if all nodes are processed
            if len(visited) == graph.num_nodes:
                break

            # Relax all neighbors
            for neighbor, weight in graph.get_neighbors(current):
                if neighbor in visited:
                    continue

                new_distance = current_dist + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current
                    # Decrease key in Fibonacci heap
                    try:
                        fib_heap.decrease_key(neighbor, new_distance)
                    except KeyError:
                        # Node already extracted, skip
                        pass

        # Validate the result
        if not validate_shortest_paths(graph, source, distances, predecessors):
            raise RuntimeError("Algorithm produced invalid results")

        return distances, predecessors

    def get_algorithm_info(self) -> dict:
        """Get information about this algorithm implementation."""
        return {
            "name": self.name,
            "time_complexity": "O(E + V log V)",
            "space_complexity": "O(V)",
            "handles_negative_weights": False,
            "optimal": True,
            "description": "Dijkstra using Fibonacci heap for better theoretical complexity",
        }


class BidirectionalDijkstra:
    """
    Bidirectional Dijkstra for finding shortest path between two specific nodes.
    Can be significantly faster than regular Dijkstra for single-pair queries.
    """

    def __init__(self):
        self.name = "Bidirectional Dijkstra"

    def shortest_path(
        self, graph: Graph, source: int, target: int
    ) -> Tuple[float, List[int]]:
        """
        Find shortest path from source to target using bidirectional search.

        Args:
            graph: Input graph
            source: Source node
            target: Target node

        Returns:
            Tuple of (distance, path)
        """
        if source < 0 or source >= graph.num_nodes:
            raise ValueError(f"Invalid source node: {source}")
        if target < 0 or target >= graph.num_nodes:
            raise ValueError(f"Invalid target node: {target}")

        if graph.has_negative_weights():
            raise ValueError("Bidirectional Dijkstra cannot handle negative weights")

        if source == target:
            return 0.0, [source]

        # Initialize forward and backward searches
        forward_dist = [float("inf")] * graph.num_nodes
        backward_dist = [float("inf")] * graph.num_nodes
        forward_pred = [None] * graph.num_nodes
        backward_pred = [None] * graph.num_nodes

        forward_dist[source] = 0.0
        backward_dist[target] = 0.0

        forward_pq = PriorityQueue()
        backward_pq = PriorityQueue()
        forward_pq.push(source, 0.0)
        backward_pq.push(target, 0.0)

        forward_visited = set()
        backward_visited = set()

        best_distance = float("inf")
        meeting_point = None

        while not forward_pq.is_empty() and not backward_pq.is_empty():
            # Expand from forward search
            if not forward_pq.is_empty():
                current, current_dist = forward_pq.pop()

                if current not in forward_visited:
                    forward_visited.add(current)

                    # Check if we've met the backward search
                    if current in backward_visited:
                        total_dist = forward_dist[current] + backward_dist[current]
                        if total_dist < best_distance:
                            best_distance = total_dist
                            meeting_point = current

                    # Expand neighbors
                    for neighbor, weight in graph.get_neighbors(current):
                        if neighbor not in forward_visited:
                            new_distance = current_dist + weight
                            if new_distance < forward_dist[neighbor]:
                                forward_dist[neighbor] = new_distance
                                forward_pred[neighbor] = current
                                forward_pq.push(neighbor, new_distance)

            # Expand from backward search
            if not backward_pq.is_empty():
                current, current_dist = backward_pq.pop()

                if current not in backward_visited:
                    backward_visited.add(current)

                    # Check if we've met the forward search
                    if current in forward_visited:
                        total_dist = forward_dist[current] + backward_dist[current]
                        if total_dist < best_distance:
                            best_distance = total_dist
                            meeting_point = current

                    # Expand neighbors (in reverse direction)
                    for u in range(graph.num_nodes):
                        if graph.has_edge(u, current) and u not in backward_visited:
                            weight = graph.get_weight(u, current)
                            new_distance = current_dist + weight
                            if new_distance < backward_dist[u]:
                                backward_dist[u] = new_distance
                                backward_pred[u] = current
                                backward_pq.push(u, new_distance)

            # Early termination condition
            if meeting_point is not None:
                # Check if we can stop (both searches have expanded beyond the meeting point)
                min_forward = min(
                    [d for d in forward_dist if d != float("inf")], default=float("inf")
                )
                min_backward = min(
                    [d for d in backward_dist if d != float("inf")],
                    default=float("inf"),
                )

                if min_forward + min_backward >= best_distance:
                    break

        if meeting_point is None:
            return float("inf"), []

        # Reconstruct path
        forward_path = []
        current = meeting_point
        while current is not None:
            forward_path.append(current)
            current = forward_pred[current]
        forward_path.reverse()

        backward_path = []
        current = backward_pred[meeting_point]
        while current is not None:
            backward_path.append(current)
            current = backward_pred[current]

        path = forward_path + backward_path
        return best_distance, path

    def get_algorithm_info(self) -> dict:
        """Get information about this algorithm implementation."""
        return {
            "name": self.name,
            "time_complexity": "O((V + E) log V)",
            "space_complexity": "O(V)",
            "handles_negative_weights": False,
            "optimal": True,
            "description": "Bidirectional search that can be faster for single-pair queries",
        }
