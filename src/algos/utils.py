"""
Utility functions for shortest path algorithms.
"""

import heapq
from typing import List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict


class PriorityQueue:
    """
    A priority queue implementation using heapq.
    Supports decrease_key operation for Dijkstra's algorithm.
    """

    def __init__(self):
        self.heap = []
        self.entry_finder = {}  # mapping of items to entries
        self.counter = 0
        self.REMOVED = "<removed-item>"  # placeholder for a removed item

    def push(self, item: Any, priority: float):
        """Add a new item or update the priority of an existing item."""
        if item in self.entry_finder:
            self.remove(item)

        entry = [priority, self.counter, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def remove(self, item: Any):
        """Mark an existing item as REMOVED."""
        if item in self.entry_finder:
            entry = self.entry_finder.pop(item)
            entry[-1] = self.REMOVED

    def pop(self) -> Tuple[Any, float]:
        """Remove and return the lowest priority item."""
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError("pop from an empty priority queue")

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self.entry_finder) == 0

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self.entry_finder)


class FibonacciHeap:
    """
    A simplified Fibonacci heap implementation for educational purposes.
    In practice, a well-optimized binary heap often performs better due to constant factors.
    """

    class Node:
        def __init__(self, key: float, value: Any):
            self.key = key
            self.value = value
            self.degree = 0
            self.marked = False
            self.parent = None
            self.child = None
            self.left = self
            self.right = self

    def __init__(self):
        self.min_node = None
        self.count = 0
        self.node_map = {}  # Maps values to nodes for decrease_key

    def insert(self, key: float, value: Any) -> "FibonacciHeap.Node":
        """Insert a new node with given key and value."""
        node = self.Node(key, value)
        self.node_map[value] = node

        if self.min_node is None:
            self.min_node = node
        else:
            self._add_to_root_list(node)
            if node.key < self.min_node.key:
                self.min_node = node

        self.count += 1
        return node

    def extract_min(self) -> Tuple[float, Any]:
        """Extract and return the minimum key-value pair."""
        if self.min_node is None:
            raise KeyError("Heap is empty")

        min_node = self.min_node

        # Add all children to root list
        if min_node.child:
            child = min_node.child
            while True:
                next_child = child.right
                child.parent = None
                self._add_to_root_list(child)
                child = next_child
                if child == min_node.child:
                    break

        # Remove min_node from root list
        self._remove_from_root_list(min_node)

        if min_node == min_node.right:
            self.min_node = None
        else:
            self.min_node = min_node.right
            self._consolidate()

        self.count -= 1
        del self.node_map[min_node.value]
        return min_node.key, min_node.value

    def decrease_key(self, value: Any, new_key: float):
        """Decrease the key of the node with given value."""
        if value not in self.node_map:
            raise KeyError(f"Value {value} not found in heap")

        node = self.node_map[value]
        if new_key > node.key:
            raise ValueError("New key is greater than current key")

        node.key = new_key
        parent = node.parent

        if parent and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)

        # Update min_node if necessary
        if node.key < self.min_node.key:
            self.min_node = node

    def is_empty(self) -> bool:
        """Check if the heap is empty."""
        return self.count == 0

    def _add_to_root_list(self, node: "FibonacciHeap.Node"):
        """Add node to the root list."""
        if self.min_node is None:
            self.min_node = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right = node
            node.right.left = node

    def _remove_from_root_list(self, node: "FibonacciHeap.Node"):
        """Remove node from the root list."""
        node.left.right = node.right
        node.right.left = node.left

    def _consolidate(self):
        """Consolidate the heap after extract_min."""
        max_degree = int(self.count**0.5) + 1
        degree_table = [None] * max_degree

        # Collect all root nodes
        root_nodes = []
        current = self.min_node
        if current:
            while True:
                root_nodes.append(current)
                current = current.right
                if current == self.min_node:
                    break

        # Consolidate nodes with same degree
        for node in root_nodes:
            degree = node.degree
            while degree < len(degree_table) and degree_table[degree] is not None:
                other = degree_table[degree]
                if node.key > other.key:
                    node, other = other, node

                self._link(other, node)
                degree_table[degree] = None
                degree += 1

            if degree < len(degree_table):
                degree_table[degree] = node

        # Find new minimum
        self.min_node = None
        for node in degree_table:
            if node is not None:
                if self.min_node is None or node.key < self.min_node.key:
                    self.min_node = node

    def _link(self, child: "FibonacciHeap.Node", parent: "FibonacciHeap.Node"):
        """Make child a child of parent."""
        self._remove_from_root_list(child)
        child.parent = parent

        if parent.child is None:
            parent.child = child
            child.left = child
            child.right = child
        else:
            child.left = parent.child
            child.right = parent.child.right
            parent.child.right = child
            child.right.left = child

        parent.degree += 1
        child.marked = False

    def _cut(self, node: "FibonacciHeap.Node", parent: "FibonacciHeap.Node"):
        """Cut node from parent and add to root list."""
        if parent.child == node:
            if node.right == node:
                parent.child = None
            else:
                parent.child = node.right

        node.left.right = node.right
        node.right.left = node.left
        parent.degree -= 1

        node.parent = None
        node.marked = False
        self._add_to_root_list(node)

    def _cascading_cut(self, node: "FibonacciHeap.Node"):
        """Perform cascading cut operation."""
        parent = node.parent
        if parent:
            if not node.marked:
                node.marked = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)


def reconstruct_path(
    predecessors: List[Optional[int]], source: int, target: int
) -> List[int]:
    """
    Reconstruct the shortest path from source to target using predecessors array.

    Args:
        predecessors: Array where predecessors[i] is the predecessor of node i
        source: Source node
        target: Target node

    Returns:
        List of nodes representing the path from source to target,
        or empty list if no path exists
    """
    if predecessors[target] is None and source != target:
        return []  # No path exists

    path = []
    current = target

    while current is not None:
        path.append(current)
        current = predecessors[current]

    path.reverse()
    return path


def validate_shortest_paths(
    graph, source: int, distances: List[float], predecessors: List[Optional[int]]
) -> bool:
    """
    Validate that the computed shortest paths are correct.

    Args:
        graph: The input graph
        source: Source node
        distances: Computed distances from source
        predecessors: Computed predecessors

    Returns:
        True if the solution is valid, False otherwise
    """
    # Check that source distance is 0
    if distances[source] != 0:
        print(f"Validation failed: source distance is {distances[source]}, should be 0")
        return False

    # Check that source has no predecessor
    if predecessors[source] is not None:
        print(
            f"Validation failed: source has predecessor {predecessors[source]}, should be None"
        )
        return False

    # Check triangle inequality for all edges - use optimized edge iteration
    for u, v, weight in graph.edges:
        if distances[u] != float("inf") and distances[u] + weight < distances[v]:
            print(f"Validation failed: triangle inequality violated for edge {u}->{v}")
            print(
                f"  distances[{u}] + weight = {distances[u]} + {weight} = {distances[u] + weight}"
            )
            print(f"  distances[{v}] = {distances[v]}")
            return False

    # Check that predecessor relationships are consistent
    # Only check nodes that have predecessors to avoid unnecessary work
    for i in range(graph.num_nodes):
        if predecessors[i] is not None:
            pred = predecessors[i]
            # Check if there's actually an edge from predecessor to node
            if not graph.has_edge(pred, i):
                print(f"Validation failed: predecessor {pred}->{i} edge doesn't exist")
                return False
            # Check if the distance is correct
            expected_dist = distances[pred] + graph.get_weight(pred, i)
            if abs(distances[i] - expected_dist) > 1e-9:
                print(f"Validation failed: distance mismatch for node {i}")
                print(f"  distances[{i}] = {distances[i]}")
                print(
                    f"  distances[{pred}] + weight = {distances[pred]} + {graph.get_weight(pred, i)} = {expected_dist}"
                )
                return False

    return True


def bellman_ford_relaxation(
    graph,
    distances: List[float],
    predecessors: List[Optional[int]],
    nodes_to_relax: Set[int] = None,
) -> bool:
    """
    Perform one iteration of Bellman-Ford relaxation.
    Used as a building block in the Tsinghua algorithm.

    Args:
        graph: The input graph
        distances: Current distance estimates
        predecessors: Current predecessor array
        nodes_to_relax: Subset of nodes to process (if None, process all)

    Returns:
        True if any distance was updated, False otherwise
    """
    updated = False

    if nodes_to_relax is None:
        # Process all nodes - use optimized edge iteration
        for u in range(graph.num_nodes):
            if distances[u] == float("inf"):
                continue

            for v, weight in graph.get_neighbors_fast(u):
                new_dist = distances[u] + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    updated = True
    else:
        # Process only specified nodes for better performance
        for u in nodes_to_relax:
            if distances[u] == float("inf"):
                continue

            for v, weight in graph.get_neighbors_fast(u):
                new_dist = distances[u] + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    updated = True

    return updated


def detect_negative_cycle(graph, distances: List[float]) -> bool:
    """
    Detect if there's a negative cycle reachable from the source.

    Args:
        graph: The input graph
        distances: Current distance estimates

    Returns:
        True if a negative cycle is detected, False otherwise
    """
    for u, v, weight in graph.edges:
        if distances[u] != float("inf") and distances[u] + weight < distances[v]:
            return True
    return False


def calculate_graph_radius(graph, source: int, max_radius: int = None) -> int:
    """
    Calculate the effective radius of the graph from a source node.
    Used for adaptive algorithm parameters.

    Args:
        graph: The input graph
        source: Source node
        max_radius: Maximum radius to consider

    Returns:
        The radius (maximum shortest path length in terms of hops)
    """
    if max_radius is None:
        max_radius = graph.num_nodes

    visited = set()
    queue = [(source, 0)]
    max_distance = 0

    while queue:
        node, dist = queue.pop(0)
        if node in visited or dist >= max_radius:
            continue

        visited.add(node)
        max_distance = max(max_distance, dist)

        for neighbor, _ in graph.get_neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, dist + 1))

    return max_distance


def estimate_sparsity(graph) -> float:
    """
    Estimate the sparsity of the graph.
    Returns a value between 0 (very dense) and 1 (very sparse).
    """
    max_edges = graph.num_nodes * (graph.num_nodes - 1)
    if not graph.directed:
        max_edges //= 2

    if max_edges == 0:
        return 1.0

    actual_edges = len(graph.edges)
    density = actual_edges / max_edges
    return 1.0 - density  # Convert density to sparsity


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    else:
        return f"{seconds:.3f} s"


def compare_distance_arrays(
    dist1: List[float], dist2: List[float], tolerance: float = 1e-9
) -> bool:
    """
    Compare two distance arrays for equality within a tolerance.

    Args:
        dist1: First distance array
        dist2: Second distance array
        tolerance: Tolerance for floating point comparison

    Returns:
        True if arrays are equal within tolerance, False otherwise
    """
    if len(dist1) != len(dist2):
        return False

    for i in range(len(dist1)):
        if abs(dist1[i] - dist2[i]) > tolerance:
            return False

    return True


def benchmark_algorithm(algorithm, graph, source: int, iterations: int = 5) -> dict:
    """
    Benchmark an algorithm with multiple runs for accurate timing.

    Args:
        algorithm: Algorithm instance to benchmark
        graph: Graph to test on
        source: Source node
        iterations: Number of iterations to run

    Returns:
        Dictionary with benchmark results
    """
    import time

    times = []
    results = []

    # Warm-up run
    try:
        distances, predecessors = algorithm.shortest_paths(graph, source)
        # Validate the warm-up result
        if not validate_shortest_paths(graph, source, distances, predecessors):
            print(f"Warning: Warm-up validation failed for {algorithm.name}")
            return {"error": "Warm-up validation failed", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}

    # Actual benchmark runs
    for i in range(iterations):
        start_time = time.perf_counter()
        try:
            distances, predecessors = algorithm.shortest_paths(graph, source)
            end_time = time.perf_counter()

            runtime = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(runtime)
            results.append((distances, predecessors))

        except Exception as e:
            return {"error": str(e), "success": False}

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Verify consistency of results
    consistent = True
    if len(results) > 1:
        reference = results[0][0]
        for distances, _ in results[1:]:
            if not compare_distance_arrays(reference, distances):
                consistent = False
                break

    return {
        "success": True,
        "iterations": iterations,
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_dev_ms": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5,
        "consistent_results": consistent,
        "algorithm_name": getattr(algorithm, "name", "Unknown"),
        "graph_size": graph.num_nodes,
        "graph_edges": len(graph.edges),
    }


def analyze_algorithm_performance(algorithm, graph, source: int) -> dict:
    """
    Analyze the performance characteristics of an algorithm on a specific graph.

    Args:
        algorithm: Algorithm instance to analyze
        graph: Graph to analyze
        source: Source node

    Returns:
        Dictionary with performance analysis
    """
    # Get algorithm information
    algo_info = getattr(algorithm, "get_algorithm_info", lambda: {})()

    # Analyze graph characteristics
    graph_analysis = {
        "num_nodes": graph.num_nodes,
        "num_edges": len(graph.edges),
        "density": graph.density(),
        "has_negative_weights": graph.has_negative_weights(),
        "is_directed": graph.directed,
        "avg_degree": len(graph.edges) / graph.num_nodes if graph.num_nodes > 0 else 0,
        "high_degree_nodes": len(graph.get_high_degree_nodes()),
        "sparsity": estimate_sparsity(graph),
    }

    # Estimate theoretical performance
    n = graph.num_nodes
    m = len(graph.edges)

    theoretical_complexity = {
        "dijkstra": m + n * (n**0.5),  # Approximate with sqrt(n) for log(n)
        "bellman_ford": n * m,
        "tsinghua": m * (n**0.33),  # log^(2/3) n approximation
    }

    # Determine algorithm type for complexity estimation
    algo_name = algo_info.get("name", "").lower()
    if "dijkstra" in algo_name:
        algo_type = "dijkstra"
    elif "bellman" in algo_name:
        algo_type = "bellman_ford"
    elif "tsinghua" in algo_name:
        algo_type = "tsinghua"
    else:
        algo_type = "unknown"

    theoretical_cost = theoretical_complexity.get(algo_type, 0)

    return {
        "algorithm_info": algo_info,
        "graph_analysis": graph_analysis,
        "theoretical_performance": {
            "complexity_type": algo_type,
            "estimated_cost": theoretical_cost,
            "expected_performance": (
                "good" if theoretical_cost < n * m * 0.5 else "moderate"
            ),
        },
        "recommendations": _generate_performance_recommendations(
            graph_analysis, algo_info
        ),
    }


def _generate_performance_recommendations(
    graph_analysis: dict, algo_info: dict
) -> list:
    """Generate performance recommendations based on graph and algorithm characteristics."""
    recommendations = []

    # Graph-based recommendations
    if graph_analysis["density"] > 0.7:
        recommendations.append("High density graph - consider sparse graph algorithms")
    elif graph_analysis["density"] < 0.1:
        recommendations.append(
            "Low density graph - good for algorithms that exploit sparsity"
        )

    if graph_analysis["has_negative_weights"]:
        recommendations.append(
            "Graph has negative weights - use Bellman-Ford or similar"
        )

    if graph_analysis["avg_degree"] > 10:
        recommendations.append(
            "High average degree - may benefit from optimized neighbor iteration"
        )

    # Algorithm-based recommendations
    if "dijkstra" in algo_info.get("name", "").lower():
        if graph_analysis["has_negative_weights"]:
            recommendations.append(
                "Dijkstra cannot handle negative weights - use Bellman-Ford"
            )

    if "tsinghua" in algo_info.get("name", "").lower():
        if graph_analysis["density"] > 0.5:
            recommendations.append(
                "Tsinghua algorithm may not show benefits on very dense graphs"
            )

    return recommendations
