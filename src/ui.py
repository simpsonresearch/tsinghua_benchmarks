"""
User Interface for the Shortest Path Algorithm Benchmark
Provides a console-based interface for running and comparing algorithms.
"""

import time
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from algos.graph import Graph
from algos.dijkstra import DijkstraAlgorithm
from algos.bellman_ford import BellmanFordAlgorithm
from algos.tsinghua import TsinghuaAlgorithm


class BenchmarkUI:
    """Console-based UI for benchmarking shortest path algorithms."""

    def __init__(self):
        self.algorithms = {
            "dijkstra": DijkstraAlgorithm(),
            "bellman_ford": BellmanFordAlgorithm(),
            "tsinghua": TsinghuaAlgorithm(),
        }
        self.current_graph = None

    def display_menu(self):
        """Display the main menu options."""
        print("\n" + "=" * 60)
        print("    SHORTEST PATH ALGORITHM BENCHMARK")
        print("=" * 60)
        print("1. Generate Random Graph")
        print("2. Load Graph from File")
        print("3. Display Current Graph Info")
        print("4. Run Single Algorithm")
        print("5. Compare All Algorithms")
        print("6. Performance Analysis")
        print("7. Visualize Graph (if small)")
        print("8. Exit")
        print("=" * 60)

    def run(self):
        """Main application loop."""
        print("Welcome to the Shortest Path Algorithm Benchmark!")

        while True:
            self.display_menu()
            choice = input("\nEnter your choice (1-8): ").strip()

            try:
                if choice == "1":
                    self.generate_random_graph()
                elif choice == "2":
                    self.load_graph_from_file()
                elif choice == "3":
                    self.display_graph_info()
                elif choice == "4":
                    self.run_single_algorithm()
                elif choice == "5":
                    self.compare_algorithms()
                elif choice == "6":
                    self.performance_analysis()
                elif choice == "7":
                    self.visualize_graph()
                elif choice == "8":
                    print("Thank you for using the benchmark tool!")
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 8.")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                input("Press Enter to continue...")

    def generate_random_graph(self):
        """Generate a random graph for testing."""
        print("\n--- Generate Random Graph ---")

        try:
            num_nodes = int(input("Enter number of nodes (10-1000): "))
            if num_nodes < 10 or num_nodes > 1000:
                print("Number of nodes must be between 10 and 1000.")
                return

            density = float(input("Enter edge density (0.1-0.9): "))
            if density < 0.1 or density > 0.9:
                print("Density must be between 0.1 and 0.9.")
                return

            directed = input("Directed graph? (y/n): ").lower().startswith("y")
            allow_negative = (
                input("Allow negative weights? (y/n): ").lower().startswith("y")
            )

            self.current_graph = Graph.generate_random(
                num_nodes=num_nodes,
                density=density,
                directed=directed,
                allow_negative_weights=allow_negative,
            )

            print(
                f"✓ Generated graph with {num_nodes} nodes and {len(self.current_graph.edges)} edges."
            )

        except ValueError:
            print("Invalid input. Please enter valid numbers.")

    def load_graph_from_file(self):
        """Load graph from a file."""
        print("\n--- Load Graph from File ---")
        filename = input("Enter filename: ").strip()

        try:
            self.current_graph = Graph.from_file(filename)
            print(f"✓ Successfully loaded graph from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except Exception as e:
            print(f"Error loading file: {e}")

    def display_graph_info(self):
        """Display information about the current graph."""
        if not self.current_graph:
            print("No graph loaded. Please generate or load a graph first.")
            return

        print("\n--- Current Graph Information ---")
        print(f"Nodes: {self.current_graph.num_nodes}")
        print(f"Edges: {len(self.current_graph.edges)}")
        print(f"Directed: {self.current_graph.directed}")
        print(f"Has negative weights: {self.current_graph.has_negative_weights()}")
        print(f"Density: {self.current_graph.density():.3f}")

        if self.current_graph.num_nodes <= 20:
            print("\nAdjacency representation:")
            self.current_graph.display()

    def run_single_algorithm(self):
        """Run a single algorithm on the current graph."""
        if not self.current_graph:
            print("No graph loaded. Please generate or load a graph first.")
            return

        print("\n--- Run Single Algorithm ---")
        print("Available algorithms:")
        for i, name in enumerate(self.algorithms.keys(), 1):
            print(f"{i}. {name.replace('_', ' ').title()}")

        try:
            choice = int(input("Select algorithm (number): ")) - 1
            algo_name = list(self.algorithms.keys())[choice]
            algorithm = self.algorithms[algo_name]

            source = int(
                input(f"Enter source node (0-{self.current_graph.num_nodes-1}): ")
            )
            if source < 0 or source >= self.current_graph.num_nodes:
                print("Invalid source node.")
                return

            print(f"\nRunning {algo_name.replace('_', ' ').title()}...")
            start_time = time.time()
            distances, predecessors = algorithm.shortest_paths(
                self.current_graph, source
            )
            end_time = time.time()

            print(f"✓ Completed in {(end_time - start_time)*1000:.2f} ms")

            # Display results for small graphs
            if self.current_graph.num_nodes <= 20:
                print("\nShortest distances from source:")
                for node in range(self.current_graph.num_nodes):
                    dist = distances[node]
                    if dist == float("inf"):
                        print(f"Node {node}: unreachable")
                    else:
                        print(f"Node {node}: {dist}")

        except (ValueError, IndexError):
            print("Invalid selection.")
        except Exception as e:
            print(f"Error running algorithm: {e}")

    def compare_algorithms(self):
        """Compare all algorithms on the current graph."""
        if not self.current_graph:
            print("No graph loaded. Please generate or load a graph first.")
            return

        source = 0  # Use node 0 as source
        print(f"\n--- Comparing All Algorithms (source: {source}) ---")

        results = {}

        for name, algorithm in self.algorithms.items():
            # Skip Bellman-Ford for graphs without negative weights if user prefers
            if name == "bellman_ford" and not self.current_graph.has_negative_weights():
                response = input(
                    f"Run Bellman-Ford on graph without negative weights? (y/n): "
                )
                if not response.lower().startswith("y"):
                    continue

            try:
                print(f"Running {name.replace('_', ' ').title()}...", end=" ")
                start_time = time.time()
                distances, predecessors = algorithm.shortest_paths(
                    self.current_graph, source
                )
                end_time = time.time()

                runtime = (end_time - start_time) * 1000  # Convert to ms
                results[name] = {
                    "runtime": runtime,
                    "distances": distances,
                    "success": True,
                }
                print(f"✓ {runtime:.2f} ms")

            except Exception as e:
                results[name] = {"success": False, "error": str(e)}
                print(f"✗ Error: {e}")

        # Display comparison
        print("\n--- Results Summary ---")
        successful_results = {k: v for k, v in results.items() if v["success"]}

        if len(successful_results) > 1:
            # Sort by runtime
            sorted_results = sorted(
                successful_results.items(), key=lambda x: x[1]["runtime"]
            )

            print(f"{'Algorithm':<15} {'Runtime (ms)':<12} {'Relative':<10}")
            print("-" * 40)

            fastest_time = sorted_results[0][1]["runtime"]
            for name, data in sorted_results:
                relative = data["runtime"] / fastest_time
                print(
                    f"{name.replace('_', ' ').title():<15} {data['runtime']:<12.2f} {relative:<10.2f}x"
                )

            # Verify results are consistent
            if self._verify_results_consistency(successful_results):
                print("\n✓ All algorithms produced consistent results.")
            else:
                print("\n⚠ Warning: Algorithms produced different results!")

    def performance_analysis(self):
        """Run performance analysis across different graph sizes."""
        print("\n--- Performance Analysis ---")
        print("This will test algorithms on graphs of increasing size.")

        sizes = [50, 100, 200, 500]
        density = 0.3
        directed = True

        results = {name: [] for name in self.algorithms.keys()}

        for size in sizes:
            print(f"\nTesting with {size} nodes...")
            test_graph = Graph.generate_random(size, density, directed, False)

            for name, algorithm in self.algorithms.items():
                try:
                    start_time = time.time()
                    algorithm.shortest_paths(test_graph, 0)
                    end_time = time.time()

                    runtime = (end_time - start_time) * 1000
                    results[name].append(runtime)
                    print(f"  {name}: {runtime:.2f} ms")

                except Exception as e:
                    results[name].append(None)
                    print(f"  {name}: Error - {e}")

        # Plot results if matplotlib is available
        try:
            self._plot_performance_results(sizes, results)
        except ImportError:
            print("Matplotlib not available for plotting.")

    def visualize_graph(self):
        """Visualize the current graph if it's small enough."""
        if not self.current_graph:
            print("No graph loaded.")
            return

        if self.current_graph.num_nodes > 50:
            print("Graph too large for visualization (>50 nodes).")
            return

        try:
            self.current_graph.visualize()
        except ImportError:
            print("Matplotlib/NetworkX not available for visualization.")

    def _verify_results_consistency(self, results: Dict) -> bool:
        """Verify that all algorithms produced the same shortest distances."""
        if len(results) < 2:
            return True

        distances_lists = [data["distances"] for data in results.values()]
        reference = distances_lists[0]

        for distances in distances_lists[1:]:
            for i in range(len(reference)):
                if (
                    abs(reference[i] - distances[i]) > 1e-9
                ):  # Allow small floating point differences
                    return False
        return True

    def _plot_performance_results(self, sizes: List[int], results: Dict):
        """Plot performance analysis results."""
        plt.figure(figsize=(10, 6))

        for name, times in results.items():
            # Filter out None values
            valid_data = [(s, t) for s, t in zip(sizes, times) if t is not None]
            if valid_data:
                valid_sizes, valid_times = zip(*valid_data)
                plt.plot(
                    valid_sizes,
                    valid_times,
                    marker="o",
                    label=name.replace("_", " ").title(),
                )

        plt.xlabel("Graph Size (nodes)")
        plt.ylabel("Runtime (ms)")
        plt.title("Shortest Path Algorithm Performance Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.show()
