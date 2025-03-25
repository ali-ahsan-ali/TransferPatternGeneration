import heapq
from collections import namedtuple
import networkx as nx
import numpy as np

# A Label represents a path from the source to a node
# node: the node this label belongs to
# cost: a d-dimensional vector of costs
# pred: pointer to the predecessor label (for path reconstruction)
Label = namedtuple("Label", ["node", "cost", "pred"])


class MultiobjectiveDijkstra:
    def __init__(self, graph, source, target=None, lower_bounds=None, upper_bound=None):
        """
        Initialize the Multiobjective Dijkstra Algorithm using NetworkX DiGraph.

        Parameters:
        - graph: NetworkX DiGraph with 'cost' attribute for each edge representing d-dimensional cost vector
        - source: Source node
        - target: Target node (for one-to-one version)
        - lower_bounds: Dictionary with nodes as keys and lower bound vectors as values (for one-to-one)
        - upper_bound: Upper bound label at target (for one-to-one)
        """
        self.graph = graph
        self.source = source
        self.target = target
        self.lower_bounds = lower_bounds if lower_bounds else {}
        self.upper_bound = upper_bound

        # Get the dimension of the cost vectors from the first edge
        first_edge = next(iter(graph.edges(data=True)))
        self.d = len(first_edge[2]["cost"])

        # Initialize data structures
        self.H = []  # Priority queue
        self.L = {}  # Efficient labels for all nodes
        self.lastProcessedLabel = {}  # Last processed label index for each arc

        # Initialize efficient labels for all nodes
        for v in graph.nodes():
            self.L[v] = []

        # Initialize lastProcessedLabel for all arcs
        for u, v in graph.edges():
            self.lastProcessedLabel[(u, v)] = 0

    def dominates(self, cost1, cost2):
        """Check if cost1 dominates cost2 (is better in all dimensions)."""
        return all(cost1[i] <= cost2[i] for i in range(len(cost1))) and any(
            cost1[i] < cost2[i] for i in range(len(cost1))
        )

    def is_dominated(self, cost, label_set):
        """Check if cost is dominated by any cost in label_set."""
        return any(self.dominates(label.cost, cost) for label in label_set)

    def insert_priority_queue(self, label):
        """Insert a label into the priority queue (ordered lexicographically)."""
        heapq.heappush(self.H, (label.cost, id(label), label))

    def extract_lexmin(self):
        """Extract the lexicographically smallest label from the priority queue."""
        if not self.H:
            return None
        _, _, label = heapq.heappop(self.H)
        return label

    def nextCandidateLabel(
        self, v, lastProcessedLabel, L, lower_bound=None, upper_bound=None
    ):
        """
        Find the next candidate label for node v by processing unprocessed labels from predecessors.

        Parameters:
        - v: Current node
        - lastProcessedLabel: Dictionary tracking last processed label index
        - L: Dictionary of efficient labels
        - lower_bound: Lower bound vector for node v (one-to-one version)
        - upper_bound: Upper bound label at target (one-to-one version)

        Returns:
        - The next candidate label for v, or None if none exists
        """
        best_label = None
        best_cost = None

        # For each incoming neighbor u
        for u in self.graph.predecessors(v):
            # Get the range of unprocessed labels (could be empty initially)
            start_idx = lastProcessedLabel.get((u, v), 0)
            end_idx = len(L[u])

            # Process all unprocessed labels from u
            for k in range(start_idx, end_idx):
                label_u = L[u][k]

                # Get the arc cost
                arc_cost = self.graph[u][v]["cost"]

                # Compute the new cost by adding the arc cost
                new_cost = tuple(label_u.cost[i] + arc_cost[i] for i in range(self.d))

                # Check if the new cost is dominated by any efficient label at v
                if self.is_dominated(new_cost, L[v]):
                    continue

                # For one-to-one version, check if new_cost + lower_bound dominates upper_bound
                if (
                    lower_bound is not None
                    and upper_bound is not None
                    and all(
                        new_cost[i] + lower_bound[i] >= upper_bound[i]
                        for i in range(self.d)
                    )
                ):
                    continue

                # Update best label if this is better
                if best_label is None or new_cost < best_cost:
                    best_cost = new_cost
                    best_label = Label(node=v, cost=new_cost, pred=label_u)

            # Update lastProcessedLabel
            lastProcessedLabel[(u, v)] = end_idx

        return best_label

    def propagate(
        self, label, w, priority_queue, L, lower_bound=None, upper_bound=None
    ):
        """
        Propagate a label to node w.

        Parameters:
        - label: The label to propagate
        - w: The target node for propagation
        - priority_queue: The priority queue
        - L: Dictionary of efficient labels
        - lower_bound: Lower bound vector for node w (one-to-one version)
        - upper_bound: Upper bound label at target (one-to-one version)

        Returns:
        - Updated priority queue
        """
        v = label.node

        # Get the arc cost
        arc_cost = self.graph[v][w]["cost"]

        # Compute the new cost
        new_cost = tuple(label.cost[i] + arc_cost[i] for i in range(self.d))

        # Check if the new label is dominated
        if self.is_dominated(new_cost, L[w]):
            return priority_queue

        # For one-to-one version, check if new_cost + lower_bound dominates upper_bound
        if (
            lower_bound is not None
            and upper_bound is not None
            and all(
                new_cost[i] + lower_bound[i] >= upper_bound[i] for i in range(self.d)
            )
        ):
            return priority_queue

        # Create new label
        new_label = Label(node=w, cost=new_cost, pred=label)

        # Insert into priority queue
        self.insert_priority_queue(new_label)

        return priority_queue

    def run(self):
        """Run the Multiobjective Dijkstra Algorithm."""
        # Initialize source label
        source_label = Label(
            node=self.source, cost=tuple(0 for _ in range(self.d)), pred=None
        )
        self.insert_priority_queue(source_label)

        # Main loop
        while self.H:
            # Extract lexicographically smallest label
            current_label = self.extract_lexmin()
            if current_label is None:
                break

            v = current_label.node

            # For one-to-one version, check if we've reached the target
            if self.target and v == self.target:
                # Update upper bound if needed
                if self.upper_bound is None or any(
                    current_label.cost[i] < self.upper_bound[i] for i in range(self.d)
                ):
                    self.upper_bound = current_label.cost

            # Add to efficient labels
            self.L[v].append(current_label)

            # Find next candidate label for v
            lower_bound = self.lower_bounds.get(v) if self.target else None
            next_label = self.nextCandidateLabel(
                v, self.lastProcessedLabel, self.L, lower_bound, self.upper_bound
            )

            if next_label:
                self.insert_priority_queue(next_label)

            # Propagate to outgoing neighbors
            for w in self.graph.successors(v):
                lower_bound = self.lower_bounds.get(w) if self.target else None
                self.H = self.propagate(
                    current_label, w, self.H, self.L, lower_bound, self.upper_bound
                )

        # Return efficient labels
        if self.target:
            return self.L[self.target]
        else:
            return self.L

    def reconstruct_path(self, label):
        """Reconstruct the path from a label."""
        path = []
        current = label
        while current:
            path.append(current.node)
            current = current.pred
        path.reverse()
        return path


# Example usage for one-to-one with lower bounds and upper bound
def one_to_one_example():
    # Create a directed graph using NetworkX
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(["s", "a", "b", "c", "d", "e", "t"])

    # Add edges with costs as attributes (2D: time, cost)
    G.add_edge("s", "a", cost=(2, 5))
    G.add_edge("s", "b", cost=(3, 2))
    G.add_edge("a", "c", cost=(3, 4))
    G.add_edge("b", "c", cost=(2, 6))
    G.add_edge("b", "d", cost=(1, 8))
    G.add_edge("c", "e", cost=(4, 3))
    G.add_edge("d", "e", cost=(5, 2))
    G.add_edge("c", "t", cost=(3, 7))
    G.add_edge("e", "t", cost=(2, 3))

    # Calculate lower bounds for one-to-one optimization
    # Here we use simple estimates: direct distance to target
    # In a real implementation, these could be more sophisticated heuristics
    lower_bounds = {
        "s": (8, 10),  # Optimistic estimate from s to t
        "a": (7, 7),  # Optimistic estimate from a to t
        "b": (6, 8),  # Optimistic estimate from b to t
        "c": (3, 7),  # Optimistic estimate from c to t
        "d": (7, 5),  # Optimistic estimate from d to t
        "e": (2, 3),  # Optimistic estimate from e to t
        "t": (0, 0),  # Target to itself is 0
    }

    # Initial upper bound at target (can be None)
    upper_bound = (15, 15)  # Initial conservative estimate

    # Run algorithm
    modjk = MultiobjectiveDijkstra(G, "s", "t", None, None)
    result = modjk.run()

    # Print results
    print("One-to-One Example Results:")
    print("Non-dominated paths from 's' to 't':")
    for i, label in enumerate(result):
        path = modjk.reconstruct_path(label)
        print(f"Path {i + 1}: {' -> '.join(path)}")
        print(f"Cost vector: {label.cost}")
        print()


if __name__ == "__main__":
    one_to_one_example()
