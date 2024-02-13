import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def generate_random_dag(n, prob):
    """
    Generates a random directed acyclic graph (DAG) with n nodes and a specified probability of having edges,
    ensuring acyclicity by adding edges in a controlled manner.

    Parameters:
    n (int): Number of nodes in the DAG.
    prob (float): Probability of having an edge between any two nodes.

    Returns:
    np.ndarray: Adjacency matrix of the generated DAG.
    """
    G = nx.DiGraph()
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < prob:
                G.add_edge(i, j)
                # Check if adding the edge creates a cycle and remove it if true
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(i, j)
    
    # Convert to adjacency matrix
    adj_matrix = nx.to_numpy_array(G, dtype=int)
    return adj_matrix

def get_parallel_exec_sequence(adj_matrix):
    """
    Calculates levels of parallelism in a DAG and ensures the graph is fully connected.

    Parameters:
    adj_matrix (np.ndarray): Adjacency matrix of the DAG.

    Returns:
    list: Levels of parallelism in the DAG.

    Raises:
    ValueError: If the graph is not fully connected or contains dangling nodes.
    """
    adj_matrix = np.array(adj_matrix)  # Ensure adj_matrix is a NumPy array
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Check if the graph is fully connected (ignoring edge direction)
    if not nx.is_weakly_connected(G):
        raise ValueError("Graph is not fully connected.")

    # Ensure the graph is a DAG
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("A Cyclic graph detected.")

    in_degrees = dict(G.in_degree())
    zero_in_degree = [node for node, degree in in_degrees.items() if degree == 0]

    topo_order = []
    while zero_in_degree:
        current_level = []
        next_level = []
        for node in zero_in_degree:
            current_level.append(node)
            for _, successor in G.edges(node):
                in_degrees[successor] -= 1
                if in_degrees[successor] == 0:
                    next_level.append(successor)

        topo_order.append(current_level)
        zero_in_degree = next_level

    return topo_order


def visualize_dag(adj):
    """
    Visualizes a DAG using NetworkX and Matplotlib with an improved layout.

    Parameters:
    adj (np.ndarray): Adjacency matrix of the DAG.
    """
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    pos = nx.layout.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", 
            edge_color='k', linewidths=1, font_size=10, arrows=True)
    plt.title("Directed Acyclic Graph (DAG)")
    plt.show()

# if __name__ == "__main__":
#     n = 7
#     prob = 0.2  # Adjusted probability for clearer demonstration
#     adj_matrix = generate_random_dag(n, prob)
#     visualize_dag(adj_matrix)
#     parallel_seq = get_parallel_exec_sequence(adj_matrix)
#     print(parallel_seq)

def test_correct_graph():
    print("Testing Correct Graph...")
    adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0, 0],  # Node 0 -> Nodes 1, 2
        [0, 0, 0, 1, 0, 0, 0],  # Node 1 -> Node 3
        [0, 0, 0, 1, 0, 0, 0],  # Node 2 -> Node 3
        [0, 0, 0, 0, 1, 0, 0],  # Node 3 -> Node 4
        [0, 0, 0, 0, 0, 1, 0],  # Node 4 -> Node 5
        [0, 0, 0, 0, 0, 0, 1],  # Node 5 -> Node 6
        [0, 0, 0, 0, 0, 0, 0],  # Node 6 has no outgoing edges
    ], dtype=int)
    visualize_dag(adj_matrix)  # Optional
    try:
        parallel_seq = get_parallel_exec_sequence(adj_matrix)
        print("Parallel Execution Sequence:", parallel_seq)
    except ValueError as e:
        print(e)

def test_disconnected_graph():
    print("\nTesting Disconnected Graph...")
    # Disconnected: Node 2 and Node 3 are disconnected from the rest
    adj_matrix = np.array([
        [0, 1, 0, 0, 0, 0, 0],  # Node 0 -> Node 1
        [0, 0, 0, 0, 0, 0, 0],  # Node 1 has no outgoing edges
        [0, 0, 0, 0, 0, 0, 0],  # Node 2 is disconnected
        [0, 0, 0, 0, 0, 0, 0],  # Node 3 is disconnected
        [0, 0, 0, 0, 0, 1, 0],  # Node 4 -> Node 5
        [0, 0, 0, 0, 0, 0, 1],  # Node 5 -> Node 6
        [0, 0, 0, 0, 0, 0, 0]   # Node 6 has no outgoing edges
    ], dtype=int)
    try:
        parallel_seq = get_parallel_exec_sequence(adj_matrix)
        print("Parallel Execution Sequence:", parallel_seq)
    except ValueError as e:
        print(e)

def test_graph_with_dangling_node():
    print("\nTesting Graph with Dangling Node...")
    # Dangling: Node 2 has no connections
    adj_matrix = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],  # Dangling node
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)
    try:
        parallel_seq = get_parallel_exec_sequence(adj_matrix)
        print("Parallel Execution Sequence:", parallel_seq)
    except ValueError as e:
        print(e)

def test_cyclic_graph():
    print("\nTesting Cyclic Graph...")
    adj_matrix = np.array([
        [0, 1, 0, 0, 0, 0, 0],  # Node 0 -> Node 1
        [0, 0, 1, 0, 0, 0, 0],  # Node 1 -> Node 2
        [0, 0, 0, 1, 0, 0, 0],  # Node 2 -> Node 3
        [0, 0, 0, 0, 1, 0, 0],  # Node 3 -> Node 4
        [0, 0, 0, 0, 0, 1, 0],  # Node 4 -> Node 5
        [0, 0, 0, 0, 0, 0, 1],  # Node 5 -> Node 6
        [1, 0, 0, 0, 0, 0, 0],  # Node 6 -> Node 0 (Creates a cycle)
    ], dtype=int)
    try:
        parallel_seq = get_parallel_exec_sequence(adj_matrix)
        print("Parallel Execution Sequence:", parallel_seq)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    test_correct_graph()
    test_disconnected_graph()
    test_graph_with_dangling_node()
    test_cyclic_graph()