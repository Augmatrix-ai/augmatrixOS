import os
import sys
import json
from .topological_sort import get_parallel_exec_sequence
from collections import deque
import copy

class QueueSet:
    def __init__(self):
        self.queue = []
        self.set = set()

    def add(self, item):
        """Add item to the queue if not already present."""
        if item not in self.set:
            self.queue.append(item)
            self.set.add(item)

    def pop(self):
        """Pop an item from the queue."""
        if self.queue:
            item = self.queue.pop(0)
            self.set.remove(item)
            return item
        return None

    def __len__(self):
        """Return the current length of the queue."""
        return len(self.queue)

    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.queue) == 0


def index_to_exec_mapping(index_to_task, sequence):
    new_sequence = []
    for par_seq in sequence:
        new_par_seq = []
        for _id in par_seq:
            new_par_seq.append(index_to_task[_id])
        new_sequence.append(new_par_seq)
    return new_sequence


def count_disconnected_components(graph):
    """
    Given a graph, counts the number of disconnected components in the graph.
    Assumes that nodes without children are the completion of a flow (may not be true for a loop).

    Args:
    - graph (dict): dictionary representing the graph

    Returns:
    - count (int): number of disconnected components in the graph
    """
    return sum(1 for children in graph.values() if len(children) == 0)


def create_adjacency_matrix(graph):
    """
    Given a graph represented as a dictionary where keys are nodes and values are lists
    of nodes that the key node points to, returns the corresponding adjacency matrix.

    Args:
    - graph (dict): a dictionary representing the graph

    Returns:
    - adj (list of lists): the adjacency matrix
    - index_to_node (dict): a dictionary mapping indices in the adjacency matrix to node names

    """
    nodes = list(graph.keys())
    n = len(nodes)
    adj = [[0] * n for _ in range(n)]
    index_to_node = {i: nodes[i] for i in range(n)}

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            adj[i][j] = 1 if node_j in graph[node_i] else 0

    return adj, index_to_node


def convert_flow_to_dependency_graph(flow):
    """
    Converts a flow dictionary into a dependency graph dictionary.

    Args:
    - flow (dict): a dictionary containing information about all blocks in the flow

    Returns:
    - graph (dict): a dictionary representing the dependency graph of the flow,
    where the keys are block ids and the values are lists of block ids that depend on them
    """

    graph = {}
    for block_id, block_info in flow["blocks"].items():
        children_block_ids = {
        connection["output_to_block_id"]
            for var_info in block_info["childrens"].values()
                for connection in var_info["connections"]
        }
        graph[block_id] = list(children_block_ids)
    return graph


def get_constant_flow(flow: dict, const_block_id: str) -> dict:
    """
    Given a flow and a constant block id, returns a sub-flow containing all blocks that
    are constant inputs to the given block id, and all blocks that they depend on.

    Args:
        flow (dict): A dictionary containing information about flow.
        const_block_id (str): The block id of the constant block to extract a sub-flow from.

    Returns:
        A dictionary containing information about all blocks in the sub-flow.

    """

    flow_blocks = flow["blocks"]

    # Create a queue to hold block ids to visit
    to_visit_q = QueueSet()

    # Add the constant block id to the queue
    to_visit_q.add(const_block_id)

    # Keep track of visited block ids
    visited = []

    # Visit all blocks reachable from the constant block id
    while to_visit_q:
        # Get the next block to visit from the queue
        visit_block_id = to_visit_q.pop()

        # Add the visited block id to the visited list
        visited.append(visit_block_id)

        # Loop through all parent variables of the visited block
        for var_id, var_info in flow_blocks[visit_block_id]["parents"].items():
            # Loop through all connections of the parent variable
            for connection in var_info["connections"]:
                # Add the input block id to the queue if it hasn't been visited yet
                if connection["input_from_block_id"] not in visited:
                    to_visit_q.add(connection["input_from_block_id"])

    # Create a new flow with only the visited blocks
    new_flow = {"blocks": {}}
    for block_id in visited:
        new_flow["blocks"][block_id] = flow_blocks[block_id]

    return new_flow

def compile_flow(flow):
    """
    Compiles the given flow by extracting constants, building parallel execution sequences for constant and non-constant flows.

    Args:
        flow (dict): The flow to be compiled, containing blocks and their dependencies.

    Returns:
        dict: The compiled flow with constant and non-constant parallel execution sequences.
    """
    flow_copy = copy.deepcopy(flow)
    constant_flow_blocks = extract_constants_from_flow(flow_copy)
    flow_copy["constant_flow_blocks"] = constant_flow_blocks
    compile_constant_flows(flow_copy)
    compile_non_constant_flows(flow_copy)
    return flow_copy

def extract_constants_from_flow(flow):
    """
    Extracts constant blocks and their dependencies from the flow.

    Args:
        flow (dict): The original workflow data.

    Returns:
        dict: A dictionary of constant flow blocks.
    """
    constant_flow_blocks = {}
    to_remove = set()
    for block_id, block_info in flow["blocks"].items():
        if block_info.get("constant_type") == "MIXED":
            process_mixed_constant_blocks(flow, block_id, constant_flow_blocks, to_remove)
    
    # Now, remove the blocks that have been processed
    for block_id in to_remove:
        del flow["blocks"][block_id]
        
    return constant_flow_blocks

def process_mixed_constant_blocks(flow, mixed_block_id, constant_flow_blocks, to_remove):
    """
    Processes mixed constant blocks to extract constants and their dependencies.

    Args:
        flow (dict): The original workflow data.
        mixed_block_id (str): The ID of the mixed constant block.
        constant_flow_blocks (dict): Accumulator for constant flow blocks.
        to_remove (set): Set of block IDs to be removed after processing.
    """
    input_constant_block_map = extract_constant_inputs(flow["blocks"][mixed_block_id])
    for constant_block_id, value_lst in input_constant_block_map.items():
        constant_flow = get_constant_flow(flow, constant_block_id)
        constant_flow_blocks[constant_block_id] = constant_flow
        # Instead of deleting here, add the blocks to the set
        to_remove.update(constant_flow["blocks"].keys())

def extract_constant_inputs(block_info):
    """
    Extracts constant inputs from a mixed constant block.

    Args:
        block_info (dict): Information about the mixed constant block.

    Returns:
        dict: A mapping from constant block IDs to their connection information.
    """
    input_constant_block_map = {}
    for var_id, var_info in block_info["parents"].items():
        for idx, connection in enumerate(var_info["connections"]):
            if connection["is_constant"]:
                input_block_id = connection["input_from_block_id"]
                input_constant_block_map.setdefault(input_block_id, []).append(
                    (block_info["id"], var_id, idx)
                )
    return input_constant_block_map

def compile_constant_flows(flow):
    """
    Compiles the constant flows into their parallel execution sequences.

    Args:
        flow (dict): The workflow data with constant flows extracted.
    """
    for constant_block_id, constant_flow in flow["constant_flow_blocks"].items():
        graph = convert_flow_to_dependency_graph(constant_flow)
        adjacency_matrix, index_to_task = create_adjacency_matrix(graph)
        parallel_exec_seq = get_parallel_exec_sequence(adjacency_matrix)
        # constant_flow["parallel_exec_sequence"] = parallel_exec_seq
        constant_flow["parallel_exec_sequence"] = index_to_exec_mapping(index_to_task, parallel_exec_seq)

def compile_non_constant_flows(flow):
    """
    Compiles the non-constant flows into their parallel execution sequences.

    Args:
        flow (dict): The workflow data with non-constant flows.
    """
    graph = convert_flow_to_dependency_graph(flow)
    adjacency_matrix, index_to_task = create_adjacency_matrix(graph)
    print(index_to_task)
    parallel_exec_seq = get_parallel_exec_sequence(adjacency_matrix)
    # flow["parallel_exec_sequence"] = parallel_exec_seq
    flow["parallel_exec_sequence"] = index_to_exec_mapping(index_to_task, parallel_exec_seq)


if __name__ == "__main__":
    flow_json = {}
    with open("../../sample_flow/aadnar_n_pan.json", "r") as fr:
        flow_json = json.load(fr)

    compiled_flow = compile_flow(flow_json)
    print(json.dumps(compiled_flow))

    with open("../../sample_flow/compiled_aadnar_n_pan.json", "w") as fw:
        fw.write(json.dumps(compiled_flow))
