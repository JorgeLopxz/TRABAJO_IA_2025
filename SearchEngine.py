# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0


def h1(current_node, objective_node) -> np.float32:
    """ Manhattan distance multiplied by minimum cost """
    global NODES_EXPANDED
    NODES_EXPANDED += 1
    return EPSILON * (abs(current_node[0] - objective_node[0]) + abs(current_node[1] - objective_node[1]))

def h2(current_node, objective_node) -> np.float32:
    """ Euclidean distance multiplied by minimum cost """
    global NODES_EXPANDED
    NODES_EXPANDED += 1
    return EPSILON * np.sqrt((current_node[0] - objective_node[0])**2 + (current_node[1] - objective_node[1])**2)


def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """ Builds an adjacency graph (not an adjacency matrix) from the detection map """
    height, width = detection_map.shape
    G = nx.DiGraph()

    for i in range(height):
        for j in range(width):
            if detection_map[i, j] > tolerance:
                continue  # No se puede transitar por esta celda

            current_node = (i, j)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # arriba, abajo, izquierda, derecha
                ni, nj = i + dx, j + dy

                if 0 <= ni < height and 0 <= nj < width:
                    if detection_map[ni, nj] <= tolerance:
                        neighbor_node = (ni, nj)
                        cost = detection_map[ni, nj]
                        G.add_edge(current_node, neighbor_node, weight=cost)

    return G



def discretize_coords(high_level_plan: np.array, boundaries: Boundaries, map_width: np.int32, map_height: np.int32) -> np.array:
    """ Converts coordinates from (lat, lon) into (x, y) indices in the grid """
    lat0, lat1 = boundaries.min_lat, boundaries.max_lat
    lon0, lon1 = boundaries.min_lon, boundaries.max_lon

    discretized = []
    for lat, lon in high_level_plan:
        i = int((lat - lat0) / (lat1 - lat0) * (map_height - 1))
        j = int((lon - lon0) / (lon1 - lon0) * (map_width - 1))
        discretized.append((i, j))

    return np.array(discretized)

def path_finding(G: nx.DiGraph,
                 heuristic_function,
                 locations: np.array, 
                 initial_location_index: np.int32, 
                 boundaries: Boundaries,
                 map_width: np.int32,
                 map_height: np.int32) -> tuple:
    """ Unified implementation of the main searching / path finding algorithm """
    plan = []
    total_cost = 0.0

    for i in range(initial_location_index, len(locations) - 1):
        start = tuple(locations[i])
        goal = tuple(locations[i + 1])

        subpath = nx.astar_path(G, start, goal, heuristic=lambda u, v: heuristic_function(u, v), weight='weight')
        plan.extend(subpath if i == initial_location_index else subpath[1:])  # Avoid repeating nodes
        total_cost += compute_path_cost(G, subpath)

    return plan, total_cost

def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """
    total_cost = 0.0
    for i in range(len(solution_plan) - 1):
        u, v = solution_plan[i], solution_plan[i + 1]
        total_cost += G[u][v]['weight']
    return total_cost
