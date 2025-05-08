# Required imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import os
from Map import Map
from Boundaries import Boundaries
from SearchEngine import build_graph, path_finding, compute_path_cost, h1, h2, discretize_coords

def plot_radar_locations(boundaries: Boundaries, radar_locations: np.array) -> None:
    """ Auxiliary function for plotting the radar locations """
    plt.figure(figsize=(8, 8))
    plt.title("Radar locations in the map")
    plt.plot([boundaries.min_lon, boundaries.max_lon, boundaries.max_lon, boundaries.min_lon, boundaries.min_lon],
             [boundaries.max_lat, boundaries.max_lat, boundaries.min_lat, boundaries.min_lat, boundaries.max_lat],
             label='Boundaries',
             linestyle='--',
             c='black')
    plt.scatter(radar_locations[:, 1], radar_locations[:, 0], label='Radars', c='green')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.show()
    return

def plot_detection_fields(detection_map: np.array, bicubic: bool=True) -> None:
    """ Auxiliary function for plotting the detection fields """
    plt.figure(figsize=(8, 8))
    plt.title("Radar detection fields")
    im = plt.imshow(X=detection_map, cmap='Greens', interpolation='bicubic' if bicubic else None)
    plt.colorbar(im, label='Detection values')
    plt.show()
    return

def plot_solution(detection_map: np.array, solution_plan: list, bicubic: bool=True) -> None:
    """ Auxiliary function for plotting the solution plan with markers in each POI """
    # Depurar el contenido de solution_plan
    print("Contenido de solution_plan:", solution_plan)
    plt.figure(figsize=(8,8))
    plt.title("Solution plan")
    for i in range(len(solution_plan)):
        # Procesar los puntos de inicio directamente si son tuplas o listas
        if isinstance(solution_plan[i], (tuple, list)) and len(solution_plan[i]) == 2:
            start_point = tuple(solution_plan[i])
        else:
            print(f"Error: Formato inesperado en solution_plan[{i}]: {solution_plan[i]}")
            return
        plt.scatter(start_point[1], start_point[0], c='black', marker='*', zorder=2)
        path_array = np.zeros(shape=(len(solution_plan[i]), 2))
        for j in range(len(path_array)):
            # Procesar directamente los puntos del camino si son números o tuplas
            if isinstance(solution_plan[i][j], (tuple, list, np.ndarray)):
                path_array[j] = solution_plan[i][j]
            elif isinstance(solution_plan[i][j], (int, float, np.integer, np.floating)):
                path_array[j] = (solution_plan[i][j], solution_plan[i][j])  # Ajustar según el formato esperado
            else:
                print(f"Error: Formato inesperado en solution_plan[{i}][{j}]: {solution_plan[i][j]}")
                return
        plt.plot(path_array[:, 1], path_array[:, 0], zorder=1)
    # Procesar el punto final directamente si es un número o una tupla
    if isinstance(solution_plan[-1][-1], (tuple, list, np.ndarray)):
        final_point = tuple(solution_plan[-1][-1])
    elif isinstance(solution_plan[-1][-1], (int, float, np.integer, np.floating)):
        final_point = (solution_plan[-1][-1], solution_plan[-1][-1])  # Ajustar según el formato esperado
    else:
        print(f"Error: Formato inesperado en solution_plan[-1][-1]: {solution_plan[-1][-1]}")
        return
    plt.scatter(final_point[1], final_point[0], c='black', marker='*', label=f'Waypoints', zorder=2)
    im = plt.imshow(X=detection_map, cmap='Greens', interpolation='bicubic' if bicubic else None)
    plt.colorbar(im, label='Detection values')
    plt.legend()
    plt.show()
    return

def parse_args() -> dict:
    """ Parses the main arguments of the program and returns them stored in a dictionary """
    json_path            = f"{os.getcwd()}/scenarios.json"
    scenario_json        = sys.argv[1]
    tolerance            = float(sys.argv[2])
    execution_parameters = {}
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            key = list(entry.keys())[0]
            if key == scenario_json:
                execution_parameters = entry[key]
                break
    execution_parameters["tolerance"] = tolerance
    return execution_parameters

# System's main function
def main() -> None:

    # Parse the input parameters (arguments) of the program (current execution)
    execution_parameters = parse_args()

    # Set the pseudo-random number generator seed (DO NOT MODIFY)
    np.random.seed(42)

    # Set boundaries
    boundaries = Boundaries(max_lat=execution_parameters['max_lat'],
                            min_lat=execution_parameters['min_lat'],
                            max_lon=execution_parameters['max_lon'],
                            min_lon=execution_parameters['min_lon'])
    
    # Define the map with its corresponding boundaries and coordinates
    M = Map(boundaries=boundaries,
            height=execution_parameters['H'],
            width=execution_parameters['W'])
    
    # Generate random radars
    n_radars = execution_parameters['n_radars']
    M.generate_radars(n_radars=n_radars)
    radar_locations = M.get_radars_locations_numpy()

    # Plot the radar locations (latitude increments from bottom to top)
    plot_radar_locations(boundaries=boundaries, radar_locations=radar_locations)

    # Compute the detection map (sets the costs for each cell)
    detection_map = M.compute_detection_map()

    # Plot the detection map (detection fields)
    plot_detection_fields(detection_map=detection_map)

    # Build the graph from the detection map
    G = build_graph(detection_map=detection_map, tolerance=execution_parameters['tolerance'])

    # Get the POI's that the plane must visit
    POIs = np.array(execution_parameters['POIs'], dtype=np.float32)

    # Validate if POIs are within map boundaries
    for poi in POIs:
        if not (boundaries.min_lat <= poi[0] <= boundaries.max_lat and boundaries.min_lon <= poi[1] <= boundaries.max_lon):
            print(f"Error: POI {poi} está fuera de los límites del mapa.")
            return

    # Discretize the POIs
    POIs_discretized = discretize_coords(POIs, boundaries, M.width, M.height)

    # Debug detection values
    for poi in POIs_discretized:
        i, j = poi
        if detection_map[i, j] > execution_parameters['tolerance']:
            print(f"Advertencia: POI {poi} tiene un valor de detección {detection_map[i, j]} que excede la tolerancia.")

    # Ajustar para manejar ambas definiciones de path_finding
    result = path_finding(G=G,
                          heuristic_function=h2,
                          locations=POIs_discretized, 
                          initial_location_index=0,
                          boundaries=boundaries,
                          map_width=M.width,
                          map_height=M.height)

    # Verificar si la función devuelve uno o dos valores
    if isinstance(result, tuple) and len(result) == 2:
        solution_plan, nodes_expanded = result
    else:
        solution_plan = result
        nodes_expanded = None  # No se devuelve el número de nodos expandido en esta versión

    # Asegurar que solution_plan sea una lista de tuplas
    solution_plan = [tuple(node) for node in solution_plan]

    # Validar el formato de solution_plan
    for node in solution_plan:
        if not isinstance(node, tuple) or len(node) != 2:
            print(f"Error: Nodo con formato inesperado en solution_plan: {node}")
            return

    # Calcular el costo del camino si no se devolvió
    if nodes_expanded is None:
        path_cost = compute_path_cost(G=G, solution_plan=solution_plan)
    else:
        path_cost = nodes_expanded

    # Some verbose of the total cost and the number of expanded nodes
    print(f"Total path cost: {path_cost}")
    print(f"Number of expanded nodes: {nodes_expanded}")

    # Plot the solution
    plot_solution(detection_map=detection_map, solution_plan=solution_plan)

if __name__ == '__main__':
    main()
