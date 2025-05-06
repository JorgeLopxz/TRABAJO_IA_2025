# Required imports
import numpy as np
from Location import Location
from Boundaries import Boundaries
from Radar import Radar
from tqdm import tqdm

# Constant that avoids setting cells to have an associated cost of zero
EPSILON = 1e-4

class Map:
    """ Class that models the map for the simulation """
    def __init__(self, 
                 boundaries: Boundaries,
                 height:     np.int32, 
                 width:      np.int32, 
                 radars:     np.array=None):
        self.boundaries = boundaries        # Boundaries of the map
        self.height     = height            # Number of coordinates in the y-axis
        self.width      = width             # Number of coordinates int the x-axis
        self.radars     = radars            # List containing the radars (objects)

    def generate_radars(self, n_radars: np.int32) -> None:
        """ Generates n-radars randomly and inserts them into the radars list """
        # Select random coordinates inside the boundaries of the map
        lat_range = np.linspace(start=self.boundaries.min_lat, stop=self.boundaries.max_lat, num=self.height)
        lon_range = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)
        rand_lats = np.random.choice(a=lat_range, size=n_radars, replace=False)
        rand_lons = np.random.choice(a=lon_range, size=n_radars, replace=False)
        self.radars = []        # Initialize 'radars' as an empty list

        # Loop for each radar that must be generated
        for i in range(n_radars):
            # Create a new radar
            new_radar = Radar(location=Location(latitude=rand_lats[i], longitude=rand_lons[i]),
                              transmission_power=np.random.uniform(low=1, high=1000000),
                              antenna_gain=np.random.uniform(low=10, high=50),
                              wavelength=np.random.uniform(low=0.001, high=10.0),
                              cross_section=np.random.uniform(low=0.1, high=10.0),
                              minimum_signal=np.random.uniform(low=1e-10, high=1e-15),
                              total_loss=np.random.randint(low=1, high=10),
                              covariance=None)

            # Insert the new radar
            self.radars.append(new_radar)
        return
    
    def get_radars_locations_numpy(self) -> np.array:
        """ Returns an array with the coordiantes (lat, lon) of each radar registered in the map """
        locations = np.zeros(shape=(len(self.radars), 2), dtype=np.float32)
        for i in range(len(self.radars)):
            locations[i] = self.radars[i].location.to_numpy()
        return locations
    
    
    def compute_detection_map(self) -> np.array:
        """ Computes the detection map for each coordinate in the map (with all the radars) """
        lat_range = np.linspace(self.boundaries.min_lat, self.boundaries.max_lat, self.height)
        lon_range = np.linspace(self.boundaries.min_lon, self.boundaries.max_lon, self.width)

        detection_map = np.zeros((self.height, self.width), dtype=np.float32)

        for i in tqdm(range(self.height), desc="Computing detection map"):
            for j in range(self.width):
                lat, lon = lat_range[i], lon_range[j]
                max_psi = 0.0

                for radar in self.radars:
                    R_max = radar.compute_max_range()
                    d = 111000 * np.sqrt((lat - radar.location.latitude)**2 + (lon - radar.location.longitude)**2)
                    if d > R_max:
                        continue

                    mu = np.array([radar.location.latitude, radar.location.longitude])
                    x = np.array([lat, lon])
                    sigma = radar.covariance
                    sigma_inv = np.linalg.inv(sigma)
                    sigma_det = np.linalg.det(sigma)

                    diff = (x - mu).reshape(1, -1)
                    exponent = -0.5 * (diff @ sigma_inv @ diff.T)[0][0]
                    psi = 1.0 / (2 * np.pi * np.sqrt(sigma_det)) * np.exp(exponent)

                    if psi > max_psi:
                        max_psi = psi

                detection_map[i, j] = max_psi

        min_val = np.min(detection_map)
        max_val = np.max(detection_map)
        scaled = ((detection_map - min_val) / (max_val - min_val)) * (1 - 1e-4) + 1e-4
        return scaled
