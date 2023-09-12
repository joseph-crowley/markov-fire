import numpy as np

from utils import calculate_fire_proximity, get_perimeter_cells

class Environment:
    def __init__(self, grid: np.ndarray):

        # initialize the grid
        self.grid = grid
        grid_shape = grid.shape

        # Grids for the environment
        self.topography_slope = np.zeros(grid_shape, dtype=float)  # Slope for topography
        self.topography_vegetation_density = np.zeros(grid_shape, dtype=float)  # Vegetation density for topography
        self.wind_vector = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=float)  # Wind vector (x, y)
        self.fuel_moisture = np.zeros(grid_shape, dtype=float)  # Moisture content in fuel
        self.fuel_type = np.empty(grid_shape, dtype=object)  # Type of fuel (e.g., 'grass', 'brush', 'timber')
        self.atmospheric_humidity = np.zeros(grid_shape, dtype=float)  # Atmospheric humidity
        self.atmospheric_temperature = np.zeros(grid_shape, dtype=float)  # Atmospheric temperature

        # Grids for fire dynamics
        self.fire_proximity = np.zeros(grid_shape, dtype=float)  # Proximity to existing fire
        self.perimeter_cells = set()  # Cells at the perimeter of the fire

        # Grids for resource allocation
        self.mobility = np.zeros(grid_shape, dtype=float)  # Mobility of firefighting resources
        self.potency = np.zeros(grid_shape, dtype=float)  # Potency of firefighting resources
        self.cost = np.zeros(grid_shape, dtype=float)  # Cost of firefighting efforts
        self.budget = 0.0  # Budget for firefighting

    def get_grid(self, grid_name: str) -> np.ndarray:
        """Get a grid by name."""
        try:
            return getattr(self, grid_name)
        except AttributeError:
            raise ValueError(f"Grid named '{grid_name}' does not exist.")

    def set_grid(self, grid_name: str, grid: np.ndarray):
        """Set a grid by name."""
        if hasattr(self, grid_name):
            setattr(self, grid_name, grid)
        else:
            raise ValueError(f"Grid named '{grid_name}' does not exist.")

    # Wrapper method to set default values for easier testing
    def set_default_values(self):
        # Topography slope: Normal distribution (mean=0.2, std_dev=0.05)
        self.topography_slope = np.random.normal(0.2, 0.05, self.topography_slope.shape)

        # Topography vegetation density: Beta distribution (a=2, b=5)
        self.topography_vegetation_density = np.random.beta(2, 5, self.topography_vegetation_density.shape)

        # Wind vector: Random direction and magnitude
        wind_magnitude = np.random.uniform(0.2, 0.8)
        wind_angle = np.random.uniform(0, 2*np.pi)
        self.wind_vector = [wind_magnitude * np.cos(wind_angle), wind_magnitude * np.sin(wind_angle)]

        # Fuel moisture: Uniform distribution [0, 0.2]
        self.fuel_moisture = np.random.uniform(0, 0.2, self.fuel_moisture.shape)

        # Fuel type: Randomly choose between 
        self.fuel_type = np.random.choice(['rothermel', 'tinder', 'retarded'], self.fuel_type.shape)

        # Atmospheric humidity: Normal distribution (mean=50, std_dev=10)
        self.atmospheric_humidity = np.random.normal(50, 10, self.atmospheric_humidity.shape)

        # Atmospheric temperature: Normal distribution (mean=25, std_dev=5)
        self.atmospheric_temperature = np.random.normal(25, 5, self.atmospheric_temperature.shape)

        # Mobility: Beta distribution (a=2, b=2)
        self.mobility = np.random.beta(2, 2, self.mobility.shape)

        # Potency: Beta distribution (a=2, b=2)
        self.potency = np.random.beta(2, 2, self.potency.shape)

        # Cost: Uniform distribution [0.1, 0.5]
        self.cost = np.random.uniform(0.1, 0.5, self.cost.shape)

        # Budget: Normal distribution (mean=1000, std_dev=100)
        self.budget = np.random.normal(1000, 100)

        # Fire proximity: calculate based on the grid
        self.fire_proximity = calculate_fire_proximity(self.grid)

        # create a log statement with the full sim info
        #print(f'SPATIAL:\n Slope: {self.topography_slope}, Vegetation Density: {self.topography_vegetation_density}, Wind Vector: {self.wind_vector}, Fuel Moisture: {self.fuel_moisture}, Fuel Type: {self.fuel_type}, Atmospheric Humidity: {self.atmospheric_humidity}, Atmospheric Temperature: {self.atmospheric_temperature}, Fire Proximity: {self.fire_proximity}, Perimeter Cells: {self.perimeter_cells}, Mobility: {self.mobility}, Potency: {self.potency}, Cost: {self.cost}, Budget: {self.budget}\n\n')


    # update the environment based on grid state
    def update(self):
        self.fire_proximity = calculate_fire_proximity(self.grid)
        self.perimeter_cells = get_perimeter_cells(self.grid)