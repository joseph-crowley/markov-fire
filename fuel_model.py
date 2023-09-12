import math

class RothermelFireSpreadModel:
    """
    A class to model the spread of wildfire based on the Rothermel model.
    
    References:
    - Rothermel Model Info: https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf
    
    Attributes:
        fuel_density (float): Fuel bed bulk density (rho).
        area_to_volume_ratio (float): Surface area to volume ratio (sigma).
        moisture_content (float): Fuel moisture content (FMC).
        reaction_velocity (float): Reaction velocity (xi).
        reaction_intensity (float): Reaction intensity (I_R).
        preignition_heat (float): Heat of preignition (Q_ig).
        heating_number (float): Effective heating number (epsilon).
        surface_area_to_volume_ratio (float): Surface area-to-volume ratio (SAVR).
        packing_ratio (float): Packing ratio (beta).
        wind_speed (float): Wind speed.
        terrain_slope (float): Slope of the terrain.
    """

    def __init__(self, 
                 fuel_density: float, 
                 area_to_volume_ratio: float, 
                 moisture_content: float, 
                 reaction_velocity: float, 
                 reaction_intensity: float,
                 preignition_heat: float, 
                 heating_number: float, 
                 surface_area_to_volume_ratio: float, 
                 packing_ratio: float, 
                 wind_speed: float, 
                 terrain_slope: float) -> None:
        """Initialize the Rothermel wildfire model with given parameters."""
        self.fuel_density = fuel_density
        self.area_to_volume_ratio = area_to_volume_ratio
        self.moisture_content = moisture_content
        self.reaction_velocity = reaction_velocity
        self.reaction_intensity = reaction_intensity
        self.preignition_heat = preignition_heat
        self.heating_number = heating_number
        self.surface_area_to_volume_ratio = surface_area_to_volume_ratio
        self.packing_ratio = packing_ratio
        self.wind_speed = wind_speed
        self.terrain_slope = terrain_slope

    def _calculate_wind_constant(self, sigma: float) -> float:
        """Calculate the wind constant (C) based on surface area to volume ratio."""
        return 7.47 * math.exp(-0.133 * sigma ** 0.55)
        
    def calculate_no_wind_no_slope_rate(self) -> float:
        """Calculate the no-wind, no-slope rate of spread (R0)."""
        return self.reaction_velocity * self.reaction_intensity / (self.fuel_density * self.preignition_heat)
        
    def calculate_wind_coefficient(self) -> float:
        """Calculate the wind coefficient (phi_w)."""
        return self._calculate_wind_constant(self.area_to_volume_ratio) * self.wind_speed ** 2 / (self.fuel_density * self.preignition_heat)
        
    def calculate_slope_coefficient(self) -> float:
        """Calculate the slope coefficient (phi_s)."""
        return self.terrain_slope / math.sin(math.atan(self.terrain_slope))
        
    def calculate_rate_of_spread(self) -> float:
        """Calculate the rate of spread (R) considering wind and slope."""
        no_wind_no_slope_rate = self.calculate_no_wind_no_slope_rate()
        wind_coefficient = self.calculate_wind_coefficient()
        slope_coefficient = self.calculate_slope_coefficient()
        return no_wind_no_slope_rate * (1 + wind_coefficient + slope_coefficient)
