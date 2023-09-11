# Info for Rothermel model: https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf
import math

class RothermelModel:
    def __init__(self, rho, sigma, FMC, xi, I_R, Q_ig, epsilon, SAVR, beta, wind_speed, slope):
        self.rho = rho  # Fuel bed bulk density
        self.sigma = sigma  # Surface area to volume ratio
        self.FMC = FMC  # Fuel moisture content
        self.xi = xi  # Reaction velocity
        self.I_R = I_R  # Reaction intensity
        self.Q_ig = Q_ig  # Heat of preignition
        self.epsilon = epsilon  # Effective heating number
        self.SAVR = SAVR  # Surface area-to-volume ratio
        self.beta = beta  # Packing ratio
        self.wind_speed = wind_speed  # Wind speed
        self.slope = slope  # Slope

    def calculate_R0(self):
        """Calculate the no-wind, no-slope rate of spread (R0)"""
        return self.xi * self.I_R / (self.rho * self.Q_ig)
        
    def calculate_phi_w(self):
        """Calculate the wind coefficient (phi_w)"""
        return self.C(self.sigma) * self.wind_speed ** 2 / (self.rho * self.Q_ig)
        
    def calculate_phi_s(self):
        """Calculate the slope coefficient (phi_s)"""
        return self.slope / math.sin(math.atan(self.slope))
        
    def C(self, sigma):
        """Calculate the wind constant C"""
        # params are from Rothermel 1972
        return 7.47 * math.exp(-0.133 * sigma ** 0.55)
        
    def calculate_R(self):
        """Calculate the rate of spread (R)"""
        R0 = self.calculate_R0()
        phi_w = self.calculate_phi_w()
        phi_s = self.calculate_phi_s()
        return R0 * (1 + phi_w + phi_s)
