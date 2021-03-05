#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title: problem parameters
project: AE4S20 STC Final Assignment MORBI-B
date: 03/02/2021
author: lmaio
"""

import numpy as np

class SpaceConstants:
    def __init__(self):
        # Constants from book (page 520)
        self.G       = 6.674e-11         # m^3 / kg*s^2          Gravitational constant
        self.c       = 2.998e8           # m / s                 Speed of radiation
        self.h       = 6.62607e10-34     # m^2 kg / s            Planck's constant
        self.k       = 1.3806e-23        # m^2 kg / s^2 K        Boltzmann's constant
        self.sigma   = 5.67051e-8        # W / m^2 K^4           Stefan-Boltzmann constant
        self.ly      = 9.4605e15         # m                     Light year
        self.AU      = 1.496e11          # m                     Astronomical unit
        self.pc      = 3.26156 * self.ly # m                     Parsec
        self.M_Earth = 5.9736e24         # kg                    Mass of earth
        self.M_Sun   = 1.989e30          # kg                    Solar mass

    def solar_flux(self, d_AU, Q_1AU=1366):
        return Q_1AU/d_AU**2

    def C2K(self, C):
        return C+273.15


class Mercury():
    def __init__(self, nu):
        self.R = 2439e3             # m Mean Radius
        self.Tss = 700              # K Max temp at min sun distance
        self.Tdd = 100              # K Min temp in eclipse
        self.albedo  = 0.12              # - Planetary Albedo
        self.aphelion = 69816900e3        # m Aphelion
        self.perihelion = 46001200e3        # m Perihelion
        self.T = 87.9691 * 24*3600  # s Orbital period
        self.area = 4 * np.pi * self.R**2

        # Orbital parameters
        self.orbit_nu = nu          # deg true anomaly
        self.orbit_e = (self.aphelion - self.perihelion) / (self.aphelion + self.perihelion)
        self.orbit_a = self.perihelion / (1 - self.orbit_e)
        self.d_sun_au = self._calc_d_to_sun(self.orbit_nu)
        self.solar_flux = SpaceConstants().solar_flux(self.d_sun_au)

        self._calc_surf_props()

    def _calc_d_to_sun(self, nu):
        return self.orbit_a * (1 - self.orbit_e**2) / (
                1 + self.orbit_e * np.cos(np.deg2rad(nu))) / SpaceConstants().AU

    def _calc_surf_props(self):
        self.alpha = 1 - self.albedo  # absorptivity\
        min_d_sun = self._calc_d_to_sun(0)
        S_max = SpaceConstants().solar_flux(min_d_sun)

        # Calculated using max solar flux
        self.epsilon = S_max * self.alpha / (2 * SpaceConstants().sigma *
                                             (self.Tss**4 + self.Tdd**4))


    def heat_flux(self, phi, gamma):
        '''
        :param phi: see figure 3
        :param gamma: see figure 2 (longitude)
        :return:
        '''
        if 90.0 < phi < 270. or 90. < gamma < 270.:
            return self.epsilon * SpaceConstants().sigma * self.Tdd**4

        else:
            T = self.epsilon * SpaceConstants().sigma * \
                ( (self.Tss - self.Tdd) * np.cos(np.deg2rad(phi)) *
                  np.cos(np.deg2rad(gamma)) + self.Tdd
                )**4
            return T

