#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title: morbi_spacecraft.py
project: AE4S20 STC Final Assignment MORBI-B
date: 27/02/2021
author: lmaio
"""
import numpy as np
import pandas as pd
from parameters import SpaceConstants, Mercury

C = SpaceConstants()
M = Mercury(nu=0) # Assumes mean anomaly of 0

class MORBI():
    def __init__(self, phi=67.31, theta=22.69, fixed_Tss=True):
        self.basic = fixed_Tss # Using basic assumptions from Q8
        # Values given in problem description
        self.alt = 9000e3         # altitude above surface
        self.per = self.alt + M.R          # m Periapsis altitude
        self.apo = self.alt + M.R          # m Apoapsis altitude
        self.i = 90                 # deg orbit inclination
        self.beta = np.nan          # deg Right ascension of ascending node
        self.T = 863 * 60           # s Orbital period
        self.T_ecl = 117 * 60       # s Eclipse length

        # Orbital values
        ecl = 180 - self.T_ecl / self.T * 180
        self.phi_ecl = [ecl, 360 - ecl]  # deg Angular range where eclipse occurs

        self.orbit_h = self.per if self.per==self.apo else None

        self.theta = theta  # solar incidence angle on panels
        self.phi = phi      # orbit position


        # Spacecraft dimensions and properties
        self.SP_eff = 0.2
        self.dims = {'F': [1, 1],
                     'R': [1, 1],
                     'W': [1, 1],
                     'Gap_dist': 0.5}

        self._properties = {'F': {'C': 40,  'e': 0.85, 'a': 0.90,
                                  'A': self.dims['F'][0]*self.dims['F'][1] },
                            'R': {'C': 40,  'e': 0.82, 'a': 0.35,
                                  'A': self.dims['R'][0]*self.dims['R'][1] },
                            'W': {          'e': 0.76, 'a': 0.09,
                                  'A': self.dims['W'][0]*self.dims['W'][1] }}
        self.P = pd.DataFrame(self._properties).T

        # Properties derived and calculated in question steps
        self.F = pd.DataFrame({'F': {'F':0, 'R':0, 'W':0, 'P':0, 'S':0}, # order of rows/cols
                               'R': {},
                               'W': {'P': 0.01},
                               'P': {},
                               'S': {}}).T
        # From ThermXL
        self.R_ir = pd.DataFrame({"W": {"F": 3.116874E-02, "R": 3.103587E-02},
                                  "F": {"W": 3.116874E-02},
                                  "R": {"W": 3.103587E-02}}
                                 ).T

        self.R_vis = pd.DataFrame({"W": {"F": 1.102113E-02, "R": 9.381133E-03},
                                   "F": {"W": 1.102113E-02},
                                   "R": {"W": 9.381133E-03}}
                                  ).T


        ###################### Question 9.1 ##########################
        # ------- Calculate View Factors --------
        self.calc_static_view_factors()
        self.planet_view_factors()
        self.space_view_factors()


        # ------- Gebhart Factors
        self.B_ir   = self.gebhart_factors(self.R_ir)
        self.B_vis  = self.gebhart_factors(self.R_vis, coeff='a')

        # ------- Radiative Couplings (remaining after ThermXL) --------
        self.radiative_couplings()


    # --------------------------------- View Factors ------------------------------
    def calc_static_view_factors(self):
        def rect_com_edge_90deg(L, N):
            return 1 / (np.pi * L) * \
                   (L * np.arctan(1 / L) + N * np.arctan(1 / N) -
                    np.sqrt(N ** 2 + L ** 2) * np.arctan(1 / np.sqrt(N ** 2 + L ** 2)) +
                    1 / 4 * np.log((1 + L ** 2) * (1 + N ** 2) / (1 + N ** 2 + L ** 2) *
                                   (L ** 2 * (1 + L ** 2 + N ** 2) /
                                    ((1 + L ** 2) * (L ** 2 + N ** 2))) ** (L ** 2) *
                                   (N ** 2 * (1 + L ** 2 + N ** 2) /
                                    ((1 + N ** 2) * (L ** 2 + N ** 2))) ** (N ** 2))
                    )

        assert round(rect_com_edge_90deg(0.1, 0.05), 5) == 0.18601  # Verify against real data

        # ---- Solar panel - wall ------- :
        # F_WR-prime: (solar panel + gap)
        Np = (self.dims['Gap_dist'] + self.dims['R'][1]) / self.dims['R'][0]
        Lp = (self.dims['W'][0] / 2) / self.dims['R'][0]
        F_WRp = rect_com_edge_90deg(Lp, Np)

        # F_Wgap
        Ng = self.dims['Gap_dist'] / self.dims['R'][0]
        Lg = (self.dims['W'][0] / 2) / self.dims['R'][0]
        F_Wgap = rect_com_edge_90deg(Lg, Ng)

        # F_WR (viewfactor_part_2 pg. 1-45)
        self.F.at['W', 'R'] = F_WRp - F_Wgap

        # F_RW (uses half area of wall)
        self.F.at['R', 'W'] = self.P.at['W', 'A'] * self.F.at['W', 'R'] / (2 * self.P.at['R', 'A'])

        # Front of solar panel same VF with wall as rear
        self.F.at['W', 'F'] = self.F.at['W', 'R']
        self.F.at['F', 'W'] = self.F.at['R', 'W']

    def planet_view_factors(self):
        # Planet to Wall (reciprocity)
        self.F.at['P', 'W'] = self.F.at['W', 'P'] * self.P.at['W', 'A'] / (M.area / 4)

        if self.basic:
            # Using equation 13 from q8.1
            lbda = 90 - self.phi - self.theta  # angle between the panel and surface normal

            if not 0 <= self.theta <= 90:
                raise NotImplementedError('See section 8.1, theta out of range')

            # Panel rear to planet:
            self.F.at['R', 'P'] = np.cos(np.deg2rad(lbda)) / (1 + (self.alt / M.R))**2

            # Reciprocity relation
            self.F.at['P', 'R'] = self.F.at['R', 'P'] * self.P.at['R', 'A'] / (np.pi * M.R**2)

            # Front of panel does not see planet
            self.F.at['F', 'P'] = 0
            self.F.at['P', 'F'] = 0

        else:
            # assuming 'best case scenario' --> Solar array perpundicular to planet
            r_SA = np.sqrt(self.P.at['R', 'A'] / np.pi)  # use circular solar array
            R = r_SA ** 2 / (M.R + self.orbit_h)
            self.F.at['P', 'R'] = 0.5 * (1 - 1 / np.sqrt(1 + R ** 2))
            if round(self.F.at['P', 'R'], 14) == 0:
                self.F.at['R', 'P'] = 0
                self.F.at['F', 'P'] = 0
                self.F.at['P', 'F'] = 0

    def space_view_factors(self):
        # Using sum of view factors = 1
        self.F.fillna(0, inplace=True)
        for row in self.F.index:
            self.F.at[row, 'S'] = 1 - sum(self.F.loc[row, ['W', 'P', 'F', 'R']])

    # --------------------------------- Gebhart Factors ------------------------------
    def gebhart_factors(self, thermxl_res, coeff='e'):
        B = pd.DataFrame()

        # Reverse-calc Gebhart factors from R-couplings
        for i, vals in thermxl_res.items():
            for j, val in vals.items():
                B.at[i, j] = val / (self.P.at[i, coeff] * self.P.at[i, 'A'])

        # No radiation between front and rear solar panel
        B.at['F', 'R'] = B.at['R', 'F'] = 0

        B.at['F', 'F'] = (1 - self.P.at['R', coeff]) * self.F.at['F', 'R'] * B.at['R', 'F'] + \
                         (1 - self.P.at['W', coeff]) * self.F.at['F', 'W'] * B.at['W', 'F']
        B.at['R', 'R'] = (1 - self.P.at['F', coeff]) * self.F.at['R', 'F'] * B.at['F', 'R'] + \
                         (1 - self.P.at['W', coeff]) * self.F.at['R', 'W'] * B.at['W', 'R']
        B.at['W', 'W'] = (1 - self.P.at['F', coeff]) * self.F.at['W', 'F'] * B.at['F', 'W'] + \
                         (1 - self.P.at['R', coeff]) * self.F.at['W', 'R'] * B.at['R', 'W']

        # Calc Gebhart Factors w/ space
        for surf in self.P.index:
            B.at[surf, 'S'] = 1 - sum(B.loc[surf, ['F', 'R', 'W']])

        return B

    # ---------------------- Radiative Couplings ---------------
    def radiative_couplings(self):
        for i in self.B_ir.index:
            for j in self.B_ir.columns:
                if i != j:
                    self.R_ir.at[i, j] = self.P.at[i, 'e'] * self.P.at[i, 'A'] * self.B_ir.at[i, j]

        # Conductive Coupling dependent of T-bar (initial T_F and T_R)
        self.R_ir.at['R', 'F'] = np.nan
        self.R_ir.at['F', 'R'] = np.nan