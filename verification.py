#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title: Unit test Verification
project: AE4S20 STC Final Assignment MORBI-B
created: 11/02/2021
author: lmaio
"""

import unittest
from thermal_analysis import mercury_dist, mercury_flux
from parameters import Mercury
from morbi_spacecraft import MORBI




class Verification(unittest.TestCase):
    def setUp(self):
        self.min_d = 1000e3
        self.max_d = 2000e3
        self.M = Mercury(nu=0.0)
        self.SC = MORBI()

    def test_q1(self):
        min_dist_mercury, max_dist_mercury = mercury_dist()
        assert min_dist_mercury < max_dist_mercury

    def test_q2(self):
        min_flux_mercury, max_flux_mercury = mercury_flux(self.min_d, self.max_d)
        assert min_flux_mercury < max_flux_mercury

    def test_mercury_alpha(self):
        self.assertAlmostEqual(self.M.alpha, 0.88, 2)

    def test_mercury_epsilon(self):
        self.assertAlmostEqual(self.M.epsilon, 0.4667, 4)



if __name__ == '__main__':
    unittest.main()
