#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title: thermal_analysis.py
project: AE4S20 STC Final Assignment MORBI-B
date: 03/02/2021
author: lmaio
"""

# ThermXL License servername:  27001@flexserv8.tudelft.nl

import numpy as np
import pandas as pd
from parameters import SpaceConstants, Mercury
from morbi_spacecraft import MORBI
import matplotlib.pyplot as plt
from tqdm import tqdm

nu = 0 #deg true anomaly for mercury in orbit

M  = Mercury(nu)
C  = SpaceConstants()


def mercury_dist():
    min_d = M.perihelion / C.AU
    max_d = M.aphelion / C.AU
    return min_d, max_d

def mercury_flux(min_d, max_d):
    min_f = C.solar_flux(max_d)
    max_f = C.solar_flux(min_d)

    return min_f, max_f

def thermal_sources(theta, phi):
    '''
    :param theta: deg
    :param phi: deg
    :return:
    '''
    # Fix possible weird angles (negative or too large)
    phi = phi % 360
    theta = theta % 360
    gamma = abs(180 - (phi % 180 + theta))
    longitude = 0 # always under sub-solar point

    Q = pd.DataFrame()
    Q.at['S', 'F'] = Q.at['S', 'R'] = Q.at['S', 'W'] = 0

    # Fluxes
    S_sol = M.solar_flux
    S_alb = S_sol * M.albedo

    # Planet warm/cold side fluxes
    S_IR = M.heat_flux(phi, longitude)

    # Correct fluxes for incidence angle
    S_sol *= np.sin(np.deg2rad(theta))
    S_IR *= np.sin(np.deg2rad(gamma))
    S_alb *= np.sin(np.deg2rad(gamma))

    # Planet "Behind" Solar Panel
    if not (180-theta < phi < 360-theta):
        # -------------------------- SOLAR -------------------------------------------------------
        if not (SC.phi_ecl[0] < phi < SC.phi_ecl[1]):  # If not in eclipse
            # Front Panel: Direct + Reflected solar flux
            Q.at['S', '$F_{sol}$'] = S_sol * SC.P.at['F', 'A'] * (SC.P.at['F', 'a'] +
                                     (1 - SC.P.at['F', 'a']) * SC.B_vis.at['F', 'F'] )

            # Generated power by solar panels
            Q.at['F', '$P_{gen}$'] = S_sol * SC.SP_eff * SC.P.at['F', 'A']

            Q.at['S', 'F'] += Q.at['S', '$F_{sol}$'] - Q.at['F', '$P_{gen}$']

            # Side wall: Reflected solar flux
            Q.at['S', '$W_{sol, refl}$'] = S_sol * (1 - SC.P.at['F', 'a']) * SC.B_vis.at['F', 'W']
            Q.at['S', 'W'] += Q.at['S', '$W_{sol, refl}$']

        # ------------------------- PLANETARY ---------------------------------------------------

        if SC.basic:
            # assuming constant surface temperature
            S_IR = C.sigma * M.Tss**4 * M.epsilon * SC.F.at['R', 'P'] * SC.P.at['R', 'A']
            Q.at['M', '$R_{IR}$'] = S_IR  * \
                             (SC.P.at['R', 'e'] + (1 - SC.P.at['R', 'e']) * SC.B_ir.at['R', 'R'] )

            Q.at['M', '$W_{IR, refl}$'] = S_IR * (1 - SC.P.at['R', 'e']) * SC.B_ir.at['R', 'W']

            # Add to main matrix used for calculations
            Q.at['S', 'R'] += Q.at['M', '$R_{IR}$']
            Q.at['S', 'W'] += Q.at['M', '$W_{IR, refl}$']

        else:
            # IR on Rear panel (including reflection back)
            Q.at['S', 'R'] += S_IR * SC.P.at['R', 'A'] * (SC.P.at['R', 'e'] +
                                                         (1 - SC.P.at['R', 'e']) *
                                                          SC.B_ir.at['R', 'R'] )
            # IR Reflection on Side Wall
            Q.at['S', 'W'] += S_IR * (1 - SC.P.at['R', 'e']) * SC.B_ir.at['R', 'W']

        # Albedo on Rear Panel (NO REFLECTIONS)
        Q.at['M', '$R_{alb}$'] = S_alb * SC.P.at['R', 'a'] * SC.P.at['R', 'A']

        Q.at['S', 'R'] += Q.at['M', '$R_{alb}$']




    # Planet "In front" of solar panel (Front panel also receiving planetary flux,
    #                                   rear panel getting nothing)
    else:
        # --- Solar Power Source ---
        if not (SC.phi_ecl[0] < phi < SC.phi_ecl[1]): # Eclipse conditions
            # Direct + Reflected solar flux
            Q.at['S', '$F_{sol}$'] = S_sol * SC.P.at['F', 'A'] * \
                                     (SC.P.at['F', 'a'] + (1 - SC.P.at['F', 'a']) *
                                      SC.B_vis.at['F', 'F'] )

            Q.at['S', 'F'] += Q.at['S', '$F_{sol}$']

            # Side wall: Reflected solar flux
            Q.at['S', 'W'] += S_sol * (1 - SC.P.at['F', 'a']) * SC.B_vis.at['F', 'W']

        # --- Planetary Power Sources ---
        # IR on Front panel (including reflection back)
        Q.at['S', 'F'] += S_IR * SC.P.at['F', 'A'] * \
                          (SC.P.at['F', 'e'] + (1 - SC.P.at['F', 'e'] * SC.B_ir.at['F', 'F']))
        # Albedo on Front Panel (NO REFLECTIONS)
        Q.at['S', 'F'] += S_alb * SC.P.at['F', 'a'] * SC.P.at['F', 'A']

        # IR Reflection on Side Wall
        Q.at['S', 'W'] += S_IR * (1 - SC.P.at['F', 'e']) * SC.B_ir.at['F', 'W']

    return Q



def F_elim(R_df, Q_df):
    # Create copies to prevent eliminating original coupling values

    def R(nodes):
        val = R_df.at[nodes[0], nodes[1:]]
        return val

    def Q(nodes):
        return Q_df.at[nodes[0], nodes[1:]]

    R_df.at['R', "Sf"] = R('FS') * R('FR') / (R('FS') + R('FR') + R('FW'))
    R_df.at['W', "Sf"] = R('FW') * R('FS') / (R('FS') + R('FR') + R('FW'))
    R_df.at['R', "Wf"] = R('FW') * R('FR') / (R('FS') + R('FR') + R('FW'))

    Q_df.at['S', "Rf"] = R('FR') / (R('FS') + R('FR') + R('FW')) * Q('SF')
    Q_df.at['S', "Wf"] = R('FW') / (R('FS') + R('FR') + R('FW')) * Q('SF')

    # Simplification of parallel couplings
    R_df.at['R', "Sp"] = R('RSf') + R('RS')
    R_df.at['W', "Sp"] = R("WSf") + R("WS")
    R_df.at['R', "Wp"] = R("RWf") + R("RW")

    Q_df.at['S', "Rp"] = Q("SRf") + Q("SR")
    Q_df.at['S', "Wp"] = Q("SWf") + Q("SW")

    return R_df, Q_df

def W_elim(R_df, Q_df, Twall=318.15):
    # Create copies to prevent eliminating original coupling values

    def R(nodes):
        val = R_df.at[nodes[0], nodes[1:]]
        return val

    def Q(nodes):
        return Q_df.at[nodes[0], nodes[1:]]


    # Fixed wall temp so Qout is also fixed
    Q_df.at['W', 'S'] = R("WSp") * C.sigma * (Twall ** 4)
    Q_df.at['S', "Wpp"] = Q("SWp") - Q('WS')  # power in minus power out

    Q_df.at['S', "Rpp"] = Q("SRp") + Q("SWpp")


    return R_df, Q_df

def node_F(Q, T_R, T_W):
    x = ( Q.at['S', 'F'] / C.sigma + SC.R_ir.at['F','R']*T_R**4 + SC.R_ir.at['F','W']*T_W**4) / \
        ( SC.R_ir.at['F','S'] + SC.R_ir.at['F', 'R'] + SC.R_ir.at['F', 'W'] )

    return x**(1/4)


def TBar(F, R):
    return (F+R)/2

def solve_network(T_bar, Q, T_W=None):
    '''
    :param T_bar: CELSIUS
    :param R: Radiant couplings
    :param Q: Power sources [W/m2]
    :param T_W: CELSIUS
    :return: T_F, T_R [celsius]
    '''
    # Convert TO Kelvin
    T_bar += 273.15

    if T_W:
        T_W += 273.15

    # Calculate Conductive Coupling
    SC.R_ir.at['F', 'R'] = SC.R_ir.at['R', 'F'] = SC.P.at['F', 'C'] / (4 * C.sigma * T_bar ** 3)

    # Eliminate node F (must come before Q_WS calc)
    SC.R_ir, Q = F_elim(SC.R_ir, Q)

    # Remove Node W
    SC.R_ir, Q = W_elim(SC.R_ir, Q, Twall=T_W)

    # Solve for T_R
    val = (Q.at['S', 'Rpp'] / (SC.R_ir.at['R', 'Sp'] * C.sigma))
    if val < 0:
        raise Warning(f'S/C requires additional heat source. T_R = {-((-val)**(1/4)):.6g} K')
    T_R = (val)**(1/4)

    T_F = node_F(Q, T_R, T_W)

    # Convert back to celsius
    T_F -= 273.15
    T_R -= 273.15

    return T_F, T_R


def FR_steady_state(theta, phi, threshold=1.0e-5, TF0=200, TR0=100, pr=False):
    def max_delta(i):
        return max(abs(results.at[i, 'dR']), abs(results.at[i, 'dF']))

    TF = TF0
    TR = TR0

    if pr:
        print(f'Initial Parameters:\n'
              f'\tT_F [C]     = {TF}\n'
              f'\tT_R [C]     = {TR}\n'
              f'\ttheta [deg] = {theta}\n'
              f'\tphi [deg]   = {phi}\n'
              f'\tmax error   = {threshold}\n')

    Q = thermal_sources(theta, phi)

    results = pd.DataFrame()
    results.at[0, 'F'] = TF
    results.at[0, 'R'] = TR
    results.at[0, 'dF'] = np.inf
    results.at[0, 'dR'] = np.inf


    i = 0
    while max_delta(i) > threshold:
        try:
            TF, TR = solve_network(TBar(TF, TR), Q, T_W=45.0)
        except Warning:
            # print('Abort: \n\t', w)
            # Not enough IR input to maintain wall temperature
            TF = TR = -1*np.inf

        results.at[i + 1, 'F'] = TF
        results.at[i + 1, 'R'] = TR
        results.at[i + 1, 'dF'] = TF - results.at[i, 'F']
        results.at[i + 1, 'dR'] = TR - results.at[i, 'R']

        if pr:
            print(f'Iteration {i + 1}\n'
                  f'\tT_F   = {TF:.6g}\n'
                  f'\tT_R   = {TR:.6g}\n'
                  f'\tError = {max_delta(i + 1):.6g}')

        i += 1

    sol = results.tail(1).reset_index()

    return sol.at[0, 'F'], sol.at[0, 'R'], results

def invalid_temps(df):
    frst = (df.F.values == -1*np.inf).argmax()
    last = (df.F.iloc[frst:].values != -1*np.inf).argmax() + frst

    return df[(df.index == frst-1) | (df.index==last)].reset_index(drop=True)


def find_opt_theta():
    def theta_max(phi, thmin=0, thmax=90, buffer=20):
        n_samples = 100
        theta_range = np.linspace(thmin, thmax, n_samples)
        temps = pd.DataFrame()
        for i, theta in enumerate(theta_range):
            # Re-initialize SC with new positions
            SC = MORBI(phi, theta)

            # Calculate temps for this condition
            temps.at[i, 'theta'] = theta
            temps.at[i, 'F'], temps.at[i, 'R'], _ = FR_steady_state(theta, phi)

            del SC

        # Find max temperature between front and rear nodes (expected front)
        temps['max_T'] = temps[['F', 'R']].max(axis=1)

        opt_idx = (temps['max_T'] >= 200).idxmax() - 1
        # Create refined theta range for next iteration
        new_thmin = temps.at[opt_idx - buffer, 'theta']
        new_thmax = temps.at[opt_idx + buffer, 'theta']

        # get exact optimum theta that is less than 200C
        max_theta = temps.at[opt_idx, 'theta']
        max_temp = temps.at[opt_idx, 'max_T']
        # new phi is when s/c panel is tangent to planet
        new_phi = 90 - max_theta

        return max_theta, new_phi, new_thmin, new_thmax, max_temp


    phi = 67.31
    th_r_min = 0
    th_r_max = 90
    th_buffer = 10

    theta_opt = pd.DataFrame()
    itermax = 100
    for i in tqdm(range(itermax)):
        theta_opt.at[i, 'phi_in'] = phi
        theta_opt.at[i, 'theta'], phi, th_r_min, th_r_max, m_T = theta_max(phi,
                                                                      th_r_min,
                                                                      th_r_max,
                                                                      th_buffer)
        theta_opt.at[i, 'phi_out'] = phi
        theta_opt.at[i, 'd_phi'] = phi - theta_opt.at[i, 'phi_in']
        theta_opt.at[i, 'T [C]'] = m_T

        if abs(theta_opt.at[i, 'd_phi']) < 1.0e-3:
            print(f'Converged before maxiter')
            break
        i += 1

    return theta_opt

def side_wall_temp(TF, TR, Q, Pdis=0):
    TF += 273.15
    TR += 273.15

    x = ((Q.at['S', 'W'] + Pdis) / C.sigma + (SC.R_ir.at['R', 'W'] * TR**4 +
                                              SC.R_ir.at['F', 'W'] * TF**4)) / (
        SC.R_ir.at['W', 'S'] + SC.R_ir.at['R', 'W'] + SC.R_ir.at['F', 'W'])

    return x**(1/4)-237.15

def side_wall_Pdis(TF, TR, TW, Q):
    TF += 273.15
    TR += 273.15
    TW += 273.15

    p = C.sigma * TW**4 * (SC.R_ir.at['W', 'S'] + SC.R_ir.at['R', 'W'] + SC.R_ir.at['F', 'W']) - \
        C.sigma * (SC.R_ir.at['R', 'W'] * TR**4 + SC.R_ir.at['F', 'W'] * TF**4) - Q.at['S', 'W']
    return p

if __name__ == '__main__': # =====================================================================
    num_fmt = '%.4g'
    global SC
    SC = MORBI()

    #%% Q1
    min_dist_mercury, max_dist_mercury = mercury_dist()

    print(f'Min dist: {min_dist_mercury:.5g} [AU]')
    print(f'Max dist: {max_dist_mercury:.5g} [AU]')

    #%% Q2
    min_flux_mercury, max_flux_mercury = mercury_flux(min_dist_mercury, max_dist_mercury)

    print(f'Min dist: {min_flux_mercury:.5g} [W/m^2]')
    print(f'Max dist: {max_flux_mercury:.5g} [W/m^2]')


    #%% Q9 ---------------------------------------------------------------------
    # ------------- Surface Properties: --------------
    with open('report/gen_tables/surf-props.tex', 'w') as f:
        f.write(SC.P.rename(columns={'a':'$\\alpha$',
                                     'A':'Area [$m^2$]',
                                     'e':'$\\epsilon$'}
                            ).to_latex(float_format=num_fmt, escape=False, na_rep='-'))

    # ------------ View Factors -----------------------
    with open('report/gen_tables/view-factors.tex', 'w') as f:
        f.write(SC.F.rename(index={'P':'M'},
                            columns={'P':'M'}).to_latex(float_format=num_fmt,
                                                        escape=False, na_rep='tbd'))

    # ------------ IR Gebhart factors -----------------
    with open('report/gen_tables/gebhart-factors.tex', 'w') as f:
        f.write(SC.B_ir.to_latex(float_format=num_fmt, escape=False, na_rep='-',
                                 columns=['F', 'R', 'W', 'S']))

    # ----------- Radiative Couplings -----------------
    with open('report/gen_tables/rad_couplings.tex', 'w') as f:
        f.write(SC.R_ir.sort_index().to_latex(float_format=num_fmt, escape=False,
                                              na_rep='-', columns=['F', 'R', 'W', 'S']))

    # ----------- Gebhart factors visible spectrum (sunlight) ------
    with open('report/gen_tables/gebhart-factors-vis.tex', 'w') as f:
        f.write(SC.B_vis.to_latex(float_format=num_fmt, escape=False, na_rep='-',
                                  columns=['F', 'R', 'W', 'S']))


    # -------- Network Solver (iterative for TF and TR estimates) ----------
    print('\n---------- Solving For Solar Panel Temps ----------')
    # Initial values don't matter, it converges fast
    theta = 22.69       # solar panel incidence angle
    phi = 67.31         # Position in orbit
    SC = MORBI(phi, theta) # ensure we're using the right values


    # ------------ Thermal Sources -----------------
    Q = thermal_sources(theta, phi)
    with open('report/gen_tables/thermal_sources-main.tex', 'w') as f:
        f.write(Q.loc[['S'], ['F', 'R', 'W']].to_latex(float_format='%.5g',
                                                       escape=False, na_rep='-'))
    with open('report/gen_tables/thermal_sources-components.tex', 'w') as f:
        f.write(Q.drop(columns=['F', 'R', 'W']
                       ).sort_index().to_latex(float_format='%.5g',
                                               escape=False, na_rep='-',
                                               columns=['$P_{gen}$', '$R_{IR}$',
                                                        '$W_{IR, refl}$','$R_{alb}$',
                                                        '$F_{sol}$', '$W_{sol, refl}$']))



    # Print results to table for report
    results = FR_steady_state(theta, phi, threshold=.0001)[2]
    print(results.tail(1))
    print()
    with open('report/gen_tables/FR_solution_wall45.tex', 'w') as f:
        results.rename(columns={'F':'$T_F$ [C]', 'R':'$T_R$ [C]',
                                'dF':'$\delta T_F$ [C]', 'dR':'$\delta T_R$ [C]'})
        caption = f'Iterations for Front and Rear panel temperatures ($\\phi = {theta}$, ' \
                  f'$\\varphi = {phi}$, $T_W = '+r'45^{\circ}$ C)'
        label = 'tab:temp-iteration'
        f.write(results.to_latex(float_format=num_fmt, escape=False, na_rep='-',
                                 label=label, caption=caption, position='H'))

    # Find max theta
    print(f'Finding optimum solar panel angle for 4-node network')
    optimum_theta = find_opt_theta()
    with open('report/gen_tables/optimum_theta_4-node.tex', 'w') as f:
        f.write(optimum_theta.rename(columns={'phi_in': r'$\varphi_{in}$',
                                              'theta': r'$\phi$',
                                              'phi_out': r'$\varphi_{out}$',
                                              'd_phi': r'd$\varphi$'}
                                     ).to_latex(float_format=num_fmt, escape=False, na_rep='-'))


    # ----------- VERIFICATION Fixed theta, full orbit analysis --------------------------------
    print(f'Solving full orbit for:\n'
          f'\ttheta [deg] = {theta}')
    n_samples = 100
    phi_range = np.linspace(0, 360, n_samples)
    theta = 22.69

    del SC
    SC = MORBI(fixed_Tss=False)

    orbit_temps = pd.DataFrame()
    for i, phi in enumerate(phi_range):
        # print(phi)
        orbit_temps.at[i, 'phi'] = phi
        orbit_temps.at[i, 'F'], orbit_temps.at[i, 'R'], _ = FR_steady_state(theta, phi)

    frozen = invalid_temps(orbit_temps)

    plt.plot(orbit_temps[['phi']], orbit_temps[['F']], color='green')
    plt.plot(frozen[['phi']], frozen[['F']], color='red')
    plt.xlabel('Phi [deg]')
    plt.ylabel('Temperature [C]')
    plt.title(r'Node F steady-state temperature over full orbit ($\theta$ ='+f' {theta} deg)')
    plt.show()


    #%% Q10 --------------------------------------------------------------------
    theta = 22.69
    phi = 67.31
    Pdis = 0

    SC = MORBI(phi, theta)

    TF, TR, _ = FR_steady_state(theta, phi)
    Q = thermal_sources(theta, phi)

    TW = side_wall_temp(TF, TR, Q)

    print(f'Side Wall temp: {TW:.5g} C')

    ## Part 2
    TW_max = 45
    Pdis_calc = side_wall_Pdis(TF, TR, TW=TW_max, Q=Q)
    print(f'Max internal dissipation: {Pdis_calc:.5g} W')

    # Save to table for report
    q10 = pd.DataFrame({0: {'TF': TF, 'TR': TR, 'Pdis': Pdis, 'TW':TW},
                        1: {'TF': TF, 'TR': TR, 'Pdis': Pdis_calc, 'TW': TW_max}
                        }).T
    with open('report/gen_tables/q10-results.tex', 'w') as f:
        f.write(q10.rename(columns={'TF': r'$T_F$ [$^{\circ}$C]',
                                    'TR': r'$T_R$ [$^{\circ}$C]',
                                    'Pdis': '$P_{dis}$ [W]',
                                    'TW': r'$T_W$ [$^{\circ}$C]'}
                           ).to_latex(float_format=num_fmt, escape=False,
                                      index=False, na_rep='-'))

    #%% Update code in report
    import shutil
    shutil.copy('thermal_analysis.py',  'report/code/thermal_analysis.py')
    shutil.copy('parameters.py',        'report/code/parameters.py')
    shutil.copy('morbi_spacecraft.py',  'report/code/morbi_spacecraft.py')
    shutil.copy('verification.py',      'report/code/verification.py')

