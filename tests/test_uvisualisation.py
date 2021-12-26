# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:12:46 2021

@author: yr3g17
"""
import numpy as np
from uncertainties import unumpy, ufloat
from uvisualisation.plotting import create_ufloatmesh, plot_ufloatmesh


def test_main():
    # BMP280 - Uncertainty in density derivations
    p_interval_hPa, p_stddev_hPa = (970, 1030), 1
    t_interval_C, t_stddev_C = (8, 25), 1

    def density(pressure_hPa, temperature_C):
        p_Pa = 100 * pressure_hPa
        t_K = 273.15 + temperature_C
        rair_JkgK = 287.05
        rho_kgm3 = p_Pa / (rair_JkgK * t_K)
        return rho_kgm3

    plotdata = create_ufloatmesh(p_interval_hPa, p_stddev_hPa,
                                 t_interval_C, t_stddev_C, density)
    labels = "Density [kg/m3]", "Pressure [hPa]", "Temperature [C]"
    plot_ufloatmesh(*plotdata, *labels)

    # MPXV7002DP - Uncertainty in dynamic pressure derivations
    # vout_interval_V, vout_stddev_V = (0.5, 4.5), 5 / 2 ** 8 / 2
    vout_interval_V, vout_stddev_V = (2.55, 2.7), 5 / 2 ** 8 / 2
    vfss_interval_V, vfss_stddev_V = (3.5, 4.5), 0.0
    # errorbudgetfactor = ufloat(0.00, 0.0625)
    errorbudgetfactor = ufloat(0.03, 0.005)

    def dynamicpressure(vout_V, vfss_V):
        vs_V = 5
        term1 = (unumpy.nominal_values(vout_V - 2.5))   # [-2.5 ... +2.5]
        term2 = np.absolute(term1).max()                # +2.5
        term3 = term1 / term2                           # [-1.0 ... +1.0]
        # Raising the absolute of term3 to -1==step error; 0==linear error
        vout_scalefactor = term3 * np.absolute(term3) ** -1
        vmeasured_V = vout_V - errorbudgetfactor * vout_scalefactor * vfss_V
        q_kgm1s2 = 5000 * ((vmeasured_V) / vs_V) - 2500
        return q_kgm1s2

    plotdata = create_ufloatmesh(vout_interval_V, vout_stddev_V,
                                 vfss_interval_V, vfss_stddev_V,
                                 dynamicpressure)
    labels = "Dynamic Pressure [kg/m/s2]", "V_Out [V]", "V_FSS [V]"
    plot_ufloatmesh(*plotdata, *labels)

    # MPXV7002DP - Uncertainty in airspeed derivations
    def eqairspeed(vout_V, vfss_V):
        q_kgm1s2 = dynamicpressure(vout_V, vfss_V)
        term1 = unumpy.nominal_values(q_kgm1s2)
        q_sig = term1 / np.absolute(term1)
        v_ms = q_sig * (2 * np.absolute(q_kgm1s2) / 1.2250) ** 0.5
        return v_ms

    plotdata = create_ufloatmesh(vout_interval_V, vout_stddev_V,
                                 vfss_interval_V, vfss_stddev_V, eqairspeed)
    labels = "Equivalent Airspeed [m/s]", "V_Out [V]", "V_FSS [V]"
    plot_ufloatmesh(*plotdata, *labels)
