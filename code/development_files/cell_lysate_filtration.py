# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:31:17 2024

@author: DanielEspinoza
"""

import matplotlib.pyplot as plt
import numpy as np

from CADETPythonSimulator.distribution_base import DistributionBase, ConstantVolumeDistribution, ConstantConcentrationDistribution
from CADETPythonSimulator.unit_operation import DistributionInlet, Outlet, DeadEndFiltration
from CADETPythonSimulator.system import FlowSystem
from CADETPythonSimulator.solver import Solver
from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.rejection import StepCutOff
from CADETPythonSimulator.viscosity import LogarithmicMixingViscosity

# def dead_end_filter(plot=False):
plot=True

rho_water = 998  # kg*m^-3

rho_debris = 1.1*rho_water

avogadro = 6.022e23

case = 1

# Sartobran P .2 micron large scale size 8
filter_area = .1  # m^2
filter_height = np.sqrt(filter_area/np.pi)*2*3  # m, assuming 1:3 aspect ratio to diameter
r_pore = (.45 + .2)/2 * 1e-6  # m
eps_filter = .8  # Filter porosity
dyn_visc = 1.05e-3  # P.s
R_m_HP = 8*dyn_visc*filter_height/eps_filter/r_pore**2
C = 1
permeability = C*r_pore**2
delta_P = .75e5  # bar, pressure drop using water at 500 l/h
Q = 1 * 1e-3/3600  # m^3/s, testing flow rate in data sheet
R_m_data_sheet = filter_area*delta_P/dyn_visc/Q
R_m_permeability = filter_height/permeability
filter_resistance = R_m_data_sheet
max_operating_pressure = 5e5  # Pa

permeate_tank_volume = 1e-9  # m^3, intial volume in permeate tank
flowrate_into_DEF = Q # m^3/s
flowrate_out_of_permeate_tank = 1e-9  # m^3/s

MW_protein = 82358.0  # g/mol
MW_water = 18  # g/mol

MWs = MW_protein*np.array([1500, 1000, 1])
cake_resistance = 1e11
# MWs = np.append(MWs)

debris_distr = np.array([.6, .4])  # 60% small debris, 40% large debris
biomass_per_liter = 379  # g/l
mass_of_fermentation_broth = 27819  # ml
protein_concentration = 9.300857233055684e-05  # mol/l, from GSK chromatography step loading
total_biomass = biomass_per_liter * (mass_of_fermentation_broth*1e-3)  # g
broth_fraction = (mass_of_fermentation_broth-total_biomass)/mass_of_fermentation_broth
protein_molar_amount = protein_concentration * mass_of_fermentation_broth*1e-3  # Protein concetration based on total broth mass
protein_mass = protein_molar_amount * MW_protein
water_mass = broth_fraction*mass_of_fermentation_broth
water_molar_amount = water_mass/MW_water
component_masses = np.append((total_biomass - protein_mass)*debris_distr[:2], np.array([protein_mass, water_mass]))
component_molar_amounts = np.append(component_masses[:2] / MWs[:2], np.array([protein_molar_amount, water_molar_amount]))
molar_component_distr = component_molar_amounts / component_molar_amounts.sum()
component_concentrations = np.append(molar_component_distr[:-2]/mass_of_fermentation_broth, protein_concentration)
mass_component_distr = component_masses/mass_of_fermentation_broth
distr = mass_component_distr

volume_to_filter = mass_of_fermentation_broth*1e-6  # m^3
time_to_filter = volume_to_filter/flowrate_into_DEF  # s

# %%
permvol = []
cakevol = []
pressure = []
permflow = []

tankvol = []
tankcon = []

component_system = CPSComponentSystem(
                    name="cell_lysate",
                    components=4,
                    pure_densities=[rho_debris]*3 + [rho_water],
                    molecular_weights=list(MWs) + [MW_water],
                    viscosities=[dyn_visc]*4,
                    specific_cake_resistances=[cake_resistance]*2 + [0, 0],
                    )

concentration_distribution = ConstantVolumeDistribution(component_system=component_system,
                                                        c=component_concentrations)
inlet = DistributionInlet(component_system=component_system, name="inlet")
inlet.distribution_function = concentration_distribution

rejectionmodell = StepCutOff(cutoff_weight=MW_protein*2)
viscositymodell = LogarithmicMixingViscosity()
filter_obj = DeadEndFiltration(
                    component_system=component_system,
                    name="deadendfilter",
                    rejection_model=rejectionmodell,
                    viscosity_model=viscositymodell,
                    membrane_area=filter_area,
                    membrane_resistance=filter_resistance,
                    )

outlet = Outlet(component_system=component_system, name="outlet")

unit_operation_list = [inlet, filter_obj, outlet]

system = FlowSystem(unit_operations=unit_operation_list)
section = [
            {
                'start': 0,
                'end': time_to_filter,
                'connections': [
                    [0, 1, 0, 0, flowrate_into_DEF],
                    [1, 2, 0, 0, flowrate_out_of_permeate_tank],
                ],
            }
        ]

system.initialize_state()

condist = ConstantConcentrationDistribution(component_system=component_system,
                                            c=[0, 0, 0])
c_init = condist.get_distribution(0, 0)

system.states['deadendfilter']['permeate_tank']['volume'] = permeate_tank_volume
system.states['deadendfilter']['permeate_tank']['c'] = c_init
solver = Solver(system, section)

solver.solve()

t = solver.time_solutions

data1 = solver.unit_solutions['deadendfilter']['cake']['pressure']['values']

data2 = solver.unit_solutions['deadendfilter']['cake']['volume']['values']

pressure.append(data1)
cakevol.append(data2)

tankvolume = solver.unit_solutions['deadendfilter']['permeate_tank']['volume']['values']
tankconcentrations = solver.unit_solutions['deadendfilter']['permeate_tank']['c']['values']

tankvol.append(tankvolume)
tankcon.append(tankconcentrations)

title = 100

if plot:
    fig, axes = plt.subplots(1, 1)
    # fig.suptitle(f'{title}% Rückhalt')
    axes.set_xlabel('$t$ [$s$]')
    axes.set_ylabel('$\Delta P$ [$Pa$]')
    axes.set_title('Pressure drop')

    sigma = title/100
    vglwerte = [(1-sigma)*(1 + sigma*time) for time in t]

    i = 0

    # print(np.linalg.norm(vglwerte-pressure[i][:, 0], np.inf))

    axes.plot(t, vglwerte, color='red', label='$\Delta P_{Vgl.}$')

    axes.plot(t, pressure[i][:, 0], 'o', color='blue', label='$\Delta P$')
    # axes[0].set_box_aspect(1)
    axes.legend()
    fig.tight_layout()

    # fig.savefig(f'{title}rejectionmulti.png')
    plt.show(block = False)

    fig, axes = plt.subplots()
    fig.suptitle(f'Tank volume')

    axes.title.set_text(f'{title}% Retention')

    axes.set_xlabel('$t$ [$s$]')
    axes.set_ylabel('$V^T$ [$liters$]')

    axes.plot(t, tankvol[i][:, 0]*1e3, 'o', label='$V^T$')
    # axes.set_box_aspect(1)
    axes.legend()
    # axes.set_xticks(t[0::2])
    fig.tight_layout()

    alpha_value = .3

    fig_c, ax_c = plt.subplots(3, 1, figsize=(2*6.4, 2*4.8))
    lines = ax_c[0].plot(t, tankconcentrations[:, :-1], alpha=alpha_value)
    ax_c_0_twin = ax_c[0].twinx()
    water_line = ax_c_0_twin.plot(t, tankconcentrations[:, -1], alpha=1, color='orange')
    ax_c[0].set_title('Concentration in permeate tank')
    # ax_c[0].set_ylabel('Protein and debris')
    ax_c_0_twin.set_ylabel('Water', color='orange')
    ax_c_0_twin.set_ylim(top=70)

    ax_c[1].plot(t, solver.unit_solutions['deadendfilter']['cake']['c_in']['values'][:, :-1], alpha=alpha_value)
    ax_c_1_twin = ax_c[1].twinx()
    ax_c_1_twin.plot(t, solver.unit_solutions['deadendfilter']['cake']['c_in']['values'][:, -1], alpha=1, color='orange')
    ax_c[1].set_title('Concentration into filter cake')
    ax_c[1].set_ylabel('Protein and debris')
    ax_c_1_twin.set_ylabel('Water', color='orange')
    # ax_c_1_twin.set_ylim(top=35)
    
    ax_c[2].plot(t, solver.unit_solutions['outlet']['inlet']['c']['values'][:, :-1], alpha=alpha_value)
    ax_c_2_twin = ax_c[2].twinx()
    ax_c_2_twin.plot(t, solver.unit_solutions['outlet']['inlet']['c']['values'][:, -1], alpha=1, color='orange')
    ax_c[2].set_title('Concentration in outlet')
    # ax_c[2].set_ylabel('Protein and debris')
    ax_c_2_twin.set_ylabel('Water', color='orange')
    ax_c_2_twin.set_ylim(top=70)

    labels = ['Debris 1', 'Debris 2', 'Proteins', 'Water']
    fig_c.legend(lines + water_line, labels, loc='outside right center', ncols=1, frameon=False)
    fig_c.tight_layout()
    fig_c.subplots_adjust(right=0.65)

    # fig.savefig('rejectionmultitankvol.png')
    plt.show(block=False)

permeate_volume = solver.unit_solutions['deadendfilter']['permeate_tank']['volume']['values'][-1]
permeate_protein_concentration = solver.unit_solutions['outlet']['inlet']['c']['values'][-1, 2]

#     return permeate_volume, permeate_protein_concentration

# if __name__ == '__main__':
#     V, c = dead_end_filter(plot=True)
#     print(f'Permeate protein concentration: {c} mol/L')