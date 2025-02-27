# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:31:17 2024

@author: DanielEspinoza
"""

import matplotlib.pyplot as plt
import numpy as np

from CADETPythonSimulator.distribution_base import DistributionBase
from CADETPythonSimulator.unit_operation import DistributionInlet, Outlet, DeadEndFiltration
from CADETPythonSimulator.system import FlowSystem
from CADETPythonSimulator.solver import Solver
from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.rejection import StepCutOff


class distribution(DistributionBase):

    a = 0.1
    set_distr = np.array([10, 0, 0])

    def get_distribution(self, t, nr):
        return self.set_distr


case1 = {'diameters': [27.5, 101.1, 315.4],  # nm
         'mass fractions': [.466, .276, .258],
         'name': 'Case 1'
         }

case2 = {'diameters': [144, 1174],  # nm
         'mass fractions': [.519, .481],
         'name': 'Case 2'
         }

rho_water = 998  # kg*m^-3

rho_debris = 1.1*rho_water

avogadro = 6.022e23

case = 1

filter_area = 1
filter_resistance = 1

permeate_tank_volume = 30  # liters

if case == 1:
    data = case1
else:
    data = case2

d = np.array(data['diameters'])
distr = np.array(data['mass fractions'])

MWs = (d*1e-9/2)**2*np.pi*rho_debris/avogadro*1000

MW_protein = 82358.0

MWs = MW_protein*np.array([1500, 1000, 1])
# MWs = np.append(MWs)

# biomass_per_liter = 379  # g/l
# mass_of_fermentation_broth = 27819  # ml
# total_biomass = biomass_per_liter * (mass_of_fermentation_broth*1e-3)  # g
# broth_fraction = (mass_of_fermentation_broth-total_biomass)/mass_of_fermentation_broth
# component_masses = total_biomass*distr
# component_distr = component_masses / mass_of_fermentation_broth
# distr = np.append(component_distr, broth_fraction)

# %%
permvol = []
cakevol = []
pressure = []
permflow = []

tankvol = []
tankcon = []

component_system = CPSComponentSystem(
                    name="cell_debris",
                    components=len(MWs),
                    pure_densities=[rho_debris]*len(MWs),
                    molecular_weights=MWs,
                    viscosities=[1]*len(MWs),
                    specific_cake_resistances=[1]*len(MWs)
                    )

rejectionmodell = StepCutOff(cutoff_weight=MWs.min()*1.1)
vol_dist = distribution()
vol_dist.set_distr = distr
inlet = DistributionInlet(component_system=component_system, name="inlet")
inlet.distribution_function = vol_dist
outlet = Outlet(component_system=component_system, name="outlet")
filter_obj = DeadEndFiltration(
                    component_system=component_system,
                    name="deadendfilter",
                    rejection_model=rejectionmodell,
                    membrane_area=filter_area,
                    membrane_resistance=filter_resistance,
                    )

unit_operation_list = [inlet, filter_obj, outlet]

system = FlowSystem(unit_operations=unit_operation_list)
section = [
            {
                'start': 0,
                'end': 110,
                'connections': [
                    [0, 1, 0, 0, 1],
                    [1, 2, 0, 0, 0.5],
                ],
            }
        ]

system.initialize_state()
system.states['deadendfilter']['permeate_tank']['tankvolume'] = permeate_tank_volume
system.states['deadendfilter']['permeate_tank']['c'] = np.array([0, 0, 0])
solver = Solver(system, section)

solver.solve()

t = solver.time_solutions
data1 = solver.unit_solutions['deadendfilter']['cake']['pressure']['values']

data2 = solver.unit_solutions['deadendfilter']['cake']['cakevolume']['values']
data3 = solver.unit_solutions['deadendfilter']['cake']['permeatevolume']['values']
data4 = solver.unit_solutions['deadendfilter']['cake']['permeatevolume']['derivatives']

pressure.append(data1)
permvol.append(data3)
cakevol.append(data2)
permflow.append(data4)

tankvolume = solver.unit_solutions['deadendfilter']['permeate_tank']['tankvolume']['values']
tankconcentrations = solver.unit_solutions['deadendfilter']['permeate_tank']['c']['values']

tankvol.append(tankvolume)
tankcon.append(tankconcentrations)

title = 100
fig, axes = plt.subplots(1, 2)
# fig.suptitle(f'{title}% Rückhalt')
axes[0].set_xlabel('$t$ [$s$]')
axes[0].set_ylabel('$\Delta P$ [$Pa$]')
axes[0].set_title('')

axes[1].set_xlabel('$t$ [$s$]')
axes[1].set_ylabel('$V$ [$m^3$]')
axes[1].set_title('Cake and permeate volume')

sigma = title/100
vglwerte = [(1-sigma)*(1 + sigma*time) for time in t]

i = 0

print(np.linalg.norm(vglwerte-pressure[i][:, 0], np.inf))

axes[0].plot(t, vglwerte, color='red', label='$\Delta P_{Vgl.}$')

axes[0].plot(t, pressure[i][:, 0], 'o', color='blue', label='$\Delta P$')
# axes[0].set_box_aspect(1)
axes[0].legend()

axes[1].plot(t, permvol[i][:, 0], 'o', color='red', label='$V^P$')

axes[1].plot(t, np.sum(cakevol[i], axis=1), 'x', color='blue', label='$V^C$')
# axes[1].set_box_aspect(1)
axes[1].legend()
# axes[0].set_xticks(t[0::2])
# axes[1].set_xticks(t[0::2])
fig.tight_layout()

# fig.savefig(f'{title}rejectionmulti.png')
plt.show(block = False)

fig, axes = plt.subplots()
fig.suptitle(f'Tank volume')

axes.title.set_text(f'{title}% Retention')

axes.set_xlabel('$t$ [$s$]')
axes.set_ylabel('$V^T$ [$m^3$]')

axes.plot(t, tankvol[i][:, 0], 'o', label='$V^T$')
# axes.set_box_aspect(1)
axes.legend()
# axes.set_xticks(t[0::2])
fig.tight_layout()

# fig.savefig('rejectionmultitankvol.png')
plt.show(block=False)
