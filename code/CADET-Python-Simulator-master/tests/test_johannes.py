# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:03:17 2024

@author: DanielEspinoza
"""

import matplotlib.pyplot as plt
import numpy as np

from CADETPythonSimulator.distribution_base import DistributionBase

class distribution(DistributionBase):

     a = 0.1
     def get_distribution(self, t, nr):
         return np.array([1-self.a, self.a])



from CADETPythonSimulator.unit_operation import DistributionInlet, Outlet, DeadEndFiltration
from CADETPythonSimulator.system import FlowSystem
from CADETPythonSimulator.solver import Solver
from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.rejection import StepCutOff

permvol = []
cakevol = []
pressure = []
permflow = []

tankvol = []
tankcon = []

for i in [0.25, 0.5, 0.75]:

    component_system = CPSComponentSystem(
                        name="test_comp",
                        components=2,
                        pure_densities=[1, 1],
                        molecular_weights=[1, 2],
                        viscosities=[1, 1],
                        specific_cake_resistances=[1, 1]
                        )

    rejectionmodell = StepCutOff(cutoff_weight=1.5)
    vol_dist = distribution()
    vol_dist.a = i
    inlet = DistributionInlet(component_system=component_system, name="inlet")
    inlet.distribution_function = vol_dist
    outlet = Outlet(component_system=component_system, name="outlet")
    filter_obj = DeadEndFiltration(
                        component_system=component_system,
                        name="deadendfilter",
                        rejection_model=rejectionmodell,
                        membrane_area=1,
                        membrane_resistance=1,
                        )

    unit_operation_list = [inlet, filter_obj, outlet]

    system = FlowSystem(unit_operations=unit_operation_list)
    section = [
                {
                    'start': 0,
                    'end': 11,
                    'connections': [
                        [0, 1, 0, 0, 1],
                        [1, 2, 0, 0, 0.5],
                    ],
                }
            ]

    system.initialize_state()
    system.states['deadendfilter']['permeate_tank']['tankvolume'] = 10
    system.states['deadendfilter']['permeate_tank']['c'] = np.array([1, 0])
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
    tankconzentrations = solver.unit_solutions['deadendfilter']['permeate_tank']['tankvolume']['values']

    tankvol.append(tankvolume)
    tankcon.append(tankconzentrations)


for i, title in enumerate([25, 50, 75]):
    fig, axes = plt.subplots(1, 2)
    #fig.suptitle(f'{title}% Rückhalt')
    axes[0].set_xlabel('$t$ [$s$]')
    axes[0].set_ylabel('$\Delta P$ [$Pa$]')

    axes[1].set_xlabel('$t$ [$s$]')
    axes[1].set_ylabel('$V$ [$m^3$]')

    sigma = title/100
    vglwerte = [(1-sigma)*(1 + sigma*time) for time in t]

    print(np.linalg.norm(vglwerte-pressure[i][:, 0], np.inf))

    axes[0].plot(t, vglwerte, color='red', label='$\Delta P_{Vgl.}$')

    axes[0].plot(t, pressure[i][:, 0], 'o', color='blue', label='$\Delta P$')
    axes[0].set_box_aspect(1)
    axes[0].legend()

    axes[1].plot(t, permvol[i][:, 0], 'o', color='red', label='$V^P$')

    axes[1].plot(t, np.sum(cakevol[i], axis=1), 'x', color='blue', label='$V^C$')
    axes[1].set_box_aspect(1)
    axes[1].legend()
    axes[0].set_xticks(t[0::2])
    axes[1].set_xticks(t[0::2])
    fig.tight_layout()

    # fig.savefig(f'{title}rejectionmulti.png')
    plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#fig.suptitle(f'Tankvolumina')
for i, title in enumerate(['25', '50', '75']):

    axes[i].title.set_text(f'{title}% Rückhalt')

    axes[i].set_xlabel('$t$ [$s$]')
    axes[i].set_ylabel('$V^T$ [$m^3$]')

    axes[i].plot(t, tankvol[i][:, 0], 'o', label='$V^T$')
    axes[i].set_box_aspect(1)
    axes[i].legend()
    axes[i].set_xticks(t[0::2])
    fig.tight_layout()

# fig.savefig('rejectionmultitankvol.png')
plt.show()
