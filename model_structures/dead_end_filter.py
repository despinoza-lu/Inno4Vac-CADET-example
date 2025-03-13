import matplotlib.pyplot as plt
import numpy as np

from CADETPythonSimulator.distribution_base import DistributionBase, ConstantVolumeDistribution, ConstantConcentrationDistribution
from CADETPythonSimulator.unit_operation import DistributionInlet, Outlet, DeadEndFiltration
from CADETPythonSimulator.system import FlowSystem
from CADETPythonSimulator.solver import Solver
from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.rejection import StepCutOff
from CADETPythonSimulator.viscosity import LogarithmicMixingViscosity

def dead_end_filter(filter_parameters, component_parameters, plot=False):
    
    plot=True

    permeate_tank_volume = 1e-9  # m^3, intial volume in permeate tank
    flowrate_out_of_permeate_tank = 1e-9  # m^3/s

    time_to_filter = filter_parameters['volume_to_filter']/filter_parameters['flowrate']  # s

    # %%
    permvol = []
    cakevol = []
    pressure = []
    permflow = []

    tankvol = []
    tankcon = []

    component_system = CPSComponentSystem(
                        name="cell_lysate",
                        components=len(component_parameters['densities']),
                        pure_densities=component_parameters['densities'],
                        molecular_weights=component_parameters['molecular_weights'],
                        viscosities=component_parameters['viscosities'],
                        specific_cake_resistances=component_parameters['cake_resistances'],
                        )

    concentration_distribution = ConstantVolumeDistribution(component_system=component_system,
                                                            c=component_parameters['concentrations'])
    inlet = DistributionInlet(component_system=component_system, name="inlet")
    inlet.distribution_function = concentration_distribution

    rejectionmodell = StepCutOff(cutoff_weight=filter_parameters['MW_cutoff'])
    viscositymodell = LogarithmicMixingViscosity()
    filter_obj = DeadEndFiltration(
                        component_system=component_system,
                        name="deadendfilter",
                        rejection_model=rejectionmodell,
                        viscosity_model=viscositymodell,
                        membrane_area=filter_parameters['area'],
                        membrane_resistance=filter_parameters['filter_resistance'],
                        )

    outlet = Outlet(component_system=component_system, name="outlet")

    unit_operation_list = [inlet, filter_obj, outlet]

    system = FlowSystem(unit_operations=unit_operation_list)
    section = [
                {
                    'start': 0,
                    'end': time_to_filter,
                    'connections': [
                        [0, 1, 0, 0, filter_parameters['flowrate']],
                        [1, 2, 0, 0, flowrate_out_of_permeate_tank],
                    ],
                }
            ]

    system.initialize_state()

    condist = ConstantConcentrationDistribution(component_system=component_system,
                                                c=[0]*(len(component_parameters['densities']) - 1))
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
        axes.set_xlabel('$t$ [$s$]')
        axes.set_ylabel('$\Delta P$ [$Pa$]')
        axes.set_title('Pressure drop')

        axes.plot(t, pressure[0][:, 0], 'o', color='blue', label='$\Delta P$')
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

        axes.plot(t, tankvol[0][:, 0]*1e3, 'o', label='$V^T$')
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
        ax_c_0_twin.set_ylim(bottom=0, top=70)

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
        ax_c[2].set_xlabel('Time [s]')
        ax_c_2_twin.set_ylim(bottom=0, top=70)

        labels = ['Debris 1', 'Debris 2', 'Proteins', 'Water']
        fig_c.legend(lines + water_line, labels, loc='outside right center', ncols=1, frameon=False)
        fig_c.tight_layout()
        fig_c.subplots_adjust(right=0.65)

        # fig.savefig('rejectionmultitankvol.png')
        plt.show(block=False)

    permeate_volume = solver.unit_solutions['deadendfilter']['permeate_tank']['volume']['values'][-1]
    permeate_protein_concentration = solver.unit_solutions['outlet']['inlet']['c']['values'][-1, 2]

    return permeate_volume, permeate_protein_concentration

if __name__ == '__main__':

    filter_area = .1  # m^2
    delta_P = .75e5  # P, pressure drop using water at 500 l/h
    dyn_visc = 1.05e-3  # P.s
    Q_deltaP = 500 * 1e-3/3600  # m^3/s, testing flow rate in data sheet
    R_m = filter_area*delta_P/dyn_visc/Q_deltaP  # Filter resistance according to Sartobran .2 micron data sheet
    filtration_flowrate = 1*1e-3  # m^3/s, operating flowrate

    concentrations = np.array([1.87916150e-12, 1.87916150e-12, 9.30085723e-05])*1e-3  # mol/m^3
    rho_water = 998  # kg*m^-3
    rho_debris = 1.1*rho_water
    densities = [rho_debris]*3 + [rho_water]
    cake_resistances = [1e11, 1e11, 0, 0]

    molecular_weights = list(82358.0e-3*np.array([1500, 1000, 1])) + [18e-3]  # kg/mol
    volume_to_filter = 27819e-6  # m^3

    DEF_parameters = {'area': filter_area,
                      'filter_resistance': R_m,
                      'flowrate': filtration_flowrate,
                      'volume_to_filter': volume_to_filter,
                      'MW_cutoff': molecular_weights[2]*2
                      }
    component_parameters = {'concentrations': concentrations,
                            'densities': densities,
                            'cake_resistances': cake_resistances,
                            'molecular_weights': molecular_weights,
                            'viscosities': [np.nan, np.nan, np.nan, dyn_visc],
                            }

    V, c = dead_end_filter(DEF_parameters, component_parameters, plot=True)
    print(f'Permeate protein concentration: {c} mol/m^3')