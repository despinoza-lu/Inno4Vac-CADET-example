from model_structures.capture_column import create_model
from model_structures.dead_end_filter import dead_end_filter
import numpy as np
from CADETProcess.simulator import Cadet
from CADETPythonSimulator.solver import Solver
import time
import matplotlib.pyplot as plt

def simulate_iex(c_load, iex_params, plot=False):
    
    process = create_model('test', c_load, *iex_params)

    process_simulator = Cadet()

    capture_sim_results = process_simulator.simulate(process)

    capture_sim_outlet_t = capture_sim_results.solution.column.outlet.coordinates['time']
    capture_sim_outlet_c = capture_sim_results.solution.column.outlet.solution
    if plot:
        fig, ax = plt.subplots()
        pH_ax = ax.twinx()
        lines = ax.plot(capture_sim_outlet_t, capture_sim_outlet_c[:, 1:])
        labels = ['Product', 'Impurity 1', 'Impurity 2', 'Displacer component']
        ax.legend(lines, labels)
        pH_ax.plot(capture_sim_outlet_t, capture_sim_outlet_c[:, 0], color='grey')
        ax.set_ylim(top=.0002)
        plt.show(block=False)
    return capture_sim_outlet_t, capture_sim_outlet_c

def simulate_DEF(filter_parameters, component_parameters, plot=False):
    V, c = dead_end_filter(filter_parameters, component_parameters, plot=plot)
    return V, c

if __name__ == '__main__':
    simulate_chrom = True
    #%% Define dead end filter parameters
    filter_area = .1  # m^2
    delta_P = .75e5  # P, pressure drop using water at 500 l/h
    dyn_visc = 1.05e-3  # P.s
    Q_deltaP = 500 * 1e-3/3600  # m^3/s, testing flow rate in data sheet
    R_m = filter_area*delta_P/dyn_visc/Q_deltaP  # Filter resistance according to Sartobran .2 micron data sheet
    filtration_flowrate = 10*1e-3/3600  # m^3/s, operating flowrate

    concentrations = np.array([1.87916150e-12, 1.87916150e-12, 9.30085723e-05])  # mol/m^3
    rho_water = 998  # kg*m^-3
    rho_debris = 1.1*rho_water
    densities = [rho_debris]*3 + [rho_water]
    cake_resistances = [1e15, 1e15, 0, 0]


    product_MW = 82358.0
    molecular_weights = list(product_MW*np.array([1500, 1000, 1])) + [18]  # kg/mol
    volume_to_filter = 28000e-6  # m^3

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

    print(f'Simulating cell clarification at a constant flowrate of {filtration_flowrate} m^3/s')
    V_permeate, c_permeate = dead_end_filter(DEF_parameters, component_parameters, plot=True)
    print('Dead end filter simulation complete.')
    print(f'Permeate protein concentration: {c_permeate} mol/l.')

    #%% Define ion-exchange parameters
    exp_number = 4
    c_distribution = np.array([0.17232376, 0.04308094, 0.13185379, 0.65274151])
    load_vol_list=[1.98, 2.22, 2.68, 3.10, 3.56, 4.00]
    wash_length_list=[8.0,8.0,8.0,8.0,12.0,12.0]
    load_volume = load_vol_list[exp_number]
    wash_volume = wash_length_list[exp_number]

    column_params = {}
    column_params['column_model'] = 'LRMP'
    column_params['kfilm'] = 1e-6
    column_params['D_particle'] = 7e-6
    column_params['L'] = 100e-3
    column_params['D'] = 8e-3
    column_params['Dax'] = 2.0e-7
    column_params['N'] = 30
    column_params['eps_c'] = 0.4
    column_params['eps_p'] = 0.8

    binding_params = {}
    binding_params['binding_model'] = 'MPLM'
    binding_params['kkin'] = [2e-1]*4
    binding_params['Keq'] = [1.6045e-2, 2.6862e-2, 3.5860e-1, 1.8854e0]
    binding_params['qmax'] = [.75e-2, .75e-2, .18e-2, .75e-2]
    binding_params['kpH1'] = 0.0
    binding_params['kpH2'] = 22.0
    binding_params['kpHref'] = 1.0

    process_params = {}
    process_params['flowrate'] = [100]*5 # cm/h
    process_params['volume'] = [load_volume, 5.0 + 1.0, wash_volume, 6.4, 6]
    process_params['pH'] = [8.5, 7.1, 6.3, 4.0]

    iex_params = [binding_params, column_params, process_params]

    c_load = c_distribution*c_permeate
    if simulate_chrom:
        t_iex, c_iex = simulate_iex(list(c_load), iex_params, plot=True)
        print('Capture column simulation complete.')