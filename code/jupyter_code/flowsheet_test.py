from model_structures import capture_column
import numpy as np
from CADETProcess.simulator import Cadet
from CADETPythonSimulator.solver import Solver
# from cell_lysate_filtration import dead_end_filter
import time

def simulate_iex(column_model, simulator, c_load, iex_params):
    
    process = column_model.create_model('test', c_load, *iex_params)

    process_simulator = Cadet()

    capture_sim_results = simulator.simulate(process)

    capture_sim_outlet_t = capture_sim_results.solution.column.outlet.coordinates['time']
    capture_sim_outlet_c = capture_sim_results.solution.column.outlet.solution
    return capture_sim_outlet_t, capture_sim_outlet_c

def simulate_DEF(filter_params, inlet_profile):
    # V, c = dead_end_filter
    # return V, c
    return 27, 9.300857233055684e-05

if __name__ == '__main__':

    # Define dead end filter parameters
    MW_protein = 82358.0
    MWs = MW_protein*np.array([1500, 1000, 1]) 

    c_permeate = simulate_DEF(0, 0)

    # Define ion-exchange parameters
    exp_number = 4
    c_distribution = np.array([0.17232376, 0.04308094, 0.13185379, 0.65274151])
    load_vol_list=[1.98, 2.22, 2.68, 3.10, 3.56, 4.00]
    wash_length_list=[8.0,8.0,8.0,8.0,12.0,12.0]
    load_volume = load_vol_list[exp_number]  # Should come from the DEF simulation
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

    iex_params = [column_params, binding_params, process_params]

    c_load = c_distribution*c_permeate

    t_iex, c_iex = simulate_iex(c_load, iex_params)