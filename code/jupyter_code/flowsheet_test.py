from model_structures import capture_column
import numpy as np
from CADETProcess.simulator import Cadet
import time

MW_GSK = 82358.0
load_vol_list=[1.98, 2.22, 2.68, 3.10, 3.56, 4.00]
wash_length_list=[8.0,8.0,8.0,8.0,12.0,12.0]
exp_number = 4

load_volume = load_vol_list[exp_number]
wash_volume = wash_length_list[exp_number]

c_load = list(np.array([1.32, .33, 1.01, 5.0])/MW_GSK)

column_params = {}

column_params['column_model'] = 'LRMP'

if column_params['column_model'] in ['GRM', 'LRMP']:
    column_params['kfilm'] = 1e-6
    column_params['D_particle'] = 7e-6
else:
    column_params['kfilm'] = 1e-8
    
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

process = capture_column.create_model('test', c_load, binding_params, column_params, process_params)

process_simulator = Cadet()

capture_sim_results = process_simulator.simulate(process)

capture_sim_outlet_t = capture_sim_results.solution.column.outlet.coordinates['time']
capture_sim_outlet_c = capture_sim_results.solution.column.outlet.solution