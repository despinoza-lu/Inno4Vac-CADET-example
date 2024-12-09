from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import GeneralizedIonExchange, StericMassAction, MobilePhaseModulator
from CADETProcess.processModel import Inlet, GeneralRateModel, LumpedRateModelWithPores, LumpedRateModelWithoutPores, Outlet
import numpy as np
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

# Column model can be either the General Rate model, the Lumped Rate moedl with pores, or the Lumped Rate model without pores.
column_model_dict = {
    'GRM': GeneralRateModel,
    'LRMP': LumpedRateModelWithPores,
    'LRM': LumpedRateModelWithoutPores
}

# Binding model can be either the Generalized Ion Exchange model, or the Mobile Phase Modulator Langmuir model. 
# Note that the Mobile Phase Modulator Langmuir model is in practice used as a Multiple Component Langmuir Model (without salt dependence) with pH dependence on the adsorption rate constant.

binding_model_dict = {
    'GIEX': GeneralizedIonExchange,
    'MPLM': MobilePhaseModulator
}


def create_model(
        name,
        c_load,
        binding_params,
        column_params,
        process_params):

    ## Component System
    component_system = ComponentSystem()
    component_system.add_component('H+')
    
    component_system.add_component('A')
    component_system.add_component('B')
    component_system.add_component('C')
    component_system.add_component('D')

    ## Binding Model
    binding_model_name = binding_params['binding_model']
    binding_model = binding_model_dict[binding_model_name](component_system, name=binding_model_name)
    binding_model.is_kinetic = True
    
    kkin1, kkin2, kkin3, kkin4 = binding_params['kkin']
    Keq1, Keq2, Keq3, Keq4 = binding_params['Keq']
    kfilm = column_params['kfilm']
    kpHref = binding_params['kpHref']
    
    binding_model.adsorption_rate = [0.0,  Keq1*kkin1, Keq2*kkin2, Keq3*kkin3, Keq4*kkin4]
    binding_model.desorption_rate = [0.0,  kkin1, kkin2, kkin3, kkin4]
    
    binding_model.ion_exchange_characteristic = [0.0]*component_system.n_comp
    binding_model.hydrophobicity = [0.0, kpHref, kpHref, kpHref, kpHref]
    
    binding_model.capacity = [.75e-2] + binding_params['qmax']
    
    #pH values
    pH_equi, pH_wash, pH_elu, pH_strip = process_params['pH']
    
    #Normalization of pH values with the equilibration pH
    pH_equi_norm = pH_equi/pH_equi
    pH_wash_norm = pH_wash/pH_equi
    pH_elu_norm = pH_elu/pH_equi
    pH_strip_norm = pH_strip/pH_equi
     
    # A second-degree polynomial term for the pH dependence on the adsoprtion rate is defined here:
    kpH1 = binding_params['kpH1']
    kpH2 = binding_params['kpH2']

    pHterm_equi = (kpH1/kpHref)*pH_equi_norm + (kpH2/kpHref)*pH_equi_norm**2
    pHterm_load = (kpH1/kpHref)*pH_equi_norm + (kpH2/kpHref)*pH_equi_norm**2
    pHterm_rinse = (kpH1/kpHref)*pH_equi_norm + (kpH2/kpHref)*pH_equi_norm**2
    pHterm_wash = (kpH1/kpHref)*pH_wash_norm + (kpH2/kpHref)*pH_wash_norm**2
    pHterm_elu = (kpH1/kpHref)*pH_elu_norm + (kpH2/kpHref)*pH_elu_norm**2
    pHterm_strip = (kpH1/kpHref)*pH_strip_norm + (kpH2/kpHref)*pH_strip_norm**2
    
    ## Inlets
    
    equilibration = Inlet(component_system, name='equilibration')
    equilibration.c = [pHterm_equi] + (component_system.n_comp-1)*[0.0]

    load = Inlet(component_system, name='load')
    load.c = [pHterm_load] + c_load

    rinse = Inlet(component_system, name='rinse')
    rinse.c = [pHterm_rinse] + (component_system.n_comp-1)*[0.0]

    wash = Inlet(component_system, name='wash')
    wash.c = [pHterm_wash] + (component_system.n_comp-1)*[0.0]

    elution = Inlet(component_system, name='elution')
    elution.c = [pHterm_elu] + (component_system.n_comp-1)*[0.0]

    stripping = Inlet(component_system, name='stripping')
    stripping.c = [pHterm_strip] + (component_system.n_comp-1)*[0.0]
    
    ## Column system
    column = column_model_dict[column_params['column_model']](component_system, name='column')
    column.binding_model = binding_model

    column.length = column_params['L']  # m
    column.diameter = column_params['D']  # m
    column.axial_dispersion = column_params['Dax'] #m2/s
    column.discretization.ncol = column_params['N']
    bed_porosity = column_params['eps_c']
    particle_porosity = column_params['eps_p'] # assumed  
    
    if column_params['column_model'] in ['GRM', 'LRMP']:
        column.bed_porosity = bed_porosity
        column.particle_radius = column_params['D_particle'] #m
        column.particle_porosity = particle_porosity   
        column.film_diffusion = component_system.n_comp*[kfilm] #m/s
        if column_params['column_model'] == 'GRM':
            if type(kfilm) == list:
                column.pore_diffusion = [1e-8] + kfilm
            else:
                column.pore_diffusion = component_system.n_comp*[kfilm]
            column.surface_diffusion = (component_system.n_comp-1)*[0.0]
    else:
        column.total_porosity = bed_porosity + (1-bed_porosity)*particle_porosity

    column.c = equilibration.c[:, 0].tolist()
    column.q = component_system.n_comp*[0.0]

    outlet = Outlet(component_system, name='outlet')

    # Setup Flow Sheet
    flow_sheet = FlowSheet(component_system)

    flow_sheet.add_unit(equilibration)
    flow_sheet.add_unit(load)
    flow_sheet.add_unit(rinse)
    flow_sheet.add_unit(wash)
    flow_sheet.add_unit(elution)
    flow_sheet.add_unit(stripping)

    flow_sheet.add_unit(column)

    flow_sheet.add_unit(outlet)

    flow_sheet.add_connection(equilibration, column)
    flow_sheet.add_connection(load, column)
    flow_sheet.add_connection(rinse, column)
    flow_sheet.add_connection(wash, column)
    flow_sheet.add_connection(elution, column)
    flow_sheet.add_connection(stripping, column)

    flow_sheet.add_connection(column, outlet)

    # Setup Process
    process = Process(flow_sheet, name)

    ## Create Events and Durations
    # Load
    u_load = process_params['flowrate'][0]    # cm/h
    u_load_SI = u_load*1e-2/3600    # m/sflow_sheet
    Q_load_SI = column.cross_section_area*u_load_SI

    CV_load = process_params['volume'][0]
    t_load = CV_load*column.volume/Q_load_SI

    process.add_event('load_start', 'flow_sheet.load.flow_rate', Q_load_SI, time=0.0)
    process.add_event('load_stop', 'flow_sheet.load.flow_rate', 0.0)
    process.add_duration('load', t_load)    # pseudo event
    process.add_event_dependency('load_stop', ['load_start', 'load'])

    # Rinse
    u_rinse = process_params['flowrate'][1]    # cm/h
    u_rinse_SI = u_rinse*1e-2/3600    # m/s
    Q_rinse_SI = column.cross_section_area*u_rinse_SI

    CV_rinse = process_params['volume'][1] 
    t_rinse = CV_rinse*column.volume/Q_rinse_SI

    process.add_event('rinse_start', 'flow_sheet.rinse.flow_rate', Q_rinse_SI)
    process.add_event_dependency('rinse_start', 'load_stop')

    process.add_event('rinse_stop', 'flow_sheet.rinse.flow_rate', 0.0)
    process.add_duration('rinse', t_rinse)
    process.add_event_dependency('rinse_stop', ['rinse_start', 'rinse'])

    # Wash
    u_wash = process_params['flowrate'][2]    # cm/h
    u_wash_SI = u_wash*1e-2/3600    # m/s
    Q_wash_SI = column.cross_section_area*u_wash_SI
    CV_wash = process_params['volume'][2]
    t_wash = CV_wash*column.volume/Q_wash_SI

    process.add_event('wash_start', 'flow_sheet.wash.flow_rate', Q_wash_SI)
    process.add_event_dependency('wash_start', 'rinse_stop')
    process.add_event('wash_stop', 'flow_sheet.wash.flow_rate', 0.0)
    process.add_duration('wash', t_wash)
    process.add_event_dependency('wash_stop', ['wash_start', 'wash'])

    # Elution
    u_elution = process_params['flowrate'][3]    # cm/h
    u_elution_SI = u_elution*1e-2/3600    # m/s
    Q_elution_SI = column.cross_section_area*u_elution_SI

    CV_elution = process_params['volume'][3] #adjusted to fit the experimental data. For some reason, the experimental elution phase seems to be longer than 6 CV.
    t_elution = CV_elution*column.volume/Q_elution_SI

    process.add_event('elution_start', 'flow_sheet.elution.flow_rate', Q_elution_SI)
    process.add_event_dependency('elution_start', 'wash_stop')
    process.add_event('elution_stop', 'flow_sheet.elution.flow_rate', 0.0)
    process.add_duration('elution', t_elution)
    process.add_event_dependency('elution_stop', ['elution_start', 'elution'])

    # Stripping
    u_stripping = process_params['flowrate'][4]    # cm/h
    u_stripping_SI = u_stripping*1e-2/3600    # m/s
    Q_stripping_SI = column.cross_section_area*u_stripping_SI
    CV_stripping = process_params['volume'][4]
    t_stripping = CV_stripping*column.volume/Q_stripping_SI

    process.add_event('stripping_start', 'flow_sheet.stripping.flow_rate', Q_stripping_SI)
    process.add_event_dependency('stripping_start', 'elution_stop')
    process.add_event('stripping_stop', 'flow_sheet.stripping.flow_rate', 0.0)
    process.add_duration('stripping', t_stripping)
    process.add_event_dependency('stripping_stop', ['stripping_start', 'stripping'])

    process.cycle_time = process.event_times[-1]
    process.flow_sheet.column.solution_recorder.write_solution_particle = True
    process.flow_sheet.column.solution_recorder.write_solution_solid = True
    return process
