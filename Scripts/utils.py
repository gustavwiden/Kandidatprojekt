import os
import json
import numpy as np
import sund
import matplotlib.pyplot as plt


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


dataset_files = {
    'HV_PK_data': 'HV_PK_data.json',
    'HV_PD_data': 'HV_PD_data.json',
    'SLE_PK_data': 'SLE_PK_data.json',
    'SLE_PD_data': 'SLE_PD_data.json',
    'SLE_CLE_PK_validation_data': 'SLE_CLE_PK_validation_data.json',
    'HV_vs_SLE_plasma_PK_response_data': 'HV_vs_SLE_plasma_PK_response_data.json',
    'HV_vs_SLE_plasma_PD_response_data': 'HV_vs_SLE_plasma_PD_response_data.json',
    'SLE_IV_dose_skin_PD_response_data': 'SLE_IV_dose_skin_PD_response_data.json',
    'SLE_SC_dose_skin_PD_response_data': 'SLE_SC_dose_skin_PD_response_data.json'   
}

model_files = {
    'HV_model': 'HV_model.txt',
    'SLE_model_1': 'SLE_model_1.txt',
    'SLE_model_10': 'SLE_model_10.txt',
    'SLE_model_80': 'SLE_model_80.txt',
    'SLE_model_400': 'SLE_model_400.txt',
    'HV_model_high': 'HV_model_high.txt'
}

parameter_files = {
    'final_params': 'final_params.json',
    'acceptable_params_1': 'acceptable_params_PL_1.csv',
    'acceptable_params_10': 'acceptable_params_PL_10.csv',
    'acceptable_params_80': 'acceptable_params_PL_80.csv',
    'acceptable_params_400': 'acceptable_params_PL_400.csv'
}


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_data(*dataset_keys):
    loaded_data = {}
    
    for key in dataset_keys:
        if key not in dataset_files:
            raise ValueError(f"'{key}' is not a valid dataset. "
                             f"Available datsets are: {list(dataset_files.keys())}")
                             
        data_path = os.path.join(base_dir, 'Data', dataset_files[key])
        with open(data_path, "r") as f:
            loaded_data[key] = json.load(f)
        
    return loaded_data


def load_models(*model_keys):
    loaded_models = {}

    for key in model_keys:
        if key not in model_files:
            raise ValueError(f"'{key}' is not a valid model. "
                             f"Available models are: {list(model_files.keys())}")
                             
        model_path = os.path.join(base_dir, 'Models', model_files[key])
        
        relative_model_path = os.path.relpath(model_path)
        
        sund.install_model(relative_model_path)
        loaded_models[key] = sund.load_model(key)
        
    return loaded_models


def load_params(*param_keys):
    loaded_params = {}
    
    for key in param_keys:
        if key not in parameter_files:
            raise ValueError(f"'{key}' is not a valid parameter set. "
                             f"Available parameter sets are: {list(parameter_files.keys())}")
                             
        param_path = os.path.join(base_dir, 'Parameters', parameter_files[key])
        if param_path.endswith('.csv'):
            loaded_params[key] = np.loadtxt(param_path, delimiter=",").tolist()
        elif param_path.endswith('.json'):
            with open(param_path, "r") as f:
                loaded_params[key] = json.load(f)
        else:
            raise ValueError(f"Unsupported file format for '{key}'. Only .csv and .json are supported.")
        
    return loaded_params


def create_simulation_objects(dataset, model, model_key, bodyweight):
    simulation_objects = {}
    
    for dose_key, data in dataset.items():
        act = sund.Activity(time_unit='h')
        
        # Add IV input if it exists
        if 'IV_in' in data['input']:
            act.add_output(
                'piecewise_constant', "IV_in",  
                t=data['input']['IV_in']['t'],  
                f=bodyweight * np.array(data['input']['IV_in']['f'])
            )
            
        # Add SC input if it exists
        if 'SC_in' in data['input']:
            act.add_output(
                'piecewise_constant', "SC_in",  
                t=data['input']['SC_in']['t'],  
                f=np.array(data['input']['SC_in']['f'])
            )
        
        if model_key == 'SLE' and "HV" in dose_key:
            final_key = dose_key.replace("HV", "SLE")
        else:
            final_key = dose_key

        simulation_objects[final_key] = sund.Simulation(models=model, activities=act, time_unit='h')
        
    return simulation_objects


def save_plot(save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    
    svg_path = os.path.join(save_dir, f"{filename}.svg")
    plt.savefig(svg_path, format='svg')
    
    png_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(png_path, format='png', dpi=600)
    
    plt.close()


def simulate(params, sim, time_vector, feature_to_plot):
    feature_idx = sim.feature_names.index(feature_to_plot)

    sim.simulate(time_vector = time_vector, parameter_values = params, reset = True)
    y_sim = sim.feature_data[:, feature_idx]

    return y_sim


def calculate_uncertainty(sim, time_vector, acceptable_params, patient_type, feature_to_plot):
    y_min = np.full_like(time_vector, 100000)
    y_max = np.full_like(time_vector, -100000)

    for params in acceptable_params:
        if patient_type == 'HV':
            adjusted_params = np.delete(params.copy(), [11, 16]) 
        elif patient_type == 'SLE':
            adjusted_params = np.delete(params.copy(), [10, 15])

        try:
            y_sim = simulate(adjusted_params, sim, time_vector, feature_to_plot)
            y_min = np.minimum(y_min, y_sim)
            y_max = np.maximum(y_max, y_sim)
            
        except RuntimeError as e:
            if "CV_ERR_FAILURE" in str(e) or "CVODE" in str(e):
                continue
            else:
                raise e
                
    return y_min, y_max


def get_response_time(y_data, response_threshold, startpoint, time_weeks, data_type):

    if data_type == 'PK':
        idx = np.where((time_weeks > startpoint) & (y_data < response_threshold))[0]

        return round(float(time_weeks[idx[0]]), 2) if len(idx) > 0 else np.nan
    elif data_type == 'PD':
        suppression_start_idx = np.where(y_data < response_threshold)[0] 

        if len(suppression_start_idx) == 0:
            return 0 
        
        suppression_end_idx = np.where((y_data > response_threshold) & (time_weeks >= time_weeks[suppression_start_idx[0]]))[0]

        return round(float(time_weeks[suppression_end_idx[0]]), 2) if len(suppression_end_idx) > 0 else np.nan
    raise ValueError("data_type must be either 'PK' or 'PD'")