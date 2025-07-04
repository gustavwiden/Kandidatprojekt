import sys
import json
import csv
import random
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import chi2
from scipy.optimize import dual_annealing
from scipy.optimize import minimize

# Define the data---------------------------------------------------
data = {
    'SBP' : {
    'time': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    'mean': [121.5424, 122.4240, 126.0144, 128.5636, 131.5991, 134.8431, 137.6694, 139.2801, 141.8642, 144.0324, 143.8378],
    'SEM': [2.6395, 2.6136, 3.4086, 3.6717, 4.4374, 4.6315, 4.7206, 4.6512, 4.8691, 5.6109, 5.0219]
    },
    'DBP' : {
    'time': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    'mean': [77.9434, 79.1608, 81.4643, 82.3094, 82.4701, 82.2884, 81.3182, 79.4995, 77.6811, 76.1601, 73.9103],
    'SEM': [1.5577, 1.4522, 1.8223, 1.7906, 2.1005, 2.0920, 1.9503, 1.6569, 1.4524, 1.6829, 1.7161]
    }
}

# Define the model------------------------------------------------------------
def bp_model(state, t, param):

    # Define the states
    SBP = state[0]
    DBP = state[1]

    # Define the parameter values  
    k1_SBP = param[0]
    k2_SBP = param[1]
    k1_DBP = param[2]
    k2_DBP = param[3]
    bSBP   = param[4]
    bDBP   = param[5]
    SBP0   = param[6]
    DBP0   = param[7]

    # Define the model variables
    MAP = DBP + ((SBP-DBP)/3)
    age = t # time is the age of the patient

    # Define the ODEs
    ddt_SBP = (k1_SBP + k2_SBP*age)*((SBP0-bSBP)/(117.86-bSBP))
    ddt_DBP = (k1_DBP + k2_DBP*age)*((DBP0-bDBP)/(75.8451-bDBP))

    return[ddt_SBP, ddt_DBP]

print('All content of data:')
print(data)
print('\nKeys in data are:')
print(data.keys())
print('\nThe content of the "time" key of the SBP dictionary:')
print(data['SBP']["time"])

def plot_data(data):
    for i in data.keys(): # loop over the keys of the dictionary
        plt.errorbar(data[i]['time'], data[i]['mean'], data[i]['SEM'], linestyle='None',marker='o', label=i) #for each key, plot the mean and SEM as errorbars at each time point
    plt.legend() # add a legend to the plot
    plt.xlabel('Time (min)') # add a label to the x-axis
    plt.ylabel('Blood pressure (mmHg)') # add a label to the y-axis

plot_data(data) # use the above-defined function to plot the data

def plot_simulation(model, param, ic, t, state_to_plot=[0,1], line_color='b'):
    sim = odeint(model, ic, t, (param,)) # Simulate the model for the initial parameters in the model file
    state_names = ['SBP simulation', 'DBP simulation'] # Define the names of the states
    colours = ['b', 'r'] # Define the colours of the lines
    for i in range(np.shape(sim)[1]): # loop over the number of columns in the sim array
        plt.plot(t, sim[:,state_to_plot[i]], label=state_names[i], color = colours[i]) # The data corresponds to Rp which is the second state

ic = [115, 60] #[SBP(0) = 115, DBP(0) = 60]
initial_parameterValues = [1, 0, 1, 0, 100, 50, ic[0], ic[1]] #[k1_SBP = 1, k2_SBP = 0, k1_DBP = 1, k2_DBP = 0, bSBP = 100, bDBP = 50]
t_year = np.arange(30, 81, 1)

#plot_simulation(bp_model, initial_parameterValues, ic, t_year)
#plot_data(data)


# Define an objective function 
def objectiveFunction(parameterValues, model, data):
    t =  data['SBP']["time"]
    ic = parameterValues[6:8]
    sim = odeint(model, ic, t, (parameterValues,)) # run a model simulation for the current parameter values

    #--------------------------------------------------------------#
    ysim_SBP = sim[:,0]

    y_SBP = np.array(data['SBP']["mean"])
    sem_SBP = np.array(data['DBP']["SEM"])

    cost_SBP = np.sum(np.square((y_SBP - ysim_SBP) / sem_SBP))

    #--------------------------------------------------------------#
    ysim_DBP = sim[:,1]

    y_DBP = np.array(data['DBP']["mean"])
    sem_DBP = np.array(data['DBP']["SEM"])

    #---------------------------------------------------------------# 
    cost_DBP = np.sum(np.square((y_DBP - ysim_DBP) / sem_DBP))

    cost = cost_SBP + cost_DBP
    return cost

sim = odeint(bp_model, ic, t_year, (initial_parameterValues,)) # Simulate the model for the initial parameters in the model file
print(sim)

ic_test = [115, 60] #[SBP(0) = 115, DBP(0) = 60]
parameterValues_test = [1, 0, 1, 0, 100, 50, ic_test[0], ic_test[1]] #[k1_SBP = 1, k2_SBP = 0, k1_DBP = 1, k2_DBP = 0, bSBP = 100, bDBP = 50]


#plot_simulation(bp_model, parameterValues_test, ic_test, t_year)
#plot_data(data)
param_startGuess = [1, 0, 1, 0, 100, 50, ic[0], ic[1]]
#parameter_OptBounds = np.array([(-1.0e6, 1.0e6)]*(len(param_startGuess)))
parameter_OptBounds = np.array([(-1.0e6, 1.0e6), (-1.0e6, 1.0e6), (-1.0e6, 1.0e6), (-1.0e6, 1.0e6),(50, 200),(20, 120),(70, 200),(50, 120)]) 
objFun_args = (bp_model, data)
niter = 0

def callback_fun(x,f,c):
    global niter
    if niter%1 == 0:
        print(f"Iter: {niter:4d}, obj:{f:3.6f}", file=sys.stdout)
    niter+=1

# Chi 2 test ---------------------------------------------------------------------
cost = objectiveFunction(parameterValues_test, bp_model, data)

dgf = len(data['SBP']["mean"])*2
chi2_limit = chi2.ppf(0.95, dgf) # Here, 0.95 corresponds to 0.05 significance level

print(f"Cost of the model is: {cost}")
print(f"The Chi2 limit is: {chi2_limit}")
if cost < chi2_limit:
    print("The model is not rejected!")
else:    
    print("The model is rejected!")

res_global = dual_annealing(func=objectiveFunction, bounds=parameter_OptBounds, args=objFun_args, x0=param_startGuess, callback=callback_fun) # This is the optimization
#print(res_global)
print(f"\nOptimized parameter values: {res_global['x']}\n")
print(f"Optimized cost: {res_global['fun']}")

print(f"chi2-limit: {chi2_limit}")

if res_global['fun'] < chi2_limit:
    print("The model is not rejected!")
else:    
    print("The model is rejected!")  

print(f"Cost of the model is: {cost}")
print(f"The Chi2 limit is: {chi2_limit}")
if cost < chi2_limit:
    print("The model is not rejected!")
else:    
    print("The model is rejected!")


fig2 = plt.figure()
plot_simulation(bp_model, res_global['x'], res_global['x'][6:8], t_year)
plot_data(data)

file_name = 'optimized_parameters.json'  # You can change the file name if you want to.

with open(file_name,'w') as file:#We save the file as a .json file, and for that we have to convert the parameter values that is currently stored as a ndarray into a traditional python list
    json.dump(res_global['x'].tolist(), file)

file_name = 'optimized_parameters.json' # Make sure that the file name is the same as the one you used to save the file

with open(file_name,'r') as file:
    optimal_ParameterValues = np.array(json.load(file))

res_global['x'] = optimal_ParameterValues # To ensure the following code will work as intended, We replace the parameter values with the ones we just loaded from the file

def target_distribution(x): # DEfine the target distribution you want to sample from, here we use the exponential of the objective function
    return np.exp(-objectiveFunction(x, bp_model, data))

def metropolis_hastings(target_dist, init_value, n_samples, sigma=1):
    samples = np.zeros((n_samples, len(init_value)))
    samples[0,:] = init_value
    burn_in = 100 # the burn-in represents the number of iterations to perform before adapting the proposal distribution
    dgf = len(data['SBP']["mean"])*2

    for i in range(1, n_samples):
        current_x = samples[i-1,:] # get the last sample
        if i < burn_in:# For the burn-in iterations, use a fixed candidate distribution, here a normal distribution.
            proposed_x = np.random.normal(current_x, 1) # calculate the next candidate sample (proposed_x) based on the previous sample.
        else: # After the burn-in period, use the covariance of past samples to determine the variance of the candidate distribution
            sigma = np.cov(samples[:i-1,:], rowvar=False)
            proposed_x = np.random.multivariate_normal(current_x, sigma) # calculate the next candidate as a random perturbation of the previous sample, with sigma based the covariance of all previous samples.

        acceptance_ratio = target_dist(proposed_x) / target_dist(current_x) # calculate the acceptance ratio, i.e the ratio of the target distribution at the proposed sample to the target distribution at the last sample

        # Accept or reject the candidate based on the acceptance probability. 
        # This probability is based on if the objective function value of the candidate vector in relation to the objective function value of the previous sample. 
        if np.random.rand() < acceptance_ratio:  # Accept proposal with probability min(1, acceptance_ratio)
             #if the proposed parameter values vector is accepted check if the corresponding model fit is acceptable. 
            if objectiveFunction(proposed_x, bp_model, data) < chi2.ppf(0.95, dgf): 
                current_x = proposed_x # if the model fit is acceptable, update the current parameter values to the proposed parameter values

        samples[i,:] = current_x # store the current parameter values in the samples array

    return samples

init_value = res_global['x']
n_samples = 10000

# Running the Metropolis-Hastings algorithm
samples = metropolis_hastings(target_distribution, init_value, n_samples)

## Plot the results
parameter_names = ['k1_SBP', 'k2_SBP', 'k1_DBP', 'k2_DBP', 'bSBP', 'bDBP', 'SBP0', 'DBP0']  # Define the names of the parameters

fig4, ax, = plt.subplots(2,4)
for i in range(len(init_value)):
    ax[m.floor(i/4),i%4].hist(samples[:,i], bins=30, density=True, alpha=0.6, color='g')
    ax[m.floor(i/4),i%4].title.set_text(parameter_names[i])
    ax[m.floor(i / 4), i % 4].set_xlabel('Parameter value')  # Set x-label for each subplot

plt.suptitle('Metropolis-Hastings algorithm')

fig4.set_figwidth(20) # set the width of the figure
fig4.set_figheight(12) # set the height of the figure

with open('TBMT42_Lab2_MCMC_results.json','r') as file:
    MCMC_results = np.array(json.load(file))

fig4, ax, = plt.subplots(2,4)
for i in range(len(init_value)):
    ax[m.floor(i/4),i%4].hist(MCMC_results[:,i], bins=30, density=True, alpha=0.6, color='g')
    ax[m.floor(i/4),i%4].title.set_text(parameter_names[i])
    ax[m.floor(i / 4), i % 4].set_xlabel('Parameter value')  # Set x-label for each subplot

plt.suptitle('Full MCMC results')
fig4.set_figwidth(20)
fig4.set_figheight(12)

def objectiveFunction_reversePL(parameterValues, model, Data, parameterIdx, polarity, threshold):
    # Calculate cost with original objective function
    cost = objectiveFunction(parameterValues, model, Data)

    # Return the parameter value at index parameterIdx.
    # Multiply by polarity = {-1, 1} to swap between finding maximum and minimum parameter value.
    v = polarity * parameterValues[parameterIdx]

    # Check if cost with the current parameter values is over the chi-2 threshold
    if cost > threshold:
        # Add penalty if the solution is above the limit.
        # Penalty grows the more over the limit the solution is.
        v = abs(v) + (cost - threshold) ** 2

    return v

# Set up
param_startGuess = res_global['x']

#parameter_OptBounds = np.array([(-1.0e6, 1.0e6)]*(len(param_startGuess)))
parameter_OptBounds = np.array([(-1.0e6, 1.0e6), (-1.0e6, 1.0e6), (-1.0e6, 1.0e6), (-1.0e6, 1.0e6),(50, 200),(20, 120),(70, 200),(50, 120)]) 
parameter_bounds = np.zeros((len(param_startGuess),2))

dgf = len(data['SBP']["mean"])*2
threshold = chi2.ppf(0.95, dgf) # Here, 0.95 corresponds to 0.05 significance level 

# Running the actual algorithm 
# for parameterIdx in range(len(param_startGuess)):
#     for polarity in [-1, 1]:
#         objFun_args = (bp_model, data, parameterIdx, polarity, threshold)
#         niter = 0
#         res_revPL = dual_annealing(func=objectiveFunction_reversePL, bounds=parameter_OptBounds, args=objFun_args, x0=param_startGuess, callback=callback_fun) # This is the optimization
#         parameter_bounds[parameterIdx,max(polarity,0)] = res_revPL['fun']

# parameter_bounds[:,0] = abs(parameter_bounds[:,0])  # Make sure that the upper bounds are positive

# print("Estimated parameter bounds:")
# print(parameter_bounds)

# # Plot the results of the reverse profile likelihood analysis ----------------------------------------------------------

# fig4, ax, = plt.subplots(2,4)
# for i in range(len(parameter_bounds)):
#     ax[m.floor(i/4),i%4].hist(MCMC_results[:,i], bins=30, density=True, alpha=0.6, color='g')
#     ax[m.floor(i/4),i%4].plot([parameter_bounds[i,:],parameter_bounds[i,:]],[[0, 0],[ax[m.floor(i/4),i%4].get_ylim()[1], ax[m.floor(i/4),i%4].get_ylim()[1]]], color='r')
#     ax[m.floor(i/4),i%4].title.set_text(parameter_names[i])
#     ax[m.floor(i / 4), i % 4].set_xlabel('Parameter value')  # Set x-label for each subplot

# fig4.set_figwidth(20)
# fig4.set_figheight(12)

def objectiveFunction_fullPL(parameterValues, model, data,parameterIdx, PL_revValue):
    t =  data['SBP']["time"]
    ic = parameterValues[6:8]
    sim = odeint(model, ic, t, (parameterValues,)) # run a model simulation for the current parameter values

    ysim_SBP = sim[:,0]

    y_SBP = np.array(data['SBP']["mean"])
    sem_SBP = np.array(data['DBP']["SEM"])

    cost_SBP = np.sum(np.square((y_SBP - ysim_SBP) / sem_SBP))
    #--------------------------------------------------------------#
    ysim_DBP = sim[:,1]

    y_DBP = np.array(data['DBP']["mean"])
    sem_DBP = np.array(data['DBP']["SEM"])

    cost_DBP = np.sum(np.square((y_DBP - ysim_DBP) / sem_DBP))
    cost = cost_SBP + cost_DBP

    # --------------------------------------------------------------
    cost = cost + 1e6*(parameterValues[parameterIdx]-PL_revValue)**2 # add very large penalty if the difference between a specific parameter value and a reference value is too large.

    return cost

param_optimalSolution = res_global['x']
parameterIdx = 6

nSteps = 10
step_size = 1

PL_revValue = np.zeros(nSteps*2)  
parameter_profile = np.zeros(nSteps*2)
count = 0
for direction in [-1, 1]:
    for step in range(nSteps):
        PL_revValue[count] = param_optimalSolution[parameterIdx] + direction*step_size*step

        objFun_args = (bp_model, data, parameterIdx, PL_revValue[count])
        niter = 0
        res_fullPL = dual_annealing(func=objectiveFunction_fullPL, bounds=parameter_OptBounds, args=objFun_args, x0=param_optimalSolution, callback=callback_fun) # This is the optimization

        parameter_profile[count] = res_fullPL['fun']
        count += 1

parameter_profile[0:nSteps] = parameter_profile[-(nSteps+1):-((2*nSteps)+1):-1]
PL_revValue[0:nSteps] = PL_revValue[-(nSteps+1):-((2*nSteps)+1):-1]

print("Parameter values:")
print(PL_revValue)

print("\nProfile values:")
print(parameter_profile)

fig6 = plt.figure()
plt.plot(PL_revValue, parameter_profile, linestyle='--', marker='o',label='Parameter profile', color='k')
plt.plot([PL_revValue[0],PL_revValue[-1]],[threshold, threshold], linestyle='--', color='r', label='Chi2 threshold')

plt.xlabel('Parameter value')
plt.ylabel('Objective function value')
plt.legend()

combined_array = np.concatenate((MCMC_results, parameter_bounds.T))
max_values = np.max(combined_array, axis=0)
min_values = np.min(combined_array, axis=0)

fig, axes = plt.subplots(2, 1, figsize=(8, 10))

for i in range(2):
    axes[i].bar(parameter_names[i*4:(i+1)*4], res_global['x'][i*4:(i+1)*4]) # Plot the parameter values as a bar plot
    axes[i].errorbar(parameter_names[i*4:(i+1)*4], res_global['x'][i*4:(i+1)*4], yerr=[res_global['x'][i*4:(i+1)*4] - min_values[i*4:(i+1)*4], max_values[i*4:(i+1)*4] - res_global['x'][i*4:(i+1)*4]], fmt='none', color='black') # Add error bars for the max and min values
    axes[i].set_xlabel('Parameters') # Set the x-axis label
    axes[i].set_ylabel('Parameter values') # Set the y-axis label
    axes[i].set_title('Optimized Parameter Values') # Set the title

plt.tight_layout()

parameterValues = MCMC_results

t =  data['SBP']["time"]
N = 10000

SBP = np.zeros((len(t),N))
DBP = np.zeros((len(t),N))
MAP = np.zeros((len(t),N))

for i in range(N):

    ic = parameterValues[i,6:8]
    sim = odeint(bp_model, ic, t, (parameterValues[i,:],)) # run a model simulation for the current parameter values

    SBP[:,i] = sim[:,0]
    DBP[:,i] = sim[:,1]

    MAP[:,i] = DBP[:,i] + ((SBP[:,i]-DBP[:,i])/3)

simulation = {
    'SBP':{
        'max': np.max(SBP, axis=1),
        'min': np.min(SBP, axis=1),
    },
    'DBP':{
        'max': np.max(DBP, axis=1),
        'min': np.min(DBP, axis=1),
    },
    'MAP':{
        'max': np.max(MAP, axis=1),
        'min': np.min(MAP, axis=1),
    }
}

fig7 = plt.figure()
j = 0
colour = ['b', 'r', 'g']

for i in simulation.keys():
    plt.plot(t, simulation[i]['max'], color=colour[j])
    plt.plot(t, simulation[i]['min'], color=colour[j])
    plt.fill_between(t, simulation[i]['max'], simulation[i]['min'], color=colour[j], alpha=0.2 , label=f"{i} simulation")
    j += 1   

plot_data(data)
plt.legend(ncol=2)

ax = plt.gca()
ax.set_ylim([60, 160])

MAP_data = {'time': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
            'mean': [92.4847210238712, 93.5597018502541, 96.2593471383376, 97.7269676924552, 98.8733702425462, 99.7520611638385,99.9519461221359,99.4019628990509,98.9772073626690,98.5700675870003, 97.0024446361806], 
            'SEM': [1.93432101510279, 1.86110659090894, 2.33853890591003, 2.38057261815908, 2.89673286115572, 2.93682324557660, 2.87054566496065, 2.69052851501081, 2.59467058140641, 2.96438227816015, 2.76490945598723]
            }

plot_data(data)
plt.errorbar(MAP_data['time'], MAP_data['mean'], MAP_data['SEM'], linestyle='None',marker='o', label='MAP')

j = 0
for i in simulation.keys():
    plt.plot(t, simulation[i]['max'], color=colour[j])
    plt.plot(t, simulation[i]['min'], color=colour[j])
    plt.fill_between(t, simulation[i]['max'], simulation[i]['min'], color=colour[j], alpha=0.2 , label=f"{i} simulation")
    j += 1  

plt.legend(loc=2, ncol=2) # add a legend to the plot
ax = plt.gca()
ax.set_ylim([60, 160])

plt.show()