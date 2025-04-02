from VF_Surrogate import * 
from fns import create_point_cloud, visualize_clouds




my_vfarm = VF_Surrogate(remove_mean=True,
                        scale_type='std',
                        test_num=0,
                        features_of_interest=['T', 'CO2', 'RH'],
                        reduced_dimension = 3)
my_vfarm.plot_split_data()

# %%


my_vfarm.perform_dimensionality_reduction()

rng = np.random.default_rng(my_vfarm.random_seed)
num_points = rng.integers(my_vfarm.n_points, size=15000)

my_vfarm.plot_dataset_XYZ(num_points=num_points, normalized=False)


for scale in ['std', 'pareto']:
# for scale in ['std', 'max', 'range', 'pareto', None]:

    my_vfarm = VF_Surrogate(remove_mean=True, 
                            scale_type=scale, 
                            test_num=0,
                            features_of_interest=['T', 'CO2', 'RH'])
    
    my_vfarm.perform_dimensionality_reduction()

    my_vfarm.plot_dataset_PCA(num_points=num_points, normalized=False)
    plt.gcf().suptitle(f'Scaling {scale}')


plt.show()

# %% We choose pareto


my_vfarm = VF_Surrogate(remove_mean=True, 
                        scale_type='pareto', 
                        test_num=0,
                        features_of_interest=['T', 'CO2', 'RH'])

my_vfarm.perform_dimensionality_reduction()


my_vfarm.plot_Lambda()


plt.show()


# %% Select 2 reduced dimensions
my_vfarm.reduced_dimension = 2


error_val, error_test = [], []
# regressors = ['poly2', 'gpr']
regressors = ['linear', 'poly2','poly3','poly5', 'svr', 'gpr', 'spline', 'ridge']

for regressor in regressors:
    my_vfarm.regressor_type = regressor
    my_vfarm.train_surrogate_model()

    results = my_vfarm.evaluate_model() 
    error_val.append(results['train_rms'])
    error_test.append(results['test_rms'])

    # Visualize
    # if regressor in ['linear', 'poly2', 'gpr']:
    #     my_vfarm.plot_model_predictions(**results)


# Compare overall prediction 
VF_Surrogate.plot_violins(error_val, error_test, regressors)

plt.show()


#Test the chosen surrogate
my_vfarm = VF_Surrogate(remove_mean=True, 
                        scale_type='pareto', 
                        test_num=0,
                        reduced_dimension=2, 
                        features_of_interest=['T', 'CO2', 'RH'],
                        regressor_type='poly2')

my_vfarm.perform_dimensionality_reduction()
my_vfarm.train_surrogate_model()
results = my_vfarm.evaluate_model() 

test_id = 0
A = results['A_test_ref'][test_id]
A = A[np.newaxis, ...]
feat_predict = my_vfarm._reconstruct(A=A, reshape=True)
feat_ref = my_vfarm.recover_original_shape(my_vfarm.test_data[:, test_id])


# Visualize clouds
pc = create_point_cloud(xyz=my_vfarm.xyz)

visualize_clouds([pc, pc], [feat_ref, feat_predict], _idx=my_vfarm.features_of_interest.index('T'))



