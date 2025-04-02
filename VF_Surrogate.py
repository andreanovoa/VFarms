# %%

import os
import re
from turtle import title
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from copy import deepcopy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams["font.family"] = "Times New Roman"



class VF_Surrogate:

    features_of_interest = ['T', 'RH', 'CO2']

    Uin = 24. + np.arange(7) * 2
    options = dict(B=0., C=0.25, D=0.5, E=0.75, A=1.0)
    test_num = 0
    voxel_size = 0.08

    random_seed = 6
    regressor_type = 'linear'

    reduced_dimension = 2
    
    scale_type = 'range'
    remove_mean = True

    # These are the properties to be defined when running the PCA and surrogate
    _Phi = None
    _Lambda = None
    _Z = None
    Z_r = None
    A_train = None
    model = None
    train_score = None



    features_labels = ['P', 'RHO', 'U', 'UX', 'UY', 'UZ', 'T', 'WT', 'H2O', 'CO2', 'RH']
    regressor_type_options = ['linear', 'poly2', 'poly3', 'poly4', 'poly5', 'spline1', 'spline2','spline3', 'gpr', 'svr', 'ridge']
    scaling_options = ['std', 'max', 'range', 'pareto', None]

    
    def __init__(self, 
                 **kwargs):
        """
        Initialize the VF_Surrogate object with project settings.
        
        Parameters:
            folder (str): Folder containing the downsampled data.
                          If None, the folder is inferred from the current working directory.
            voxel_size (float): Voxel size for the downsampled data.
            normalize (bool): Whether to apply normalization.
            test_num (int): Selects which test-case split to use (0 or 1).
            random_seed (int): Seed for reproducible random sampling.
        """
        if 'folder' not in kwargs.keys():
            if 'NewData' in os.getcwd():
                self.folder = 'Downsampled/'
            else:
                self.folder = 'NewData/Downsampled/'
        else:
            self.folder = kwargs['folder']

        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                print(f'Attribute {key} not recognized')


        # ==== Load the data ======
        input_parameters, features_all, self.xyz = self.load_data()
        self.n_points = self.xyz.shape[0] 
        self.n_cases = len(input_parameters)

        # ==== Split the dataset into training and testing sets based on predefined cases ====
        
        if self.test_num == 0:
            extrapolation_cases = [(lam, 24) for lam in self.options.values()]
            interpolation_cases = [(self.options['C'], 28), (self.options['A'], 34),
                                   (self.options['E'], 26), (self.options['D'], 36)]
        elif self.test_num == 1:
            extrapolation_cases = [(lam, 36) for lam in self.options.values()]
            extrapolation_cases += [(self.options['A'], u) for u in self.Uin]
            interpolation_cases = [(lam, 32) for lam in self.options.values()][:-1]
            interpolation_cases += [(lam, 28) for lam in self.options.values()][:-1]
        else:
            extrapolation_cases, interpolation_cases = [], []

        self.extrapolation_cases = extrapolation_cases
        self.interpolation_cases = interpolation_cases

        test_cases = set(extrapolation_cases + interpolation_cases)
        test_indices = [input_parameters.index(case) for case in test_cases]
        train_indices = list(set(range(len(input_parameters))) - set(test_indices))

        self.train_data = features_all[:, train_indices]
        self.test_data = features_all[:, test_indices]

        self.train_input_parameters = [input_parameters[ii] for ii in train_indices]
        self.test_input_parameters = [input_parameters[ii] for ii in test_indices]

        # ==== Define the normalization from the training  cases ====

        self.feat_shift = np.zeros((features_all.shape[0], 1))
        self.feat_scale = np.ones((features_all.shape[0], 1))

        for ii, key in enumerate(self.features_of_interest):
            idx = [ii * self.n_points, (ii + 1) * self.n_points]
            feat = features_all[idx[0]:idx[1]].copy()

            # Define the normalization factors if requested
            if self.remove_mean:
                shift =  np.mean(feat, axis=1, keepdims=True)
                self.feat_shift[idx[0]:idx[1]] += shift

            if self.scale_type is not None:
                scale = VF_Surrogate._get_norm(feat - self.feat_shift[idx[0]:idx[1]], scale_type=self.scale_type, axis=1)
                if np.min(scale) < 1e-9:
                    scale += 1e-9

                self.feat_scale[idx[0]:idx[1]] *= scale


    def print_details(self):
        print("---------- Vertical Farm Surogate Model ---------------\n",
              f"Test {self.test_num}: Training on {len(self.train_input_parameters)} out of {self.n_cases} cases.\n",
              f"Quantities of interest: {[f'{feat}' for feat in self.features_of_interest]}\n"
              f"Dataset scaling and shift: {self.scale_type}, {self.remove_mean}\n",
              f"Number of latent dimensions: {self.reduced_dimension}\n",
              f"Regressor type: {self.regressor_type}\n",
              f"Score on training set: {self.train_score:.5}\n",
              )

    def copy(self):
        return deepcopy(self)

    def get_property(self, data=None, key='T', normalized=False):
        if data is None:
            data = self.train_data.copy()
        if normalized:
            data = self._normalize_data(data)
        ii = self.features_of_interest.index(key)
        if data.shape[0] == self.n_points:
            return data[:, ii]
        else:
            return data[ii * self.n_points:(ii + 1) * self.n_points]
        

    @staticmethod
    def _get_norm(ff, scale_type='range', axis=None):
        if scale_type == 'range':
            return np.max(ff, axis=axis, keepdims=True) - np.min(ff, axis=axis, keepdims=True)
        elif scale_type == 'max':
            return np.max(ff, axis=axis, keepdims=True)
        elif scale_type == 'std':
            return np.std(ff, axis=axis, keepdims=True)
        elif scale_type == 'pareto':
            return np.sqrt(np.std(ff, axis=axis, keepdims=True))
        else:
            raise NotImplementedError(f'scale_type {scale_type} not defined')

    # -----------------------------
    # Main Methods of the Class
    # -----------------------------
        
    def load_data(self):
        """
        Load the downsampled data, optionally normalize it, and extract features.
        """

        input_parameters = []
        features_all_list = []

        flag = True

        for case_option, lambda_sink in self.options.items():
            data = np.load(f'{self.folder}Option{case_option}-downsampled_{self.voxel_size}.npy')

            n_case, _, n_points = data.shape
            features_all = data[:, :-3]
            
            features = np.empty((n_points * len(self.features_of_interest), n_case))
            # if flag:
            #     xyz_feat = np.empty((n_points , 3))
            
            for ii, foi in enumerate(self.features_of_interest):
                feat = features_all[:,  VF_Surrogate.features_labels.index(foi)]
                features[ii * n_points:(ii + 1) * n_points, :] = feat.T

                if flag: 
                    xyz_feat = data[ii, -3:].T
                    flag = False

            features_all_list.append(features)
            input_parameters.extend([(lambda_sink, uin) for uin in self.Uin])
            

        features_all = np.concatenate(features_all_list, axis=1)

        return input_parameters, features_all, xyz_feat

    
    # -----------------------------
    # PCA-specific methods
    # -----------------------------

    def perform_dimensionality_reduction(self):
        """
        Perform PCA on the normalized training data and compute the POD coefficients.
        
        Parameters:
            r (int): Number of modes to retain.
        """
        # Normalize training data.

        X_n = self._normalize_data(self.train_data)

        _K = np.dot(X_n.T, X_n)
        self._Lambda, self._Phi = np.linalg.eig(_K)
        self._Z = (X_n @ self._Phi) / np.sqrt(self._Lambda + 1e-12)

        
        self.A_train = self._project_dataset(self.train_data)



    @property
    def Z_r(self):
        return self._Z[:, :self.reduced_dimension]



    def _normalize_data(self, X):
        if X.shape[0] == self.n_points:
            shift = self.recover_original_shape(self.feat_shift)
            scale = self.recover_original_shape(self.feat_scale)
            return  (X - shift) / scale
        else:
            return (X - self.feat_shift) / self.feat_scale   

    def _un_normalize_data(self, X):
        return (X * self.feat_scale) + self.feat_shift

    
    def _project_dataset(self, X):
        X_n = self._normalize_data(X)
        return np.dot(X_n.T, self.Z_r)


    def _reconstruct(self, A, reshape=False):

        if A.ndim == 1:
            A = A[np.newaxis, :]

        X_n = self.Z_r @ A.T 
        feat = self._un_normalize_data(X_n)

        if reshape:
            feat = self.recover_original_shape(feat)

        return feat
        


    # -----------------------------
    # Surrogate-specific methods
    # -----------------------------

    def train_surrogate_model(self):
        """
        Train a suite of surrogate regression models.
        """
        df_train = VF_Surrogate._format_input_params(self.train_input_parameters)
        preprocessor = VF_Surrogate._set_preprocessor(labels=['lambda_sink', 'Uin'])

        regressor = VF_Surrogate._set_regressor(reg_type=self.regressor_type)
        self.model = Pipeline(steps=[('preprocessor', preprocessor), 
                                ('regressor', regressor)])
        self.model.fit(df_train, self.A_train)
        self.train_score = self.model.score(df_train, self.A_train)
        print(f"Train score: {self.train_score:.4f}")

    @staticmethod
    def _format_input_params(params):
        return pd.DataFrame(params, columns=['lambda_sink', 'Uin'])
    
    @staticmethod
    def _set_regressor(reg_type='linear', multi=True):
        if reg_type == 'linear':
            reg = LinearRegression()
        elif reg_type == 'logistic':
            reg = LogisticRegression()
        elif 'poly' in reg_type:
            degree = ''.join(reg_type.split('poly'))
            degree = int(degree) if degree else 3
            reg = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                            ('linreg', LinearRegression())])
        elif 'spline' in reg_type:
            degree = ''.join(reg_type.split('spline'))
            degree = int(degree) if degree else 3
            reg = Pipeline([('spline', SplineTransformer(degree=degree, n_knots=5, 
                                                         include_bias=False, extrapolation='linear')),
                            ('linreg', LinearRegression())])
        elif reg_type == 'gpr':
            reg = GaussianProcessRegressor(n_restarts_optimizer=5)
        elif reg_type == 'svr':
            reg = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif reg_type == 'ridge':
            reg = Ridge(alpha=1.0, solver='saga')
        else:
            raise ValueError(f'Invalid regressor type {reg_type}')

        if multi:
            return MultiOutputRegressor(reg)
        else:
            return reg

    @staticmethod
    def _set_preprocessor(labels=['lambda_sink', 'Uin']):
        if labels[0] == 'option':
            t1 = OneHotEncoder(handle_unknown='ignore')
        else:
            t1 = StandardScaler()
        return ColumnTransformer(transformers=[('t1', t1,  [labels[0]]), 
                                               ('t2', StandardScaler(), [labels[1]])])


    def predict_features(self, input_params, reshape=False):

        df = VF_Surrogate._format_input_params(input_params)
        
        A_pred = self.model.predict(df)
        reconstructed_feat = self._reconstruct(A_pred, reshape=reshape)

        return A_pred, reconstructed_feat
    

    def recover_original_shape(self, features):
        return features.reshape(self.n_points, len(self.features_of_interest), -1, order='F')


    def evaluate_model(self):
        """
        Evaluate a single surrogate model using RMS error on both coefficient space and
        reconstructed features.

        """

        results = dict(A_train_ref=self.A_train, 
                       A_test_ref=self._project_dataset(self.test_data)
                       )

        for key in ['train', 'test']:
            in_params = getattr(self, f'{key}_input_parameters')
            feat_ref = getattr(self, f'{key}_data')
            A_ref = results[f'A_{key}_ref']

            A_pred, reconstructed_feat = self.predict_features(input_params=in_params)
            results[f'A_{key}_pred'] = A_pred

            rms_model = np.empty((A_pred.shape[0], 2))
            for jj in range(len(A_pred)):
                rms_model[jj, 0] = self._RMS(A_ref[jj], A_pred[jj])
                rms_model[jj, 1] = self._RMS(feat_ref[:, jj], reconstructed_feat[:, jj])
            
            results[f'{key}_rms'] = rms_model

        return results

    # -----------------------------
    # Plotting functions
    # -----------------------------


    @staticmethod
    def _model_title(model_name):
        if 'poly' in model_name:
            return f'Polynomial Regression order {model_name[-1]}'
        elif 'spline' in model_name:
            return f'Spline Regression order {model_name[-1]}'
        elif 'gpr' in model_name:
            return "Gaussian Process Regression"
        else:
            return model_name.capitalize() + ' Regression'

    @staticmethod
    def _RMS(y_true, y_pred):
        y_true, y_pred = [x.squeeze() for x in [y_true, y_pred]]
        rms_error = np.sqrt(np.mean((y_true - y_pred)**2))
        norm_true = np.sqrt(np.mean(y_true**2))
        return rms_error / (norm_true + 1e-8)


    def plot_model_predictions(self, A_test_pred, A_test_ref, **kwargs):
        """
        Plot 3D surfaces of the predicted POD coefficients along with training and test data.
        """

        df_train = self._format_input_params(self.train_input_parameters)
        df_test = self._format_input_params(self.test_input_parameters)
        # Y_test = self.model.predict(df_test)

        x2 = np.linspace(20, 38, 30)
        x1 = np.linspace(-0.1, 1.1, 30)
        X1, X2 = np.meshgrid(x1, x2)
        X_surf = np.column_stack([X1.ravel(), X2.ravel()])
        df_Xsurf = self._format_input_params(X_surf)
        Y_surf = self.model.predict(df_Xsurf)

        _, axs = plt.subplots(1, self.reduced_dimension, layout='constrained',
                              figsize=(4 * self.reduced_dimension, 4), subplot_kw={"projection": "3d"})
        for ii, ax in enumerate(axs):
            ax.plot_surface(X1, X2, Y_surf[:, ii].reshape(X1.shape), cmap='viridis', alpha=0.4)
            ax.scatter(df_train.values[:, 0], df_train.values[:, 1], self.A_train[:, ii],
                        color='k', s=20, label="Training data")
            ax.scatter(df_test.values[:, 0], df_test.values[:, 1], A_test_ref[:, ii],
                        color='w', marker='*', s=20, label="Test data", edgecolor='k')
            ax.scatter(df_test.values[:, 0], df_test.values[:, 1], A_test_pred[:, ii],
                        color='r', s=50, alpha=.5, label="Test prediction")
            ax.set(xlabel="$\lambda_{sink}$", ylabel="$U_{in}$", zlabel=f"$\\alpha_{{{ii+1}}}$")
            ax.legend(frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(-0.01, 1.))
        plt.suptitle(VF_Surrogate._model_title(self.regressor_type), fontsize=16)


    @staticmethod
    def plot_violins(validation_errors, test_errors, models):
        """
        Create violin plots of the RMS error distributions.
        """

        if isinstance(validation_errors, list):

            validation_errors = np.array(validation_errors)
            test_errors = np.array(test_errors)


        fig, axs = plt.subplots(1, 2, figsize=(2*len(models), 4), layout='constrained', sharex=True)
        test_color = "#E69F00"  
        train_color = "#0072B2" 
        data_types = ['PCA', 'features']
        for kk, ax in enumerate(axs):
            parts_train = ax.violinplot(validation_errors[:, :, kk].T, positions=np.arange(len(models)),
                                         widths=0.4, showmeans=True, showextrema=True)
            parts_test = ax.violinplot(test_errors[:, :, kk].T, positions=np.arange(len(models)) + 0.5,
                                        widths=0.4, showmeans=True, showextrema=True)
            for ll in range(validation_errors.shape[1]):
                ax.scatter(np.arange(len(models)), validation_errors[:, ll, kk],
                           color=train_color, s=18, alpha=0.8, edgecolors='black', linewidth=0.5,
                           label="Train" if ll == 0 else "")
            for ll in range(test_errors.shape[1]):
                ax.scatter(np.arange(len(models)) + 0.5, test_errors[:, ll, kk],
                           color=test_color, s=18, alpha=0.8, edgecolors='black', linewidth=0.5,
                           label="Test" if ll == 0 else "")
            for pc in parts_train['bodies']:
                pc.set_facecolor(train_color)
                pc.set_edgecolor('black')
                pc.set_alpha(0.5)
            for pc in parts_test['bodies']:
                pc.set_facecolor(test_color)
                pc.set_edgecolor('black')
                pc.set_alpha(0.5)
            for part in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
                parts_train[part].set_edgecolor(train_color)
                parts_train[part].set_linewidth(2)
                parts_test[part].set_edgecolor(test_color)
                parts_test[part].set_linewidth(2)
            ax.set_xticks(np.arange(len(models)) + 0.25)
            ax.set_xticklabels(models, rotation=20, fontsize=12)
            ax.set_title(f"{data_types[kk]}", fontsize=14)
            ax.set_ylabel("RMS Error", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=train_color, markersize=10, label='Train'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=test_color, markersize=10, label='Test')]
        fig.legend(handles, ['Train', 'Test'], loc="upper right", fontsize=12, bbox_to_anchor=(0.95, 0.98))



    def plot_dataset_XYZ(self, normalized=True, num_points=50000, cmaps=['BuPu', 'YlOrBr', 'YlGnBu']):
        """
        Visualize selected features from the dataset in 2D.
        """

        if isinstance(num_points, int):
            rng = np.random.default_rng(self.random_seed)
            num_points = rng.integers(self.n_points, size=num_points)


        _, axs = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True, layout='constrained')
        for ax, c_label, cmap in zip(axs, self.features_of_interest, cmaps):
            color_data = self.get_property(key=c_label, normalized=normalized) 
            sm = ax.scatter(self.xyz[num_points, 0], self.xyz[num_points, 1],
                            c=color_data[num_points, 0], s=2, cmap=truncate_colormap(cmap, 0.2, 0.9, len(num_points)))
            plt.colorbar(sm, ax=ax, label=c_label, orientation='horizontal', location='top')
            ax.set(xlabel='X', ylabel='Y')



    def plot_dataset_PCA(self, nrows=2, normalized=True, num_points=50000, cmaps=['BuPu', 'YlOrBr', 'YlGnBu']):


        if isinstance(num_points, int):
            rng = np.random.default_rng(self.random_seed)
            num_points = rng.integers(self.n_points, size=num_points)

        if self.reduced_dimension <= 2:
            nrows = 1

        _, grid = plt.subplots(nrows, 3, figsize=(10, 3*nrows), sharex=True, sharey='row', layout='constrained')

        if nrows == 1:
            grid = [grid]

        for ii, axs in enumerate(grid):
            for ax, c_label, cmap in zip(axs, self.features_of_interest, cmaps):
                color_data = self.get_property(key=c_label, normalized=normalized) 
                sm = ax.scatter(self._Z[num_points, 0], self._Z[num_points, ii+1],
                                c=color_data[num_points, 0], s=2, cmap=truncate_colormap(cmap, 0.2, 0.9, len(num_points)))
                if ii == 0:
                    plt.colorbar(sm, ax=ax, label=c_label, orientation='horizontal', location='top')
                else:
                    ax.set(xlabel='Score 1')

            axs[0].set(ylabel=f'Score {ii+2}')


    
    def plot_split_data(self):

        plt.figure(figsize=(4, 4))
        plt.scatter(*zip(*self.train_input_parameters), color='k', alpha=0.4, label='Training cases')
        plt.scatter(*zip(*self.extrapolation_cases), s=50, marker='^', color='c', label='Extrapolation test cases')
        plt.scatter(*zip(*self.interpolation_cases), s=50, marker='*', color='r', label='Interpolation test cases')
        plt.xlabel('$\lambda_{sink}$')
        plt.ylabel('$U_{in}$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    def plot_Phi(self):
        plt.figure()
        plt.imshow(self._Phi, cmap='RdBu')
        plt.yticks(np.arange(self._Phi.shape[0]), labels=self.train_input_parameters)
        plt.xlabel('Score id')
        plt.colorbar()


    def plot_Lambda(self, max_mode = None):
        
        c_rel = 'C0'
        c_cum = 'orange'
        L = self._Lambda
        r = self.reduced_dimension

        if max_mode is None:
            max_mode = len(L)
        

        relative_energy = L / np.sum(L)
        cumulative_energy = np.cumsum(relative_energy)
        pcs = np.arange(1, L.size + 1)

        _, ax1 = plt.subplots(figsize=(5, 4))
        ax1.axvspan(0.5, r + 0.5, color='k', alpha=0.2, label=f'First {r} modes')
        ax1.set(xlim=(0.5, max_mode))


        # Bar plot for relative energy
        ax1.bar(pcs - 0.25, relative_energy, width=0.5, label='Relative Energy', color=c_rel)
        ax1.set_xlabel('Principal Component (PC)')
        ax1.set_ylabel('Relative Energy', color=c_rel)
        ax1.tick_params(axis='y', labelcolor=c_rel)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Second axis for cumulative energy
        ax2 = ax1.twinx()
        ax2.plot(pcs, cumulative_energy, color=c_cum, marker='o', label='Cumulative Energy')
        ax2.set_ylabel('Cumulative Energy', color=c_cum)
        ax2.tick_params(axis='y', labelcolor=c_cum)
        ax2.axhline(0.99, color='k', linestyle='--', linewidth=1)
        ax2.text(max_mode-1, 0.99, '99% threshold', color='k', va='bottom', ha='right')
        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')




    # PLOTS FOR THE UI 
    
    def plot_split_data_go(self, **kwargs):
        fig = go.Figure(layout=create_go_figure_layout(**kwargs))

        # Training cases
        fig.add_trace(go.Scatter(
            x=[pt[0] for pt in self.train_input_parameters],
            y=[pt[1] for pt in self.train_input_parameters],
            mode='markers',
            name='Training cases',
            marker=dict(color='gray', opacity=0.6, size=6),
        ))

        # Extrapolation cases
        fig.add_trace(go.Scatter(
            x=[pt[0] for pt in self.extrapolation_cases],
            y=[pt[1] for pt in self.extrapolation_cases],
            mode='markers',
            name='Extrapolation test cases',
            marker=dict(color='deepskyblue', symbol='triangle-up', size=10, line=dict(width=1, color='black')),
        ))

        # Interpolation cases
        fig.add_trace(go.Scatter(
            x=[pt[0] for pt in self.interpolation_cases],
            y=[pt[1] for pt in self.interpolation_cases],
            mode='markers',
            name='Interpolation test cases',
            marker=dict(color='red', symbol='star', size=12, line=dict(width=1, color='black')),
        ))

        fig.update_layout(
            xaxis=dict(
                title=r"λsink",
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                title="Uin",
                gridcolor='rgba(0,0,0,0.1)'
            ),
            legend=dict(
                orientation="v",
                x=1,
                y=1,
                xanchor="left",
                yanchor="top",
                bgcolor='rgba(255,255,255,0.7)',
                borderwidth=1
            ),
            width=800,
            height=450,
        )

        return fig



    def plot_model_predictions_go(self, A_test_pred, A_test_ref, **kwargs):
        df_train = self._format_input_params(self.train_input_parameters)
        df_test = self._format_input_params(self.test_input_parameters)

        x1 = np.linspace(-0.1, 1.1, 30)
        x2 = np.linspace(20, 38, 30)
        X1, X2 = np.meshgrid(x1, x2)
        df_surf = self._format_input_params(np.column_stack([X1.ravel(), X2.ravel()]))
        Y_surf = self.model.predict(df_surf)

        train_kwargs = dict(mode='markers', name='Training data', 
                            marker=dict(color='gray', size=4))
        test_kwargs = dict(mode='markers', name='Test data',
                        marker=dict(color='cyan', size=6, symbol='diamond-open'))
        pred_kwargs = dict(mode='markers', name='Test prediction',
                        marker=dict(color='red', size=6, opacity=0.6))

        input_params = [df_train, df_test, df_test]
        all_kwargs = [train_kwargs, test_kwargs, pred_kwargs]

        # Create subplot with multiple scenes
        fig = make_subplots(
            rows=1, cols=self.reduced_dimension,
            specs=[[{'type': 'scene'}] * self.reduced_dimension],
            subplot_titles=[f"Principal component {ii}" for ii in range(self.reduced_dimension)]
        )

        for ii in range(self.reduced_dimension):
            scene_name = f"scene{ii+1}" if ii > 0 else "scene"
            show_legend = ii == 0

            # Surface
            fig.add_trace(go.Surface(
                x=X1, y=X2, z=Y_surf[:, ii].reshape(X1.shape),
                colorscale='Viridis', opacity=0.5, showscale=False
            ), row=1, col=ii+1)

            ZZ = [self.A_train[:, ii], A_test_ref[:, ii], A_test_pred[:, ii]]
            for xx, zz, args in zip(input_params, ZZ, all_kwargs):
                fig.add_trace(go.Scatter3d(
                    x=xx.iloc[:, 0], y=xx.iloc[:, 1], z=zz, 
                    **args, showlegend=show_legend
                ), row=1, col=ii+1)

            fig.update_layout({
                scene_name: dict(
                    xaxis=dict(title="λsink"),
                    yaxis=dict(title="Uin"),
                    zaxis=dict(title=f"α{ii+1}"),
                    aspectmode='cube'
                )
            })


        fig.update_layout(
            **create_go_figure_layout(**kwargs)
        )
        fig.update_layout(
            title=self._model_title(self.regressor_type) + f" with {self.reduced_dimension} dimensions",
            height=600, width=800 * self.reduced_dimension,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        return fig

    @staticmethod
    def plot_violins_go(validation_errors, test_errors, models, **kwargs):
        """
        Create interactive violin plots of RMS error distributions using Plotly.
        """

        if isinstance(validation_errors, list):
            validation_errors = np.array(validation_errors)
            test_errors = np.array(test_errors)
        elif validation_errors.ndim == 2:
            validation_errors = np.array([validation_errors])
            test_errors = np.array([test_errors])

        test_kwargs = dict(fillcolor="#E69F00", legendgroup="Test") #, side='positive')
        train_kwargs = dict(fillcolor="#0072B2", legendgroup="Train")#, side='negative')

        # Flatten for global min/max range
        combined = np.concatenate([validation_errors.flatten(), test_errors.flatten()])
        y_min, y_max = np.min(combined), np.max(combined)
        padding = 0.05 * (y_max - y_min)

        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"RMS Error Distribution - {dt}" for dt in ['PCA', 'features']])

        for kk in range(2):  # PCA and features
            for ii, model in enumerate(models):

                # (np.array([results['train_rms']]).shape)
                for key, yy, args in zip(['Train', 'Test'], [validation_errors, test_errors], 
                                         [train_kwargs, test_kwargs]):
                    fig.add_trace(
                        go.Violin(y=yy[ii, :, kk], 
                                  name=f"{model} {key}", 
                                  **args, 
                                  line_color='black', opacity=0.5, meanline_visible=True,
                                  showlegend=False, 
                                  width=0.5,
                                    # points="all",          # show each data point
                                    # jitter=0.2,             # optional: adds random horizontal noise for clarity
                                    scalemode="count",     # avoids scaling to area (useful with few points)
                                    ),
                        row=1, col=kk + 1
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=yy[ii, :,  kk],
                            x=[f"{model} {key}"] * len(yy[ii, :, kk]),
                            mode="markers",
                            name=f"{model}",
                            marker=dict(size=6, color=args.get("fillcolor", "gray"), line=dict(width=1, color="black")),
                            showlegend=False
                        ),
                        row=1, col=kk + 1
                    )
        
        fig.update_layout(**create_go_figure_layout(**kwargs),
                          height=500,
                            width=800,
                            violingap=0,
                         violinmode='group'
                    )
        fig.update_layout(title='Training and test root mean squared errors')
        fig.update_yaxes(title_text="RMS Error", showgrid=True, gridcolor='rgba(0,0,0,0.1)', row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', row=1, col=2)
        
        return fig


    def plot_dataset_PCA_go(self, nrows=2, normalized=True, num_points=50000, cmaps=None, **kwargs):
        if cmaps is None:
            cmaps = ['BuPu', 'YlOrBr', 'YlGnBu']  

        if isinstance(num_points, int):
            rng = np.random.default_rng(self.random_seed)
            num_points = rng.integers(self.n_points, size=num_points)

        if self.reduced_dimension <= 2:
            nrows = 1

        fig = make_subplots(
            rows=nrows, cols=3,
            shared_xaxes=True,
            shared_yaxes='rows',
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
            subplot_titles=[""] * (nrows * 3)  # hide subplot titles
        )

        for ii in range(nrows):  # for PC2, PC3, ...
            for jj, (c_label, cmap) in enumerate(zip(self.features_of_interest, cmaps)):
                color_data = self.get_property(key=c_label, normalized=normalized)
                fig.add_trace(
                    go.Scattergl(
                        x=self._Z[num_points, 0],
                        y=self._Z[num_points, ii + 1],
                        mode='markers',
                        marker=dict(
                            color=color_data[num_points, 0],
                            colorscale=cmap,
                            size=2,
                            showscale=True,
                            colorbar=dict(
                                title=dict(text=c_label, side='top'),
                                orientation='h',
                                x=0.33 * jj + 0.17,  # spread colorbars across
                                y=1.1 + 0.03 * ii,  # stack vertically if needed
                                len=0.25,
                                thickness=10,
                                xanchor='center'
                            )
                        ),
                        name=c_label,
                        showlegend=False
                    ),
                    row=ii + 1, col=jj + 1
                )

                # Axis labels
                if ii == nrows - 1:
                    fig.update_xaxes(title_text="Score 1", row=ii + 1, col=jj + 1)
            fig.update_yaxes(title_text=f"Score {ii + 2}", row=ii + 1, col=1)

        fig.update_layout(
            **create_go_figure_layout(**kwargs),
            height=400 * nrows,
            width=1000,
        )

        return fig



    def plot_dataset_XYZ_go(self, normalized=True, num_points=50000, cmaps=None, **kwargs):
        if cmaps is None:
            cmaps = ['BuPu', 'YlOrBr', 'YlGnBu']

        if isinstance(num_points, int):
            rng = np.random.default_rng(self.random_seed)
            num_points = rng.integers(self.n_points, size=num_points)

        fig = make_subplots(rows=1, cols=3, shared_xaxes=True, shared_yaxes=True)

        for ii, (c_label, cmap) in enumerate(zip(self.features_of_interest, cmaps)):
            color_data = self.get_property(key=c_label, normalized=normalized)
            fig.add_trace(
                go.Scattergl(
                    x=self.xyz[num_points, 0],
                    y=self.xyz[num_points, 1],
                    mode='markers',
                    marker=dict(
                        color=color_data[num_points, 0],
                        colorscale=cmap,
                        size=2,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text=c_label,side='top'),
                            orientation='h',
                            x=0.33 * ii + 0.16,  # adjust to center over each subplot
                            y=1.12,
                            len=0.25,
                            thickness=10,
                            xanchor='center',
                        )
                    ),
                    name=c_label,
                    showlegend=False
                ),
                row=1, col=ii + 1
            )

        fig.update_layout(
            **create_go_figure_layout(**kwargs),
            height=400,
            width=1000,
        )


        fig.update_xaxes(title_text="X", row=1, col=2)
        fig.update_yaxes(title_text="Y", row=1, col=1)

        return fig



    def plot_predicted_vfarm_go(self, features,
                                fast=True,  
                                display_features=None,
                                normalized=False, cmaps=None, **kwargs):
 

        if not isinstance(features, list):
            features = [features]
        

        if display_features is None:
            display_features = self.features_of_interest
            if cmaps is None:
                cmaps = ['BuPu', 'YlOrBr', 'YlGnBu']
        elif not isinstance(display_features, list):
            display_features = [display_features]

        
        ncols = len(features)
        nrows = len(display_features)
        

        if cmaps is None:
            cmaps = ['viridis'] * ncols
        elif not isinstance(cmaps, list):
            cmaps = [cmaps]

        fig = make_subplots(
            rows=nrows, cols=ncols,
            specs=[[{'type': 'scene'} for _ in range(ncols)] for _ in range(nrows)],
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
            # subplot_titles=self.features_of_interest
        )

        

        xx, yy, zz = self.xyz[:self.n_points, :].T


        if fast: 
            num_points = int(self.n_points * .15)
        else:
            num_points = int(self.n_points * .5)


        rng = np.random.default_rng(self.random_seed)
        num_points = rng.choice(self.n_points, size=num_points, replace=False)

        xx, yy, zz = [_y[num_points] for _y in [xx, yy, zz]]

        for ii, feat in enumerate(features):
            cmap = cmaps[ii]

            for jj, c_label in enumerate(display_features):

                # Get color data for each subplot
                color_data = self.get_property(data=feat, key=c_label, normalized=normalized)
                color_data = color_data[num_points]
                if color_data.ndim > 1:
                    color_data = color_data.squeeze()

                lbl = c_label if cmap != 'Reds' else c_label + ' Error %'

                kk, batch = 0, len(num_points)
                while kk <= len(num_points)-batch: 
                    fig.add_trace(
                        go.Scatter3d(
                            x=xx[kk:kk+batch],
                            y=yy[kk:kk+batch],
                            z=zz[kk:kk+batch],
                            mode='markers',
                            marker=dict(
                                color=color_data[kk:kk+batch],
                                colorscale=cmap,
                                size=2,
                                showscale=kk == 0,
                                colorbar=dict(
                                    title=dict(text=lbl, side='top'),
                                    orientation='h',
                                    # x=0.33 * (ii+1),  
                                    # y=1.1 * (jj + 1.),  
                                    x=0.33 * ii + 0.17,  # spread colorbars across
                                    y=1.1 + 0.03 * jj,  # stack vertically if needed 
                                    len=0.5 / ncols,
                                    thickness=10,
                                    xanchor='center'
                                ) if kk == 0 else dict()
                            ),
                            name='',
                            showlegend=False
                        ),
                        row= jj+1, col=ii + 1
                        )
                    kk += batch


        fig.update_layout(
            **create_go_figure_layout(**kwargs),
            height=500*nrows,
            width=500*ncols,
        )

        return fig





def create_go_figure_layout(title="", font_family="Times New Roman", font_size=13, **kwargs):
    # args = dict(family=font_family, size=font_size)
    return dict(
        font=dict(family=font_family, size=font_size), 
        title=dict(text=title, font=dict(size=font_size + 2, family=font_family)),      
        legend=dict(font=dict(size=font_size+ 2, family=font_family)),
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            xaxis=dict(
                title_font=dict(size=font_size+2),
                tickfont=dict(size=font_size)
            ),
            yaxis=dict(
                title_font=dict(size=font_size+2),
                tickfont=dict(size=font_size)
            ),
            zaxis=dict(
                title_font=dict(size=font_size+2),
                tickfont=dict(size=font_size))
            )
        )
    # layout_defaults.update(kwargs)
    # return go.Figure(layout=layout_defaults)





def truncate_colormap(cmap, minval=0.2, maxval=0.9, n=100):
    """Truncate a colormap to use only a subset of its range."""

    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


# %%
