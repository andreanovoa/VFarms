import CGNS.MAP
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

import copy
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVR

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler




def model_title(model_name):
    if 'poly' in model_name:
        try: 
            order = int(model_name[-1])
        except:
            order = 3
        return f'Polinomial Regression order {order}'
    elif 'spline' in model_name:
        try: 
            order = int(model_name[-1])
        except:
            order = 3
        return f'Spline Regression order {order}'
    elif 'gpr' in model_name:
        return "Gaussian Process Regression"
    else:
        return model_name.capitalize() + ' Regression'



def load_new_data(folder='NewData/', filename='FFF-69-Uin30.cgns', verbose=False):
    """
    
    Returns np.array of data with shape = (14, 4479168) with this ordering:
        # 1-3: XYZ
        # 4: Pressure, 
        # 5: Density, 
        # 6-9: VelocityMagnitude, VelocityX, VelocityY, VelocityZ, 
        # 10: Temperature, 
        # 11: Wall_Temperature, 
        # 12: Mass_fraction_of_h2o, 
        # 13: Mass_fraction_of_co2, 
        # 14: Relative_Humidity, 
    
    """

    (tree,links,paths)=CGNS.MAP.load(folder+filename)

    level_1_nodes = tree[2]
    base_node = level_1_nodes[1]
    zone_nodes = base_node[2]
    local_nodes = zone_nodes[0]
    sub_nodes = local_nodes[2]

    if verbose:
        print(f'Tree has {len(tree[2])} child nodes')
        print(f'Level 1 Node List has {len(level_1_nodes)} child nodes') 
        print(f'Base Node List has {len(base_node[2])} child nodes') 
        print(f'Zone Node List has {len(zone_nodes)} child nodes') 
        print(f'Local Node List has {len(local_nodes)} child nodes') 
        print(f'Sub Node List has {len(sub_nodes)} child nodes') 

    
    zones = []
    zones_data = []
    for zone in sub_nodes:
        #print(f'Checking on child node: {zone[0]}')
        if zone[0].startswith('FlowSolution'):
            if verbose: print(f'Found a solution data: {zone[0]} node has {len(zone[2])} child nodes') 
            zones.append(zone[0])      
            zones_data.append(zone[2])
    if verbose: print(f'The data has: {len(zones_data)} zone(s)') # Normally there should only be 1 zones
    data = []
    i = 0
    if verbose: print(f'In zone {i}, there are {len(zones_data[0])} variables stored. These are:')
    for dat in zones_data[0]:
        if verbose: print(dat[0])
        data.append(dat[1])

    return np.array(data)  


def RMS(y_true, y_pred):
    y_true, y_pred = [x.squeeze() for x in [y_true, y_pred]]
    rms_error = np.sqrt(np.mean((y_true - y_pred)**2))
    # Compute the norm of the true field (RMS of true field)
    norm_true = np.sqrt(np.mean(y_true**2))
    return rms_error / (norm_true + 1e-6)


# def get_norm(ff, norm_type='range', axis=None):
#     if norm_type == 'range':
#         return np.max(ff, axis=axis) - np.min(ff, axis=axis)
#     elif norm_type == 'max':
#         return np.max(ff, axis=axis)
#     elif norm_type == 'std':
#         return np.std(ff, axis=axis)
#     elif norm_type == 'pareto':
#         return np.sqrt(np.std(ff, axis=axis))
#     else:
#         raise NotImplementedError(f'norm_type {norm_type} not defined')


def visualize_clouds(_pc_list, _features_list, _idx=6, shift=True):
    if not isinstance(_pc_list, list):
        _pc_list = [_pc_list]
    if not isinstance(_features_list, list):
        _features_list = [_features_list]
    pc_list = []
    ii = 0

    # Translate the point cloud along the X-axis
    xyz = np.asarray(_pc_list[0].points)
    if shift:
        offset = np.array([max(xyz[:, 0]) - min(xyz[:, 0]) + 1, 0, 0])  # Offset to separate the point clouds

    for _pc, _f in zip(_pc_list, _features_list):
        # Select one feature to visualize [default Temperature]
        if _f.ndim > 1:
            _f = _f[:, _idx].squeeze()

        # Normalize the feature values for color mapping
        f_n = _f - _f.min(axis=0)
        f_n /= f_n.max(axis=0)

        cmap = plt.get_cmap("viridis")  

        _pc.colors = o3d.utility.Vector3dVector(cmap(f_n)[:, :3])
        
        if ii > 0 and shift:
            _pc = copy.deepcopy(_pc)  # Properly clone the point cloud
            _pc.translate(offset * ii)

        pc_list.append(_pc)
        ii += 1

    # Visualize the point cloudso3d
    # o3d.open3d.visualization
    o3d.visualization.draw_geometries(pc_list)      





def create_point_cloud(data=None, xyz=None):

    
    # Create a point cloud
    pc = o3d.geometry.PointCloud()


    # Extract coordinate points (X, Y, Z) and properties (Pressure, Density, etc.) 
    if data is not None:
        assert xyz is None
        xyz = data[-3:, :].T  # Shape: (4479168, 3)
        feat = data[:-3, :].T  # Shape: (4479168, n_features)
        pc.points = o3d.utility.Vector3dVector(xyz)

        return pc, feat
    else:
        
        pc.points = o3d.utility.Vector3dVector(xyz)

        return pc



# %%  Downsample the point cloud

def down_sample_data_idx(pc, voxel_size = 0.05):

    pcd_downsampled = pc.voxel_down_sample(voxel_size=voxel_size)
    xyz_downsampled = np.asarray(pcd_downsampled.points)

    # Use KDTree to find the nearest neighbors of downsampled points in the original data
    xyz = np.asarray(pc.points)
    tree = cKDTree(xyz)
    _, indices = tree.query(xyz_downsampled)

    return indices