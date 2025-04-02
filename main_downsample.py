
from fns import *

cwd = os.getcwd()
if 'NewData' in cwd:
    folder = ''
else:
    folder = 'NewData/'

print(f"Folder: '{folder}'")

# Load the data and create the point cloud
data = load_new_data(folder=folder)
pc, features = create_point_cloud(data)


# Get the indices of the downsampled point cloud from the base case (B, .30)
# Downsample the point cloud. Original point cloud: 4,479,168 points. 
# voxel_size = 0.10  #  378,279 points
voxel_size = 0.08  #  700,416 points  
# voxel_size = 0.07  # 1,059,192 points
# voxel_size = 0.05  # 2,620,936 points

idx_d = down_sample_data_idx(pc, voxel_size=voxel_size)   

    
# Save the downsampled data for each case option
downsample_folder = folder+'Downsampled/'
os.makedirs(downsample_folder, exist_ok=True)

min_uin = 24
for case_option in ['A', 'B', 'C', 'D', 'E']:
    row_data = []
    for ii in range(7):
        uin = min_uin + ii * 2
        if case_option != 'B':
            file = f'Uin{uin}_Cycle/FFF-69-Uin{uin}-Option{case_option}.cgns'
        else:
            file = f'FFF-69-Uin{uin}.cgns'

        data = load_new_data(folder=folder, filename=file)

        assert data.shape[1] == 4479168, f"{file} - Data shape: {data.shape} != 4479168. NOT OK."

        row_data.append(data[:, idx_d])

    row_data = np.array(row_data)
    np.save(f'{downsample_folder}Option{case_option}-downsampled_{voxel_size}', row_data)


# Visualize clouds
pc_d, features_d = create_point_cloud(data[:, idx_d])
visualize_clouds([pc, pc_d], [features, features_d])

print('Done')