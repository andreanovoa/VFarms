from matplotlib.artist import kwdoc
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from VF_Surrogate import VF_Surrogate
import streamlit as st

def time_to_lambda(hours):
    return hours / 24.

st.set_page_config(layout="wide", page_title="Vertical Farm Surrogate Model")

st.title("Vertical Farm Surrogate Model")
# st.latex()

# Sidebar options for adjusting model parameters
st.sidebar.header("Model Settings")
st.sidebar.subheader("(default parameters are selected)")

features_of_interest = st.sidebar.multiselect("Select features of interest", 
                                              options=VF_Surrogate.features_labels, 
                                              default=VF_Surrogate.features_of_interest)

reduced_dimension = st.sidebar.slider('Number of reduced dimensions', 
                                      min_value=1, 
                                      max_value=20,
                                      value=2, 
                                      step=1)

regressor_type = st.sidebar.selectbox("Select regressor type", 
                                      options=VF_Surrogate.regressor_type_options,
                                      index=VF_Surrogate.regressor_type_options.index('poly2'))

norm_type = st.sidebar.selectbox("Scaling type",
                                 options=VF_Surrogate.scaling_options, 
                                 index=VF_Surrogate.scaling_options.index('pareto'))


st.sidebar.header("Figures Settings")

font_size = st.sidebar.slider("Global plot font size", 8, 24, 13)
font_family = st.sidebar.selectbox(
    "Font family",
    options=["Arial", "Courier New", "Times New Roman", "Verdana", "Helvetica", "Georgia", "Lucida Console"],
    index=0
)


kwargs = dict(font_size=font_size, 
              font_family=font_family)


# Row layout for run/save
st.markdown("### ⚙️ Simulation Control")

# col1, col2, col3 = st.columns([1, 1, 1])

if st.button("▶️ Run new simulation"):
    st.session_state.pod = VF_Surrogate(scale_type=norm_type,
                                        features_of_interest=features_of_interest,
                                        regressor_type=regressor_type,
                                        reduced_dimension=reduced_dimension)
    st.session_state.pod.perform_dimensionality_reduction()
    st.session_state.pod.train_surrogate_model()
    with open("vf_surrogate.pkl", "wb") as f:
        pickle.dump(st.session_state.pod, f)
    st.success("New simulation run and saved ✅")

if st.button("💾 Save simulation"):
    if "pod" in st.session_state:
        with open("vf_surrogate.pkl", "wb") as f:
            pickle.dump(st.session_state.pod, f)
        st.success("Simulation saved ✅")
    else:
        st.warning("Run simulation first before saving.")


if st.button("📂 Load simulation"):
    if os.path.exists("vf_surrogate.pkl"):
        with open("vf_surrogate.pkl", "rb") as f:
            st.session_state.pod = pickle.load(f)
        st.success("Loaded existing simulation from file ✅")
    else:
        st.warning("Run simulation and save it before loading.")


st.markdown("---")
st.markdown("### 📈 Visualizations on the available dataset")

if st.button("Visualize data split"):
    st.markdown("#### 🔹 Train/Test Split")
    fig = st.session_state.pod.plot_split_data_go(**kwargs)
    st.plotly_chart(fig, use_container_width=False)

    
if st.button("Visualize dataset projection (XYZ and PCA scores)"):
    st.markdown("#### 🔹 Dataset Projections XYZ and PCA")

    rng = np.random.default_rng(st.session_state.pod.random_seed)
    num_points = rng.choice(st.session_state.pod.n_points, size=10000, replace=False)


    fig1 = st.session_state.pod.plot_dataset_XYZ_go(num_points=num_points, 
                                                    normalized=False, **kwargs)
    st.plotly_chart(fig1, use_container_width=False)

    fig2 = st.session_state.pod.plot_dataset_PCA_go(num_points=num_points,
                                                    normalized=False, **kwargs)
    st.plotly_chart(fig2, use_container_width=False)




if st.button("Visualize the surrogate model on the test set"):

    st.markdown("#### 🔹 Surrogate Model Predictions")

    results = st.session_state.pod.evaluate_model()
    fig1 = st.session_state.pod.plot_model_predictions_go(results['A_test_pred'], 
                                                          results['A_test_ref'],
                                                          **kwargs)

    st.plotly_chart(fig1, use_container_width=False)

    fig2 = VF_Surrogate.plot_violins_go(results['train_rms'], 
                                        results['test_rms'], 
                                        [regressor_type],
                                        **kwargs
                                        )
    st.plotly_chart(fig2, use_container_width=False)



st.markdown("---")
st.markdown("""
            ### 🧪 Test the model  
            Predict the reduced components and features for specific **U<sub>in</sub>** and **λ<sub>sink</sub>**.  
            If the selected options are in the test set, we will also compare the prediction to the ground truth.
            """, unsafe_allow_html=True)

col1, col2, _ = st.columns([1, 1, 5])  # You can tweak the proportions

with col1:
    inlet_velocity = st.number_input(
        "Inlet velocity (Uin)", min_value=0.0, max_value=100.0, value=30.0, step=0.5
    )

with col2:
    time_of_day = st.number_input(
        "Time of day (hours)", min_value=0, max_value=24, value=12
    )

    lambda_sink = time_to_lambda(time_of_day)



if st.button("Run test"):
    # st.write(f'tests in trainning: {st.session_state.pod.test_input_parameters}')
    
    selected_input_params = (lambda_sink, inlet_velocity)

    if selected_input_params in st.session_state.pod.test_input_parameters:
        # st.write(f"Selected properties in test set -- Visualizing both, prediction and true farm.")
        ii = st.session_state.pod.test_input_parameters.index(selected_input_params)
        reference_case = st.session_state.pod.test_data[:, ii]
        st.session_state.reference_case = st.session_state.pod.recover_original_shape(reference_case)
    elif selected_input_params in st.session_state.pod.train_input_parameters:
        # st.write(f"Selected properties in training set -- Visualizing both, prediction and true farm.")
        ii = st.session_state.pod.train_input_parameters.index(selected_input_params)
        reference_case = st.session_state.pod.train_data[:, ii]
        st.session_state.reference_case = st.session_state.pod.recover_original_shape(reference_case)
    else:
        st.session_state.reference_case = None
    

    # Predict values
    
    st.session_state.reconstructed_feat = st.session_state.pod.predict_features(input_params=[selected_input_params], 
                                                                reshape=True)[1]

    st.success(f"Test with U_in={inlet_velocity}, lamda_sink={lambda_sink} complete. ")


col1, _ = st.columns([2, 5])  # You can tweak the proportions

with col1:
    display_feats = st.multiselect("Choose property to visualize", 
                                  options=VF_Surrogate.features_of_interest, 
                                  default='T')


if st.button("Visualize test"):

    col1, col2 = st.columns([2, 3]) 
    
    fig = st.session_state.pod.plot_predicted_vfarm_go(features=st.session_state.reconstructed_feat, 
                                                       display_features=display_feats, 
                                                       normalized=False,
                                                       **kwargs)


    with col1:
        st.markdown("##### 🔹 Surrogate Model Prediction")
        st.plotly_chart(fig, use_container_width=True)

    with col2: 
        st.markdown("##### 🔹 Reference Simulation")
            
        if st.session_state.reference_case is not None:
            # fig1 = st.session_state.pod.plot_predicted_vfarm_go(features=st.session_state.reference_case, 
            #                                                     display_features=display_feats, 
            #                                                     normalized=False)
            
            diff = (st.session_state.reference_case - st.session_state.reconstructed_feat) / st.session_state.reference_case * 100
            fig2 = st.session_state.pod.plot_predicted_vfarm_go(features=[st.session_state.reference_case, diff], 
                                                                display_features=display_feats, 
                                                                normalized=False,
                                                                cmaps=['viridis', 'Reds'],
                                                                **kwargs)
            # st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.write("No reference case available for the selected parameters.")

    st.success(f"Test Visualization with U_in={inlet_velocity}, lamda_sink={lambda_sink} complete. ")

