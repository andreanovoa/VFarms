
# Surrogate Modelling of Vertical Farms

This project focuses on creating surrogate models for vertical farms, enabling faster simulations and experimentation.

---

## Installation

Install the required Python packages with

```bash
pip install -r requirements.txt
```
NB. The code was developed in python 3.11.

_______

## 🗂️ Project Structure

- `main_downsample.py` – Pre-processes the data and performs downsampling.
- `main_surrogate.py` – Handles the surrogate modeling workflow.
  
- `VF_Surrogate.py` – Contains the VF_Surrogate class for surrogate modeling.
- `vfarms_ui.py` – Streamlit UI wrapper for interacting with the model.
- `fns.py` – Auxiliary functions used throughout the project.


-------

## 🚀 Running the Application

To launch the Streamlit UI, run:
``` bash
streamlit run vfarms_ui.py
``` 
