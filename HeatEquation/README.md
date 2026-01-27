# Heat Equation in 1D

This folder contains the scripts to reproduce the results in Section 5.3 "Heat Equation". 
The important scripts are:
- **generate_test_data.py**: This script create the paired test data (x,y)
- **train_uncond.py**: This script trains the unconditional diffusion model
- **train_conditional.py**: for training the conditional diffusion model 
- **train_sgt.py**: for training the guidance term
- **evaluate_funDPS.py**: evaluating the FunDPS guidance approximation
- **evaluate_conditional.py**: evaluate the trained conditional diffusion model
- **evaluate_sgt.pt**: evaluate the trained guidance term using our SGT framework