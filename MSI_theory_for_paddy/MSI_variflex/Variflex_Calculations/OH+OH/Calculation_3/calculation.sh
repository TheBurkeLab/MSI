cd /home/leil/MSI/MSI_variflex/Variflex_Calculations/OH+OH/Calculation_3/nominal_B1_Energy_1_0.0 
echo Running nominal Variflex...
./variflex.exe 
echo Running perturbed Variflex...
cd /home/leil/MSI/MSI_variflex/Variflex_Calculations/OH+OH/Calculation_3/perturb_B1_Energy_1_175.0 
./variflex.exe 
cd /home/leil/MSI/MSI_variflex/Variflex_Calculations/OH+OH/Calculation_3/perturb_B1_Symmetry_1_0.5 
./variflex.exe 
