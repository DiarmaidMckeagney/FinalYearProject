This is my Final Year Project titled "Comparison of Datasets for Anomaly Detection
Development". It implemnents three ML algorithms on the VNFCYBERDATA Dataset ("Ayodele, B.; Buttigieg, V. The VNF Cybersecurity Dataset for Research (VNFCYBERDATA). Data 2024, 9, 132. https://doi.org/10.3390/data9110132") and on the BETH dataset (K. Highnam, K. Arulkumaran, Z. Hanif, and N. R. Jennings, “Beth dataset:
Real cybersecurity data for unsupervised anomaly detection research,”
in CEUR Workshop Proc, vol. 3095, 2021, pp. 1–12. [Online]. Available:
https://ceur-ws.org/Vol-3095/paper1.pdf.). It compares these datasets and tries to find a feature set and hyperparameter values to train good models for the VNFCYBERDATA dataset.

NOTE: The files for the VNF Dataset and BETH dataset are stored in github LFS (Large File System). To pull these files, run `git lfs pull` or `git lfs fetch`. 

To run the program, run the main() function in either the VNFModelRunner.py file, or in the BETHModelRunner.py file. I would advise you to only run the part you want to test (Feature Importance, Feature Selection, etc..) and comment out the rest. This is due to the large amounts of models being trained. It takes over two hours to run the VNFModelRunner.py fully.
