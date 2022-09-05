# A_Prot_Paper

## Description
**A Prot paper related materials:**

The MSAs used for the A-Prot are at:

```
A_Prot_Paper/A_Prot_msa/
```

The predicted pdb structure files of the A-Prot are at:

```
A_Prot_Paper/A_Prot_predicted_pdb/
```

## Requirements
- python 3.6
- esm 0.3.0 (https://github.com/facebookresearch/esm)
- pytorch 1.7.1
- einops
- pyrosetta 4 (http://www.pyrosetta.org/)

## Usages
(1) Using MSA to predict distogram and anglegram for structure modeling.
```
python run_a_prot.py --input_path T0998.json --output_path T0998.npz --conv_model_path a_prot_resnet_weights.pth
```
The input file should be in format .fa or .a3m like MSA file, or .json file like ours.
The ```a_prot_resnet_weights.pth``` trained network weights can be downloaded from 
```
https://drive.google.com/drive/folders/1JLSsSzKu3NBTKg9KwSlXCB_8q3E-g2Gv?usp=sharing
```

(2) Run structure modeling
```
python run_structure_modeling.py --input_path T0998.npz --output_path T0998.pdb
```

## Reference

- Hong, Y., Lee, J. & Ko, J. A-Prot: protein structure modeling using MSA transformer. BMC Bioinformatics 23, 93 (2022). https://doi.org/10.1186/s12859-022-04628-8

- J Yang, I Anishchenko, H Park, Z Peng, S Ovchinnikov, D Baker. Improved protein structure prediction using predicted inter-residue orientations. (2020) PNAS. 117(3): 1496-1503

- Roshan Rao, Jason Liu, Robert Verkuil, Joshua Meier, John F. Canny, Pieter Abbeel, Tom Sercu, Alexander Rives. MSA Transformer. bioRxiv 2021.02.12.430858; doi: https://doi.org/10.1101/2021.02.12.430858

## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

