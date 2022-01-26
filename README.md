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
Using MSA to predict distogram and anglegram for trrosetta structure modeling.
```
python run_a_prot.py --input_path T0998.json --output_path T0998.npz --conv_model_path a_prot_resnet_weights.pth
```
The input file should be in format .fa or .a3m like MSA file, or .json file like ours.
The ```a_prot_resnet_weights.pth``` trained network weights can be downloaded from ```https://drive.google.com/drive/folders/1JLSsSzKu3NBTKg9KwSlXCB_8q3E-g2Gv?usp=sharing```

## Reference
J Yang, I Anishchenko, H Park, Z Peng, S Ovchinnikov, D Baker. Improved protein structure prediction using predicted inter-residue orientations. (2020) PNAS. 117(3): 1496-1503

Yiyu Hong, Juyong Lee, Junsu Ko. A-Prot: Protein structure modeling using MSA transformer. bioRxiv 2021.09.10.459866; doi: https://doi.org/10.1101/2021.09.10.459866




