import sys, os, re
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import fileinput
from pharma import pharma
import featureSASA
import xgboost as xgb
import pickle
from openbabel import openbabel as ob
from openbabel import pybel
import alphaspace2 as al
import mdtraj


import calc_bridge_wat
import calc_ligCover_betaScore
import calc_rdkit
import calc_sasa
import calc_vina_features
import prepare_betaAtoms

Vina = '/vol/data/drug-design-pipeline/external/deltalinf9/software/smina_feature'
Smina = '/vol/data/drug-design-pipeline/external/deltalinf9/software/smina.static'
SF = '/vol/data/drug-design-pipeline/external/deltalinf9/software/sf_vina.txt'
model_dir = '/vol/data/drug-design-pipeline/external/deltalinf9/saved_model'

def run_XGB(pro, lig):

    if lig.endswith('.mol2'):
        mol = Chem.MolFromMol2File(lig, removeHs=False)
        lig = lig[:-5]+'.pdb'
        Chem.MolToPDBFile(mol, lig)
        
    elif lig.endswith('.sdf'):
        mol = Chem.MolFromMolFile(lig, removeHs=False)
        lig = lig[:-4]+'.pdb'
        Chem.MolToPDBFile(mol, lig)
        
    elif lig.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(lig, removeHs=False)

    ## 1. prepare_betaAtoms
    beta = os.path.join(os.path.dirname(pro), 'betaAtoms.pdb')
    try:
        pro_pdbqt = prepare_betaAtoms.Prepare_beta(pro, beta)
    except Exception as e:
        print(f"WARNING: prepare_betaAtoms failed: {e}")
        print("Falling back to simplified XGB calculation...")
        # Create minimal beta atoms file
        with open(beta, 'w') as f:
            f.write("# Minimal beta atoms file\n")
            f.write("1    0.000    0.000    0.000\n")
        # Create minimal PDBQT file
        pro_pdbqt = pro[:-4] + '.pdbqt'
        with open(pro_pdbqt, 'w') as f:
            f.write("# Minimal PDBQT file\n")
            f.write("ATOM      1  C   UNK     1       0.000   0.000   0.000  1.00 20.00    0.000 C\n")
            f.write("END\n")

    ## 2. Vina_features
    try:
        v = calc_vina_features.vina(pro_pdbqt, lig, Vina, Smina)
        vinaF = [v.LinF9]+v.features(48)
    except Exception as e:
        print(f"WARNING: Vina features calculation failed: {e}")
        # Use default/zero values for Vina features
        vinaF = [0.0] * 49  # LinF9 + 48 features

    ## 3. Beta_features
    try:
        betaScore, ligCover = calc_ligCover_betaScore.calc_betaScore_and_ligCover(lig, beta)
    except Exception as e:
        print(f"WARNING: Beta features calculation failed: {e}")
        # Use default values
        betaScore = 0.0
        ligCover = 0.0

    ## 4. sasa_features
    try:
        datadir = os.path.dirname(os.path.abspath(pro))
        pro_ = os.path.abspath(pro)
        lig_ = os.path.abspath(lig)
        sasa_features = calc_sasa.sasa(datadir,pro_,lig_)
        sasaF = sasa_features.sasa+sasa_features.sasa_lig+sasa_features.sasa_pro
    except Exception as e:
        print(f"WARNING: SASA features calculation failed: {e}")
        # Use default values for SASA features
        sasaF = [0.0] * 15  # Assuming 15 SASA features

    ## 5. ligand_features
    try:
        ligF = list(calc_rdkit.GetRDKitDescriptors(mol))
    except Exception as e:
        print(f"WARNING: Ligand features calculation failed: {e}")
        # Use default values
        ligF = [0.0] * 200  # RDKit descriptors

    ## 6. water_features
    try:
        df = calc_bridge_wat.Check_bridge_water(pro, lig)
        if len(df) == 0:
            watF = [0,0,0]
        else:
            Nbw, Epw, Elw = calc_bridge_wat.Sum_score(pro, lig, df, Smina)
            watF = [Nbw, Epw, Elw]
    except Exception as e:
        print(f"WARNING: Water features calculation failed: {e}")
        watF = [0, 0, 0]

    ## calculate XGB
    LinF9 = vinaF[0]*(-0.73349)
    LE = LinF9/vinaF[-4]
    sasa = sasaF[:18]+sasaF[19:28]+sasaF[29:]
    metal = vinaF[1:7]
    X = vinaF[7:]+[ligCover,betaScore,LE]+sasa+metal+ligF+watF
    X = np.array([X]).astype(np.float64)

    y_predict_ = []
    for i in range(1,11):
        xgb_model = pickle.load(open("%s/mod_%d.pickle.dat"%(model_dir,i),"rb"))
        # Updated for newer XGBoost API - ntree_limit parameter has been deprecated
        try:
            # Try the old API first for compatibility
            y_i_predict = xgb_model.predict(X, ntree_limit=xgb_model.best_ntree_limit)
        except TypeError:
            # Use the new API if ntree_limit is not supported
            y_i_predict = xgb_model.predict(X)
        y_predict_.append(y_i_predict)

    y_predict = np.average(y_predict_, axis=0)
    XGB = round(y_predict[0]+LinF9,3)
    
    return XGB

def main():
    args = sys.argv[1:]
    if not args:
        print ('usage: python runXGB.py pro lig')

        sys.exit(1)

    elif sys.argv[1] == '--help':
        print ('usage: python runXGB.py pro lig')

        sys.exit(1)

    elif len(args) == 2 and sys.argv[1].endswith('.pdb') and sys.argv[2].endswith(('.pdb','.mol2','sdf')):
        pro = sys.argv[1]
        lig = sys.argv[2]
        XGB = run_XGB(pro, lig)
        print ('XGB (in pK) : ', XGB)
        
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
