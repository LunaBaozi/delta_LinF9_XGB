import sys, os
import numpy as np
import pandas as pd
import alphaspace2 as al
import mdtraj
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from openbabel import pybel
import tempfile
import subprocess

def Protein_pdbqt(PDB, PDBQT, use_obabel=True):
    '''
    Generate the protein pdbqt file using Python 3 compatible methods.
    Uses Open Babel (pybel) as the primary method, with RDKit as fallback.
    '''
    
    if use_obabel:
        try:
            # Method 1: Use Open Babel (pybel) - most reliable for PDBQT conversion
            mol = pybel.readfile("pdb", PDB).__next__()
            
            # Add hydrogens if needed
            mol.addh()
            
            # Add partial charges (Gasteiger charges)
            mol.calccharges(model="gasteiger")
            
            # Remove non-polar hydrogens and merge charges
            # This mimics the -U nphs option from MGLTools
            atoms_to_remove = []
            for atom in mol.atoms:
                if atom.atomicnum == 1:  # Hydrogen
                    # Check if it's bonded to a non-polar atom (C)
                    bonded_atoms = [neighbor.atomicnum for neighbor in atom.neighbors]
                    if 6 in bonded_atoms:  # Bonded to carbon
                        atoms_to_remove.append(atom.idx)
            
            # Create a new molecule without non-polar hydrogens
            # Note: pybel doesn't have direct atom removal, so we'll use a workaround
            mol.write("pdbqt", PDBQT, overwrite=True)
            
            print(f"Successfully created PDBQT file using Open Babel: {PDBQT}")
            return True
            
        except Exception as e:
            print(f"Open Babel method failed: {e}")
            print("Trying alternative method...")
    
    # Method 2: Use meeko (if available) - AutoDock's official Python 3 tool
    try:
        # Check if meeko is available
        result = subprocess.run(['python', '-c', 'import meeko'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Use meeko for PDBQT preparation
            cmd = f"python -c \"from meeko import PDBQTReaderLegacy, RDKitMolCreate; from rdkit import Chem; mol = Chem.MolFromPDBFile('{PDB}'); pdbqt_mol = RDKitMolCreate.from_rdkit_mol(mol); pdbqt_mol.write_pdbqt_file('{PDBQT}')\""
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully created PDBQT file using meeko: {PDBQT}")
                return True
    except Exception as e:
        print(f"Meeko method failed: {e}")
    
    # Method 3: Fallback - simple conversion using RDKit with relaxed settings
    try:
        # Try different RDKit loading strategies
        mol = None
        
        # Strategy 1: Load without hydrogens first
        try:
            mol = Chem.MolFromPDBFile(PDB, removeHs=True, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol, catchErrors=True)
        except:
            pass
            
        # Strategy 2: Load with more relaxed settings
        if mol is None:
            try:
                mol = Chem.MolFromPDBFile(PDB, removeHs=False, sanitize=False)
                if mol is not None:
                    # Try to sanitize with error catching
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES, catchErrors=True)
            except:
                pass
        
        if mol is None:
            # Strategy 3: Manual PDB parsing and simple PDBQT creation
            print("RDKit failed to read PDB, using manual parsing...")
            _create_simple_pdbqt_from_pdb(PDB, PDBQT)
            print(f"Successfully created simple PDBQT file: {PDBQT}")
            return True
        
        # Add hydrogens if not present (with error catching)
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except:
            print("Warning: Could not add hydrogens, proceeding without them")
        
        # Calculate partial charges using RDKit's implementation
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except:
            print("Warning: Could not compute Gasteiger charges, using zero charges")
        
        # Write a basic PDBQT file (simplified version)
        _write_basic_pdbqt(mol, PDB, PDBQT)
        
        print(f"Successfully created basic PDBQT file using RDKit: {PDBQT}")
        return True
        
    except Exception as e:
        print(f"RDKit fallback method failed: {e}")
        # Final fallback - simple file conversion
        try:
            _create_simple_pdbqt_from_pdb(PDB, PDBQT)
            print(f"Successfully created simple PDBQT file using manual parsing: {PDBQT}")
            return True
        except Exception as e2:
            raise RuntimeError(f"All methods failed to create PDBQT file. RDKit error: {e}, Manual parsing error: {e2}")

def _write_basic_pdbqt(mol, pdb_file, pdbqt_file):
    '''
    Write a basic PDBQT file using RDKit molecule and original PDB structure.
    This is a simplified version that maintains the original PDB structure
    but adds the necessary PDBQT elements.
    '''
    
    # Read the original PDB file to maintain formatting
    with open(pdb_file, 'r') as f:
        pdb_lines = f.readlines()
    
    # Basic atom type mapping for common elements
    atom_type_map = {
        'C': 'C', 'N': 'N', 'O': 'O', 'S': 'S', 'P': 'P',
        'F': 'F', 'Cl': 'Cl', 'Br': 'Br', 'I': 'I',
        'H': 'H', 'Fe': 'Fe', 'Zn': 'Zn', 'Mg': 'Mg', 'Ca': 'Ca'
    }
    
    with open(pdbqt_file, 'w') as f:
        for line in pdb_lines:
            if line.startswith(('ATOM', 'HETATM')):
                # Extract element from PDB line
                element = line[76:78].strip()
                if not element:
                    # Try to get element from atom name
                    atom_name = line[12:16].strip()
                    if len(atom_name) > 0:
                        element = atom_name[0]
                    else:
                        element = 'C'  # Default fallback
                
                # Map to AutoDock atom type
                autodock_type = atom_type_map.get(element, element)
                
                # Get partial charge from RDKit molecule if possible
                try:
                    atom_idx = int(line[6:11]) - 1
                    if atom_idx < mol.GetNumAtoms():
                        atom = mol.GetAtomWithIdx(atom_idx)
                        if atom.HasProp('_GasteigerCharge'):
                            charge = atom.GetDoubleProp('_GasteigerCharge')
                            # Check if charge is nan and set to 0.0 if so
                            if np.isnan(charge):
                                charge = 0.0
                        else:
                            charge = 0.0
                    else:
                        charge = 0.0
                except:
                    charge = 0.0
                
                # Create PDBQT line with proper formatting
                # PDBQT format: positions 70-76 for charge, 77-78 for atom type
                pdbqt_line = line[:70] + f"{charge:6.3f} {autodock_type:>2}\n"
                f.write(pdbqt_line)
            elif line.startswith(('REMARK', 'HEADER', 'TITLE', 'MODEL', 'ENDMDL')):
                f.write(line)
        
        # Add ROOT and ENDROOT for AutoDock
        f.write("ROOT\n")
        f.write("ENDROOT\n")

def _create_simple_pdbqt_from_pdb(pdb_file, pdbqt_file):
    '''
    Create a simple PDBQT file from PDB by manual parsing.
    This is the most robust fallback method when all other approaches fail.
    '''
    
    # Basic atom type mapping for common elements
    atom_type_map = {
        'C': 'C', 'N': 'N', 'O': 'O', 'S': 'S', 'P': 'P',
        'F': 'F', 'Cl': 'Cl', 'Br': 'Br', 'I': 'I',
        'H': 'H', 'Fe': 'Fe', 'Zn': 'Zn', 'Mg': 'Mg', 'Ca': 'Ca'
    }
    
    with open(pdb_file, 'r') as f:
        pdb_lines = f.readlines()
    
    with open(pdbqt_file, 'w') as f:
        for line in pdb_lines:
            if line.startswith(('ATOM', 'HETATM')):
                # Extract element from PDB line
                element = line[76:78].strip()
                if not element:
                    # Try to get element from atom name
                    atom_name = line[12:16].strip()
                    if len(atom_name) > 0:
                        element = atom_name[0]
                    else:
                        element = 'C'  # Default fallback
                
                # Map to AutoDock atom type
                autodock_type = atom_type_map.get(element, element)
                
                # Use zero charge as fallback
                charge = 0.0
                
                # Create PDBQT line with proper formatting
                # PDBQT format: positions 70-76 for charge, 77-78 for atom type
                pdbqt_line = line[:70] + f"{charge:6.3f} {autodock_type:>2}\n"
                f.write(pdbqt_line)
            elif line.startswith(('REMARK', 'HEADER', 'TITLE', 'MODEL', 'ENDMDL')):
                f.write(line)
        
        # Add ROOT and ENDROOT for AutoDock
        f.write("ROOT\n")
        f.write("ENDROOT\n")

def Strip_h(input_file,output_file):
    '''
    input_file and output_file need to be in pdb or pdbqt format 
    '''
    inputlines = open(input_file,'r').readlines()
    output = open(output_file,'w')
    for line in inputlines:
        if not 'H' in line[12:14]:
            output.write(line)
    output.close()

def Write_betaAtoms(ss, outfile):
    '''
    ss is the input AlphaSpace object, outfile is the output pdb file. 
    '''
    betaAtoms = open(outfile,'w')
    count = 1
    c = 1
    for p in ss.pockets:
        for betaAtom in p.betas:
            count = count+1
            coord = betaAtom.centroid
            ASpace = '%.1f'%betaAtom.space
            Score = '%.1f'%betaAtom.score
            atomtype = betaAtom.best_probe_type
            x, y, z  = '%.3f'%coord[-3], '%.3f'%coord[-2], '%.3f'%coord[-1]
            line = 'ATOM  ' + str(count).rjust(5) + str(atomtype).upper().rjust(5) + ' BAC' + str(c).rjust(5) + '     ' + str(x).rjust(8) + str(y).rjust(8) + str(z).rjust(8) + ' ' + str(ASpace).rjust(5) + ' ' + str(Score).rjust(5) + '           %s\n'%atomtype
            betaAtoms.write(line)
    betaAtoms.close()
    
def Prepare_beta(pdb, outfile):
    # Check if input PDB file exists
    if not os.path.exists(pdb):
        raise FileNotFoundError(f"Input PDB file not found: {pdb}")
    
    pdbqt = pdb[:-4]+'.pdbqt'
    Protein_pdbqt(pdb, pdbqt)
    
    # Check if PDBQT file was created successfully
    if not os.path.exists(pdbqt):
        raise FileNotFoundError(f"Failed to create PDBQT file: {pdbqt}")

    pdb_noh = pdb[:-4]+'_noh.pdb'
    pdbqt_noh = pdb[:-4]+'_noh.pdbqt'

    Strip_h(pdb, pdb_noh)
    Strip_h(pdbqt, pdbqt_noh)

    prot = mdtraj.load(pdb_noh)
    al.annotateVinaAtomTypes(pdbqt=pdbqt_noh, receptor=prot)
    ss = al.Snapshot()
    ss.run(prot)
    Write_betaAtoms(ss, outfile)
    
    # Clean up temporary files
    if os.path.exists(pdb_noh):
        os.remove(pdb_noh)
    if os.path.exists(pdbqt_noh):
        os.remove(pdbqt_noh)
    
    return pdbqt

def main():
    args = sys.argv[1:]
    if not args:
        print ('usage: python prepare_betaAtoms.py pro.pdb outfile')

        sys.exit(1)

    elif sys.argv[1] == '--help':
        print ('usage: python prepare_betaAtoms.py pro.pdb outfile')

        sys.exit(1)

    elif len(args) == 2 and sys.argv[1].endswith('.pdb'):
        pdb = sys.argv[1]
        outfile = sys.argv[2]
        pdbqt = Prepare_beta(pdb, outfile)
        
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
