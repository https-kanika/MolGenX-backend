from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import py3Dmol
from IPython.display import display
import os
import requests

def visualize_simple(compounds, show_protein=True,pdb_id=None):
    output_dir = "compound_visualizations"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
    # 1. Generate individual 2D images for each compound (as PNG)
    print("Visualizing top compounds:")
    
    for i, comp in enumerate(compounds):
        if comp['molecule'] is not None:
            mol = comp['molecule']
            img = Draw.MolToImage(
                mol,
                size=(500, 500),
                legend=f"Compound {i+1}\nScore: {comp['score']:.2f}"
            )
            filename_2d = f"{output_dir}/compound_{i+1}_2D.png"
            img.save(filename_2d)
            print(f"2D visualization saved as '{filename_2d}'")
            try:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_3d)
                pdb_str = Chem.MolToPDBBlock(mol_3d)
                pdb_filename = f"{output_dir}/compound_{i+1}_3D.pdb"
                with open(pdb_filename, 'w') as f:
                    f.write(pdb_str)
                print(f"3D structure saved as PDB: '{pdb_filename}'")

                sdf_filename = f"{output_dir}/compound_{i+1}_3D.sdf"
                writer = Chem.SDWriter(sdf_filename)
                writer.write(mol_3d)
                writer.close()
                print(f"3D structure saved as SDF: '{sdf_filename}'")
                
                # Save HTML file with embedded 3D viewer
                html_filename = f"{output_dir}/compound_{i+1}_3D_viewer.html"
                
                view = py3Dmol.view(width=600, height=500)
                view.addModel(pdb_str, 'pdb')
                view.setStyle({'stick': {'radius': 0.2, 'colorscheme': 'cyanCarbon'}})
                view.addStyle({'atom': 'C'}, {'sphere': {'radius': 0.4, 'color': 'cyan'}})
                view.addStyle({'atom': 'O'}, {'sphere': {'radius': 0.4, 'color': 'red'}})
                view.addStyle({'atom': 'N'}, {'sphere': {'radius': 0.4, 'color': 'blue'}})
                view.addStyle({'atom': 'S'}, {'sphere': {'radius': 0.4, 'color': 'yellow'}})
                view.addStyle({'atom': 'Cl'}, {'sphere': {'radius': 0.4, 'color': 'green'}})
                view.setBackgroundColor('white')
                view.zoomTo()
                
                html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Compound {i+1} - 3D Structure</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
</head>
<body>
    <h2>Compound {i+1}</h2>
    <p>SMILES: {comp['smiles']}</p>
    <p>Score: {comp['score']:.2f}</p>
    <div id="container" style="width: 600px; height: 500px; position: relative;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById("container"));
        let pdbData = `{pdb_str}`;
        viewer.addModel(pdbData, "pdb");
        viewer.setStyle({{}}, {{"stick": {{"radius": 0.2, "colorscheme": "cyanCarbon"}}}});
        viewer.addStyle({{"atom": "C"}}, {{"sphere": {{"radius": 0.4, "color": "cyan"}}}});
        viewer.addStyle({{"atom": "O"}}, {{"sphere": {{"radius": 0.4, "color": "red"}}}});
        viewer.addStyle({{"atom": "N"}}, {{"sphere": {{"radius": 0.4, "color": "blue"}}}});
        viewer.addStyle({{"atom": "S"}}, {{"sphere": {{"radius": 0.4, "color": "yellow"}}}});
        viewer.addStyle({{"atom": "Cl"}}, {{"sphere": {{"radius": 0.4, "color": "green"}}}});
        viewer.setBackgroundColor("white");
        viewer.zoomTo();
        viewer.render();
    </script>
</body>
</html>
"""
                
                with open(html_filename, 'w') as f:
                    f.write(html_content)
                print(f"Interactive 3D viewer saved as HTML: '{html_filename}'")
                
                # Display in notebook if in interactive environment
                try:
                    display(img)
                    display(view)
                    print(f"Compound {i+1}: {comp['smiles']}")
                    print("Properties:")
                    for metric, value in comp['metrics'].items():
                        print(f"  {metric}: {value:.3f}")
                    print("-" * 50)
                except Exception as display_error:
                    print(f"Note: Could not display in interactive environment")
            
            except Exception as e:
                print(f"Error generating 3D visualization for compound {i+1}: {e}")

    mols = [comp['molecule'] for comp in compounds if comp['molecule'] is not None]
    if mols:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(300, 300),
            legends=[f"Compound {i+1}\nScore: {compounds[i]['score']:.2f}" 
                    for i in range(len(mols))]
        )

        grid_filename = f"{output_dir}/all_compounds_grid.png"
        img.save(grid_filename)
        print(f"Grid visualization saved as '{grid_filename}'")
    
    # Show the target protein (if requested)
    if show_protein:
        try:
           
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(pdb_url)
            
            if response.status_code == 200:
                protein_pdb_filename = f"{output_dir}/target_protein.pdb"
                with open(protein_pdb_filename, 'w') as f:
                    f.write(response.text)
                print(f"Protein structure saved as PDB: '{protein_pdb_filename}'")
                protein_html_filename = f"{output_dir}/target_protein.html"
                protein_html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EGFR Kinase Domain - 3D Structure</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
</head>
<body>
    <h2>EGFR Kinase Domain (PDB ID: {pdb_id})</h2>
    <div id="container" style="width: 800px; height: 600px; position: relative;"></div>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById("container"));
        viewer.addPDBURL("https://files.rcsb.org/download/{pdb_id}.pdb", function() {{
            viewer.setStyle({{}}, {{"cartoon": {{"colorscheme": "spectrum"}}}});
            viewer.setStyle({{"hetflag": true}}, {{"stick": {{"colorscheme": "greenCarbon"}}}});
            viewer.zoomTo();
            viewer.render();
        }});
    </script>
</body>
</html>
"""
                
                with open(protein_html_filename, 'w') as f:
                    f.write(protein_html_content)
                print(f"Interactive protein viewer saved as HTML: '{protein_html_filename}'")
                
                view = py3Dmol.view(query=f'pdb:{pdb_id}', width=800, height=600)
                view.setStyle({'cartoon': {'colorscheme': 'spectrum'}})
                view.zoomTo()
                
                try:
                    display(view)
                    print(f"Target Protein: EGFR kinase domain (PDB ID: {pdb_id})")
                except:
                    pass
            else:
                print(f"Could not fetch protein structure: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error displaying protein: {str(e)}")