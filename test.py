import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

class MolGenXComparativeTest:
    def __init__(self, api_url="http://localhost:3000"):
        """Initialize the comparative test harness"""
        self.api_url = api_url
        self.results_dir = Path("comparative_results")
        self.results_dir.mkdir(exist_ok=True)
        self.protein_library = self._load_protein_library()
        
    def _load_protein_library(self):
        """Load reference data for proteins and their known ligands"""
        return {
            "EGFR": {
                "pdb_id": "1M17",
                "known_compounds": [
                    {"name": "Erlotinib", "smiles": "CNC(=O)C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1", "activity": 0.2},
                    {"name": "Gefitinib", "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "activity": 0.033},
                    {"name": "Osimertinib", "smiles": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1", "activity": 0.001}
                ],
                "description": "Epidermal Growth Factor Receptor - Cancer drug target"
            },
            "COX2": {
                "pdb_id": "5KIR",
                "known_compounds": [
                    {"name": "Celecoxib", "smiles": "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "activity": 0.04},
                    {"name": "Rofecoxib", "smiles": "CS(=O)(=O)c1ccc(cc1)C1=C(C(=O)OC1)c1ccccc1", "activity": 0.018},
                    {"name": "Etoricoxib", "smiles": "CCOC1=C(C=CC(=C1)C1=NC(=C(S1)C(=O)C)C1=CC=C(C=C1)Cl)C", "activity": 0.0015}
                ],
                "description": "Cyclooxygenase-2 - Anti-inflammatory target"
            },
            "ACE2": {
                "pdb_id": "6M0J",
                "known_compounds": [
                    {"name": "Lisinopril", "smiles": "NCCCCC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C(N)CC1=CC=CC=C1)C(=O)O", "activity": 0.1},
                    {"name": "Enalaprilat", "smiles": "CCOC(=O)C(CCC(=O)O)NC(C)C(=O)N1CCCC1C(=O)O", "activity": 0.036},
                    {"name": "Captopril", "smiles": "CC(CS)C(=O)N1CCCC1C(=O)O", "activity": 0.023}
                ],
                "description": "Angiotensin-converting enzyme 2 - Cardiovascular disease target"
            },
            "HIV_Protease": {
                "pdb_id": "6B3S",
                "known_compounds": [
                    {"name": "Saquinavir", "smiles": "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(O)C(Cc1ccccc1)NC(=O)C(CC(N)=O)NC(=O)c1ccc2ccccc2n1", "activity": 0.004},
                    {"name": "Ritonavir", "smiles": "CC(C)c1nc(CN(C)C(=O)NC(C(=O)NC(Cc2ccccc2)CC(O)C(Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1", "activity": 0.015},
                    {"name": "Darunavir", "smiles": "CC(C)CN(CC(O)C(Cc1ccccc1)NC(=O)OC1COC2CCCCC2O1)S(=O)(=O)C1CC1", "activity": 0.0001}
                ],
                "description": "HIV-1 Protease - Antiviral target"
            }
        }
    
    def calculate_similarity(self, smiles1, smiles2):
        """Calculate Tanimoto similarity between two SMILES strings"""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if mol1 is None or mol2 is None:
                return 0.0
                
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0
    
    def evaluate_protein(self, protein_key, num_top_compounds=10):
        """Run MolGenX on a protein and compare to known compounds"""
        protein_data = self.protein_library.get(protein_key)
        if not protein_data:
            print(f"Error: Protein {protein_key} not found in library")
            return None
            
        print(f"\n{'='*80}")
        print(f"Evaluating {protein_key}: {protein_data['description']}")
        print(f"PDB ID: {protein_data['pdb_id']}")
        print(f"Known compounds: {len(protein_data['known_compounds'])}")
        print(f"{'='*80}\n")
        
        # Define protein sequences for known PDB IDs
        protein_sequences = {
            "1M17": "FKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLIMQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQG",
            "5KIR": "MLFRVTLRWKLFSNGKKYTFSDPQMPNSTWYQLLASDLLIGNRMTSPVCPSLLTQEVIAAPDELPFILVHYNSDHALFPLLQAGLAENNEIDVRYLEDKPIPVIFKASFDLKSWKLVKIQVSGRGPTGVSDLLSSVLPSPEAQYVSYLRELCYRGLFGRARAMELLLGLSPDPNKLNNQQVFSSLFAGIDLSSLSLHLLEIYKDPLPSSEISAVCPLEVRGYGAAVLKSLVSPLISGYFEIDSVSWHLRSNVSQLDLSAYFEEQPLLIQRGSAPADPHFLLHIEALEPVLCRWQLQAAPLGIRLLFNSGVGPAKDMISPSLHYQDLARNYQQQQLVQRAFNRFLQKMFIPRTSLDMSQALDIQNMALFIQRELRSSIRLLRSALSLLRTLLALGPGGLGEAFGLVCAASLSTMFLGLLVVLYVGHDTAIRQARSAGVALMTGAFLHLGALCAGLLLADVNLVTSPITRLLLLLSPPALPSAPLLVALGAVMGPGGLAFLLTHRGVLNAPPLCLLVILATHSLFGPQAVFGLCFLAVICSVVLGIVLCLNPWTAWVPSLCLLVICAFCFCALPFAPGHMLFLDHALAPGGLFMASGFLLGFAPAWSLALGLLALGVTLCLSLAFPLGPGRLCLLVTGLFLVAALLALSSCLTVQDFCALAPPTLQVSFLLLCSVLLSVTMGAASAGQLCECAFGSFPGSETKAFLSLSLNSDSEAEAEPLLSHFVLLSACEAPAASSRPSAAPPTGHGALPLGSKEPGDTATARPWFTPGPRLALCFQDLMRQNPLAFSSAGTQCTTSGWLLSPRGVEAQRAACDVDTCARGRAPARGGRGCSRPTSDLSHVLRTALRSYRPAAELQEGLRRLGAGPDSPAQPVQAERSLSMDKAVLTGWSALQTFRGRPSVRLDYAHSLLLSEHRPAAEALQGLRRLLRARSPAQPVPQDTSLSLDRGVQTGCDALQTFRGHPDVRLDHSSSLLNPGRLRGAFDQLWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRG",
            "6M0J": "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYADQSIKVRISLKSALGDKAYEWNDNEMYLFRSSVAYAMRQYFLKVKNQMILFGEEDVRVANLKPRISFNFFVTAPKNVSDIIPRTEVEKAIRMSRSRINDAFRLNDNSLEFLGIQPTLGPPNQPPVSIWLIVFGVVMGVIVVGIVILIFTGIRDRKKKNKARSGENPYASIDISKGENNPGFQNTDDVQTSF",
            "6B3S": "PQITLWKRPLVTIRIGGQLKEALLDTGADDTVLEEMNLPGKWKPKMIGGIGGFIKVRQYDQIPVEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
        }
        protein_sequence = protein_sequences.get(protein_data["pdb_id"], "")
    
        # Prepare the API payload with BOTH pdb_id and protein
        payload = {
            "pdb_id": protein_data["pdb_id"],
            "protein": protein_sequence,
            "weights": {
                "druglikeness": 1.0,
                "synthetic_accessibility": 0.8,
                "lipinski_violations": 0.7,
                "toxicity": 1.2,
                "binding_affinity": 1.5,
                "solubility": 0.6
            },
            "generate_visualizations": False
        }

        # Call the API
        print(f"Calling MolGenX API for {protein_key}...")
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_url}/api/optimize",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            elapsed_time = time.time() - start_time
            print(f"API call completed in {elapsed_time:.2f} seconds")
            
            if response.status_code != 200:
                print(f"Error: API returned status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
            # Process the results
            result_data = response.json()
            compounds_df = pd.read_json(result_data["optimized_compounds"])
            
            # Save raw results
            raw_results_file = self.results_dir / f"{protein_key}_raw_results.json"
            with open(raw_results_file, "w") as f:
                json.dump(result_data, f, indent=2)
            
            print(f"Generated {len(compounds_df)} candidate compounds")
            
            # Compare with known compounds
            comparison_results = []
            for known in protein_data["known_compounds"]:
                known_smiles = known["smiles"]
                
                # Find most similar generated compounds
                similarities = []
                for _, row in compounds_df.iterrows():
                    similarity = self.calculate_similarity(known_smiles, row["smiles"])
                    similarities.append({
                        "rank": row["rank"],
                        "generated_smiles": row["smiles"],
                        "similarity": similarity,
                        "druglikeness": row["druglikeness"],
                        "binding_affinity": row["binding_affinity"],
                        "toxicity": row["toxicity"]
                    })
                
                # Sort by similarity and get top matches
                similarities.sort(key=lambda x: x["similarity"], reverse=True)
                top_similarities = similarities[:num_top_compounds]
                
                comparison_results.append({
                    "known_compound": known["name"],
                    "known_smiles": known_smiles,
                    "known_activity": known["activity"],
                    "top_matches": top_similarities,
                    "max_similarity": top_similarities[0]["similarity"] if top_similarities else 0
                })
            
            # Create comprehensive results
            evaluation_results = {
                "protein": protein_key,
                "pdb_id": protein_data["pdb_id"],
                "description": protein_data["description"],
                "generated_compounds": len(compounds_df),
                "comparison": comparison_results,
                "avg_max_similarity": np.mean([r["max_similarity"] for r in comparison_results]),
                "elapsed_time": elapsed_time
            }
            
            # Save evaluation results
            eval_file = self.results_dir / f"{protein_key}_evaluation.json"
            with open(eval_file, "w") as f:
                json.dump(evaluation_results, f, indent=2)
            
            # Generate visualization
            self._visualize_comparison(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None
    
    def _visualize_comparison(self, evaluation_results):
        """Create visualizations for the evaluation results"""
        protein_key = evaluation_results["protein"]
        
        # 1. Similarity scores for each known compound
        similarities = [result["max_similarity"] for result in evaluation_results["comparison"]]
        compound_names = [result["known_compound"] for result in evaluation_results["comparison"]]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(compound_names, similarities, color=sns.color_palette("viridis", len(similarities)))
        plt.ylabel("Max Similarity Score (Tanimoto)")
        plt.title(f"Similarity to Known {protein_key} Ligands")
        plt.ylim(0, 1.0)
        plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label="Good Similarity (0.7)")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', 
                    ha='center', va='bottom', rotation=0)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{protein_key}_similarities.png")
        plt.close()
        
        # 2. Properties of top compounds compared to known compounds
        plt.figure(figsize=(12, 8))
        
        for i, result in enumerate(evaluation_results["comparison"]):
            known_name = result["known_compound"]
            top_match = result["top_matches"][0]
            
            plt.subplot(len(evaluation_results["comparison"]), 1, i+1)
            properties = ["similarity", "druglikeness", "binding_affinity"]
            values = [top_match[prop] for prop in properties]
            
            bars = plt.bar(properties, values, color=sns.color_palette("Paired", len(properties)))
            plt.title(f"Top Match for {known_name}")
            plt.tight_layout()
        
        plt.savefig(self.results_dir / f"{protein_key}_properties.png")
        plt.close()
    
    def run_comparative_analysis(self, proteins=None):
        """Run evaluation for multiple proteins and create a summary"""
        if proteins is None:
            proteins = list(self.protein_library.keys())
            
        print(f"Running comparative analysis for {len(proteins)} proteins")
        
        all_results = []
        for protein in proteins:
            results = self.evaluate_protein(protein)
            if results:
                all_results.append(results)
                
        # Create summary report
        if all_results:
            summary = {
                "tests_run": len(all_results),
                "total_proteins": len(proteins),
                "average_similarity": np.mean([r["avg_max_similarity"] for r in all_results]),
                "individual_results": [
                    {
                        "protein": r["protein"],
                        "avg_similarity": r["avg_max_similarity"],
                        "generated_compounds": r["generated_compounds"],
                        "elapsed_time": r["elapsed_time"]
                    } for r in all_results
                ]
            }
            
            # Save summary
            summary_file = self.results_dir / "comparative_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
                
            # Create summary visualization
            self._create_summary_chart(summary)
            
            # Generate HTML report
            self._generate_html_report(all_results, summary)
            
            print(f"\n{'='*80}")
            print(f"COMPARATIVE ANALYSIS COMPLETE")
            print(f"{'='*80}")
            print(f"Overall average similarity: {summary['average_similarity']:.2f}")
            print(f"Detailed results saved to: {self.results_dir}")
            
    def _create_summary_chart(self, summary):
        """Create a summary chart for all evaluated proteins"""
        results = summary["individual_results"]
        
        proteins = [r["protein"] for r in results]
        similarities = [r["avg_similarity"] for r in results]
        
        plt.figure(figsize=(12, 6))
        
        # 1. Similarity by protein
        plt.subplot(1, 2, 1)
        bars = plt.bar(proteins, similarities, color=sns.color_palette("viridis", len(proteins)))
        plt.ylabel("Average Max Similarity")
        plt.title("Average Similarity by Protein Target")
        plt.ylim(0, 1.0)
        plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', 
                    ha='center', va='bottom', rotation=0)
        
        # 2. Compounds per target
        plt.subplot(1, 2, 2)
        compounds = [r["generated_compounds"] for r in results]
        plt.bar(proteins, compounds, color=sns.color_palette("muted", len(proteins)))
        plt.ylabel("Number of Compounds Generated")
        plt.title("Generated Compounds by Protein Target")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "comparative_summary.png")
        plt.close()
    
    def _generate_html_report(self, all_results, summary):
        """Generate an HTML report with the results"""
        # Create HTML content as before
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>MolGenX Comparative Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .protein-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 30px; }}
                .protein-header {{ background-color: #e9e9e9; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
                .compound-comparison {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .known-compound {{ flex: 1; min-width: 300px; border: 1px solid #ccc; border-radius: 5px; padding: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .similarity-high {{ color: green; font-weight: bold; }}
                .similarity-medium {{ color: orange; }}
                .similarity-low {{ color: red; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>MolGenX Comparative Analysis Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Tests Run:</strong> {summary["tests_run"]}/{summary["total_proteins"]} proteins</p>
                <p><strong>Overall Average Similarity:</strong> {summary["average_similarity"]:.2f}</p>
                
                <img src="comparative_summary.png" alt="Summary Chart">
            </div>
        """
        
        # Add individual protein results
        for result in all_results:
            protein = result["protein"]
            html_content += f"""
            <div class="protein-card">
                <div class="protein-header">
                    <h2>{protein}: {result["description"]}</h2>
                    <p><strong>PDB ID:</strong> {result["pdb_id"]}</p>
                    <p><strong>Generated Compounds:</strong> {result["generated_compounds"]}</p>
                    <p><strong>Average Max Similarity:</strong> {result["avg_max_similarity"]:.2f}</p>
                    <p><strong>Analysis Time:</strong> {result["elapsed_time"]:.2f} seconds</p>
                </div>
                
                <img src="{protein}_similarities.png" alt="{protein} Similarities">
                <img src="{protein}_properties.png" alt="{protein} Properties">
                
                <h3>Comparison with Known Compounds</h3>
                <div class="compound-comparison">
            """
            
            # Add each known compound comparison
            for comparison in result["comparison"]:
                known_name = comparison["known_compound"]
                max_similarity = comparison["max_similarity"]
                
                # Determine similarity class
                sim_class = "similarity-low"
                if max_similarity >= 0.7:
                    sim_class = "similarity-high"
                elif max_similarity >= 0.5:
                    sim_class = "similarity-medium"
                
                # Use 'uM' instead of 'μM' to avoid encoding issues
                html_content += f"""
                <div class="known-compound">
                    <h4>{known_name}</h4>
                    <p><strong>SMILES:</strong> <span title="{comparison["known_smiles"]}">{comparison["known_smiles"][:50]}...</span></p>
                    <p><strong>Known Activity:</strong> {comparison["known_activity"]} uM</p>
                    <p><strong>Max Similarity:</strong> <span class="{sim_class}">{max_similarity:.2f}</span></p>
                    
                    <h5>Top Matching Compounds</h5>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Similarity</th>
                            <th>Druglikeness</th>
                            <th>Binding Affinity</th>
                        </tr>
                """
                
                # Add top matches
                for match in comparison["top_matches"][:3]:  # Show top 3
                    html_content += f"""
                    <tr>
                        <td>{match["rank"]}</td>
                        <td>{match["similarity"]:.2f}</td>
                        <td>{match["druglikeness"]:.2f}</td>
                        <td>{match["binding_affinity"]:.2f}</td>
                    </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write the HTML file with explicit UTF-8 encoding
        html_file = self.results_dir / "comparative_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"Generated HTML report: {html_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comparative analysis for MolGenX API")
    parser.add_argument("--url", default="http://localhost:3000", help="API base URL")
    parser.add_argument("--proteins", nargs="+", help="Specific proteins to test (default: all)")
    args = parser.parse_args()
    
    test = MolGenXComparativeTest(api_url=args.url)
    test.run_comparative_analysis(proteins=args.proteins)