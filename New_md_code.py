!pip install MDAnalysis plotly pandas numpy matplotlib

import os
import csv
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from MDAnalysis.analysis import align

# =====================================================================
# 0. DIRECTORIES & FILE PATHS
# =====================================================================

# --- NEW: Define and create a dedicated output directory in Kaggle ---
OUTPUT_DIR = "/kaggle/working/md_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"All outputs will be saved to: {OUTPUT_DIR}")

# Trajectory and Topology paths
pdb_trajectory_r = "/kaggle/input/datasets/kirushikesan/dcd-and-one-prmtop-file/Ref_pep_third_batch_0_200_ns_autoimage.dcd"
prmtop_path_r = "/kaggle/input/datasets/kirushikesan/dcd-and-one-prmtop-file/complex_YDVD_amber_2.prmtop"

pdb_trajectory_1 = "/kaggle/input/datasets/kirushikesan/all-dcd-files/Pep_1_third_batch_0_200_ns.dcd"
prmtop_path_1 = "/kaggle/input/datasets/kirushikesan/all-complex-prmtop-files/complex_5_1.prmtop"

pdb_trajectory_2 = "/kaggle/input/datasets/kirushikesan/all-dcd-files/Pep_2_third_batch_0_200_ns.dcd"
prmtop_path_2 = "/kaggle/input/datasets/kirushikesan/all-complex-prmtop-files/complex_pep_5_2.prmtop"

pdb_trajectory_3 = "/kaggle/input/datasets/kirushikesan/all-dcd-files/pep_3_third_batch_0_200_ns_MD_file.dcd"
prmtop_path_3 = "/kaggle/input/datasets/kirushikesan/all-complex-prmtop-files/complex_pep_3.prmtop"

pdb_trajectory_4 = "/kaggle/input/datasets/kirushikesan/all-dcd-files/pep_4_third_batch_0_200_ns_MD_file_autoimage.dcd"
prmtop_path_4 = "/kaggle/input/datasets/kirushikesan/all-complex-prmtop-files/5_4_bestpose_complex.prmtop"


# =====================================================================
# HIGH-RESOLUTION EXPORT SETTINGS (matplotlib-based, no Chrome needed)
# =====================================================================
PNG_DPI = 300          # 300 = print-quality
PNG_FIGSIZE = (10, 6)  # inches

def save_line_plot(x_r, y_r, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, 
                   title, xlabel, ylabel, base_name, 
                   name_r, name_1, name_2, name_3, name_4, 
                   marker=None, line_shape='linear'):
    """
    Saves an interactive Plotly HTML AND a high-resolution matplotlib PNG 
    superimposing 5 distinct systems into the OUTPUT_DIR.
    """
    # --- Interactive HTML (Plotly) ---
    fig = go.Figure()
    mode = 'lines+markers' if marker else 'lines'
 
    datasets = [
        (x_r, y_r, name_r), (x_1, y_1, name_1), 
        (x_2, y_2, name_2), (x_3, y_3, name_3), (x_4, y_4, name_4)
    ]
    
    for x, y, name in datasets:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode=mode,
            line=dict(shape='hv') if line_shape == 'hv' else None,
            marker=dict(size=marker) if marker else None,
            name=name
        ))
        
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white',
        hovermode='x unified'
    )
    fig.show()
    fig.write_html(os.path.join(OUTPUT_DIR, f"{base_name}.html"))

    # --- High-resolution PNG (Matplotlib) ---
    plt.figure(figsize=PNG_FIGSIZE)
    drawstyle = 'steps-post' if line_shape == 'hv' else 'default'
    
    for x, y, name in datasets:
        if marker:
            plt.plot(x, y, linewidth=2, marker='o', markersize=3, drawstyle=drawstyle, label=name)
        else:
            plt.plot(x, y, linewidth=2, drawstyle=drawstyle, label=name)

    plt.legend() 
    plt.title(title, fontweight='bold', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.png"), dpi=PNG_DPI)
    plt.close()

    return fig


def save_line_plot_1(x, y, title, xlabel, ylabel, base_name, marker=None, line_shape='linear'):
    """
    Saves an interactive Plotly HTML AND a high-resolution matplotlib PNG for single traces.
    """
    fig = go.Figure()
    mode = 'lines+markers' if marker else 'lines'
   
    fig.add_trace(go.Scatter(
        x=x, y=y, mode=mode,
        line=dict(shape='hv') if line_shape == 'hv' else None,
        marker=dict(size=marker) if marker else None
    ))
    
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white',
        hovermode='x unified'
    )
    fig.show()
    fig.write_html(os.path.join(OUTPUT_DIR, f"{base_name}.html"))

    plt.figure(figsize=PNG_FIGSIZE)
    drawstyle = 'steps-post' if line_shape == 'hv' else 'default'
    if marker:
        plt.plot(x, y, linewidth=2, marker='o', markersize=3, drawstyle=drawstyle)
    else:
        plt.plot(x, y, linewidth=2, drawstyle=drawstyle)
        
    plt.title(title, fontweight='bold', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.png"), dpi=PNG_DPI)
    plt.close()

    return fig


# =====================================================================
# LOAD & ALIGN UNIVERSES
# =====================================================================
u_r = mda.Universe(prmtop_path_r, pdb_trajectory_r, in_memory=True)
ref_r = mda.Universe(prmtop_path_r, pdb_trajectory_r, in_memory=True)
align.AlignTraj(u_r, ref_r, select="protein and name CA", ref_frame=0, in_memory=True).run()

u_1 = mda.Universe(prmtop_path_1, pdb_trajectory_1, in_memory=True)
ref_1 = mda.Universe(prmtop_path_1, pdb_trajectory_1, in_memory=True)
align.AlignTraj(u_1, ref_1, select="protein and name CA", ref_frame=0, in_memory=True).run()

u_2 = mda.Universe(prmtop_path_2, pdb_trajectory_2, in_memory=True)
ref_2 = mda.Universe(prmtop_path_2, pdb_trajectory_2, in_memory=True)
align.AlignTraj(u_2, ref_2, select="protein and name CA", ref_frame=0, in_memory=True).run()

u_3 = mda.Universe(prmtop_path_3, pdb_trajectory_3, in_memory=True)
ref_3 = mda.Universe(prmtop_path_3, pdb_trajectory_3, in_memory=True)
align.AlignTraj(u_3, ref_3, select="protein and name CA", ref_frame=0, in_memory=True).run()

u_4 = mda.Universe(prmtop_path_4, pdb_trajectory_4, in_memory=True)
ref_4 = mda.Universe(prmtop_path_4, pdb_trajectory_4, in_memory=True)
align.AlignTraj(u_4, ref_4, select="protein and name CA", ref_frame=0, in_memory=True).run()


# =====================================================================
# LIGAND SELECTIONS
# =====================================================================
def set_ligand_sel(u):
    return "resname UNL" if len(u.select_atoms("resname UNL")) > 0 else "protein and resid 167-171"

ligand_sel_r = set_ligand_sel(u_r)
ligand_sel_1 = set_ligand_sel(u_1)
ligand_sel_2 = set_ligand_sel(u_2)
ligand_sel_3 = set_ligand_sel(u_3)
ligand_sel_4 = set_ligand_sel(u_4)


# =====================================================================
# 1. LIGAND RMSD
# =====================================================================
print("Calculating Ligand RMSD...")

R_ligand_r = rms.RMSD(u_r, ref_r, select=ligand_sel_r, ref_frame=0).run()
frames_r = R_ligand_r.results.rmsd[:, 1] * 100000 * 0.002
ligand_rmsd_r = R_ligand_r.results.rmsd[:, 2]
pd.DataFrame({'Frame_ns': frames_r, 'Ligand_RMSD_A': ligand_rmsd_r}).to_csv(os.path.join(OUTPUT_DIR, "ligand_rmsd_r.csv"), index=False)

R_ligand_1 = rms.RMSD(u_1, ref_1, select=ligand_sel_1, ref_frame=0).run()
frames_1 = R_ligand_1.results.rmsd[:, 1] / 1000
ligand_rmsd_1 = R_ligand_1.results.rmsd[:, 2]
pd.DataFrame({'Frame_ns': frames_1, 'Ligand_RMSD_A': ligand_rmsd_1}).to_csv(os.path.join(OUTPUT_DIR, "ligand_rmsd_1.csv"), index=False)

R_ligand_2 = rms.RMSD(u_2, ref_2, select=ligand_sel_2, ref_frame=0).run()
frames_2 = R_ligand_2.results.rmsd[:, 1] / 1000
ligand_rmsd_2 = R_ligand_2.results.rmsd[:, 2]
pd.DataFrame({'Frame_ns': frames_2, 'Ligand_RMSD_A': ligand_rmsd_2}).to_csv(os.path.join(OUTPUT_DIR, "ligand_rmsd_2.csv"), index=False)

R_ligand_3 = rms.RMSD(u_3, ref_3, select=ligand_sel_3, ref_frame=0).run()
frames_3 = R_ligand_3.results.rmsd[:, 1] * 100000 * 0.002
ligand_rmsd_3 = R_ligand_3.results.rmsd[:, 2]
pd.DataFrame({'Frame_ns': frames_3, 'Ligand_RMSD_A': ligand_rmsd_3}).to_csv(os.path.join(OUTPUT_DIR, "ligand_rmsd_3.csv"), index=False)

R_ligand_4 = rms.RMSD(u_4, ref_4, select=ligand_sel_4, ref_frame=0).run()
frames_4 = R_ligand_4.results.rmsd[:, 1] * 100000 * 0.002
ligand_rmsd_4 = R_ligand_4.results.rmsd[:, 2]
pd.DataFrame({'Frame_ns': frames_4, 'Ligand_RMSD_A': ligand_rmsd_4}).to_csv(os.path.join(OUTPUT_DIR, "ligand_rmsd_4.csv"), index=False)

save_line_plot(
    frames_r, ligand_rmsd_r, frames_1, ligand_rmsd_1, frames_2, ligand_rmsd_2, frames_3, ligand_rmsd_3, frames_4, ligand_rmsd_4,
    name_r="Reference RMSD", name_1="Peptide1 RMSD", name_2="Peptide2 RMSD", name_3="Peptide3 RMSD", name_4="Peptide4 RMSD",
    title="Superimposed Ligand RMSD over Time", xlabel='ns', ylabel='RMSD (Å)',
    base_name="Superimposed_Ligand_RMSD_over_Time"
)


# =====================================================================
# 2. PROTEIN RMSD
# =====================================================================
print("Calculating Protein RMSD...")

protein_rmsd_r = rms.RMSD(u_r, ref_r, select=f"protein and backbone and not ({ligand_sel_r})", ref_frame=0).run().results.rmsd[:, 2]
protein_rmsd_1 = rms.RMSD(u_1, ref_1, select=f"protein and backbone and not ({ligand_sel_1})", ref_frame=0).run().results.rmsd[:, 2]
protein_rmsd_2 = rms.RMSD(u_2, ref_2, select=f"protein and backbone and not ({ligand_sel_2})", ref_frame=0).run().results.rmsd[:, 2]
protein_rmsd_3 = rms.RMSD(u_3, ref_3, select=f"protein and backbone and not ({ligand_sel_3})", ref_frame=0).run().results.rmsd[:, 2]
protein_rmsd_4 = rms.RMSD(u_4, ref_4, select=f"protein and backbone and not ({ligand_sel_4})", ref_frame=0).run().results.rmsd[:, 2]

save_line_plot(
    frames_r, protein_rmsd_r, frames_1, protein_rmsd_1, frames_2, protein_rmsd_2, frames_3, protein_rmsd_3, frames_4, protein_rmsd_4,
    name_r="Reference", name_1="Peptide 1", name_2="Peptide 2", name_3="Peptide 3", name_4="Peptide 4",
    title='Superimposed Protein Backbone RMSD over Time', xlabel='ns', ylabel='RMSD (Å)',
    base_name="Superimposed_Protein_RMSD_over_Time"
)


# =====================================================================
# 3. HYDROGEN BOND ANALYSIS
# =====================================================================
print("Running Hydrogen Bond Analysis...")

def run_hba(u, sel1, sel2, prefix):
    hb = HBA(universe=u, between=[sel1, sel2], d_a_cutoff=3.5, d_h_a_angle_cutoff=150).run()
    all_data = []
    counts = np.zeros(len(u.trajectory))
    
    for bond in hb.results.hbonds:
        f_idx = int(bond[0])
        d_atom, a_atom = u.atoms[int(bond[1])], u.atoms[int(bond[3])]
        counts[f_idx] += 1
        all_data.append([
            f_idx, f"{d_atom.resname}{d_atom.resid}", d_atom.name,
            f"{a_atom.resname}{a_atom.resid}", a_atom.name, round(bond[4], 3), round(bond[5], 3)
        ])
        
    pd.DataFrame(all_data, columns=["frame", "donor_res", "donor_atom", "acceptor_res", "acceptor_atom", "distance_a", "angle_deg"]).to_csv(os.path.join(OUTPUT_DIR, f"all_frames_hbonds_{prefix}.csv"), index=False)
    df_count = pd.DataFrame({"frame": np.arange(len(u.trajectory)), "count": counts})
    df_count.to_csv(os.path.join(OUTPUT_DIR, f"hbond_count_{prefix}.csv"), index=False)
    return df_count

df_hb_r = run_hba(u_r, 'protein', 'resname UNK' if len(u_r.select_atoms("resname UNK")) > 0 else 'resname UNL', 'r')
df_hb_1 = run_hba(u_1, 'protein and not resid 167-171', 'resid 167-171', '1')
df_hb_2 = run_hba(u_2, 'protein and not resid 167-171', 'resid 167-171', '2')
df_hb_3 = run_hba(u_3, 'protein and not resid 167-171', 'resid 167-171', '3')
df_hb_4 = run_hba(u_4, 'protein', 'resname UNL', '4')

save_line_plot(
    df_hb_r["frame"], df_hb_r["count"], df_hb_1["frame"], df_hb_1["count"], df_hb_2["frame"], df_hb_2["count"], df_hb_3["frame"], df_hb_3["count"], df_hb_4["frame"], df_hb_4["count"],
    name_r="Reference", name_1="Peptide 1", name_2="Peptide 2", name_3="Peptide 3", name_4="Peptide 4",
    title="Hydrogen Bond Analysis Over Time", xlabel="Frame", ylabel="Count of H-bonds",
    base_name="Hbond_Analysis_Superimposed", marker=6, line_shape='hv'
)


# =====================================================================
# 4. RMSF ANALYSIS
# =====================================================================
print("Calculating RMSF...")

def run_rmsf(u, prefix,ligand_sel):
    ca = u.select_atoms(f"protein and backbone and not ({ligand_sel})")
    vals = rms.RMSF(ca).run().results.rmsf
    resids = ca.resids
    pd.DataFrame({'Residue_Number': resids, 'RMSF_Angstrom': vals}).to_csv(os.path.join(OUTPUT_DIR, f"RMSF_{prefix}.csv"), index=False)
    return resids, vals

res_ids_r, rmsf_r = run_rmsf(u_r, 'ref',ligand_sel_r)
res_ids_1, rmsf_1 = run_rmsf(u_1, 'pep1',ligand_sel_1)
res_ids_2, rmsf_2 = run_rmsf(u_2, 'pep2',ligand_sel_2)
res_ids_3, rmsf_3 = run_rmsf(u_3, 'pep3',ligand_sel_3)
res_ids_4, rmsf_4 = run_rmsf(u_4, 'pep4',ligand_sel_4)

save_line_plot(
    res_ids_r, rmsf_r, res_ids_1, rmsf_1, res_ids_2, rmsf_2, res_ids_3, rmsf_3, res_ids_4, rmsf_4,
    name_r="Reference", name_1="Peptide 1", name_2="Peptide 2", name_3="Peptide 3", name_4="Peptide 4",
    title="Per-Residue Root-Mean-Square Fluctuation (RMSF)", xlabel="Residue Index", ylabel="RMSF (Å)",
    base_name="Superimposed_RMSF"
)


# =====================================================================
# 5. RADIUS OF GYRATION (Rg)
# =====================================================================
print("Calculating Radius of Gyration...")

rg_r = [u_r.select_atoms(f"protein and backbone and not ({ligand_sel_r})").radius_of_gyration() for _ in u_r.trajectory]
rg_1 = [u_1.select_atoms(f"protein and backbone and not ({ligand_sel_1})").radius_of_gyration() for _ in u_1.trajectory]
rg_2 = [u_2.select_atoms(f"protein and backbone and not ({ligand_sel_2})").radius_of_gyration() for _ in u_2.trajectory]
rg_3 = [u_3.select_atoms(f"protein and backbone and not ({ligand_sel_3})").radius_of_gyration() for _ in u_3.trajectory]
rg_4 = [u_4.select_atoms(f"protein and backbone and not ({ligand_sel_4})").radius_of_gyration() for _ in u_4.trajectory]

# Export Rg CSVs
pd.DataFrame({'Frame_ns': frames_r, 'Rg_A': rg_r}).to_csv(os.path.join(OUTPUT_DIR, "Rg_ref.csv"), index=False)
pd.DataFrame({'Frame_ns': frames_1, 'Rg_A': rg_1}).to_csv(os.path.join(OUTPUT_DIR, "Rg_pep1.csv"), index=False)
pd.DataFrame({'Frame_ns': frames_2, 'Rg_A': rg_2}).to_csv(os.path.join(OUTPUT_DIR, "Rg_pep2.csv"), index=False)
pd.DataFrame({'Frame_ns': frames_3, 'Rg_A': rg_3}).to_csv(os.path.join(OUTPUT_DIR, "Rg_pep3.csv"), index=False)
pd.DataFrame({'Frame_ns': frames_4, 'Rg_A': rg_4}).to_csv(os.path.join(OUTPUT_DIR, "Rg_pep4.csv"), index=False)

# Superimposed Rg Plot Call
save_line_plot(
    frames_r, rg_r, frames_1, rg_1, frames_2, rg_2, frames_3, rg_3, frames_4, rg_4,
    name_r="Reference", name_1="Peptide 1", name_2="Peptide 2", name_3="Peptide 3", name_4="Peptide 4",
    title="Radius of Gyration (Rg) Timeline", xlabel="ns", ylabel="Radius of Gyration (Å)",
    base_name="Superimposed_Rg_over_Time"
)

print(f"\n=== ALL ANALYSIS & SUPERIMPOSED PLOTS COMPLETED AND SAVED TO: {OUTPUT_DIR} ===")
