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
# 0. FILE PATHS & UNIVERSES 
# =====================================================================

pdb_trajectory_r = "/kaggle/input/datasets/kirushikesan/new-pep-5-autoimaged-dcd-file/Pep_5_MD_autoimaged_first_batch.dcd"
prmtop_path_r = "/kaggle/input/datasets/kirushikesan/new-pep-5-input-files/Optimized_pep_5_complex_amber.prmtop"


# =====================================================================
# HIGH-RESOLUTION EXPORT SETTINGS (matplotlib-based, no Chrome needed)
# =====================================================================
PNG_DPI = 300          # 300 = print-quality; bump to 600 for even higher-res
PNG_FIGSIZE = (10, 6)  # inches; combined with DPI gives final pixel size


def save_line_plot_1(x, y, title, xlabel, ylabel, base_name,
                    marker=None, line_shape='linear'):
    """
    Saves an interactive Plotly HTML (bold title) AND a high-resolution
    matplotlib PNG (bold title) for the same data, sharing one base filename.
    """
    # --- Interactive HTML (Plotly) ---
    fig = go.Figure()
    mode = 'lines+markers' if marker else 'lines'
   
    if line_shape == 'hv':
        fig.add_trace(go.Scatter(x=x, y=y, mode=mode,
                                  line=dict(shape='hv'),
                                  marker=dict(size=marker) if marker else None))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode=mode,
                                  marker=dict(size=marker) if marker else None))
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white',
        hovermode='x unified'
    )
    fig.show()
    fig.write_html(f"{base_name}.html")

    # --- High-resolution PNG (matplotlib, no Chrome dependency) ---

    plt.figure(figsize=PNG_FIGSIZE)
    if marker:
        plt.plot(x, y, linewidth=2, marker='o', markersize=3,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default')
    else:
        plt.plot(x, y, linewidth=2,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default')
    plt.title(title, fontweight='bold', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_name}.png", dpi=PNG_DPI)
    plt.close()

    return fig




u_r = mda.Universe(prmtop_path_r, pdb_trajectory_r)
ref_r = mda.Universe(prmtop_path_r, pdb_trajectory_r)  # Reference frame 0

Align_u_r = align.AlignTraj(u_r, u_r, select="protein and name CA", ref_frame=0, in_memory=True).run()
Align_ref_r = align.AlignTraj(ref_r, ref_r, select="protein and name CA", ref_frame=0, in_memory=True).run()




# Determine Ligand Residue Name (defaults to UNL, falls back to LIG)
ligand_sel_r = "resname UNL"
if len(u_r.select_atoms(ligand_sel_r)) == 0:
    ligand_sel_r = "protein and resid 167-171"

print(f"Ligand selection: {ligand_sel_r} ({len(u_r.select_atoms(ligand_sel_r))} atoms found)")






# =====================================================================
# 1. LIGAND RMSD
# =====================================================================
print("Calculating Ligand RMSD...")

R_ligand_r = rms.RMSD(u_r, ref_r, select="protein and name CA",groupselections=["resname UNL"], ref_frame=0).run()

frames_r = R_ligand_r.results.rmsd[:, 0] * 100000*0.002
ligand_rmsd_r = R_ligand_r.results.rmsd[:, 3]

df_ligand_r = pd.DataFrame({'Frame': frames_r, 'Ligand_RMSD_A': ligand_rmsd_r})
df_ligand_r.to_csv("ligand_rmsd_r.csv", index=False)

fig_ligand_rmsd= save_line_plot_1(
    frames_r, ligand_rmsd_r,
    title="Peptide 4's Ligand RMSD over Time",
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 4 Ligand RMSD over Time"
)





# =====================================================================
# PROTEIN RMSD
# =====================================================================
R_protein_r = rms.RMSD(u_r, ref_r, select="protein and backbone ", ref_frame=0).run()
protein_rmsd_r = R_protein_r.results.rmsd[:, 2]

df_protein_r = pd.DataFrame({'Frame': frames_r, 'Protein_RMSD_A': protein_rmsd_r})
df_protein_r.to_csv("protein_rmsd_r.csv", index=False)

fig_protein_rmsd= save_line_plot_1(
    frames_r, protein_rmsd_r,
    title="Peptide 4's Protein RMSD over Time",
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 4 Protein RMSD over Time"
)


# =====================================================================
# RMSF
# =====================================================================
calphas_r= u_r.select_atoms("protein and name CA")

print("Calculating RMSF...")
rmsf_analysis_r= rms.RMSF(calphas_r).run()
res_ids_r = calphas_r.resids
rmsf_values_r = rmsf_analysis_r.results.rmsf

df_rmsf_r = pd.DataFrame({
    'Residue_Number': res_ids_r,
    'RMSF_Angstrom': rmsf_values_r
})
df_rmsf_r.to_csv("RMSF_for_Reference_protein.csv", index=False)

fig_rmsf = save_line_plot_1(
    res_ids_r, rmsf_values_r,
    title="Per-Residue Root-Mean-Square Fluctuation (RMSF) for Peptide 4 protein",
    xlabel="Residue Index", ylabel="RMSF (Å)",
    base_name="RMSF_for_Peptide_4_one_micro",
     marker=4
)
print("All tasks executed and figures saved successfully!")



print("All tasks executed and figures saved successfully!")

# =====================================================================
# RADIUS OF GYRATION
# =====================================================================
backbone_r = u_r.select_atoms("protein and name CA")
rg_values_r = [backbone_r.radius_of_gyration() for ts in u_r.trajectory]

fig_rg = save_line_plot_1(
    frames_r, rg_values_r,
    title="Radius of Gyration (Rg) Timeline for Peptide 4's protein",
    xlabel="ns", ylabel="Radius of Gyration (Å)",
    base_name="Rg for Peptide 4 protein one micro"
)



# =====================================================================
# 2. HYDROGEN BOND ANALYSIS (Peptide resid 1-5 <-> Target)
# =====================================================================
print("Running Hydrogen Bond Analysis...")

hbonds_r = HBA(
    universe=u_r,
    between=['protein', 'resname UNL'],
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=150
)
hbonds_r.run()

all_hbond_data_r = []
total_frames_r = len(u_r.trajectory)
hbond_count_r = np.zeros(total_frames_r)

for bond_r in hbonds_r.results.hbonds:
    frame_idx_r = int(bond_r[0])
    donor_atom_r = u_r.atoms[int(bond_r[1])]
    acceptor_atom_r = u_r.atoms[int(bond_r[3])]
    distance_r = bond_r[4]
    angle_r = bond_r[5]

    hbond_count_r[frame_idx_r] += 1
    all_hbond_data_r.append([
        frame_idx_r,
        f"{donor_atom_r.resname}{donor_atom_r.resid}",
        donor_atom_r.name,
        f"{acceptor_atom_r.resname}{acceptor_atom_r.resid}",
        acceptor_atom_r.name,
        round(distance_r, 3),
        round(angle_r, 3)
    ])

with open("all_frames_hbonds_r.csv", "w", newline="") as f_r:
    writer_r = csv.writer(f_r)
    writer_r.writerow(["frame", "donor_res", "donor_atom", "acceptor_res", "acceptor_atom", "distance_a", "angle_deg"])
    writer_r.writerows(all_hbond_data_r)


with open("hbond_count_r.csv", "w", newline="") as f_r:
    writer_r = csv.writer(f_r)
    writer_r.writerow(["frame", "count"])
    for i in range(total_frames_r):
        writer_r.writerow([i, int(hbond_count_r[i])])

df_hb_r = pd.read_csv("hbond_count_r.csv")
fig_hb = save_line_plot_1(
    frames_r, df_hb_r["count"],
    title="Hydrogen Bond Analysis (Peptide 4 [resid 1-5] <-> Target Protein)",
    xlabel="Frame", ylabel="Count of H-bonds",
    base_name=" Peptide 4's h-bond_analysis one micro",
     marker=6, line_shape='hv'
)



print("Finished! Output files created: ligand_rmsd.csv, ligand_rmsd.html, all_frames_hbonds.csv, hbond_count.csv, h-bond_analysis.html")

