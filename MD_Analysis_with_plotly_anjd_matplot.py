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

pdb_trajectory_r = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_MD.dcd"
prmtop_path_r = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_box_2.prmtop"

pdb_trajectory_1 = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_MD.dcd"
prmtop_path_1 = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_box_2.prmtop"

pdb_trajectory_2 = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_MD.dcd"
prmtop_path_2 = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_box_2.prmtop"

pdb_trajectory_3 = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_MD.dcd"
prmtop_path_3 = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_box_2.prmtop"

pdb_trajectory_4 = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_MD.dcd"
prmtop_path_4 = "/kaggle/input/datasets/kirushi/ddddd-file/5NN8_complex_box_2.prmtop"




# =====================================================================
# HIGH-RESOLUTION EXPORT SETTINGS (matplotlib-based, no Chrome needed)
# =====================================================================
PNG_DPI = 300          # 300 = print-quality; bump to 600 for even higher-res
PNG_FIGSIZE = (10, 6)  # inches; combined with DPI gives final pixel size

def save_line_plot(x_r,y_r,x_1,y_1,x_2,y_2,x_3,y_3,x_4,y_4, title, xlabel, ylabel, base_name , name_r,name_1,name_2,name_3,name_4 ,
                    marker=None, line_shape='linear',):
    """
    Saves an interactive Plotly HTML (bold title) AND a high-resolution
    matplotlib PNG (bold title) for the same data, sharing one base filename.
    """
    # --- Interactive HTML (Plotly) ---
    fig = go.Figure()
    mode = 'lines+markers' if marker else 'lines'
 
    if line_shape == 'hv':
        fig.add_trace(go.Scatter(x=x_r, y=y_r, mode=mode,
                                  line=dict(shape='hv'),
                                  marker=dict(size=marker) if marker else None,name=name_r))
        fig.add_trace(go.Scatter(x=x_1, y=y_1, mode=mode,
                                  line=dict(shape='hv'),
                                  marker=dict(size=marker) if marker else None,name=name_1))
        fig.add_trace(go.Scatter(x=x_2, y=y_2, mode=mode,
                                  line=dict(shape='hv'),
                                  marker=dict(size=marker) if marker else None,name=name_2))
        fig.add_trace(go.Scatter(x=x_3, y=y_3, mode=mode,
                                  line=dict(shape='hv'),
                                  marker=dict(size=marker) if marker else None,name=name_3))
        fig.add_trace(go.Scatter(x=x_4, y=y_4, mode=mode,
                                  line=dict(shape='hv'),
                                  marker=dict(size=marker) if marker else None,name=name_4))
        
    else:
        fig.add_trace(go.Scatter(x=x_r, y=y_r, mode=mode,
                                marker=dict(size=marker) if marker else None,name=name_r))
        fig.add_trace(go.Scatter(x=x_1, y=y_1, mode=mode,
                                 marker=dict(size=marker) if marker else None,name=name_1))
        fig.add_trace(go.Scatter(x=x_2, y=y_2, mode=mode,
                                marker=dict(size=marker) if marker else None,name=name_2))
        fig.add_trace(go.Scatter(x=x_3, y=y_3, mode=mode,
                                 marker=dict(size=marker) if marker else None,name=name_3))
        fig.add_trace(go.Scatter(x=x_4, y=y_4, mode=mode,
                                  marker=dict(size=marker) if marker else None,name=name_4))
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
        plt.plot(x_r, y_r, linewidth=2, marker='o', markersize=3,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_r)
        plt.plot(x_1, y_1, linewidth=2, marker='o', markersize=3,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_1)
        plt.plot(x_2, y_2, linewidth=2, marker='o', markersize=3,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_2)
        plt.plot(x_3, y_3, linewidth=2, marker='o', markersize=3,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_3)
        plt.plot(x_4, y_4, linewidth=2, marker='o', markersize=3,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_4)
         
    else:
        plt.plot(x_r, y_r, linewidth=2,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_r)
        plt.plot(x_1, y_1, linewidth=2,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_1)
        plt.plot(x_2, y_2, linewidth=2,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_2)
        plt.plot(x_3, y_3, linewidth=2,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_3)
        plt.plot(x_4, y_4, linewidth=2,
                  drawstyle='steps-post' if line_shape == 'hv' else 'default',label=name_4)

    plt.legend() 
    plt.title(title, fontweight='bold', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_name}.png", dpi=PNG_DPI)
    plt.close()

    return fig



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




u_r = mda.Universe(prmtop_path_r, pdb_trajectory_r, in_memory=True)
ref_r = mda.Universe(prmtop_path_r, pdb_trajectory_r, in_memory=True)  # Reference frame 0

Align_u_r = align.AlignTraj(u_r, u_r, select="protein and name CA", ref_frame=0, in_memory=True).run()
Align_ref_r = align.AlignTraj(ref_r, ref_r, select="protein and name CA", ref_frame=0, in_memory=True).run()



u_1 = mda.Universe(prmtop_path_1, pdb_trajectory_1, in_memory=True)
ref_1 = mda.Universe(prmtop_path_1, pdb_trajectory_1, in_memory=True)  # Reference frame 0

Align_u_1 = align.AlignTraj(u_1, u_1, select="protein and name CA", ref_frame=0, in_memory=True).run()
Align_ref_1 = align.AlignTraj(ref_1, ref_1, select="protein and name CA", ref_frame=0, in_memory=True).run()


u_2 = mda.Universe(prmtop_path_2, pdb_trajectory_2, in_memory=True)
ref_2 = mda.Universe(prmtop_path_2, pdb_trajectory_2, in_memory=True)  # Reference frame 0

Align_u_2 = align.AlignTraj(u_2, u_2, select="protein and name CA", ref_frame=0, in_memory=True).run()
Align_ref_2 = align.AlignTraj(ref_2, ref_2, select="protein and name CA", ref_frame=0, in_memory=True).run()


u_3 = mda.Universe(prmtop_path_3, pdb_trajectory_3, in_memory=True)
ref_3 = mda.Universe(prmtop_path_3, pdb_trajectory_3, in_memory=True)  # Reference frame 0

Align_u_3 = align.AlignTraj(u_3, u_3, select="protein and name CA", ref_frame=0, in_memory=True).run()
Align_ref_3 = align.AlignTraj(ref_3, ref_3, select="protein and name CA", ref_frame=0, in_memory=True).run()

u_4 = mda.Universe(prmtop_path_4, pdb_trajectory_4, in_memory=True)
ref_4 = mda.Universe(prmtop_path_4, pdb_trajectory_4, in_memory=True)  # Reference frame 0

Align_u_4 = align.AlignTraj(u_4, u_4, select="protein and name CA", ref_frame=0, in_memory=True).run()
Align_ref_4 = align.AlignTraj(ref_4, ref_4, select="protein and name CA", ref_frame=0, in_memory=True).run()



# Determine Ligand Residue Name (defaults to UNL, falls back to LIG)
ligand_sel_r = "resname UNL"
if len(u_r.select_atoms(ligand_sel_r)) == 0:
    ligand_sel_r = "protein and resid 167-171"

print(f"Ligand selection: {ligand_sel_r} ({len(u_r.select_atoms(ligand_sel_r))} atoms found)")



ligand_sel_1 = "resname UNL"
if len(u_1.select_atoms(ligand_sel_1)) == 0:
    ligand_sel_1 = "protein and resid 167-171"

print(f"Ligand selection: {ligand_sel_1} ({len(u_1.select_atoms(ligand_sel_1))} atoms found)")




ligand_sel_2 = "resname UNL"
if len(u_2.select_atoms(ligand_sel_2)) == 0:
    ligand_sel_2 = "protein and resid 167-171"

print(f"Ligand selection: {ligand_sel_2} ({len(u_2.select_atoms(ligand_sel_2))} atoms found)")




ligand_sel_3 = "resname UNL"
if len(u_3.select_atoms(ligand_sel_3)) == 0:
    ligand_sel_3 = "protein and resid 167-171"

print(f"Ligand selection: {ligand_sel_3} ({len(u_3.select_atoms(ligand_sel_3))} atoms found)")



ligand_sel_4 = "resname UNL"
if len(u_4.select_atoms(ligand_sel_4)) == 0:
    ligand_sel_4 = "protein and resid 167-171"

print(f"Ligand selection: {ligand_sel_4} ({len(u_4.select_atoms(ligand_sel_4))} atoms found)")


# =====================================================================
# 1. LIGAND RMSD
# =====================================================================
print("Calculating Ligand RMSD...")

R_ligand_r = rms.RMSD(u_r, ref_r, select=ligand_sel_r, ref_frame=0).run()

frames_r = R_ligand_r.results.rmsd[:, 1] / 1000.0
ligand_rmsd_r = R_ligand_r.results.rmsd[:, 2]

df_ligand_r = pd.DataFrame({'Frame': frames_r, 'Ligand_RMSD_A': ligand_rmsd_r})
df_ligand_r.to_csv("ligand_rmsd_r.csv", index=False)

fig_ligand_rmsd= save_line_plot_1(
    frames_r, ligand_rmsd_r,
    title=' Reference Ligand RMSD over Time',
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Reference Ligand RMSD over Time"
)


R_ligand_1 = rms.RMSD(u_1, ref_1, select=ligand_sel_1, ref_frame=0).run()

frames_1 = R_ligand_1.results.rmsd[:, 1] / 1000.0
ligand_rmsd_1 = R_ligand_1.results.rmsd[:, 2]

df_ligand_1 = pd.DataFrame({'Frame': frames_1, 'Ligand_RMSD_A': ligand_rmsd_1})
df_ligand_1.to_csv("ligand_rmsd_1.csv", index=False)

fig_ligand_rmsd = save_line_plot_1(
    frames_1, ligand_rmsd_1,
    title="Peptide 1 RMSD over Time",
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 1 RMSD over Time"
)


R_ligand_2 = rms.RMSD(u_2, ref_2, select=ligand_sel_2, ref_frame=0).run()

frames_2 = R_ligand_2.results.rmsd[:, 1] / 1000.0
ligand_rmsd_2 = R_ligand_2.results.rmsd[:, 2]

df_ligand_2 = pd.DataFrame({'Frame': frames_2, 'Ligand_RMSD_A': ligand_rmsd_2})
df_ligand_2.to_csv("ligand_rmsd_2.csv", index=False)

fig_ligand_rmsd = save_line_plot_1(
    frames_2, ligand_rmsd_2,
    title='Peptide 2 RMSD over Time',
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 2 RMSD over Time"
)


R_ligand_3 = rms.RMSD(u_3, ref_3, select=ligand_sel_3, ref_frame=0).run()

frames_3 = R_ligand_3.results.rmsd[:, 1] / 1000.0
ligand_rmsd_3 = R_ligand_3.results.rmsd[:, 2]

df_ligand_3 = pd.DataFrame({'Frame': frames_3, 'Ligand_RMSD_A': ligand_rmsd_3})
df_ligand_3.to_csv("ligand_rmsd_3.csv", index=False)

fig_ligand_rmsd = save_line_plot_1(
    frames_3, ligand_rmsd_3,
    title='Peptide 3 RMSD over Time',
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 3 RMSD over Time"
)


R_ligand_4 = rms.RMSD(u_4, ref_4, select=ligand_sel_4, ref_frame=0).run()

frames_4 = R_ligand_4.results.rmsd[:, 1] / 1000.0
ligand_rmsd_4 = R_ligand_4.results.rmsd[:, 2]

df_ligand_4 = pd.DataFrame({'Frame': frames_4, 'Ligand_RMSD_A': ligand_rmsd_4})
df_ligand_4.to_csv("ligand_rmsd_4.csv", index=False)

fig_ligand_rmsd = save_line_plot_1(
    frames_4, ligand_rmsd_4,
    title='Peptide 4 RMSD over Time',
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 4 RMSD over Time"
)



fig_ligand_rmsd = save_line_plot(
    frames_r, ligand_rmsd_r,frames_1, ligand_rmsd_1,frames_2, ligand_rmsd_2,frames_3, ligand_rmsd_3,frames_4, ligand_rmsd_4,
    name_r="Reference RMSD",name_1="Peptide1 RMSD",name_2="Peptide2 RMSD",name_3="Peptide3 RMSD",name_4="Peptide4 RMSD",
    title="Superimposed RMSD over Time",
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Superimposed RMSD over Time"
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
    title='Reference Protein RMSD over Time',
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Reference Protein RMSD over Time"
)


R_protein_1 = rms.RMSD(u_1, ref_1, select="protein and backbone ", ref_frame=0).run()
protein_rmsd_1 = R_protein_1.results.rmsd[:, 2]

df_protein_1 = pd.DataFrame({'Frame': frames_1, 'Protein_RMSD_A': protein_rmsd_1})
df_protein_1.to_csv("protein_rmsd_1.csv", index=False)

fig_protein_rmsd= save_line_plot_1(
    frames_1, protein_rmsd_1,
    title="Peptide 1's Protein RMSD over Time",
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 1's Protein RMSD over Time"
)


R_protein_2 = rms.RMSD(u_2, ref_2, select="protein and backbone ", ref_frame=0).run()
protein_rmsd_2 = R_protein_2.results.rmsd[:, 2]

df_protein_2 = pd.DataFrame({'Frame': frames_2, 'Protein_RMSD_A': protein_rmsd_2})
df_protein_2.to_csv("protein_rmsd_2.csv", index=False)

fig_protein_rmsd= save_line_plot_1(
    frames_2, protein_rmsd_2,
    title="Peptide 2's Protein RMSD over Time",
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 2's Protein RMSD over Time"
)


R_protein_3 = rms.RMSD(u_3, ref_3, select="protein and backbone ", ref_frame=0).run()
protein_rmsd_3 = R_protein_3.results.rmsd[:, 2]

df_protein_3 = pd.DataFrame({'Frame': frames_3, 'Protein_RMSD_A': protein_rmsd_3})
df_protein_3.to_csv("protein_rmsd_3.csv", index=False)

fig_protein_rmsd= save_line_plot_1(
    frames_3, protein_rmsd_3,
    title="Peptide 3's Protein RMSD over Time",
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 3's Protein RMSD over Time"
)


R_protein_4 = rms.RMSD(u_4, ref_4, select="protein and backbone ", ref_frame=0).run()
protein_rmsd_4 = R_protein_4.results.rmsd[:, 2]

df_protein_4 = pd.DataFrame({'Frame': frames_4, 'Protein_RMSD_A': protein_rmsd_4})
df_protein_4.to_csv("protein_rmsd_4.csv", index=False)

fig_protein_rmsd= save_line_plot_1(
    frames_4, protein_rmsd_4,
    title="Peptide 4's Protein RMSD over Time",
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Peptide 4's Protein RMSD over Time"
)



fig_protein_rmsd= save_line_plot(
    frames_r, protein_rmsd_r,frames_1, protein_rmsd_1,frames_2, protein_rmsd_2,frames_3, protein_rmsd_3,frames_4, protein_rmsd_4,
    name_r="Reference RMSD",name_1="Peptide1 RMSD",name_2="Peptide2 RMSD",name_3="Peptide3 RMSD",name_4="Peptide4 RMSD",
    title='Superimposed RMSD over Time',
    xlabel='ns', ylabel='RMSD (Å)',
    base_name="Superimposed RMSD over Time",
    
)

# =====================================================================
# 2. HYDROGEN BOND ANALYSIS (Peptide resid 1-5 <-> Target)
# =====================================================================
print("Running Hydrogen Bond Analysis...")

hbonds_r = HBA(
    universe=u_r,
    between=['protein', 'resname UNK'],
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
    df_hb_r["frame"], df_hb_r["count"],
    title="Hydrogen Bond Analysis (Reference Peptide [resid 1-5] <-> Target Protein)",
    xlabel="Frame", ylabel="Count of H-bonds",
    base_name="h-bond_analysis_0_200_ns",
     marker=6, line_shape='hv'
)




hbonds_1 = HBA(
    universe=u_1,
    between=['protein and not resid 167-171', 'resid 167-171'],
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=150
)
hbonds_1.run()

all_hbond_data_1 = []
total_frames_1 = len(u_1.trajectory)
hbond_count_1 = np.zeros(total_frames_1)

for bond_1 in hbonds_1.results.hbonds:
    frame_idx_1 = int(bond_1[0])
    donor_atom_1 = u_1.atoms[int(bond_1[1])]
    acceptor_atom_1 = u_1.atoms[int(bond_1[3])]
    distance_1 = bond_1[4]
    angle_1 = bond_1[5]

    hbond_count_1[frame_idx_1] += 1
    all_hbond_data_1.append([
        frame_idx_1,
        f"{donor_atom_1.resname}{donor_atom_1.resid}",
        donor_atom_1.name,
        f"{acceptor_atom_1.resname}{acceptor_atom_1.resid}",
        acceptor_atom_1.name,
        round(distance_1, 3),
        round(angle_1, 3)
    ])

with open("all_frames_hbonds_1.csv", "w", newline="") as f_1:
    writer_1 = csv.writer(f_1)
    writer_1.writerow(["frame", "donor_res", "donor_atom", "acceptor_res", "acceptor_atom", "distance_a", "angle_deg"])
    writer_1.writerows(all_hbond_data_1)

with open("hbond_count_1.csv", "w", newline="") as f_1:
    writer_1 = csv.writer(f_1)
    writer_1.writerow(["frame", "count"])
    for i in range(total_frames_1):
        writer_1.writerow([i, int(hbond_count_1[i])])

df_hb_1 = pd.read_csv("hbond_count_1.csv")
fig_hb = save_line_plot_1(
    df_hb_1["frame"], df_hb_1["count"],
    title="Hydrogen Bond Analysis ( Peptide 1 [resid 1-5] <-> Target Protein)",
    xlabel="Frame", ylabel="Count of H-bonds",
    base_name="Peptide 1's h-bond_analysis_0_200_ns",
     marker=6, line_shape='hv'
)


hbonds_2 = HBA(
    universe=u_2,
    between=['protein and not resid 167-171', 'resid 167-171'],
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=150
)
hbonds_2.run()

all_hbond_data_2 = []
total_frames_2 = len(u_2.trajectory)
hbond_count_2 = np.zeros(total_frames_2)

for bond_2 in hbonds_2.results.hbonds:
    frame_idx_2 = int(bond_2[0])
    donor_atom_2 = u_2.atoms[int(bond_2[1])]
    acceptor_atom_2 = u_2.atoms[int(bond_2[3])]
    distance_2 = bond_2[4]
    angle_2 = bond_2[5]

    hbond_count_2[frame_idx_2] += 1
    all_hbond_data_2.append([
        frame_idx_2,
        f"{donor_atom_2.resname}{donor_atom_2.resid}",
        donor_atom_2.name,
        f"{acceptor_atom_2.resname}{acceptor_atom_2.resid}",
        acceptor_atom_2.name,
        round(distance_2, 3),
        round(angle_2, 3)
    ])

with open("all_frames_hbonds_2.csv", "w", newline="") as f_2:
    writer_2 = csv.writer(f_2)
    writer_2.writerow(["frame", "donor_res", "donor_atom", "acceptor_res", "acceptor_atom", "distance_a", "angle_deg"])
    writer_2.writerows(all_hbond_data_2)

with open("hbond_count_2.csv", "w", newline="") as f_2:
    writer_2 = csv.writer(f_2)
    writer_2.writerow(["frame", "count"])
    for i in range(total_frames_2):
        writer_2.writerow([i, int(hbond_count_2[i])])

df_hb_2 = pd.read_csv("hbond_count_2.csv")
fig_hb = save_line_plot_1(
    df_hb_2["frame"], df_hb_2["count"],
    title="Hydrogen Bond Analysis ( Peptide 2 [resid 1-5] <-> Target Protein)",
    xlabel="Frame", ylabel="Count of H-bonds",
    base_name="Peptide 2's h-bond_analysis_0_200_ns",
     marker=6, line_shape='hv'
)



hbonds_3 = HBA(
    universe=u_3,
    between=['protein and not resid 167-171', 'resid 167-171'],
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=150
)
hbonds_3.run()

all_hbond_data_3 = []
total_frames_3 = len(u_3.trajectory)
hbond_count_3 = np.zeros(total_frames_3)

for bond_3 in hbonds_3.results.hbonds:
    frame_idx_3 = int(bond_3[0])
    donor_atom_3 = u_3.atoms[int(bond_3[1])]
    acceptor_atom_3 = u_3.atoms[int(bond_3[3])]
    distance_3 = bond_3[4]
    angle_3 = bond_3[5]

    hbond_count_3[frame_idx_3] += 1
    all_hbond_data_3.append([
        frame_idx_3,
        f"{donor_atom_3.resname}{donor_atom_3.resid}",
        donor_atom_3.name,
        f"{acceptor_atom_3.resname}{acceptor_atom_3.resid}",
        acceptor_atom_3.name,
        round(distance_3, 3),
        round(angle_3, 3)
    ])

with open("all_frames_hbonds_3.csv", "w", newline="") as f_3:
    writer_3 = csv.writer(f_3)
    writer_3.writerow(["frame", "donor_res", "donor_atom", "acceptor_res", "acceptor_atom", "distance_a", "angle_deg"])
    writer_3.writerows(all_hbond_data_3)

with open("hbond_count_3.csv", "w", newline="") as f_3:
    writer_3 = csv.writer(f_3)
    writer_3.writerow(["frame", "count"])
    for i in range(total_frames_3):
        writer_3.writerow([i, int(hbond_count_3[i])])

df_hb_3 = pd.read_csv("hbond_count_3.csv")
fig_hb_3 = save_line_plot_1(
    df_hb_3["frame"], df_hb_3["count"],
    title="Hydrogen Bond Analysis ( Peptide 3 [resid 1-5] <-> Target Protein)",
    xlabel="Frame", ylabel="Count of H-bonds",
    base_name="Peptide 3's h-bond_analysis_0_200_ns",
     marker=6, line_shape='hv'
)



hbonds_4 = HBA(
    universe=u_4,
    between=['protein', 'resname UNL'],
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=150
)
hbonds_4.run()

all_hbond_data_4 = []
total_frames_4 = len(u_4.trajectory)
hbond_count_4 = np.zeros(total_frames_4)

for bond_4 in hbonds_4.results.hbonds:
    frame_idx_4 = int(bond_4[0])
    donor_atom_4 = u_4.atoms[int(bond_4[1])]
    acceptor_atom_4 = u_4.atoms[int(bond_4[3])]
    distance_4 = bond_4[4]
    angle_4 = bond_4[5]

    hbond_count_4[frame_idx_4] += 1
    all_hbond_data_4.append([
        frame_idx_4,
        f"{donor_atom_4.resname}{donor_atom_4.resid}",
        donor_atom_4.name,
        f"{acceptor_atom_4.resname}{acceptor_atom_4.resid}",
        acceptor_atom_4.name,
        round(distance_4, 3),
        round(angle_4, 3)
    ])

with open("all_frames_hbonds_4.csv", "w", newline="") as f_4:
    writer_4 = csv.writer(f_4)
    writer_4.writerow(["frame", "donor_res", "donor_atom", "acceptor_res", "acceptor_atom", "distance_a", "angle_deg"])
    writer_4.writerows(all_hbond_data_4)

with open("hbond_count_4.csv", "w", newline="") as f_4:
    writer_4 = csv.writer(f_4)
    writer_4.writerow(["frame", "count"])
    for i in range(total_frames_4):
        writer_4.writerow([i, int(hbond_count_4[i])])

df_hb_4 = pd.read_csv("hbond_count_4.csv")
fig_hb_4 = save_line_plot_1(
    df_hb_4["frame"], df_hb_4["count"],
    title="Hydrogen Bond Analysis ( Peptide 4 [resid 1-5] <-> Target Protein)",
    xlabel="Frame", ylabel="Count of H-bonds",
    base_name="Peptide 4's h-bond_analysis_0_200_ns",
     marker=6, line_shape='hv'
)


fig_hb = save_line_plot(
    df_hb_r["frame"], df_hb_r["count"],df_hb_1["frame"], df_hb_1["count"],df_hb_2["frame"], df_hb_2["count"],df_hb_3["frame"], df_hb_3["count"],df_hb_4["frame"], df_hb_4["count"],
    name_r="Reference",name_1="Peptide1",name_2="Peptide2",name_3="Peptide3",name_4="Peptide4",
    title="Hydrogen Bond Analysis (Reference Peptide [resid 1-5] <-> Target Protein)",
    xlabel="Frame", ylabel="Count of H-bonds",
    base_name="H-bond_analysis over time",
     marker=6, line_shape='hv'
)


print("Finished! Output files created: ligand_rmsd.csv, ligand_rmsd.html, all_frames_hbonds.csv, hbond_count.csv, h-bond_analysis.html")


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
    title="Per-Residue Root-Mean-Square Fluctuation (RMSF) for Reference protein",
    xlabel="Residue Index", ylabel="RMSF (Å)",
    base_name="RMSF_for_Reference_protein",
     marker=4
)
print("All tasks executed and figures saved successfully!")


calphas_1 = u_1.select_atoms("protein and name CA")

print("Calculating RMSF...")
rmsf_analysis_1 = rms.RMSF(calphas_1).run()
res_ids_1 = calphas_1.resids
rmsf_values_1 = rmsf_analysis_1.results.rmsf

df_rmsf_1 = pd.DataFrame({
    'Residue_Number': res_ids_1,
    'RMSF_Angstrom': rmsf_values_1
})
df_rmsf_1.to_csv("RMSF_for_Peptide_1_protein.csv", index=False)

fig_rmsf = save_line_plot_1(
    res_ids_1, rmsf_values_1,
    title="Per-Residue Root-Mean-Square Fluctuation (RMSF) for peptide 1",
    xlabel="Residue Index ", ylabel="RMSF (Å)",
    base_name="RMSF_for_Peptide_1_protein",
     marker=4
)
print("All tasks executed and figures saved successfully!")


calphas_2 = u_2.select_atoms("protein and name CA")

print("Calculating RMSF...")
rmsf_analysis_2 = rms.RMSF(calphas_2).run()
res_ids_2 = calphas_2.resids
rmsf_values_2 = rmsf_analysis_2.results.rmsf

df_rmsf_2 = pd.DataFrame({
    'Residue_Number': res_ids_2,
    'RMSF_Angstrom': rmsf_values_2
})
df_rmsf_2.to_csv("RMSF_for_Peptide_2_protein.csv", index=False)

fig_rmsf = save_line_plot_1(
    res_ids_2, rmsf_values_2,
    title="Per-Residue Root-Mean-Square Fluctuation (RMSF) for peptide 2",
    xlabel="Residue Index ", ylabel="RMSF (Å)",
    base_name="RMSF_for_Peptide_2_protein",
     marker=4
)




calphas_3 = u_3.select_atoms("protein and name CA")

print("Calculating RMSF...")
rmsf_analysis_3 = rms.RMSF(calphas_3).run()
res_ids_3 = calphas_3.resids
rmsf_values_3 = rmsf_analysis_3.results.rmsf

df_rmsf_3 = pd.DataFrame({
    'Residue_Number': res_ids_3,
    'RMSF_Angstrom': rmsf_values_3
})
df_rmsf_3.to_csv("RMSF_for_Peptide_3_protein.csv", index=False)

fig_rmsf = save_line_plot_1(
    res_ids_3, rmsf_values_3,
    title="Per-Residue Root-Mean-Square Fluctuation (RMSF) for peptide 3",
    xlabel="Residue Index ", ylabel="RMSF (Å)",
    base_name="RMSF_for_Peptide_3_protein",
     marker=4
)
print("All tasks executed and figures saved successfully!")


calphas_4 = u_4.select_atoms("protein and name CA")

print("Calculating RMSF...")
rmsf_analysis_4 = rms.RMSF(calphas_4).run()
res_ids_4 = calphas_4.resids
rmsf_values_4 = rmsf_analysis_4.results.rmsf

df_rmsf_4 = pd.DataFrame({
    'Residue_Number': res_ids_4,
    'RMSF_Angstrom': rmsf_values_4
})
df_rmsf_4.to_csv("RMSF_for_Peptide_4_protein.csv", index=False)

fig_rmsf = save_line_plot_1(
    res_ids_4, rmsf_values_4,
    title="Per-Residue Root-Mean-Square Fluctuation (RMSF) for peptide 4",
    xlabel="Residue Index ", ylabel="RMSF (Å)",
    base_name="RMSF_for_Peptide_4_protein",
     marker=4
)


fig_rmsf = save_line_plot(
    res_ids_r, rmsf_values_r,res_ids_1, rmsf_values_1,res_ids_2, rmsf_values_2,res_ids_3, rmsf_values_3,res_ids_4, rmsf_values_4,
    name_r="Reference protein's RMSF",name_1="Peptide1 protein's RMSF",name_2="Peptide2 protein's RMSF",name_3="Peptide3 protein's RMSF",name_4="Peptide4 protein's RMSF",
    title="Per-Residue Root-Mean-Square Fluctuation (RMSF)",
    xlabel="Residue Index", ylabel="RMSF (Å)",
    base_name="Superimposed RMSF",
     marker=4
)



print("All tasks executed and figures saved successfully!")

# =====================================================================
# RADIUS OF GYRATION
# =====================================================================
backbone_r = u_r.select_atoms("protein and name CA")
rg_values_r = [backbone_r.radius_of_gyration() for ts in u_r.trajectory]

fig_rg = save_line_plot_1(
    frames_r, rg_values_r,
    title="Radius of Gyration (Rg) Timeline",
    xlabel="ns", ylabel="Radius of Gyration (Å)",
    base_name="Rg for reference protein"
)



backbone_1 = u_1.select_atoms("protein and name CA")
rg_values_1 = [backbone_1.radius_of_gyration() for ts in u_1.trajectory]

fig_rg = save_line_plot_1(
    frames_1, rg_values_1,
    title="Radius of Gyration (Rg) Timeline",
    xlabel="ns", ylabel="Radius of Gyration (Å)",
    base_name="Rg for peptide1"
)



backbone_2 = u_2.select_atoms("protein and name CA")
rg_values_2 = [backbone_2.radius_of_gyration() for ts in u_2.trajectory]

fig_rg = save_line_plot_1(
    frames_2, rg_values_2,
    title="Radius of Gyration (Rg) Timeline",
    xlabel="ns", ylabel="Radius of Gyration (Å)",
    base_name="Rg for protein of peptide 2"
    
)


backbone_3 = u_3.select_atoms("protein and name CA")
rg_values_3 = [backbone_3.radius_of_gyration() for ts in u_3.trajectory]

fig_rg = save_line_plot_1(
    frames_3, rg_values_3,
    title="Radius of Gyration (Rg) Timeline",
    xlabel="ns", ylabel="Radius of Gyration (Å)",
    base_name="Rg for protein of peptide 3"
    
)


backbone_4 = u_4.select_atoms("protein and name CA")
rg_values_4 = [backbone_4.radius_of_gyration() for ts in u_4.trajectory]

fig_rg = save_line_plot_1(
    frames_4, rg_values_4,
    title="Radius of Gyration (Rg) Timeline",
    xlabel="ns", ylabel="Radius of Gyration (Å)",
    base_name="Rg for protein of peptide 4"
    
)


fig_rg = save_line_plot(
    frames_r, rg_values_r,frames_1, rg_values_1,frames_2, rg_values_2,frames_3, rg_values_3,frames_4, rg_values_4,
    name_r="Reference protein's Rg",name_1="Peptide1 protein's Rg",name_2="Peptide2 protein's Rg",name_3="Peptide3 protein's Rg",name_4="Peptide4 protein's Rg",
    title="Radius of Gyration (Rg) over time",
    xlabel="ns", ylabel="Radius of Gyration (Å)",
    base_name="Superimposed Rg"
)
