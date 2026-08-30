!pip install plotly MDAnalysis numpy

import plotly.graph_objects as go
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis import rms
from MDAnalysis.analysis import align, pca
import numpy as np
from scipy.ndimage import gaussian_filter



prmtop_t="/kaggle/input/datasets/kirushikesan/3-oxo-files/reference_complex (1).prmtop"
trajec_t="/kaggle/input/datasets/kirushikesan/3-oxo-files/trajectory_3_oxo_DCD_100ns (1).dcd"



u_t=mda.Universe(prmtop_t,trajec_t)
RMSD_t=rms.RMSD(u_t,u_t,select="protein and name CA",ref_frame=0).run()
RMSD_time=RMSD_t.results.rmsd[:, 1]/1000 
RMSD_time2=RMSD_t.results.rmsd[:, 1]/1000 +100
combined_t=np.concatenate([RMSD_time, RMSD_time2])

#pep_1
prmtop="/kaggle/input/datasets/kirushikesan/3-oxo-files/reference_complex (1).prmtop"
trajectory_1_1_dcd="/kaggle/input/datasets/kirushikesan/right-aligned-files-for-refpep1-to-pep4-dcd-files/ref_3_OXO.dcd"




#pep_2
prmtop_2="/kaggle/input/datasets/kirushikesan/pep-1-files/complex_5_2.prmtop"
trajectory_2_1_dcd="/kaggle/input/datasets/kirushikesan/pep-1-cpp-aligned-dcd-file/pep_1 (1).dcd"




#pep_3
prmtop_3="/kaggle/input/datasets/kirushikesan/pep-2-correct-files/complex_pro_pep.prmtop"
trajectory_3_1_dcd="/kaggle/input/datasets/kirushikesan/right-aligned-files-for-refpep1-to-pep4-dcd-files/0_200_ns_MD_DCD_file_cpp.dcd"



#pep_4
prmtop_4="/kaggle/input/datasets/kirushikesan/pep-3-files/new_pep_complex.prmtop"
trajectory_4_1_dcd="/kaggle/input/datasets/kirushikesan/right-aligned-files-for-refpep1-to-pep4-dcd-files/pep_3.dcd"





#pep_5
prmtop_5="/kaggle/input/datasets/kirushikesan/pep-4-files/5_4_bestpose_complex (1).prmtop"
trajectory_5_1_dcd="/kaggle/input/datasets/kirushikesan/right-aligned-files-for-refpep1-to-pep4-dcd-files/pep_4.dcd"





u=mda.Universe(prmtop,trajectory_1_1_dcd)
u_2=mda.Universe(prmtop_2,trajectory_2_1_dcd)
u_3=mda.Universe(prmtop_3,trajectory_3_1_dcd)
u_4=mda.Universe(prmtop_4,trajectory_4_1_dcd)
u_5=mda.Universe(prmtop_5,trajectory_5_1_dcd)


Align=align.AlignTraj(u,u,select="protein and name CA",ref_frame=0,in_memory=True).run()
Align_2=align.AlignTraj(u_2,u_2,select="protein and name CA and not (resid 167-171)",ref_frame=0,in_memory=True).run()
Align_3=align.AlignTraj(u_3,u_3,select="protein and name CA and not (resid 167-171)",ref_frame=0,in_memory=True).run()
Align_4=align.AlignTraj(u_4,u_4,select="protein and name CA and not (resid 167-171)",ref_frame=0,in_memory=True).run()
Align_5=align.AlignTraj(u_5,u_5,select="protein and name CA",ref_frame=0,in_memory=True).run()

#protein RMSD
RMSD_protein=rms.RMSD(u,u,select="protein and name CA",ref_frame=0).run()
RMSD_protein_time_steps=RMSD_protein.results.rmsd[:, 1]/1000 
RMSD_protein_results=RMSD_protein.results.rmsd[:, 2]

RMSD_2_protein=rms.RMSD(u_2,u_2,select="protein and name CA and not resid 167-171",ref_frame=0).run()
RMSD_2_protein_time_steps=RMSD_2_protein.results.rmsd[: , 1]/1000
RMSD_2_protein_results=RMSD_2_protein.results.rmsd[: , 2]

RMSD_3_protein=rms.RMSD(u_3,u_3,select="protein and name CA and not resid 167-171",ref_frame=0).run()
RMSD_3_protein_time_steps=RMSD_3_protein.results.rmsd[: , 1]/1000
RMSD_3_protein_results=RMSD_3_protein.results.rmsd[: , 2]

RMSD_4_protein=rms.RMSD(u_4,u_4,select="protein and name CA and not resid 167-171",ref_frame=0).run()
RMSD_4_protein_time_steps=RMSD_4_protein.results.rmsd[: , 1]/1000
RMSD_4_protein_results=RMSD_4_protein.results.rmsd[: , 2]

RMSD_5_protein=rms.RMSD(u_5,u_5,select="protein and name CA",ref_frame=0).run()
RMSD_5_protein_time_steps=RMSD_5_protein.results.rmsd[: , 1]/1000
RMSD_5_protein_results=RMSD_5_protein.results.rmsd[: , 2]


#protein rmsd plot
fig=go.Figure()
fig.add_trace(go.Scatter(x=combined_t,y=RMSD_protein_results,name="protein_rmsd_1"))
fig.add_trace(go.Scatter(x=combined_t,y=RMSD_2_protein_results,name="protein_rmsd_2"))
fig.add_trace(go.Scatter(x=combined_t,y=RMSD_3_protein_results,name="protein_rmsd_3"))
fig.add_trace(go.Scatter(x=combined_t,y=RMSD_4_protein_results,name="protein_rmsd_4"))
fig.add_trace(go.Scatter(x=combined_t,y=RMSD_5_protein_results,name="protein_rmsd_5"))
fig.update_layout(title="Protein RMSD",xaxis_title="ns",yaxis_title="RMSD in angstrom")

fig.write_html("Protein_rmsd")
fig.show()

#ligand RMSD
RMSD_ligand=rms.RMSD(u,u,select="resname UNK",ref_frame=0).run()
time_steps=RMSD_ligand.results.rmsd[:, 1]/1000
RMSD_ligand_results=RMSD_ligand.results.rmsd[:, 2]

RMSD_2_ligand=rms.RMSD(u_2,u_2,select="resid 167-171",ref_frame=0).run()
time_steps_2=RMSD_2_ligand.results.rmsd[: ,1]/1000
RMSD_2_ligand_results=RMSD_2_ligand.results.rmsd[: ,2]

RMSD_3_ligand=rms.RMSD(u_3,u_3,select="resid 167-171",ref_frame=0).run()
time_steps_3=RMSD_3_ligand.results.rmsd[: ,1]/1000
RMSD_3_ligand_results=RMSD_3_ligand.results.rmsd[: ,2]

RMSD_4_ligand=rms.RMSD(u_4,u_4,select="resid 167-171",ref_frame=0).run()
time_steps_4=RMSD_4_ligand.results.rmsd[: ,1]/1000
RMSD_4_ligand_results=RMSD_4_ligand.results.rmsd[: ,2]

RMSD_5_ligand=rms.RMSD(u_5,u_5,select="resname UNL",ref_frame=0).run()
time_steps_5=RMSD_5_ligand.results.rmsd[: ,1]/1000
RMSD_5_ligand_results=RMSD_5_ligand.results.rmsd[: ,2]

# ligand rmsd plot

fig_2=go.Figure()
fig_2.add_trace(go.Scatter(x=combined_t,y=RMSD_ligand_results,name="Ligand_rmsd_1"))
fig_2.add_trace(go.Scatter(x=combined_t,y=RMSD_2_ligand_results,name="Ligand_rmsd_2"))
fig_2.add_trace(go.Scatter(x=combined_t,y=RMSD_3_ligand_results,name="Ligand_rmsd_3"))
fig_2.add_trace(go.Scatter(x=combined_t,y=RMSD_4_ligand_results,name="Ligand_rmsd_4"))
fig_2.add_trace(go.Scatter(x=combined_t,y=RMSD_5_ligand_results,name="Ligand_rmsd_5"))
fig_2.update_layout(title="Ligand rmsd",xaxis_title="ns",yaxis_title="RMSD in angstrom")

fig_2.write_html("Ligand_rmsd")
fig_2.show()

#RMSF analysis

calphas = u.select_atoms("protein and name CA")
calphas_2 = u_2.select_atoms("protein and name CA and not (resid 167-171)")
calphas_3 = u_3.select_atoms("protein and name CA and not (resid 167-171)")
calphas_4 = u_4.select_atoms("protein and name CA and not (resid 167-171)")
calphas_5 = u_5.select_atoms("protein and name CA")


RMSF=rms.RMSF(calphas).run()
res_ids = calphas.resids
rmsf_values = RMSF.results.rmsf


RMSF_2=rms.RMSF(calphas_2).run()
res_ids_2 = calphas_2.resids
rmsf_values_2 = RMSF_2.results.rmsf

RMSF_3=rms.RMSF(calphas_3).run()
res_ids_3 = calphas_3.resids
rmsf_values_3 = RMSF_3.results.rmsf

RMSF_4=rms.RMSF(calphas_4).run()
res_ids_4 = calphas_4.resids
rmsf_values_4 = RMSF_4.results.rmsf

RMSF_5=rms.RMSF(calphas_5).run()
res_ids_5 = calphas_5.resids
rmsf_values_5 = RMSF_5.results.rmsf

#RMSF plot

fig_3=go.Figure()
fig_3.add_trace(go.Scatter(x=res_ids,y=rmsf_values,name="protein_rmsf_1"))
fig_3.add_trace(go.Scatter(x=res_ids_2,y=rmsf_values_2,name="protein_rmsf_2"))
fig_3.add_trace(go.Scatter(x=res_ids_3,y=rmsf_values_3,name="protein_rmsf_3"))
fig_3.add_trace(go.Scatter(x=res_ids_4,y=rmsf_values_4,name="protein_rmsf_4"))
fig_3.add_trace(go.Scatter(x=res_ids_5,y=rmsf_values_5,name="protein_rmsf_5"))
fig_3.update_layout(title="Protein RMSF ",xaxis_title="ns",yaxis_title="RMSF in angstrom")

fig_3.write_html("Protein_RMSF")
fig_3.show()

#Rg analysis

backbone=u.select_atoms("protein and name CA")
rg_values = [backbone.radius_of_gyration() for ts in u.trajectory]

backbone2=u_2.select_atoms("protein and name CA and not (resid 167-171)")
rg2_values = [backbone2.radius_of_gyration() for ts in u_2.trajectory]

backbone3=u_3.select_atoms("protein and name CA and not (resid 167-171)")
rg3_values = [backbone3.radius_of_gyration() for ts in u_3.trajectory]

backbone4=u_4.select_atoms("protein and name CA and not (resid 167-171)")
rg4_values = [backbone4.radius_of_gyration() for ts in u_4.trajectory]

backbone5=u_5.select_atoms("protein and name CA")
rg5_values = [backbone5.radius_of_gyration() for ts in u_5.trajectory]

#Rg plot

fig_4=go.Figure()
fig_4.add_trace(go.Scatter(x=combined_t,y=rg_values,name="Rg_for_protein_1"))
fig_4.add_trace(go.Scatter(x=combined_t,y=rg2_values,name="Rg_for_protein_2"))
fig_4.add_trace(go.Scatter(x=combined_t,y=rg3_values,name="Rg_for_protein_3"))
fig_4.add_trace(go.Scatter(x=combined_t,y=rg4_values,name="Rg_for_protein_4"))
fig_4.add_trace(go.Scatter(x=combined_t,y=rg5_values,name="Rg_for_protein_5"))
fig_4.update_layout(title="Protein Rg ",xaxis_title="ns",yaxis_title="Rg in angstrom")

fig_4.write_html("Protein_Rg")
fig_4.show()

#RMSD analysis for complex

RMSD_complex=rms.RMSD(u,u,select="protein and name CA",groupselections=[f"(protein and name CA) or (resname UNK)"],ref_frame=0).run()
RMSD_complex_time_step=RMSD_complex.results.rmsd[:, 1]/1000
RMSD_results=RMSD_complex.results.rmsd[: ,3]

RMSD_complex_2=rms.RMSD(u_2,u_2,select="protein and name CA and not (resid 167-171)",groupselections=[f"(protein and name CA and not resid 167-171) or (resid 167-171 and not name H*)"],ref_frame=0).run()
RMSD_complex_time_step_2=RMSD_complex_2.results.rmsd[: ,1]/1000
RMSD_results_2=RMSD_complex_2.results.rmsd[: ,3]

RMSD_complex_3=rms.RMSD(u_3,u_3,select="protein and name CA and not (resid 167-171)",groupselections=[f"(protein and name CA and not resid 167-171) or (resid 167-171 and not name H*)"],ref_frame=0).run()
RMSD_complex_time_step_3=RMSD_complex_3.results.rmsd[: ,1]/1000
RMSD_results_3=RMSD_complex_3.results.rmsd[: ,3]

RMSD_complex_4=rms.RMSD(u_4,u_4,select="protein and name CA and not (resid 167-171)",groupselections=[f"(protein and name CA and not resid 167-171) or (resid 167-171 and not name H*)"],ref_frame=0).run()
RMSD_complex_time_step_4=RMSD_complex_4.results.rmsd[: ,1]/1000
RMSD_results_4=RMSD_complex_4.results.rmsd[: ,3]

RMSD_complex_5=rms.RMSD(u_5,u_5,select="protein and name CA",groupselections=[f"(protein and name CA) or (resname UNL)"],ref_frame=0).run()
RMSD_complex_time_step_5=RMSD_complex_5.results.rmsd[: ,1]/1000
RMSD_results_5=RMSD_complex_5.results.rmsd[: ,3]

#complex plots
fig_5=go.Figure()
fig_5.add_trace(go.Scatter(x=combined_t,y=RMSD_results,name="RMSD_for_complex_1"))
fig_5.add_trace(go.Scatter(x=combined_t,y=RMSD_results_2,name="RMSD_for_complex_2"))
fig_5.add_trace(go.Scatter(x=combined_t,y=RMSD_results_3,name="RMSD_for_complex_3"))
fig_5.add_trace(go.Scatter(x=combined_t,y=RMSD_results_4,name="RMSD_for_complex_4"))
fig_5.add_trace(go.Scatter(x=combined_t,y=RMSD_results_5,name="RMSD_for_complex_5"))
fig_5.update_layout(title="Complex rmsd",xaxis_title="ns",yaxis_title="RMSD in angstrom")

fig_5.write_html("Complex_rmsd")
fig_5.show()

#Hydrogen bond analysis

hbonds = HBA(
    universe=u,
    between=['protein and not (resid 167-171)', "resname UNK"],
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=150
)
hbonds.run()

total_frames = len(u.trajectory)
hbond_count = np.zeros(total_frames)

for bond in hbonds.results.hbonds:
    frame_idx = int(bond[0])
    hbond_count[frame_idx] += 1




hbonds2 = HBA(
    universe=u_2,
    between=['protein and not (resid 167-171)', "resid 167-171"],
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=150
)
hbonds2.run()

total_frames_2 = len(u_2.trajectory)
hbond_count_2 = np.zeros(total_frames_2)

for bond in hbonds2.results.hbonds:
    frame_idx_2 = int(bond[0])
    hbond_count_2[frame_idx_2] += 1




hbonds3=HBA(universe=u_3,between=["protein and not (resid 167-171)","resid 167-171"],d_a_cutoff=3.5,d_h_a_angle_cutoff=150).run()

total_frame_3=len(u_3.trajectory)
hbond_count_3=np.zeros(total_frame_3)

for bond in hbonds3.results.hbonds:
    frame_idx_3=int(bond[0])
    hbond_count_3[frame_idx_3]+=1





hbonds4=HBA(universe=u_4,between=["protein and not (resid 167-171)","resid 167-171"],d_a_cutoff=3.5,d_h_a_angle_cutoff=150).run()

total_frame_4=len(u_4.trajectory)
hbond_count_4=np.zeros(total_frame_4)

for bond in hbonds4.results.hbonds:
    frame_idx_4=int(bond[0])
    hbond_count_4[frame_idx_4]+=1


hbonds5=HBA(universe=u_5,between=["protein and not (resid 167-171)","resname UNL"],d_a_cutoff=3.5,d_h_a_angle_cutoff=150).run()

total_frame_5=len(u_5.trajectory)
hbond_count_5=np.zeros(total_frame_5)

for bond in hbonds5.results.hbonds:
    frame_idx_5=int(bond[0])
    hbond_count_5[frame_idx_5]+=1



fig_6=go.Figure()
fig_6.add_trace(go.Scatter(x=combined_t,y=hbond_count,name="hbond_1"))
fig_6.add_trace(go.Scatter(x=combined_t,y=hbond_count_2,name="hbond_2"))
fig_6.add_trace(go.Scatter(x=combined_t,y=hbond_count_3,name="hbond_3"))
fig_6.add_trace(go.Scatter(x=combined_t,y=hbond_count_4,name="hbond_4"))
fig_6.add_trace(go.Scatter(x=combined_t,y=hbond_count_5,name="hbond_5"))
fig_6.update_layout(title="Hydrogen bond",xaxis_title="ns",yaxis_title="Number of hydrogen bonds")

fig_6.write_html("Hydrogen_bond")
fig_6.show()









#PCA analysis

# Run the PCA analysis
pca_projection_1 = pca.PCA(u, select="protein and name CA", align=False).run()
# Project the trajectory onto the computed PCA space
transformed = pca_projection_1.transform(u.select_atoms("protein and name CA"))
# Slice the first two principal components
pc1_1 = transformed[:, 0]  # PC1 projection
pc1_2 = transformed[:, 1]  # PC2 projection


# System 2
sel_2 = u_2.select_atoms("protein and name CA and not (resid 167-171)")
pca_projection_2 = pca.PCA(u_2, select="protein and name CA and not (resid 167-171)", align=False).run()
trans_2 = pca_projection_2.transform(sel_2)
pc2_1 = trans_2[:, 0]
pc2_2 = trans_2[:, 1]

# System 3
sel_3 = u_3.select_atoms("protein and name CA and not (resid 167-171)")
pca_projection_3 = pca.PCA(u_3, select="protein and name CA and not (resid 167-171)", align=False).run()
trans_3 = pca_projection_3.transform(sel_3)
pc3_1 = trans_3[:, 0]
pc3_2 = trans_3[:, 1]

# System 4
sel_4 = u_4.select_atoms("protein and name CA and not (resid 167-171)")
pca_projection_4 = pca.PCA(u_4, select="protein and name CA and not (resid 167-171)", align=False).run()
trans_4 = pca_projection_4.transform(sel_4)
pc4_1 = trans_4[:, 0]
pc4_2 = trans_4[:, 1]

# System 5
sel_5 = u_5.select_atoms("protein and name CA")
pca_projection_5 = pca.PCA(u_5, select="protein and name CA", align=False).run()
trans_5 = pca_projection_5.transform(sel_5)
pc5_1 = trans_5[:, 0]
pc5_2 = trans_5[:, 1]


def calculate_fel(pca_1,pca_2,bins=70,T=300.0):
    kB=0.008314462

    counts,xedges,yedges=np.histogram2d(pca_1,pca_2,bins=bins)
    P_smooth=gaussian_filter(counts,sigma=1.0,mode="nearest")
    P=P_smooth/np.sum(P_smooth)
    P[P == 0] = np.nan
    G = -kB * T * np.log(P)
    G-=np.nanmin(G)
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    # Return grid coordinates and transposed 2D matrix (Y=rows, X=cols)
    return x_centers, y_centers, G.T

x_centers_1, y_centers_1, G_matrix_1 = calculate_fel(
    pc1_1, pc1_2, bins=70, T=300.0
)


x_centers_2, y_centers_2, G_matrix_2 = calculate_fel(
    pc2_1, pc2_2, bins=70, T=300.0
)

x_centers_3, y_centers_3, G_matrix_3 = calculate_fel(
   pc3_1, pc3_2, bins=70, T=300.0
)

x_centers_4, y_centers_4, G_matrix_4 = calculate_fel(
    pc4_1, pc4_2, bins=70, T=300.0
)

x_centers_5, y_centers_5, G_matrix_5 = calculate_fel(
    pc5_1, pc5_2, bins=70, T=300.0
)





fig_pca = go.Figure()
fig_pca.add_trace(go.Surface(
            x=x_centers_1,
            y=y_centers_1,
            z=G_matrix_1,
            colorscale="Rainbow",
            colorbar=dict(title="G (kJ/mol)"),
        ))


fig_pca.update_layout(
    title="3D Free Energy Landscape for reference ligand",
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="G (kJ/mol)",
        zaxis=dict(autorange="reversed")
    ),
    autosize=True,  # Omit width/height to auto-fit container
)

fig_pca.write_html("Chk_Inverted_PCA_analysis_for_ref_plot")


fig_pca2 = go.Figure()
fig_pca2.add_trace(go.Surface(
            x=x_centers_2,
            y=y_centers_2,
            z=G_matrix_2,
            colorscale="Rainbow",
            colorbar=dict(title="G (kJ/mol)"),
        ))


fig_pca2.update_layout(
    title="3D Free Energy Landscape for peptide 1",
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="G (kJ/mol)",
        zaxis=dict(autorange="reversed")
    ),
    autosize=True,  # Omit width/height to auto-fit container
)

fig_pca2.write_html("Chk_Inverted_PCA_analysis_for_pep1_plot")


fig_pca3 = go.Figure()
fig_pca3.add_trace(go.Surface(
            x=x_centers_3,
            y=y_centers_3,
            z=G_matrix_3,
            colorscale="Rainbow",
            colorbar=dict(title="G (kJ/mol)"),
        ))


fig_pca3.update_layout(
    title="3D Free Energy Landscape for peptide 2",
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="G (kJ/mol)",
        zaxis=dict(autorange="reversed")
    ),
    autosize=True,  # Omit width/height to auto-fit container
)

fig_pca3.write_html("Chk_Inverted_PCA_analysis_for_pep2_plot")



fig_pca4 = go.Figure()
fig_pca4.add_trace(go.Surface(
            x=x_centers_4,
            y=y_centers_4,
            z=G_matrix_4,
            colorscale="Rainbow",
            colorbar=dict(title="G (kJ/mol)"),
        ))

fig_pca4.update_layout(
    title="3D Free Energy Landscape for peptide 3",
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="G (kJ/mol)",
        zaxis=dict(autorange="reversed")
    ),
    autosize=True,  # Omit width/height to auto-fit container
)

fig_pca4.write_html("Chk_Inverted_PCA_analysis_for_pep3_plot")




fig_pca5 = go.Figure()
fig_pca5.add_trace(go.Surface(
            x=x_centers_5,
            y=y_centers_5,
            z=G_matrix_5,
            colorscale="Rainbow",
            colorbar=dict(title="G (kJ/mol)"),
        ))

# Responsive layout (fills browser/notebook width automatically)
fig_pca5.update_layout(
    title="3D Free Energy Landscape for peptide 4",
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="G (kJ/mol)",
        zaxis=dict(autorange="reversed")
    ),
    autosize=True,  # Omit width/height to auto-fit container
)

fig_pca5.write_html("Chk_Inverted_PCA_analysis_for_pep4_plot")

