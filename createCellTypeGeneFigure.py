#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:46:30 2025

@author: zjpeters
"""
import pandas as pd
from pathlib import Path
import numpy as np
import anndata
import time
import matplotlib.pyplot as plt
import SimpleITK as sitk
import csv
import os
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

# location to store output images
derivatives = os.path.join('/','media','zjpeters','Expansion','traxManuscript','derivatives')
#%% Create lists of possible genes

shortGeneList = ['Tnc','Itga5','Tln1','Pxn','Plaur','Itga10', 'Tsn']
fullGeneList = ['Tsn', 'Dram1','Fgr','Ifih1','Sp100','Map4k4','Itga5','Tnc','Tln1','Cd84','Cd33','Pxn','Ctsc','Mtmr10',
                'Lyn','Tapbp','Il10ra','Ctsa','Slc13a3','Gpsm3','Ptbp1','Stxbp2','Efs','Arhgef1','Adam17',
                'Pcolce2','P2ry2','Ret','Arhgap9','Ptafr','Trpv4','Tmed7','Plaur','Isg20','Mpeg1','Bach1',
                'Sulf1','Sfrp1','Emp1','Myrf','Hpgds','Col4a5','Itga10','Gbp6']
#%% prepare allen brain cell atlas environment


def create_expression_dataframe(ad, gf):
    """
    this code is taken from:
    https://alleninstitute.github.io/abc_atlas_access/notebooks/merfish_tutorial_part_1.html
    """
    gdata = ad[:, gf.index].to_df()
    gdata.columns = gf.gene_symbol
    joined = section.join(gdata)
    return joined

# importing data
dataLocation = Path('/media/zjpeters/Expansion/allenBrainCellAtlas/sourcedata/abc_atlas')
abc_cache = AbcProjectCache.from_cache_dir(dataLocation)

abc_cache.current_manifest
# cell information
cell = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850',
    file_name='cell_metadata',
    dtype={"cell_label": str}
)
cell.set_index('cell_label', inplace=True)

# cluster/taxonomy information
cluster_details = abc_cache.get_metadata_dataframe(
    directory='WMB-taxonomy',
    file_name='cluster_to_cluster_annotation_membership_pivoted',
    keep_default_na=False
)
cluster_details.set_index('cluster_alias', inplace=True)

cluster_colors = abc_cache.get_metadata_dataframe(directory='WMB-taxonomy', file_name='cluster_to_cluster_annotation_membership_color')
cluster_colors.set_index('cluster_alias', inplace=True)

cell_extended = cell.join(cluster_details, on='cluster_alias')
cell_extended = cell_extended.join(cluster_colors, on='cluster_alias')

# gene information
gene = abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850', file_name='gene')
gene.set_index('gene_identifier', inplace=True)

# # load h5ad file
# abc_cache.list_data_files('MERFISH-C57BL6J-638850')
# file = abc_cache.get_data_path(directory='MERFISH-C57BL6J-638850', file_name='C57BL6J-638850/log2')

#### NOTE: the imputed file below is 50.2gb
imputed_h5ad_path = abc_cache.get_data_path('MERFISH-C57BL6J-638850-imputed', 'C57BL6J-638850-imputed/log2')

adata = anndata.read_h5ad(imputed_h5ad_path, backed='r')

# select for a single section (59 sections in this dataset)
pred = (cell_extended['brain_section_label'] == 'C57BL6J-638850.38')
section = cell_extended[pred]

# create list of potential class types and neurotransmitters
classTypes = np.unique(section['class'])
neurotransmitter_types = np.unique(section['neurotransmitter'])
# del section
#%% look for genes from list in the data

pred = [x in fullGeneList for x in adata.var.gene_symbol]
gene_filtered = adata.var[pred]
del pred
asubset = adata[:, gene_filtered.index].to_memory()
del adata

#%% reorder gene list so Tsn is first
# first find index of Tsn
tsn_idx = gene_filtered[gene_filtered['gene_symbol'] == 'Tsn'].index[0]
gene_filtered_idxs = gene_filtered.index
# create new list where remaining genes are in same order, but Tsn is first
new_gene_filtered_idxs = [tsn_idx]
for i in gene_filtered_idxs:
    if i != tsn_idx:
        new_gene_filtered_idxs.append(i)
# perform reindexing of data
gene_filtered = gene_filtered.reindex(new_gene_filtered_idxs)
#%% load ccf registered coordinates for all cellsannotation

cell = abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850', file_name='cell_metadata_with_cluster_annotation')
cell.rename(columns={'x': 'x_section',
                     'y': 'y_section',
                     'z': 'z_section'},
            inplace=True)
cell.set_index('cell_label', inplace=True)

# #%% 
# reconstructed_coords = abc_cache.get_metadata_dataframe(
#     directory='MERFISH-C57BL6J-638850-CCF',
#     file_name='reconstructed_coordinates',
#     dtype={"cell_label": str}
# )
# reconstructed_coords.rename(columns={'x': 'x_reconstructed',
#                                      'y': 'y_reconstructed',
#                                      'z': 'z_reconstructed'},
#                             inplace=True)
# reconstructed_coords.set_index('cell_label', inplace=True)

#%% import ccf coordinates
ccf_coords = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850-CCF',
    file_name='ccf_coordinates',
    dtype={"cell_label": str}
)
ccf_coords.rename(columns={'x': 'x_ccf',
                           'y': 'y_ccf',
                           'z': 'z_ccf'},
                  inplace=True)
# ccf_coords.drop(['parcellation_index'], axis=1, inplace=True)
ccf_coords.set_index('cell_label', inplace=True)

#%% import parcellation information

parcellation_term = abc_cache.get_metadata_dataframe(directory='Allen-CCF-2020', file_name='parcellation_term')
parcellation_term.set_index('label', inplace=True)

parcellation_annotation = abc_cache.get_metadata_dataframe(directory='Allen-CCF-2020',
                                                           file_name='parcellation_to_parcellation_term_membership_acronym')
parcellation_annotation.set_index('parcellation_index', inplace=True)
parcellation_annotation.columns = ['parcellation_%s'% x for x in  parcellation_annotation.columns]

#%% need to restrict original data based on cells contained in ccf
# loop through ccf coordinates and find which are present in merfish data
ccf_cells_idx = [x in ccf_coords.index for x in cell.index]
adata_ccf_idx = [x in ccf_coords.index for x in asubset.obs.index]

#%% mask cells 
ccf_cells = cell_extended[ccf_cells_idx]
ccf_cells = ccf_cells.join(ccf_coords)

asubset_ccf = asubset[adata_ccf_idx,:]

section_ccf = ccf_cells[ccf_cells['brain_section_label'] == 'C57BL6J-638850.38']
del ccf_cells_idx
del adata_ccf_idx
#%% use the parcellation_annotation to display only data from certain regions

hpfAnnot = parcellation_annotation[parcellation_annotation['parcellation_division'] == 'HPF']
hpfIdxs = hpfAnnot.index
del hpfAnnot
#%% create subset of only hippocampal formation
hpfMask = [ x in hpfIdxs for x in section_ccf['parcellation_index']]
section_hpf = section_ccf[hpfMask]

adata_hpf_idx = [x in section_hpf.index for x in asubset_ccf.obs.index]
asubset_hpf = asubset_ccf[adata_hpf_idx]
del section_hpf

#%% create figure that includes the three genes present from the short list
# include the genes as columns and four cell types (exc, inh, astro, vascular) as rows
cellTypesOfInterest = {'Excitatory': ['01 IT-ET Glut', '02 NP-CT-L6b Glut', '03 OB-CR Glut', 
                        '04 DG-IMN Glut', '13 CNU-HYa Glut', '14 HY Glut', '15 HY Gnrh1 Glut',
                        '16 HY MM Glut', '17 MH-LH Glut', '18 TH Glut', '19 MB Glut',
                        '23 P Glut', '24 MY Glut'],
                       'Inhibitory': ['05 OB-IMN GABA','06 CTX-CGE GABA', '07 CTX-MGE GABA', 
                        '08 CNU-MGE GABA', '09 CNU-LGE GABA', '11 CNU-HYa GABA', 
                        '12 HY GABA', '20 MB GABA', '27 MY GABA'],
                       'Astrocytes': ['30 Astro-Epen'],
                       'Vascular': ['33 Vascular']}

#%% limit data points to only those within the hippocampal formation
def restrictToHPF(inputDataFrame, hemispheres='single'):
    """
    Generates a subset of a dataframe restricted to the coordinates included
    in one hemisphere of the mouse hippocampal formation

    Parameters
    ----------
    inputDataFrame : pandas dataframe
        A pandas dataframe from ABC data.

    Returns
    -------
    hpfDataFrame : pandas dataframe
        A pandas dataframe formatted for ABC data restricted using coordinates.

    """
    hpfDataFrame = inputDataFrame[inputDataFrame['x'] > 2.0]
    if hemispheres == 'single':
        hpfDataFrame = hpfDataFrame[hpfDataFrame['x'] < 5.0]
    elif hemispheres == 'double':
        hpfDataFrame = hpfDataFrame[hpfDataFrame['x'] < 9.0]
    hpfDataFrame = hpfDataFrame[hpfDataFrame['y'] > 2.5]
    hpfDataFrame = hpfDataFrame[hpfDataFrame['y'] < 5.0]
    return hpfDataFrame

#%% find maximum expression value across genes

for i in range(len(gene_filtered.gene_symbol)):
    geneName = gene_filtered.gene_symbol[i]
    print(geneName)
    gf = asubset.var[asubset.var.gene_symbol == geneName]
    geneDataFrame = create_expression_dataframe(asubset_hpf, gf)
    geneDataFrame = restrictToHPF(geneDataFrame)
    print(np.max(geneDataFrame[geneName]))

#%% plotting to multiple figures
# for a set of 25 genes, breaking it into 5 figures of 5 genes each,
# plotted with genes along y-axis, cell types along x-axis

# don't need to repeat the following line since it's run above
section_hpf = restrictToHPF(section_ccf, hemispheres='single')

plt.close('all')
# , figsize=(8.5, 9.5)

fig,ax = plt.subplots(5, len(cellTypesOfInterest), figsize=(9.5, 9.5))
figNumber = 1
geneNumber = 0
for i in range(len(gene_filtered.gene_symbol)):
    geneName = gene_filtered.gene_symbol[i]
    print(geneName)
    gf = asubset.var[asubset.var.gene_symbol == geneName]
    geneDataFrame = create_expression_dataframe(asubset_hpf, gf)
    geneDataFrame = restrictToHPF(geneDataFrame)
    # loop over the cell type groups
    for j in enumerate(cellTypesOfInterest):
        ax[0, j[0]].set_title(j[1])
        cellTypeMask = [x in cellTypesOfInterest[j[1]] for x in geneDataFrame['class']]
        ax[geneNumber, 0].set_ylabel(geneName, rotation='horizontal', horizontalalignment='right')
        cellTypeDataFrame = geneDataFrame[cellTypeMask]
        ax[geneNumber, j[0]].scatter(section_hpf['x'], section_hpf['y'], c='tab:grey', s=3, alpha=0.1)
        sc = ax[geneNumber, j[0]].scatter(cellTypeDataFrame['x'], cellTypeDataFrame['y'], c=cellTypeDataFrame[geneName], s=1, cmap='Reds', vmin=0, vmax=8)
        ax[geneNumber, j[0]].yaxis.set_inverted(True)
        ax[geneNumber, j[0]].set_aspect('equal')
        ax[geneNumber, j[0]].tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            left=False,
            labelleft=False)
    if geneNumber == 4:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        fig.colorbar(sc, cax=cbar_ax, fraction=0.015, pad=0.04)
        plt.show()
        plt.savefig(os.path.join(derivatives, f'threeGeneFourCellTypes_vertical_{figNumber}.png'), bbox_inches='tight', dpi=300)
        fig,ax = plt.subplots(5, len(cellTypesOfInterest), figsize=(9.5, 9.5))
        geneNumber = 0
        figNumber +=1
    else:
        geneNumber += 1

