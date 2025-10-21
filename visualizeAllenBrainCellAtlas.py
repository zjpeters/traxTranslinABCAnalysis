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

shortGeneList = ['Tnc','Itga5','Tln1','Pxn','Plaur','Itga10']
fullGeneList = ['Dram1','Fgr','Ifih1','Sp100','Map4k4','Itga5','Tnc','Tln1','Cd84','Cd33','Pxn','Ctsc','Mtmr10',
                'Lyn','Tapbp','Il10ra','Ctsa','Slc13a3','Gpsm3','Ptbp1','Stxbp2','Efs','Arhgef1','Adam17',
                'Pcolce2','P2ry2','Ret','Arhgap9','Ptafr','Trpv4','Tmed7','Plaur','Isg20','Mpeg1','Bach1',
                'Sulf1','Sfrp1','Emp1','Myrf','Hpgds','Col4a5','Itga10','Gbp6']
#%% prepare allen brain cell atlas environment
"""
this code is mainly taken from:

https://alleninstitute.github.io/abc_atlas_access/notebooks/merfish_tutorial_part_1.html
"""
# functions
def plot_section(xx, yy, cc = None, val = None, fig_width = 8, fig_height = 8, cmap = None):
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)
    if cmap is not None:
        plt.scatter(xx, yy, s=0.5, c=val, marker='.', cmap=cmap)
    elif cc is not None:
        plt.scatter(xx, yy, s=0.5, color=cc, marker='.')
    ax.set_ylim(11, 0)
    ax.set_xlim(0, 11)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def create_expression_dataframe(ad, gf):
    gdata = ad[:, gf.index].to_df()
    gdata.columns = gf.gene_symbol
    joined = section.join(gdata)
    return joined

def aggregate_by_metadata(df, gnames, value, sort = False):
    grouped = df.groupby(value)[gnames].mean()
    if sort:
        grouped = grouped.sort_values(by=gnames[0], ascending=False)
    return grouped

def plot_heatmap(df, fig_width = 8, fig_height = 4, cmap = plt.cm.magma_r):

    arr = df.to_numpy()

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)

    im = ax.imshow(arr, cmap=cmap, aspect='auto', vmin=0, vmax=5)
    xlabs = df.columns.values
    ylabs = df.index.values

    ax.set_xticks(range(len(xlabs)))
    ax.set_xticklabels(xlabs)

    ax.set_yticks(range(len(ylabs)))
    res = ax.set_yticklabels(ylabs)
    
    return im

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

pred = [x in shortGeneList for x in adata.var.gene_symbol]
gene_filtered = adata.var[pred]
del pred
asubset = adata[:, gene_filtered.index].to_memory()
del adata
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
#%% turn above code into a function

def plotHippocampalGeneExpression(
        geneName, cellType=None, neurotransmitter=None, 
        singleHemisphere=True, saveImage=False, filetype='png'):
    """
    Plots the expression of a given gene contained in the asubset variable 
    created above.
    
    Parameters
    ----------
    geneName : str
        Gene name for a gene contained in the asubset variable generated above.
    cellType : str, optional
        One of the cell type contained in the 'class' column of the data. 
        The default is None.
    neurotransmitter : str, optional
        One of the neurotransmitters contained in the 'neurotransmitter column
        of the data. The default is None.
    singleHemisphere : bool, optional
        Whether to display only the left hemisphere or both hemispheres of the
        hippocampus. The default is True.
    saveImage : bool, optional
        Whether to save image to default derivatives. The default is False.

    Returns
    -------
    None.

    """
    gf = asubset.var[asubset.var.gene_symbol == geneName]
    geneDataFrame = create_expression_dataframe(asubset_hpf, gf)
    if cellType != None:
        geneDataFrame = geneDataFrame[geneDataFrame['class'] == cellType]
    if neurotransmitter != None:
        geneDataFrame = geneDataFrame[geneDataFrame['neurotransmitter'] == neurotransmitter]
    bgGrey = np.empty_like(section_ccf['x'])
    bgGrey[:] = 0.5
    if np.any(geneDataFrame[geneName]):
        fig, ax = plt.subplots()
        ax.scatter(section_ccf['x'], section_ccf['y'], c='tab:grey', s=3, alpha=0.1)
        sc = ax.scatter(geneDataFrame['x'], geneDataFrame['y'], c=geneDataFrame[geneName], s=3, cmap='Reds')
        # ax.yaxis.set_inverted(True)
        ax.set_aspect('equal')
        # these x and y limits restrict plot to single hemisphere of hippocampus
        if singleHemisphere == True:
            ax.set_ylim(2.8, 5.0)
            ax.set_xlim(2.0, 5.0)
            plt.colorbar(sc,fraction=0.03, pad=0.04)
        else:
            ax.set_ylim(2.8, 5.0)
            ax.set_xlim(2.0, 9.0)
            plt.colorbar(sc,fraction=0.015, pad=0.04)
        ax.yaxis.set_inverted(True)
        plt.axis('off')
        if cellType  == None and neurotransmitter == None:
            plt.title(f'Expression of {geneName} in the Hippocampal formation')
            savePath = os.path.join(derivatives, filetype, 'hippocampalFormation',f'{geneName}_expression_in_Hippocampal_formation.{filetype}')
        elif cellType != None and neurotransmitter == None:
            plt.title(f'Expression of {geneName} in {cellType} in the Hippocampal formation')
            savePath = os.path.join(derivatives, filetype, 'cellTypes', f'{cellType}', f'{geneName}_expression_in_{cellType}_in_Hippocampal_formation.{filetype}')
        elif cellType == None and neurotransmitter != None:
            plt.title(f'Expression of {geneName} in {neurotransmitter} in the Hippocampal formation')
            savePath = os.path.join(derivatives, filetype, 'neurotransmitters', f'{geneName}_expression_in_{neurotransmitter}_Hippocampal_formation.{filetype}')
        elif cellType != None and neurotransmitter != None:
            plt.title(f'Expression of {geneName} in {cellType} and {neurotransmitter} in the Hippocampal formation')
            savePath = os.path.join(derivatives, filetype, f'{geneName}_expression_in_{cellType}_in_{neurotransmitter}_in_Hippocampal_formation.{filetype}')
        plt.show()
        if saveImage==True:
            plt.savefig(savePath, bbox_inches='tight', dpi=300)
            plt.close()
    else:
        if cellType  == None and neurotransmitter == None:
            print(f'No cells expressing {gene} in Hippocampal formation')
        elif cellType != None and neurotransmitter == None:
            print(f'No cells expressing {gene} in {cellType} in Hippocampal formation')
        elif cellType == None and neurotransmitter != None:
            print(f'No cells expressing {gene} in {neurotransmitter} in Hippocampal formation')
        elif cellType != None and neurotransmitter != None:
            print(f'No cells expressing {gene} in {cellType} in {neurotransmitter} in Hippocampal formation')
# plotHippocampalGeneExpression('Tnc', cellType='30 Astro-Epen', singleHemisphere=True, saveImage=True, filetype='png')

#%% plot each of the genes in each of the hippocampal formation

for gene in gene_filtered.gene_symbol:
    plotHippocampalGeneExpression(gene, saveImage=True)

#%% plot each of the genes in each of the cell types

for gene in gene_filtered.gene_symbol:
    for classType in classTypes:
        plotHippocampalGeneExpression(gene, cellType=classType, saveImage=True)

#%% plot each of the genes in each of the neurotransmitters

for gene in gene_filtered.gene_symbol:
    for nt in neurotransmitter_types:
        plotHippocampalGeneExpression(gene, neurotransmitter=nt, saveImage=True)

#%% create figure that includes the three genes present from the short list
# include the genes as columns and four cell types (exc, inh, astro, vascular) as rows
pred = [x in shortGeneList for x in adata.var.gene_symbol]
gene_filtered = adata.var[pred]

asubset = adata[:, gene_filtered.index].to_memory()

gf = asubset.var[asubset.var.gene_symbol == geneName]
geneDataFrame = create_expression_dataframe(asubset_hpf, gf)
#%% plotting
plt.close('all')
fig,ax = plt.subplots(4,3)
ax[0].scatter(section_ccf['x'], section_ccf['y'], c='tab:grey', s=3, alpha=0.1)
sc = ax[0].scatter(geneDataFrame['x'], geneDataFrame['y'], c=geneDataFrame[geneName], s=3, cmap='Reds')
# ax.yaxis.set_inverted(True)
ax.set_aspect('equal')