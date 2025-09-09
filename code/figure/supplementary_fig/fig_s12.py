# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 下午6:39
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig_s12.py
# @IDE     : PyCharm
# @REMARKS : Description text
import pandas as pd
import numpy as np
import baltic as bt
import dendropy
from io import StringIO
from tqdm import tqdm
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
mpl.rcParams['pdf.fonttype']=42

# Define function to calculate entropy
def ent(data):
    prob1 = pd.value_counts(data) / len(data)
    return sum(np.log2(prob1) * prob1 * (-1))

# Define function to calculate information gain
def gain(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    e2 = sum(e1 * p1)
    return ent(data[str2]) - e2

def setAbsoluteTime(tree, dates):
    for i in tree.Objects:
        if i.is_leaf():
            i.absoluteTime=float(dates.loc[i.name, 'numeric date'])
        else:
            i.absoluteTime=float(dates.loc[i.traits['label'], 'numeric date'])
    tree.mostRecent = float(dates['numeric date'].max())
    return tree

df_all = pd.read_csv(r'./result/BY_IG.csv',index_col=0)

df = pd.read_csv(r'./result/BY_site_kind.csv',index_col=0)
print(df.shape[0])
print(df.columns)

aa_color = {'G':'#ECB9D1', 'A':'#B3C6E6', 'V':'#569EB1', 'L':'#7BC9D8', 'I':'#ACD8E4', 'F':'#8D69B9', 'P':'#C1B1D3',
            'W':'#ED8730', 'S':'#F4BE80', 'Y':'#DE776E', 'C':'#F09D98', 'M':'#CC4DCC', 'N':'#3D75B0', 'Q':'#F6442C', 'T':'#BD9E95',
            'D':'#539E3A', 'E':'#A9DD90',
            'K':'#BCBD3E', 'R':'#19B3B3', 'H':'#DBDB94',
            '--':'#ED8730','---':'#84594D','nogap':'#8D69B9'}

dates = pd.read_csv('./result/dates.tsv', sep='\t', index_col=0)

tree_file = './result/treetime_paper/timetree.nexus'

df_color = pd.read_csv('./result/BY_node_color_paper.csv', index_col=0)

tree = dendropy.Tree.get(path=tree_file, schema='nexus', extract_comment_metadata=True)
treeString = tree.as_string(schema='newick', suppress_rooting=True, suppress_annotations=True, annotations_as_nhx=False)

ll=bt.loadNewick(StringIO(treeString), tip_regex='_([0-9\-]+)$', absoluteTime=False)
ll=setAbsoluteTime(ll, dates)

node_color = df_color.set_index('tree_name')['color'].to_dict()
# print(node_color)
treeHeight = ll.treeHeight
print(treeHeight)

#指定厘米换算单位
cm = 2.54
# 画关键位点散点
fig = plt.figure(figsize=(18.4/cm, 18.4/cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(3,6)
trans = ScaledTranslation(-30/300, 30/300, fig.dpi_scale_trans)

with mpl.rc_context({'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):

    fig_num = 'ABCDE'

    for i , p in enumerate([56,232]):

        ax = fig.add_subplot(spec[0, i+1])
        ax.text(0.0, 1.0, '', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',fontfamily='Arial')
        y_attr = lambda k: -k.y
        x_attr = lambda k: k.absoluteTime

        node_aa = df.set_index('tree_name')['pos_'+str(p)].to_dict()
        def fetch_leaf_color(k):
            return aa_color.get(node_aa[k.name], 'black')

        color_func = lambda k: fetch_leaf_color(k) if k.branchType == 'leaf' else 'black'
        print(color_func)
        ll.plotTree(ax, width=0.7, x_attr=x_attr, y_attr=y_attr, alpha=0.9)
        ll.plotPoints(ax, x_attr=x_attr, y_attr=y_attr, size=0.5, colour=color_func, zorder=100, alpha=0.9)

        ax.set_xticks([])
        ax.set_yticks([])
        if p != 56:
            aa_mode = df_all.groupby('new_cluster')['pos_'+str(p)].apply(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
            aa_fore = aa_mode[aa_mode['new_cluster'] == 3]['pos_'+str(p)].iloc[0] if not aa_mode[aa_mode['new_cluster'] == 3].empty else None
            aa_after = aa_mode[aa_mode['new_cluster'] == 2]['pos_'+str(p)].iloc[0] if not aa_mode[aa_mode['new_cluster'] == 2].empty else None
            ax.set_title('Site ' + aa_fore+str(p)+aa_after)
        if p == 56:
            ax.set_title('Site ' + 'N' + str(p) + 'DT')

        ax.axhline(y=-2295.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.axhline(y=-3182.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.text(1990, (-2295.5 - 3182.5) / 2, 'BJ93', fontsize=6)
        ax.axhline(y=-3204.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.text(2012, -3204.5 - 1 / 2, 'YA88', fontsize=6)

        [ax.spines[loc].set_visible(False) for loc in ax.spines]
        ax.grid(False)

    fig_num = 'CDEAB'
    for i, p in enumerate([48,108,116,150,165,172,202,229,298,312]):
        if i <= 4:
            ax = fig.add_subplot(spec[1, i+1])
        else:
            ax = fig.add_subplot(spec[2, i+1-5])
        ax.text(0.0, 1.0, '', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
                fontfamily='Arial')
        y_attr = lambda k: -k.y
        x_attr = lambda k: k.absoluteTime

        node_aa = df.set_index('tree_name')['pos_' + str(p)].to_dict()


        def fetch_leaf_color(k):
            return aa_color.get(node_aa[k.name], 'black')


        color_func = lambda k: fetch_leaf_color(k) if k.branchType == 'leaf' else 'black'
        print(color_func)
        ll.plotTree(ax, width=0.7, x_attr=x_attr, y_attr=y_attr, alpha=0.9)
        ll.plotPoints(ax, x_attr=x_attr, y_attr=y_attr, size=0.5, colour=color_func, zorder=100, alpha=0.9)

        ax.set_xticks([])
        ax.set_yticks([])
        if p == 48:
            ax.set_title('Site ' + 'KR' + str(p) + 'R')
        elif p == 108:
                    ax.set_title('Site ' + 'AP' + str(p) + 'P')
        elif p == 116:
                    ax.set_title('Site ' + 'NK' + str(p) + 'K')
        else:
            aa_mode = df_all.groupby('new_cluster')['pos_' + str(p)].apply(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
            aa_fore = aa_mode[aa_mode['new_cluster'] == 2]['pos_' + str(p)].iloc[0] if not aa_mode[aa_mode['new_cluster'] == 2].empty else None
            aa_after = aa_mode[aa_mode['new_cluster'] == 1]['pos_' + str(p)].iloc[0] if not aa_mode[aa_mode['new_cluster'] == 1].empty else None
            ax.set_title('Site ' + aa_fore + str(p) + aa_after)

        ax.axhline(y=-1.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.axhline(y=-2295.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.text(2000, (-1.5 - 2295.5) / 2, 'WI10', fontsize=6)
        ax.axhline(y=-3182.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.text(1990, (-2295.5 - 3182.5) / 2, 'BJ93', fontsize=6)

        [ax.spines[loc].set_visible(False) for loc in ax.spines]
        ax.grid(False)


    fig_num = 'AB '
    fig_title = ['YA88-BJ93', 'BJ93-WI10', 'BJ93-WI10']
    for i in range(0,3):
        ax = fig.add_subplot(spec[i, 0])
        ax.text(0.0, 1.0, fig_num[i], transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
                fontfamily='Arial')

        y_attr = lambda k: -k.y
        x_attr = lambda k: k.absoluteTime


        def fetch_leaf_color(k):
            return node_color.get(k.name, 'black')


        color_func = lambda k: fetch_leaf_color(k) if k.branchType == 'leaf' else 'black'
        print(color_func)
        ll.plotTree(ax, width=0.7, x_attr=x_attr, y_attr=y_attr, alpha=0.9)
        ll.plotPoints(ax, x_attr=x_attr, y_attr=y_attr, size=0.5, colour=color_func, zorder=100, alpha=0.9)

        ct = 0
        for leaf in ll.getExternal():
            y = leaf.y

            color = node_color.get(leaf.name, 'k')

            rect = plt.Rectangle(
                (2025, -leaf.y - 1 / 2),
                1.5,
                1,
                facecolor=color,
                edgecolor=color,
                lw=.0,
                alpha=0.9
            )
            ax.add_patch(rect)  ## add rectangle to plot

        ax.set_title(fig_title[i])
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.axhline(y=-1.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.axhline(y=-2295.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.text(2000, (-1.5-2295.5) / 2, 'WI10',fontsize=6)
        ax.axhline(y=-3182.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.text(1990, (-2295.5-3182.5) / 2, 'BJ93',fontsize=6)
        ax.axhline(y=-3204.5 - 1 / 2, color='k', linestyle='--', linewidth=0.8)
        ax.text(2012, -3204.5 - 1 / 2, 'YA88', fontsize=6)

        [ax.spines[loc].set_visible(False) for loc in ax.spines]
        ax.grid(False)

    ax = fig.add_subplot(spec[0, 4])
    legend_color = {'G': '#ECB9D1', 'A': '#B3C6E6', 'V': '#569EB1', 'L': '#7BC9D8', 'I': '#ACD8E4', 'F': '#A3A3A3','P': '#C1B1D3',
                    'W': '#7F7F7F', 'S': '#F4BE80', 'Y': '#DE776E', 'C': '#F09D98', 'M': '#CC4DCC', 'N': '#3D75B0',
                    'Q': '#F6442C', 'T': '#BD9E95',
                    'D': '#539E3A', 'E': '#A9DD90',
                    'K': '#BCBD3E', 'R': '#19B3B3', 'H': '#DBDB94',
                    }
    legend_order = ['N','R','V','A','L','I','D','E',
                    'Q','S','Y','C','M','G','T','W','F','P',
                    'K','H']
    legend_color = dict(sorted(legend_color.items(), key=lambda x: legend_order.index(x[0])))
    handles = []
    for k, v in legend_color.items():
        handles.append(Line2D([0], [0], lw=0, marker='o', markeredgecolor='none', markersize=6, markerfacecolor=v, label=k, alpha=.9))
    ax.legend(handles=handles, frameon=False, fontsize=6, loc='center', ncols=3,handletextpad=0.2,columnspacing=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[loc].set_visible(False) for loc in ax.spines]
    ax.grid(False)

plt.savefig(r"./figure/fig_BY_keypoints_tree.pdf")
plt.show()
