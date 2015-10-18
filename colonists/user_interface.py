# coding: utf-8
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# Colors table
colors_csv = '''
group,color,r,g,b
light,grey,140,140,140
light,blue,136,189,230
light,orange,251,178,88
light,green,144,205,151
light,pink,246,170,201
light,brown,191,165,84
light,purple,188,153,199
light,yellow,237,221,70
light,red,240,126,110
medium,grey,77,77,77
medium,blue,93,165,218
medium,orange,250,154,58
medium,green,96,189,104
medium,pink,241,124,176
medium,brown,178,145,47
medium,purple,178,118,178
medium,yellow,222,207,63
medium,red,241,88,84
dark,grey,0,0,0
dark,blue,38,93,171
dark,orange,223,92,36
dark,green,5,151,72
dark,pink,229,18,111
dark,brown,157,114,42
dark,purple,123,58,150
dark,yellow,199,180,46
dark,red,203,32,39
'''.strip()

df_colors = pd.read_csv(BytesIO(colors_csv)).set_index(['group', 'color'])
df_colors['hex'] = df_colors.apply(lambda x: '#%02x%02x%02x' % tuple(x),
                                   axis=1)


TERRAIN_COLORS = pd.Series([('light', 'red'),  # clay
                            ('light', 'green'),  # sheep
                            ('medium', 'grey'),  # ore
                            ('light', 'yellow'),  # wheat
                            ('dark', 'green'),  # wood
                            ('medium', 'brown'),  # desert
                            ('light', 'red'),  # clay port
                            ('light', 'green'),  # sheep port
                            ('light', 'grey'),  # ore port
                            ('light', 'yellow'),  # wheat port
                            ('medium', 'green'),  # wood port
                            ('light', 'orange'),  # 3:1 port
                            ('medium', 'blue'),  # sea
                            ],
                            index=['clay', 'sheep', 'ore', 'wheat',
                                   'wood',
                                   'desert',
                                   'clay_port', 'sheep_port',
                                   'ore_port', 'wheat_port',
                                   'wood_port', 'three_to_one_port',
                                   'sea'])


def plot_hexes(df_nodes, df_hexes, df_hex_paths, axis=None, labelby='node',
               colorby='terrain', region_colors=None, terrain_colors=None):
    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 6))

    df_nodes.dropna().plot(kind='scatter', x='x', y='y', ax=axis,
                           s=16 ** 2)

    if region_colors is None:
        region_colors = pd.Series(['brown', 'blue', 'white'],
                                  index=['land', 'sea', 'port'])

    if terrain_colors is None:
        terrain_colors = TERRAIN_COLORS.map(lambda v: df_colors.loc[v].hex)

    for hex_i, df_i in df_hex_paths.groupby('hex'):
        hex_info = df_hexes.iloc[hex_i]
        if colorby == 'terrain':
            color = terrain_colors[hex_info.terrain]
        elif colorby == 'region':
            color = region_colors[hex_info.region]
        else:
            raise ValueError('Invalid hex attribute to color by.  Must be '
                             'either "terrain" or "region" (not %s)' % colorby)
        alpha = .9 if hex_info.region == 'port' else 1.0
        poly = Polygon(df_i[['x', 'y']].values, facecolor=color, alpha=alpha)
        axis.add_patch(poly)
        center = df_i[['x', 'y']].mean()

        text_kwargs = {'fontsize': 14}
        if labelby == 'node':
            # Label each hex using hex index.
            label = str(hex_i)
        elif labelby == 'collect_index':
            # Label each land hex using collect index (i.e., dice number), and
            # each port based on trade ratio.
            label = (int(hex_info.collect_index)
                     if (hex_info.collect_index == hex_info.collect_index)
                     else '')
            if label in (6, 8):
                text_kwargs['fontweight'] = 'bold'
                text_kwargs['fontsize'] = 18
            elif label in (5, 9):
                text_kwargs['fontsize'] = 16
            elif label in (2, 12):
                text_kwargs['fontsize'] = 10
            elif hex_info.region == 'port':
                if hex_info.terrain == 'three_to_one_port':
                    label = '3:1'
                else:
                    label = '2:1'
        else:
            raise ValueError('Invalid hex attribute to label by.  Must be '
                             'either "node" or "collect_index" (not %s)' %
                             labelby)
        axis.text(center.x, center.y, label, ha='center', va='center',
                  **text_kwargs)
    axis.set_aspect(True)
    return axis
