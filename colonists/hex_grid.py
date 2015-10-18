# coding: utf-8
import numpy as np
import pandas as pd


def get_row_edges(df_nodes):
    frames = []
    for i, df_i in df_nodes.groupby('row'):
        frame = pd.DataFrame([[df_i.index[k], df_i.index[k + 1],
                               df_i.row_i.iloc[k], df_i.row_i.iloc[k + 1]]
                               for k in xrange(df_i.index.shape[0] - 1)],
                              columns=['index_left', 'index_right',
                                       'column_left', 'column_right'])
        frame.insert(0, 'row', i)
        frames.append(frame)
    return pd.concat(frames)


def get_vertical_edges(df_nodes):
    even_rows = (df_nodes.row & 0x1 == 0)
    even_columns = (df_nodes.row_i & 0x1 == 0)
    odd_rows = (df_nodes.row & 0x1 == 1)
    odd_columns = (df_nodes.row_i & 0x1 == 1)

    even_row_columns = (df_nodes[even_rows & even_columns]
                        .reset_index().join(df_nodes[odd_rows & even_columns]
                                            .reset_index(), lsuffix='_top',
                                            rsuffix='_bottom'))

    top = df_nodes[odd_rows & odd_columns].reset_index()
    bottom = df_nodes[(df_nodes.row > 0) & even_rows & odd_columns].reset_index()
    odd_row_columns = (top.iloc[:bottom.shape[0]]
                       .join(bottom, lsuffix='_top', rsuffix='_bottom'))

    all_columns = pd.concat([even_row_columns, odd_row_columns])
    return all_columns.sort(all_columns.columns.values.tolist())


def get_edges(df_nodes):
    row_edges = get_vertical_edges(df_nodes)
    vertical_edges = get_vertical_edges(df_nodes)

    new_labels = ['source', 'target']
    labels = ['index_left', 'index_right']
    row_edges = get_row_edges(df_nodes)[labels].rename(columns=dict(zip(labels, new_labels)))
    labels = ['index_top', 'index_bottom']
    vertical_edges = get_vertical_edges(df_nodes)[labels].rename(columns=dict(zip(labels, new_labels)))
    edges = pd.concat([row_edges, vertical_edges])
    return edges.sort(edges.columns.values.tolist())


def get_nodes(row_count, row_length, unit_offset=.25, offset_scale=1.7):
    df_nodes = pd.DataFrame(np.arange(row_count * row_length) / row_length,
                            columns=['row'])
    df_nodes['row_i'] = df_nodes.index.values % row_length

    even_rows = (df_nodes.row & 0x1 == 0)
    odd_rows = (df_nodes.row & 0x1 == 1)
    points = (df_nodes.row_i & 0x1 == 1)

    offset = (even_rows.astype(int) * ((-2 * points) + 1) +
              odd_rows.astype(int) * ((2 * points) - 1)) * unit_offset
    df_nodes['x'] = df_nodes.row_i
    df_nodes['y'] = (df_nodes.row + offset + unit_offset) * offset_scale
    return df_nodes


def get_hex_links_a(df_nodes):
    frames = []

    for i, df_i in df_nodes.groupby('row'):
        # Left-aligned rows
        for j, k in enumerate(xrange(0, df_i.shape[0] - 2, 2)):
            hex_nodes = df_i.iloc[k:k + 3].copy()
            hex_nodes['hex_row'] = 2 * (i // 2)
            hex_nodes['hex_row_i'] = j
            frames.append(hex_nodes)

        if i > 0 and i < df_nodes.row.max():
            # "Indented" rows
            for j, k in enumerate(xrange(1, df_i.shape[0] - 2, 2)):
                hex_nodes = df_i.iloc[k:k + 3].copy()
                hex_nodes['hex_row'] = 2 *((i - 1) // 2) + 1
                hex_nodes['hex_row_i'] = j
                frames.append(hex_nodes)

    df_hex_links = (pd.concat(frames)[['hex_row', 'hex_row_i', 'row', 'row_i']]
                    .sort(['hex_row', 'hex_row_i', 'row', 'row_i']))

    df_hex_index = (df_hex_links[['hex_row', 'hex_row_i']].drop_duplicates()
                    .reset_index(drop=True).reset_index()
                    .set_index(['hex_row', 'hex_row_i']))
    return df_hex_links, df_hex_index


def get_hex_links(df_nodes):
    df_hex_links, df_hex_index = get_hex_links_a(df_nodes)
    df_hex_links['hex'] = (df_hex_links[['hex_row', 'hex_row_i']]
                           .apply(lambda row: df_hex_index
                                  .loc[row['hex_row'],
                                       row['hex_row_i']]['index'], axis=1))
    node_index = df_nodes.reset_index().set_index(['row', 'row_i'])['index']
    df_hex_links['node'] = (df_hex_links
                            .apply(lambda row: node_index.loc[row['row'],
                                                              row['row_i']],
                                   axis=1))
    return (df_hex_links[['hex', 'node', 'hex_row', 'hex_row_i']]
            .sort(['hex', 'node', 'hex_row',
                   'hex_row_i'])).reset_index(drop=True)


def get_hex_paths(df_nodes):
    df_hex_links = get_hex_links(df_nodes)
    df_hex_link_xy = df_hex_links.join(df_nodes[['x', 'y']].iloc[df_hex_links.node.values]
                                       .reset_index(drop=True)).sort(['hex', 'y', 'x'])
    df_hex_link_xy.insert(1, 'vertex_i',
                          np.tile([0, 1, 5, 2, 4, 3],
                                  df_hex_link_xy.hex.unique().shape[0]))
    return df_hex_link_xy.sort(['hex', 'vertex_i']).reset_index(drop=True)


class HexGrid(object):
    def __init__(self, row_count, row_length, unit_offset=.25,
                 offset_scale=1.7):
        '''
        Create hex grid, including:

         - Hex vertices (i.e, nodes), including x/y position.
         - Edges between nodes.
         - Links between nodes and adjancent hexes (by node and hex index).
         - Polygon definition of each hex.
        '''
        # Hex vertices (i.e, nodes), including x/y position.
        self.df_nodes = get_nodes(row_count, row_length, unit_offset, offset_scale)
        # Edges between nodes.
        self.df_edges = get_edges(self.df_nodes)
        # Links between nodes and adjancent hexes (by node and hex index).
        self.df_hex_links = get_hex_links(self.df_nodes)
        # Polygon definition of each hex.
        self.df_hex_paths = get_hex_paths(self.df_nodes)

    @property
    def size(self):
        return self.df_hex_paths.hex.unique().max() + 1
