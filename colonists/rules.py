# coding: utf-8
import pandas as pd
import numpy as np


CENTER_HEX_INDEX = 26
ORDERED_COLLECT_INDEXES = [5, 2, 6, 3, 8, 10, 9, 12, 11,
                           4, 8, 10, 9, 4, 5, 6, 3, 11]


TERRAIN_TYPES = pd.DataFrame(['clay', 'sheep', 'ore', 'wheat',
                              'wood', 'desert',
                              'clay_port', 'sheep_port',
                              'ore_port', 'wheat_port',
                              'wood_port', 'three_to_one_port',
                              'sea'], columns=['terrain'])

# Define default map.
DEFAULT_MAP = dict(land=(range(10, 13) + range(17, 21) + range(24, 29)
                         + range(32, 36) + range(40, 43)),
                   border=(range(2, 6) + [13, 21, 29, 36, 43]
                           + range(50, 46, -1) + [39, 31, 23, 16, 9]),
                   terrain_counts=[3, 4, 3, 4, 4, 1, 1, 1, 1, 1, 1, 4, 9])


def get_empty_node_contents(df_nodes):
    '''
    Return data frame, where each row indicates the elements present at
    corresponding node.
    '''
    return pd.DataFrame(df_nodes.shape[0] * [5 * [0]],
                                    columns=['camp', 'village', 'walls',
                                             'castle', 'knight'],
                                    index=df_nodes.index)


def land_adjacent(df_hexes, df_hex_links, node_index):
    '''
    Check if node is adjacent to at least one land hex.
    '''
    hex_neighbours = (df_hex_links.loc[df_hex_links.node == node_index]
                      .join(df_hexes, on='hex'))
    return hex_neighbours.loc[hex_neighbours.region == 'land'].shape[0] > 0


def place_camp(df_nodes, df_edges, df_hexes, df_hex_links, node_index,
               df_node_contents, inplace=False):
    '''
    Update node contents to reflect a new camp at the node specified by `node_index`.

    An `IndexError` exception is raised if the specified node is not adjacent to
    any land tile.

    A `ValueError` exception is raised if the specified node or any of its immediate
    neighbours is already occupied.
    '''
    if not land_adjacent(df_hexes, df_hex_links, node_index):
        raise IndexError('Node %s is not adjancent to any land hex.' % node_index)

    # Check if selected position is occupied by an existing camp or village.
    if df_node_contents.loc[node_index, ['camp', 'village']].values.sum() > 0:
        raise ValueError('Node %s is already occupied by a camp or village.' %
                         node_index)

    # Check if neighbours of selected position are occupied.
    neighbours = (df_edges.loc[(df_edges.source == node_index) |
                               (df_edges.target == node_index)].stack()
                  .reset_index(level=1, drop=True))
    neighbours = neighbours[neighbours != node_index]

    if df_node_contents.loc[neighbours, ['camp', 'village']].values.sum() > 0:
        raise ValueError('Immediate neighbour of Node %s is already occupied by '
                         'a camp or village.' % node_index)

    if not inplace:
        df_node_contents = df_node_contents.copy()
    df_node_contents.loc[node_index, 'camp'] = 1
    if not inplace:
        return df_node_contents


def assign_region_hex_indexes(df_hexes, inplace=False):
    if not inplace:
        df_hexes = df_hexes.copy()

    df_hexes.reset_index(inplace=True)
    df_hexes.loc[df_hexes.sort(['region', 'index']).index.values,
                 'region_hex_i'] = np.arange(df_hexes.shape[0])

    region_count_prefix_sum = df_hexes.groupby('region')['terrain'].count().cumsum()
    region_count_exclusive_prefix_sum = region_count_prefix_sum.copy()
    region_count_exclusive_prefix_sum[0] = 0
    region_count_exclusive_prefix_sum[1:] = region_count_prefix_sum[:-1]

    df_hexes['region_hex_i'] -= region_count_exclusive_prefix_sum[df_hexes.region].values
    df_hexes.drop('index', axis=1, inplace=True)

    if not inplace:
        return df_hexes


def shuffle_regions(df_hexes, inplace=False):
    if not inplace:
        df_hexes = df_hexes.copy()
    df_hexes['rand'] = np.random.rand(df_hexes.shape[0])
    df_hexes['region_hex_rand_i'] = 0

    region_hex_index = np.arange(df_hexes.shape[0])
    df_hexes.loc[df_hexes.sort(['region', 'rand']).index,
                 'region_hex_rand_i'] = region_hex_index

    region_count_prefix_sum = df_hexes.groupby('region')['terrain'].count().cumsum()
    region_count_exclusive_prefix_sum = region_count_prefix_sum.copy()
    region_count_exclusive_prefix_sum[0] = 0
    region_count_exclusive_prefix_sum[1:] = region_count_prefix_sum[:-1]
    region_count_exclusive_prefix_sum

    df_hexes['region_hex_rand_i'] -= region_count_exclusive_prefix_sum[df_hexes.region].values

    for region, df_i in df_hexes.groupby('region'):
        terrain = df_i.terrain.copy()
        indexes = df_i.region_hex_rand_i.copy()
        df_hexes.loc[indexes.index, 'terrain'] = terrain.values[indexes]

    df_hexes.drop(['region_hex_rand_i', 'rand'], axis=1, inplace=True)

    if not inplace:
        return df_hexes


def get_hex_roll_order(clockwise=True, shift=0):
    '''
    Return list of ordered hex indexes, representing order in which
    dice roll numbers should be assigned.

    Arguments
    ---------

     - `clockwise`: If `True`, assign dice numbers in clockwise order
       starting from outer ring of hexes.  Otherwise, assign in
       counter-clockwise order.
     - `shift`: Controls starting outer hex.
    '''
    inner_ring = pd.DataFrame([[32, 40], [41, 42], [35, 20],
                               [20, 12], [11, 10], [17, 32]],
                              index=[33, 34, 27, 19, 18, 25],
                              columns=['clockwise',
                                       'counter_clockwise',
                                       # 'clockwise',
                                      ])
    outer_ring = [32, 40, 41, 42, 35, 28, 20, 12, 11, 10, 17, 24]

    inner_sequence = np.roll(inner_ring.index, shift)

    if clockwise:
        inner_sequence = inner_sequence[::-1]

    if clockwise:
        outer_sequence = outer_ring[::-1]
        outer_start = inner_ring.loc[inner_sequence[-1]].clockwise
    else:
        outer_sequence = outer_ring[:]
        outer_start = inner_ring.loc[inner_sequence[-1]].counter_clockwise
    outer_sequence = np.roll(outer_sequence,
                             -outer_sequence.index(outer_start))

    return np.concatenate([[26], inner_sequence, outer_sequence])[::-1]


def assign_collect_index(df_hexes, hex_roll_order, inplace=False):
    # Land hexes in collect assignment order.
    # __N.B.,__ The desert tile is *not* assigned a roll number.
    df_land_hexes = df_hexes.loc[hex_roll_order].copy()
    df_land_hexes.loc[df_land_hexes.terrain != 'desert',
                      'collect_index'] = ORDERED_COLLECT_INDEXES
    df_land_hexes.loc[df_land_hexes.terrain == 'desert',
                      'collect_index'] = np.NaN
    if not inplace:
        df_hexes = df_hexes.copy()
    if 'collect_index' in df_hexes:
        df_hexes.drop('collect_index', axis=1, inplace=True)
    df_hexes['collect_index'] = df_land_hexes['collect_index']
    if not inplace:
        return df_hexes


def mark_port_nodes(df_nodes, df_hex_paths, df_hex_links, df_hexes,
                    inplace=False):
    '''
    Find the pair of nodes associated with each port.

    Some port hexes may have more than two vertices (i.e., nodes) contacting
    adjacent land hexes.  In such cases, according to the rules, only two nodes
    must be selected, which are on the side of the hex port facing the most
    (i.e., 4 instead of 3) land hexes.
    '''
    if not inplace:
        df_nodes = df_nodes.copy()

    # Compute center position of board
    center = df_hex_paths.loc[df_hex_paths.hex == CENTER_HEX_INDEX, ['x', 'y']].mean()

    # Compute distance from each node (i.e., hex vertex) to center of the board.
    nodes_to_center = np.sqrt(((df_nodes[['x', 'y']] - center) ** 2).sum(axis=1))
    df_hex_links = df_hex_links.copy()
    df_hex_links['region'] = df_hexes.loc[df_hex_links.hex, 'region'].values
    df_hex_links['distance_to_center'] = nodes_to_center.loc[df_hex_links['node']].values

    # Get two vertices of each port hex that are closest to the center of the island.
    df_port_nodes = (df_hex_links.loc[df_hex_links.region == 'port']
                     .sort(['hex', 'distance_to_center'])
                     .groupby('hex').head(2)[['hex', 'node']])
    df_nodes['port_hex'] = np.NaN
    df_nodes.loc[df_port_nodes.node, 'port_hex'] = df_port_nodes.hex.values

    if not inplace:
        return df_nodes


def get_hexes(hex_count, map_=None, port_offset=0):
    df_hexes = pd.DataFrame(['sea'] * hex_count, index=range(hex_count),
                            columns=['terrain'])

    if map_ is None:
        map_ = DEFAULT_MAP

    # Set region type of each hex based on `
    df_hexes.loc[map_['land'] + map_['border'],
                 'terrain'] = (TERRAIN_TYPES
                               .values.repeat(map_['terrain_counts']))

    df_hexes['region'] = 'sea'
    df_hexes.loc[map_['land'], 'region'] = 'land'
    df_hexes.loc[map_['border'], 'region'] = 'port'

    # Assign every other border position as a *sea* tile
    # (as opposed to a *port*).
    border_terrain = df_hexes.loc[(df_hexes.region == 'port')
                                  & (df_hexes.terrain != 'sea'),
                                  'terrain'].copy()
    df_hexes.loc[map_['border'][(port_offset & 0x1)::2], ['terrain', 'region']] = 'sea'
    df_hexes.loc[df_hexes.region == 'port', 'terrain'] = border_terrain.values
    return df_hexes
