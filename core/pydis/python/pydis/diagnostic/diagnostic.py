"""@package docstring

"""

import numpy as np
from ..disnet import DisNet
from framework.disnet_manager import DisNetManager

def SegLength(G: DisNet, state: dict):
    """
    Set length for each dislocation segment
    """
    for tag in G.all_segments_dict():
        tag1, tag2 = tag
        seg = G.all_segments_dict()[(tag1,tag2)]
        tag1 = seg.source_tag
        tag2 = seg.target_tag

        node1 = G.nodes(tag1)
        R1 = node1.R.copy()
        node2 = G.nodes(tag2)
        R2 = node2.R.copy()
        R2 = G.cell.closest_image(Rref=R1, R=R2) 
        seg.line_vec = R2-R1
        seg.length = np.linalg.norm(seg.line_vec)

        vel = state['vel_dict']
        v1 = 0.0
        v2 = 0.0
        if tag1 in vel.keys():
            v1 = vel[tag1]
        if tag2 in vel.keys():
            v2 = vel[tag2]
        seg.vel = (v1+v2) / 2.0
    return 0

def ComputeSegSumProp(G: DisNet, state: dict, props=['DislDen', 'Lp_dot']):
    """
    Compute summation of segments properties
    List of supported properties:
    DislDen:       Dislocation density scalar
    DislDenTensor: Dislocation density tensor
    Lp_dot:        Velocity gradient tensor
    """
    SegLength(G, state)

    prop_list = {'DislDen': np.zeros(1), 'DislDenTensor': np.zeros([3,3]), 'Lp_dot': np.zeros([3,3])}
    
    volume = np.linalg.det(G.cell.h)

    for seg in G.all_segments_dict().values():
        if 'DislDen' in props:
            prop_list['DislDen'][0] += seg.length / volume
        if 'DislDenTensor' in props:
            prop_list['DislDenTensor'] += np.outer(seg.burg_vec, seg.line_vec) / volume
        if 'Lp_dot' in props:
            prop_list['Lp_dot'] += np.dot(np.cross(seg.vel, seg.line_vec), seg.plane_normal)*np.outer(seg.burg_vec, seg.plane_normal)/volume #/ state['burgmag']

    prop_list['DislDenTensor'] = prop_list['DislDenTensor'].reshape([9])
    prop_list['Lp_dot'] = prop_list['Lp_dot'].reshape([9])

    output = np.array([])
    for prop in props:
        output = np.concatenate((output, prop_list[prop]))
    return output
