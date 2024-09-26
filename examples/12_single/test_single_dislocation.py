import numpy as np
import sys, os

pydis_paths = ['../../python', '../../lib', '../../core/pydis/python']
[sys.path.append(os.path.abspath(path)) for path in pydis_paths if not path in sys.path]
np.set_printoptions(threshold=20, edgeitems=5)

from framework.disnet_manager import DisNetManager
from pydis import DisNode, DisNet, Cell, CellList
from pydis import CalForce, MobilityLaw, TimeIntegration, Topology
from pydis import Collision, Remesh, VisualizeNetwork, SimulateNetwork
from pydis.diagnostic.diagnostic import ComputeSegSumProp

def init_single_dislocation(box_length=8.0, burg_vec=np.array([1.0,0.0,0.0]), pbc=False):
    '''Generate an initial straight dislocation configuration
    '''
    print("init_single_dislocation: length = %f" % (box_length))
    cell = Cell(h=box_length*np.eye(3), is_periodic=[pbc,pbc,pbc])

    rn    = np.array([[0.0,  0.00*box_length, 0.0,         DisNode.Constraints.UNCONSTRAINED],
                      [0.0,  0.25*box_length, 0.0,         DisNode.Constraints.UNCONSTRAINED],
                      [0.0,  0.50*box_length, 0.0,         DisNode.Constraints.UNCONSTRAINED],
                      [0.0,  0.75*box_length, 0.0,         DisNode.Constraints.UNCONSTRAINED]])
    rn[:,0:3] += cell.center()

    N = rn.shape[0]
    links = np.zeros((N, 8))
    for i in range(N):
        pn = np.array([0.0, 0.0, 1.0])
        links[i,:] = np.concatenate(([i, (i+1)%N], burg_vec, pn))

    return DisNetManager(DisNet(cell=cell, rn=rn, links=links))

def main():
    global net, sim, state

    Lbox = 1000.0
    net = init_single_dislocation(box_length=Lbox, pbc=True)
    nbrlist = CellList(cell=net.cell, n_div=[8,8,8])

    vis = VisualizeNetwork()

    state = {"burgmag": 3e-10, "mu": 50e9, "nu": 0.3, "a": 1.0, "maxseg": 0.04*Lbox, "minseg": 0.01*Lbox, "rann": 3.0}

    calforce  = CalForce(force_mode='LineTension', state=state)
    mobility  = MobilityLaw(mobility_law='SimpleGlide', state=state)
    timeint   = TimeIntegration(integrator='EulerForward', dt=1.0e-8, state=state)
    topology  = Topology(split_mode='MaxDiss', state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Proximity', state=state, nbrlist=nbrlist)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    diag      = [ComputeSegSumProp]

    sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint,
                          topology=topology, collision=collision, remesh=remesh, vis=vis, diag=diag,
                          state=state, max_step=100, loading_mode="stress",
                          applied_stress=np.array([0.0, 0.0, 0.0, 0.0, -4.0e8, 0.0]),
                          print_freq=1, plot_freq=10, plot_pause_seconds=0.01,
                          write_freq=10, write_dir='output', save_state=False)
    sim.run(net, state)

    return net.is_sane()


if __name__ == "__main__":
    main()

    # explore the network after simulation
    G  = net.get_disnet()
    
    os.makedirs('output', exist_ok=True)
    net.write_json('output/single_dislocation.json')
