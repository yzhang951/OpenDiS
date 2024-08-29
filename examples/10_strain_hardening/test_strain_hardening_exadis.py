import os, sys
import numpy as np

# Import pyexadis
pyexadis_paths = ['../../python', '../../lib', '../../core/pydis/python', '../../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]
np.set_printoptions(threshold=20, edgeitems=5)

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')

def init_from_paradis_data_file(datafile):
    G = ExaDisNet()
    G.read_paradis(datafile)
    net = DisNetManager(G)
    restart = None
    return net, restart

def example_fcc_Cu_15um_1e3():
    """example_fcc_Cu_15um_1e3:
    Example of a 15um simulation of fcc Cu loaded at a
    strain rate of 1e3/s using the subcycling integrator.
    E.g. see Bertin et al., MSMSE 27 (7), 075014 (2019)
    """
    
    pyexadis.initialize()
    
    state = {
        "crystal": 'fcc',
        "burgmag": 2.55e-10,
        "mu": 54.6e9,
        "nu": 0.324,
        "a": 6.0,
        "maxseg": 2000.0,
        "minseg": 300.0,
        "rtol": 10.0,
        "rann": 10.0,
        "nextdt": 1e-10,
        "maxdt": 1e-9,
    }
    
    output_dir = 'output_fcc_Cu_15um_1e3'
    
    restart_id = sys.argv[1] if len(sys.argv) > 1 else None

    if restart_id is None:
        # Initial configuration from file
        data_filename = '180chains_16.10e.data'
        print(f"init from {data_filename}")
        net, restart = init_from_paradis_data_file(data_filename)
    else:
        # Restart configuration
        restart_filename = f'restart.{restart_id}.exadis'
        print(f"restart from {restart_filename}")
        net, restart = read_restart(state=state, restart_file=os.path.join(output_dir, restart_filename))
    
    vis = None
    
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    
    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint, 
                              collision=collision, topology=topology, remesh=remesh, vis=vis,
                              loading_mode='strain_rate', erate=1e3, edir=np.array([0.,0.,1.]),
                              max_strain=0.01, burgmag=state["burgmag"], state=state,
                              print_freq=1, plot_freq=10, plot_pause_seconds=0.0001,
                              write_freq=100, write_dir=output_dir, restart=restart)
    sim.run(net, state)
    
    pyexadis.finalize()
    

if __name__ == "__main__":
    example_fcc_Cu_15um_1e3()
