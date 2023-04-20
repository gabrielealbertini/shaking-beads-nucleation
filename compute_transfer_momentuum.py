#!/usr/Abin/env python

from typing import List
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.pyplot as plt
import os

class Bead:
    def __init__(self,
                 radius : float,
                 position : np.ndarray,
                 velocity : np.ndarray,
                 angular_velocity: np.ndarray):
        
        """
        Position and velocity in the dish frame (not the lab frame)
        """

        self.radius = radius
        self.position = position
        self.velocity = velocity
        self.angular_velocity = angular_velocity

        
def find_neighbours(beads : list[Bead],
                    my_bead_id : int,
                    thres_dist : float) -> list:
    """
    beads [list of bead]
    my_bead_id [int]
    thres_dist [float]
    """
    my_bead = beads[my_bead_id]

    neighbours_id = []
    for bead_id, bead in enumerate(beads):
        if bead_id == my_bead_id:
            continue
        # take only projection in xy plane
        dist = np.linalg.norm(bead.position[:2] - my_bead.position[:2])
        if dist < thres_dist:
            neighbours_id.append(bead_id)
            
    return neighbours_id

def compute_transfer_angular_momentuum(beads : list[Bead],
                                       my_bead_id : int,
                                       thres_dist : float,
                                       test : bool=False) -> list:
    """
    idea:
    from rotation of beads in contact compute if this rotation provides vertical uplift
    """
    momentuum_perpendicular = []

    ref_bead = beads[my_bead_id]

    neighbours_list = find_neighbours(beads,
                                      my_bead_id,
                                      thres_dist)

    if test:
        X=[]
        Y=[]
        U=[]
        V=[]
        C=[]
        
    for bead_id in neighbours_list:
        bead = beads[bead_id]

        # compute angular velocity perpendicualr to target bead direction (project onto xy plane)
        direction = bead.position - ref_bead.position

        # xy projection
        direction = direction[0:2]

        # direction in perpendicular direction
        perpendicular = np.array([-direction[1],direction[0]])

        # normalize
        perpendicular = perpendicular/np.linalg.norm(perpendicular)

        # add z dimension
        perpendicular = list(perpendicular)+[0]
        # compute angular velocity in perpendicular component
        mom_perp = np.dot(bead.angular_velocity-ref_bead.angular_velocity,
                          perpendicular)

        momentuum_perpendicular.append(mom_perp)
        
        if test:
            X.append(bead.position[0])
            X.append(bead.position[0])
            Y.append(bead.position[1])
            Y.append(bead.position[1])
            
            U.append(perpendicular[0]*mom_perp)
            V.append(perpendicular[1]*mom_perp)
            C.append(0)

            
            U.append(bead.angular_velocity[0])
            V.append(bead.angular_velocity[1])
            C.append(1)
    if test:
        plt.quiver(X,Y,U,V,C,cmap='tab10',clim=[0,10])

    return momentuum_perpendicular

def plot_transfer_angular_momentum(beads: List[Bead], r: float, ref_bead: int=0, folder: str="", prefix: str="") -> None:
    """
        Plot in the same folder:
            * the reference bead and the influence of its neighbors (nucleation)
            * the angular momentum transfer for each bead, highlighting the largest one (momentuum)

        Args:
            beads: list of Bead of a frame
            r: float, radius of the beads
            ref_bead: int, index of the reference bead
            folder: str, folder where to save the plots
            prefix: str, prefix for the name of the files
    """
    # find beads within 10% of radius distance
    beads_within = 2.1*r
    neighbours_id = find_neighbours(beads, ref_bead, beads_within)

    plt.close()
    plt.figure()

    for bead_id, bead in enumerate(beads):
        if bead_id in neighbours_id:
            clr = 'g'
        elif bead_id==ref_bead:
            clr = 'r'
        else:
            clr='k'
        if 1:
            ax = plt.gca()
            circ = plt.Circle([bead.position[0],
                            bead.position[1]],
                            bead.radius)
            
            # Make sure the reference bead is on top
            zorder = -10
            if bead_id == ref_bead:
                zorder = -9
            
            coll= PatchCollection([circ], zorder=zorder, color=clr)

            ax.add_collection(coll)
        else:
            plt.scatter(bead.position[0],
                        bead.position[1],
                        s=100*bead.radius,color=clr)

    
    ax = plt.gca()
    ax.add_patch(plt.Circle([beads[ref_bead].position[0],
                             beads[ref_bead].position[1]],
                            beads_within, color='g',alpha=0.6))
    #ax.set_aspect('equal')
    #plt.show()
    #print('verify visually')

    #############
    # test compute tesnfer angular momentuum

    momentuum_list = compute_transfer_angular_momentuum(beads, ref_bead, beads_within,test=True)

    # loop over all particles and find the one with largest transfer of angular momentuum
    momentuum = []
    for ref_bead_loc,bead in enumerate(beads):
        momentuum_list = compute_transfer_angular_momentuum(beads, ref_bead_loc, beads_within)
        momentuum_amplitude = np.sum(momentuum_list)

        momentuum.append(momentuum_amplitude)
    idx_max = np.argmax(momentuum)
    
    plt.scatter(beads[idx_max].position[0],
                beads[idx_max].position[1],
                marker='x',
                c='r', s=15, label='bead with max momentuum')

    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.legend()
    plt.title(f"{prefix}nucleation\nGreen circle: neighbors, Red circle: nucleating bead z={beads[ref_bead].position[2]:.2f}")
    plt.legend()
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{prefix}{ref_bead}_{idx_max}_nucleation.png')
    
    print('verify visually')
   
    plt.close()
    plt.figure()   

    plt.scatter([beads[i].position[0] for i in range(len(beads))],
                [beads[i].position[1] for i in range(len(beads))],
                c=momentuum)

    plt.scatter(beads[idx_max].position[0],
                beads[idx_max].position[1],
                c='r', s=5)
    print(beads[idx_max].position[2])
    
    ax = plt.gca()
    ax.set_aspect('equal')
    
    #plt.show()
    # plt.legend()
    plt.colorbar()
    plt.title(f"{prefix}momentuum")
    plt.savefig(f'{folder}/{prefix}{ref_bead}_{idx_max}_momentuum.png')

    plt.close()

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    #################
    # test bead calss
    
    r,pos,vel,ang_vel = [0.5,np.array([0,2,3]), np.array([-0.1,0.3,0.1]), np.array([0.2,0.5,0.1])] 
    bd = Bead(r,pos,vel,ang_vel)
    if r != bd.radius or (pos != bd.position).all() or (vel != bd.velocity).all() or (ang_vel != bd.angular_velocity).all():
        print('r,pos,vel,ang_vel',r,pos,vel,ang_vel)
        print('not correclty stored into bead object')
        print(bd)
        raise RuntimeError('bead object data not save properly: check bead __init__')
    else:
        print('bead object data test passed')

    ######################
    # test find neighbours

    # create N beads with random postion within box 1x1x0.01
    import random 

    N = 100

    Lx = 1.
    Ly = 1.
    Lz = 0.01

    vmax  = 0.1
    avmax = 0.2
    
    r = 0.1
    
    beads = []
    
    for i in range(N):
        pos = np.array([random.uniform(0,Lx),
                        random.uniform(0,Ly),
                        random.uniform(0,Lz)])
        vel = np.array([random.uniform(-vmax,vmax) for i in range(3)])
        ang_vel = np.array([random.uniform(-avmax,avmax) for i in range(3)])

        beads.append(Bead(r,pos,vel,ang_vel))

    plot_transfer_angular_momentum(beads, r)

    
