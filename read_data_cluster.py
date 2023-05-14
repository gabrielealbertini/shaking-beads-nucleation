"""
    read_data_cluster.py:

    Read data from cluster and save it locally,
    then plot the angular momentum transfer for each simulation
"""

import os
from pathlib import Path
import time
from matplotlib import pyplot as plt
import numpy as np
import read_binary
from compute_transfer_momentuum import Bead, compute_transfer_angular_momentuum, compute_transfer_angular_momentuum_numba, plot_transfer_angular_momentum
import imageio
import glob
import numba


# fast math does not increase performance
@numba.njit(fastmath=True, parallel=False)
def _compute_ang_momentum(data_beads, n, first_frame_nuc, cr):
    r = 0.5
    frame_window_before = 6000
    frame_window_after = 2000#3000
    beads_within = cr*r

    step = 1
    frames = [] 
    agm = np.zeros(((data_beads.shape[0] - 500)//step, n))
    zpos = np.zeros(((data_beads.shape[0] - 500)//step, n))
    for i, frame_number in enumerate(range(0, data_beads.shape[0] - 500, step)):
    # agm = np.zeros(((frame_window_after + frame_window_before)//step, n))
    # zpos = np.zeros(((frame_window_after + frame_window_before)//step, n))
    #for i, frame_number in enumerate(range(first_frame_nuc - frame_window_before, first_frame_nuc + frame_window_after, step)):
        # beads = []
        frames.append(frame_number)
        # for j in range(self.n):
        #     beads.append(Bead(r, self.data_beads[frame_number, 1, j, :],
        #                         self.data_beads[frame_number, 2, j, :],
        #                 self.data_beads[frame_number, 3, j, :]))  
        
        # compute angular momentum
        for ref_bead in range(n):
            momentuum_list = compute_transfer_angular_momentuum_numba(data_beads[frame_number, :, :, :], ref_bead, beads_within)
            # print(momentuum_list)
            agm[i, ref_bead] = np.sum(np.array(momentuum_list))
            zpos[i, ref_bead] = data_beads[frame_number, 1, ref_bead, 2] #beads[ref_bead].position[2]

    return frames, agm, zpos

class Simulation:
    """
        Class to handle simulation results of the shaking-beads experiment
    """
    def __init__(self, base_folder: str, n: int, v: int, it: int):
        """
            init function

            the data is stored in the folder base_folder/n/v/it

            the data will be read in the variable data_beads

            Args:
                base_folder: str
                    path to the folder containing the data on the cluster
                n: int
                    number of beads
                v: int
                    shaking velocity
                it: int
                    iteration number
        """
        self.base_folder = Path(base_folder)
        self.n = n
        self.v = v
        self.it = it

        self.data_beads = None

    def get_sim_path(self) -> Path:
        # Path of the data on the cluster
        return self.base_folder / f"{self.n:04}" / f"{self.v:04}" / f"{self.it:04}"  

    def get_local_data_path(self) -> str:
        # Path of the data on the local machine
        return f"data_beads_{self.n}_{self.v}_{self.it}.npz"
    
    def read_data_cluster(self) -> None:
        """
            Read data from cluster

            The data is stored in the variable data_beads
        """
        binary_data, header_size, _ = read_binary.read_bins(f'{self.get_sim_path()}/')
        time = binary_data[:, 0]*read_binary.tau
        index = np.argmin(abs(time))

        # Dish x,y positions and vx, vy velocities
        cxs = binary_data[index:, 1].reshape(-1, 1)
        cys = binary_data[index:, 2].reshape(-1, 1)
        cvxs = binary_data[index:, 5].reshape(-1, 1)
        cvys = binary_data[index:, 6].reshape(-1, 1)

        # Beads x,y,z positions
        rpxs = binary_data[index:, header_size + 0:-1:13]
        rpys = binary_data[index:, header_size + 1:-1:13]
        rpzs = binary_data[index:, header_size + 2:-1:13]
        # Converting to dish frame
        rpxs = rpxs - cxs
        rpys = rpys - cys
        pos = np.stack((rpxs, rpys, rpzs), axis=2)

        # Beads vx,vy,vz velocities
        rvxs = binary_data[index:, header_size + 7:-1:13]
        rvys = binary_data[index:, header_size + 8:-1:13]
        rvzs = binary_data[index:, header_size + 9:-1:13]
        # Converting to dish frame
        rvxs = rvxs - cvxs
        rvys = rvys - cvys
        vel = np.stack((rvxs, rvys, rvzs), axis=2)

        # Beads angular velocities
        roxs = binary_data[index:, header_size + 10:-1:13]
        roys = binary_data[index:, header_size + 11:-1:13]
        rozs = binary_data[index:, header_size + 12:-1:13]
        ang_vel = np.stack((roxs, roys, rozs), axis=2)

        # Frame numbers with the right dimensions
        frames = np.repeat(binary_data[index:, -1][:, np.newaxis], (rpxs.shape[1]), axis = 1)
        frames = np.repeat(frames[:, :, np.newaxis], (3), axis = 2)
        print(frames.shape)
        print(pos.shape)

        # Merge all these data in one numpy array of shape (# of frames, 4 [frame, pos, vel, ang_vel], # of beads, 3 [x,y,z])
        data_beads = np.stack((frames, pos, vel, ang_vel), axis=1)
        print(data_beads.shape)

        self.data_beads = data_beads

    def save_data_locally(self) -> None:
        """
            Save data locally

            The data can be easily shared in one file instead of several files.
            It also improves the performance for loading it then.
        """
        path = self.get_local_data_path()
        np.savez(path, data_beads=self.data_beads)
        print(f"Saved to {path}")

    def load_local_data(self) -> None:
        """
            Load data from local file

            The data is stored in the variable data_beads

            The file should be created with save_data_locally and in the same folder as the script
        """
        path = self.get_local_data_path()
        self.data_beads = np.load(path)['data_beads']
        print(f"Loaded from {path}")

    def plot_transfer_momentuum(self) -> None:
        """
            Plot in the same folder:
                * the nucleating bead and the influence of its neighbors (nucleation)
                * the angular momentum transfer for each bead, highlighting the largest one (momentuum)
        """
        # find frame where there is nucleation
        # threshold is quite low to make sure to get the moment when the bead is not fully up
        idx = np.where(self.data_beads[:, 1, :, 2] > (0.5 + np.sin(np.pi/3)))#/2))
        print(idx)
        first_frame_nuc = idx[0][0]
        first_bead_nuc = idx[1][0]

        frame_window = 10
        folder = f"{self.n}_{self.v}_{self.it}"
        for frame_number in range(first_frame_nuc - frame_window, first_frame_nuc + frame_window):
        # convert data of this first frame of nucleation to bead objects
            beads = []
            r = 0.5 # distances normalized by bead diameter in the simulations
            for i in range(self.n):
                beads.append(Bead(r, self.data_beads[frame_number, 1, i, :],
                                    self.data_beads[frame_number, 2, i, :],
                                    self.data_beads[frame_number, 3, i, :]))
                
            prefix = f"{frame_number}_"
            plot_transfer_angular_momentum(beads, r, first_bead_nuc, folder, prefix)

        frame_array = glob.glob(f"{folder}/*nucleation.png")
        frame_array.sort()
        print(frame_array)
        self.make_gif(folder, frame_array)

    def make_gif(self, folder, frame_array):
        frames = []
        for frame in frame_array:
            image = imageio.v2.imread(frame)
            frames.append(image)
        imageio.mimsave(f'{folder}/{folder}.gif', # output gif
                frames,          # array of input frames
                fps = 1) 
        

   


    # @numba.jit(nopython=False)
    def plot_angular_momentum_over_time(self):
        """
            Plot angular momentum over time for each bead
        """
        cr = 2.
        idx = np.where(self.data_beads[:, 1, :, 2] > (0.5 + np.sin(np.pi/3)))#/2))
        print(idx)
        first_frame_nuc = idx[0][0]
        first_bead_nuc = idx[1][0]

        print(len(self.data_beads))
        # exit()
        
        file = f"{self.n}_{self.v}_{self.it}_agm.npz"
        if os.path.isfile(file):
            data = np.load(file)
            frames = data['frames']
            agm = data['agm']
            zpos = data['zpos']
            print("Loaded agm data")
        else:
            time_start = time.perf_counter()
            frames, agm, zpos = _compute_ang_momentum(self.data_beads, self.n, first_frame_nuc, cr)
            time_end = time.perf_counter()
            np.savez(file, frames=frames, agm=agm, zpos=zpos)
            print("Saved agm data")
            print(f"Time elapsed: {time_end - time_start:0.4f} seconds")

        # idx_2 = np.argsort(zpos[-1, :])[::-1][:10]
        idx_2 = []
        for idx_b in idx[1]:
            if idx_b != first_bead_nuc and idx_b not in idx_2:
                idx_2.append(idx_b) 
            if len(idx_2) == 10:
                break
        print(idx_2)

        plt.figure()
        fig, axs = plt.subplots(2, 1, figsize=(45, 15))
        for j in range(self.n):
            args = {"color": "gray", "zorder": 0}
            if j == first_bead_nuc:
                args = {"label": "First nucleating bead", "zorder": 10}
            elif j in idx_2:
                args = {"label": "Other bead of interest", "zorder": 10}
            axs[0].plot(frames, agm[:, j], **args)
            axs[1].plot(frames, zpos[:, j], **args)

        axs[0].set_xlabel("Frame number")
        axs[0].set_ylabel("Projected angular momentum (rad/tau)")
        axs[1].set_xlabel("Frame number")
        axs[1].set_ylabel("Z position (particle diameter)")
        for ax in axs:
            ax.axvline(first_frame_nuc, color="black", linestyle="--", label='nucleation')
            ax.grid()
            ax.legend()
        
        plt.suptitle(f"{self.n}_{self.v}_{self.it} 1000 fps simulation, contact radius = {cr}")

        plt.tight_layout()

        plt.savefig(f"{self.n}_{self.v}_{self.it}_angular_momentum_evolution.png")
        plt.close()

        # plot z position of nucleating bead too
    

def job(base_folder, n, v, it):
    sim = Simulation(base_folder, 107, v, it)

    # to read data from cluster and save it locally
    # sim.read_data_cluster()
    # sim.save_data_locally()

    # study momentum transfer
    sim.load_local_data()
    # sim.plot_transfer_momentuum()
    sim.plot_angular_momentum_over_time()

if __name__ == "__main__":
    base_folder = "/n/holyscratch01/shared/adjellouli/simulations_reorganized/mechanisms/nucleation_mechanism_1000fps/d_amp=1.600"

    import multiprocessing as mp
    with mp.get_context('spawn').Pool() as p:
        res = []
        for it in range(3):
            for v in [205]:#[205, 240]:
                res.append(p.apply_async(job, args=(base_folder, 107, v, it)))

        [r.get() for r in res]
           

    # to start interactive session on the cluster
    # salloc -p test -t 0-08:00 --mem 32000 -c 4
