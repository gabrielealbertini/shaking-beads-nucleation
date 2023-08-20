"""
    read_data_cluster.py:

    Read data from cluster and save it locally,
    then plot the angular momentum transfer for each simulation
"""

import os
from pathlib import Path
import pickle
import time
from matplotlib import pyplot as plt
import numpy as np
import read_binary
from compute_transfer_momentuum import Bead, compute_transfer_angular_momentuum, compute_transfer_angular_momentuum_numba, plot_transfer_angular_momentum
import imageio
import glob
import numba
from scipy.signal import savgol_filter

# fast math does not increase performance
@numba.njit(fastmath=True, parallel=False)
def _compute_ang_momentum(data_beads, n, first_frame_nuc, cr):
    r = 0.5
    frame_window_before = 100#6000
    frame_window_after =100 #2000#3000
    beads_within = cr*r

    step = 1
    frames = [] 
    agm = np.zeros(((data_beads.shape[0] - 500)//step, n))
    ag_screw = np.zeros(((data_beads.shape[0] - 500)//step, n))
    zpos = np.zeros(((data_beads.shape[0] - 500)//step, n))
    for i, frame_number in enumerate(range(0, data_beads.shape[0] - 500, step)):
    # first_frame_nuc = 34230
    # agm = np.zeros(((frame_window_after + frame_window_before)//step, n))
    # zpos = np.zeros(((frame_window_after + frame_window_before)//step, n))
    # for i, frame_number in enumerate(range(first_frame_nuc - frame_window_before, first_frame_nuc + frame_window_after, step)):
        # beads = []
        frames.append(frame_number)
        # for j in range(self.n):
        #     beads.append(Bead(r, self.data_beads[frame_number, 1, j, :],
        #                         self.data_beads[frame_number, 2, j, :],
        #                 self.data_beads[frame_number, 3, j, :]))  
        
        # compute angular momentum
        for ref_bead in range(n):
            momentuum_list, momentuum_spin_z = compute_transfer_angular_momentuum_numba(data_beads[frame_number, :, :, :], ref_bead, beads_within)
            # print(momentuum_list)
            agm[i, ref_bead] = np.sum(momentuum_list) # - np.dot(ag_bead, perpendicular)
            ag_screw[i, ref_bead] = np.sum(momentuum_spin_z)
            zpos[i, ref_bead] = data_beads[frame_number, 1, ref_bead, 2] #beads[ref_bead].position[2]

    return frames, agm, ag_screw, zpos

def compute_ang_momentum(data_beads, n, first_frame_nuc, cr):
    r = 0.5
    frame_window_before = 50
    frame_window_after = 50#3000
    beads_within = cr*r

    step = 1
    frames = [] 
    agm = np.zeros(((data_beads.shape[0] - 500)//step, n))
    zpos = np.zeros(((data_beads.shape[0] - 500)//step, n))
    for i, frame_number in enumerate(range(0, data_beads.shape[0] - 500, step)):
    # agm = np.zeros(((frame_window_after + frame_window_before)//step, n))
    # zpos = np.zeros(((frame_window_after + frame_window_before)//step, n))
    # for i, frame_number in enumerate(range(first_frame_nuc - frame_window_before, first_frame_nuc + frame_window_after, step)):
        beads = []
        frames.append(frame_number)
        for j in range(n):
            beads.append(Bead(r, data_beads[frame_number, 1, j, :],
                                data_beads[frame_number, 2, j, :],
                        data_beads[frame_number, 3, j, :]))  
        
        # compute angular momentum
        for ref_bead in range(n):
            momentuum_list = compute_transfer_angular_momentuum(beads, ref_bead, beads_within)
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
        return f"/n/holyscratch01/shared/fpollet/mechanism/ALL/data_beads_{self.n}_{self.v}_{self.it}.npz"
    
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
        if not os.path.isfile(path):
            self.read_data_cluster()
            self.save_data_locally()
        else:
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
        cr = 2.05
        rolling_avg_nb = 50
        idx = np.where(self.data_beads[:, 1, :, 2] > (0.5 + np.sin(np.pi/3)))#/2))
        print(idx)
        first_frame_nuc = idx[0][0]
        first_bead_nuc = idx[1][0]

        print(len(self.data_beads))
        # exit()
        
        file = f"/n/holyscratch01/shared/fpollet/mechanism/PAM/{self.n}_{self.v}_{self.it}_agm-{cr}.npz"
        if os.path.isfile(file):
            data = np.load(file)
            frames = data['frames']
            agm = data['agm']
            zpos = data['zpos']
            neighboring_spin = data['am_screw']
            print("Loaded agm data")
        else:
            time_start = time.perf_counter()
            frames, agm, neighboring_spin, zpos = _compute_ang_momentum(self.data_beads, self.n, first_frame_nuc, cr)
            time_end = time.perf_counter()
            np.savez(file, frames=frames, agm=agm, am_screw=neighboring_spin, zpos=zpos)
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

        # frames, agm, zpos = compute_ang_momentum(self.data_beads, self.n, first_frame_nuc, cr)

        plt.figure()
        fig, axs = plt.subplots(4, 1, figsize=(45, 45), sharex=True)
        slice_fr = slice(0, len(frames))
        #slice_fr = slice(25000, 26500)

        # print(slice_fr)
        # print(agm.shape)
        # print(agm[slice_fr, :])

        # compute rolling average for each bead with agm
        # https://stackoverflow.com/questions/27427618/how-can-i-simply-calculate-the-rolling-moving-variance-of-a-time-series-in-pytho
        def moving_average(a, rolling_avg_nb) :
            ret = np.cumsum(a, axis=0, dtype=float)
            ret[rolling_avg_nb:, :] = ret[rolling_avg_nb:, :] - ret[:-rolling_avg_nb, :]
            return ret[rolling_avg_nb - 1:, :] / rolling_avg_nb
        rolling_avg = moving_average(agm[slice_fr], rolling_avg_nb=rolling_avg_nb)
        savgol_signal = savgol_filter(agm[slice_fr], 21, 3, axis=0, mode='interp') 
        Aw = np.lib.stride_tricks.sliding_window_view(agm[slice_fr], rolling_avg_nb, axis=0)
        rolling_std = np.std(Aw, axis=-1)
        # testa = np.array([[1, 2],[1, 2], [1,2], [1,3]])
        # Aw = np.lib.stride_tricks.sliding_window_view(testa, 2, axis=0)
        # print(np.std(Aw, axis=-1))
        # exit()

        ke_trans = np.sum(self.data_beads[:, 2, :, :]**2, axis=2)
        ke_rot = np.sum(self.data_beads[:, 3, :, :]**2, axis=2)
        d_center = np.sqrt(np.sum(self.data_beads[:, 1, :, :2]**2, axis=2))  # 2d
        spin_z = self.data_beads[:, 3, :, 2]


        # slice_fr = slice(0, 100)
        preferred_bead = 59#40#103#first_bead_nuc#57#104
        for j in range(self.n):
            args = {"color": "gray", "zorder": 0}
            if j == first_bead_nuc:
                args = {"label": "First nucleating bead", "zorder": 10}
            elif j in idx_2:
                args = {"label": "Other bead of interest", "zorder": 10}

            # if j == 69: # 89 interesting for it 1
            if j == preferred_bead:#86:#26:#104:  #first_bead_nuc:
                axs[0].plot(frames[slice_fr], agm[slice_fr, j], **args)
                axs[1].plot(frames[slice_fr], zpos[slice_fr, j], **args)
                # axs[2].plot(frames[slice_fr][rolling_avg_nb-1:], rolling_avg[:, j], **args)
                # axs[3].plot(frames[slice_fr][rolling_avg_nb-1:], rolling_std[:, j], **args)
                # axs[2].plot(frames[slice_fr][rolling_avg_nb//2-1:-rolling_avg_nb//2], rolling_avg[:, j], **args)
                # axs[3].plot(frames[slice_fr][rolling_avg_nb//2-1:-rolling_avg_nb//2], rolling_std[:, j], **args)
                axs[2].plot(frames[slice_fr], savgol_signal[:, j], **args)
                axs[3].plot(frames[slice_fr], spin_z[frames[slice_fr], j], **args)
                # axs[4].plot(frames[slice_fr], ke_trans[frames[slice_fr], j], **args)
                # axs[5].plot(frames[slice_fr], ke_rot[frames[slice_fr], j], **args)
                # axs[6].plot(frames[slice_fr], d_center[frames[slice_fr], j], **args)
                # axs[7].plot(frames[slice_fr], screweffect[frames[slice_fr], j], **args)
                # axs[7].plot(frames[slice_fr], spin_z[frames[slice_fr], j], **args)
                # axs[8].plot(frames[slice_fr], neighboring_spin[slice_fr, j], **args)
                # print(agm[slice_fr, j])

        axs[0].set_xlabel("Frame number")
        axs[0].set_ylabel("Projected angular momentum (rad/tau)")
        axs[1].set_xlabel("Frame number")
        axs[1].set_ylabel("Z position (particle diameter)")
        # axs[1].set_ylim(0.4, 2.)
        axs[0].set_ylim(-1.5, 1.5)

        axs[2].axhline(0, color="black", linestyle="--")
        axs[3].axhline(0, color="black", linestyle="--")

        # for ax in axs:
        #     ax.axvline(first_frame_nuc, color="black", linestyle="--", label='nucleation')
        #     ax.grid()
        #     ax.legend()
        
        plt.suptitle(f"{self.n}_{self.v}_{self.it}_{preferred_bead} 1000 fps simulation, contact radius = {cr}")

        plt.tight_layout()

        # pikcle figure

        plt.savefig(f"/n/holyscratch01/shared/fpollet/mechanism/Output/{self.n}_{self.v}_{self.it}_{preferred_bead}_angular_momentum_evolution_full.png")

        
        with open(f"/n/holyscratch01/shared/fpollet/mechanism/Output/{self.n}_{self.v}_{self.it}_{preferred_bead}_angular_momentum_evolution.pkl", "wb") as f:
            pickle.dump(plt.gcf(), f)
            
        plt.close()

        # plot z position of nucleating bead too

    def plot_time_amplitude(self):
        cr = 2.05
        threshold = (0.5 + np.sin(np.pi/3))
        low_threshold = 0.6
        print(threshold)
        idx = np.where(self.data_beads[:, 1, :, 2] > threshold)#/2))
        # print(idx)
        frames_nuc = idx[0]
        beads_nuc = idx[1]

        file = f"/n/holyscratch01/shared/fpollet/mechanism/PAM/{self.n}_{self.v}_{self.it}_agm-{cr}.npz"
        if os.path.isfile(file):
            data = np.load(file)
            frames = data['frames']
            agm = data['agm']
            zpos = data['zpos']
            spin_z = data['am_screw']
            # spin_z = self.data_beads[:, 3, :, 2]
            print("Loaded agm data")
        rolling_avg_nb = 20#20#50
        def moving_average(a, rolling_avg_nb=10) :
            ret = np.cumsum(a, axis=0, dtype=float)
            ret[rolling_avg_nb:, :] = ret[rolling_avg_nb:, :] - ret[:-rolling_avg_nb, :]
            return ret[rolling_avg_nb - 1:, :] / rolling_avg_nb
        slice_fr = slice(0, len(frames))
        #rolling_avg = moving_average(agm[slice_fr], rolling_avg_nb=rolling_avg_nb)
        savgol_agm = savgol_filter(agm[slice_fr], rolling_avg_nb + 1, 3, axis=0, mode='interp') 

        times = []
        amps = []
        spins = []
        z_amps = []
        i_idx = []
        i_frames_idx = []

        times_nuc = []
        amps_nuc = []
        spins_nuc = []

        print(frames_nuc[0])
        print(beads_nuc[0])

        # reduce get same results with one additional dim
        # print()
        margin = 1000
        # 1000 fps
        frame_acc = self.v/30*1000
        for i in range(self.n):
            # if i != 3:
            #     continue
            # print(i)
            arr = savgol_agm[slice_fr, i]#[:frames_nuc[0] + margin]
            zpos_short = zpos[slice_fr, i]#[rolling_avg_nb//2-1:-rolling_avg_nb//2]#[:frames_nuc[0] + margin]
            spin_z_short = spin_z[slice_fr, i]#[rolling_avg_nb//2-1:-rolling_avg_nb//2]#[:frames_nuc[0] + margin]
            assert zpos_short.shape == arr.shape
            # compute time between zeros of rolling_avg
            # zero_indices = np.where(np.isclose(arr, 0, atol=0.01))[0]
            zero_indices = np.where((arr[1:] > 0) != (arr[:-1] > 0))[0]

            # Calculate index differences
            index_diff = np.diff(zero_indices)

            # if i == 102:
            #     print(zero_indices[(zero_indices > 26253) & (zero_indices < 26382)])

            # convert index to time at some point

            # extract max between every two zero
            # Shift zero_indices by 1 to get the starting indices for each interval
            starting_indices = zero_indices[:-1] + 1

            # Calculate the maximum value within each interval
            max_values = np.maximum.reduceat(arr, starting_indices) 
            if i == 59:
                print(arr[17320:17335])
            max_values_z = np.maximum.reduceat(zpos_short, starting_indices)
            max_values_spin = np.maximum.reduceat(spin_z_short, starting_indices)


            # if i in beads_nuc:
            #     times_nuc.extend(index_diff)
            #     amps_nuc.extend(max_values)
            # else:
            #     times.extend(index_diff)
            #     amps.extend(max_values)

            # add when it is an event leading to nucleation
            # or  value starting ind below threshold
            # and value end ind above threshold

            zpos_startings = zpos_short[zero_indices] > low_threshold#threshold

            # z_chg_all = (zpos_startings[:-1] == False) & (zpos_startings[1:] == True)
            # z_chg_all = zpos_startings[:-1] != zpos_startings[1:]
            # z_chg_all = zero_indices[:-1] > frames_nuc[0] # keep only before nucleation
            # print(np.where(z_chg_all))

            # remove flying beads, denucleation...
            # == false apres nucleation

            # if i == beads_nuc[0]:
            #     event_nucleation_mask = (zero_indices[:-1] <= frames_nuc[0]) & (zero_indices[1:] > frames_nuc[0])
            # else:
            event_nucleation_mask = np.zeros(index_diff.shape, dtype=bool)

            events_selected_mask = ((zpos_startings[:-1] == False) & (zero_indices[:-1] < (frames_nuc[0] - margin))  & (zero_indices[:-1] > frame_acc)) | event_nucleation_mask # keep only the ones below threshold at the beginning of the interval

            if i == beads_nuc[0]:
                print(zero_indices.shape)
                frame_low_before_nuc = zero_indices[np.nonzero((zpos_short[zero_indices] < low_threshold)[:-1] & (zero_indices[:-1] <= frames_nuc[0]))[0][-1]]
                # print(frame_low_before_nuc[0], frame_low_before_nuc[0])
                frame_nuc_0 = frames_nuc[0]
                frame_deb_nuc = frame_low_before_nuc - (frame_nuc_0 - frame_low_before_nuc)

                print(frame_deb_nuc, frame_low_before_nuc, frame_nuc_0)

                event_nucleation_mask = (zero_indices[:-1] <= frame_nuc_0) & (zero_indices[:-1] > frame_deb_nuc)

                idx_max_nuc = np.argmax(max_values[event_nucleation_mask])
                time_max_nuc = index_diff[event_nucleation_mask][idx_max_nuc]
                amp_max_nuc = max_values[event_nucleation_mask][idx_max_nuc]
                spin_max_nuc = max_values_spin[event_nucleation_mask][idx_max_nuc]


                times_nuc.append(time_max_nuc)
                amps_nuc.append(amp_max_nuc)
                spins_nuc.append(spin_max_nuc)

                print(zero_indices[:-1][event_nucleation_mask][idx_max_nuc])
                print(time_max_nuc, amp_max_nuc, spin_max_nuc)
            
            # z_chg_all = zero_indices > z_chg

            # denuc include as well: issue

            # see why no nucleation event found\
            # see to remove other nucleation event
            # why frame number so low? just indice of changes in rolling avg so much less than frames
            # why simulation so long???

            # times_nuc.extend(index_diff[event_nucleation_mask])
            # amps_nuc.extend(max_values[event_nucleation_mask])
            # spins_nuc.extend(max_values_spin[event_nucleation_mask])

            # if (max_values[events_selected_mask] > 1.7).any() and (index_diff[events_selected_mask] > 50).any():
            if (max_values[events_selected_mask] > 1.65).any() and (max_values[events_selected_mask] < 1.7).any():
                mask_select = np.where((max_values[events_selected_mask] > 1.65) & (max_values[events_selected_mask] < 1.7))[0]
                print(mask_select)
                # np.argmax(max_values_z[events_selected_mask]))
                print("Anomaly", self.it, i, zero_indices[:-1][events_selected_mask][mask_select])

            times.extend(index_diff[events_selected_mask])
            amps.extend(max_values[events_selected_mask])
            z_amps.extend(max_values_z[events_selected_mask])
            spins.extend(max_values_spin[events_selected_mask])
            # spins.extend(spin_z_short[zero_indices][z_chg_all])
            i_idx.extend([i]*len(max_values[events_selected_mask]))
            i_frames_idx.extend(zero_indices[:-1][events_selected_mask])


            # zpos to cut to match rolling avg

        times = np.array(times)
        amps = np.array(amps)
        z_amps = np.array(z_amps)
        order = np.argsort(z_amps)
        print(order.shape)

        return self.it, times, amps, z_amps, i_idx, i_frames_idx, times_nuc, amps_nuc, spins, spins_nuc

        plt.scatter(times[order], amps[order], label='Other beads', zorder=0, c=z_amps[order], cmap='viridis', s=10) # plot( 'o', color='blue',  alpha=0.05,
        # annotate with bead number
        plt.plot(times_nuc, amps_nuc, '+', color='red', label=f'Nuc bead', zorder=10) # {self.it}')
        if len(times_nuc) > 0:
            plt.annotate(f"{self.it}", (times_nuc[0], amps_nuc[0]), color='black', fontsize=10)
        else:
            print("No nucleation event found", self.it)
        print(len(times))
        for time, amp, z, i_id in zip(times, amps, z_amps, i_idx):
            if z > 1:
                plt.annotate(f"{self.it} {i_id}", (time, amp), color='black', fontsize=10)

        # plt.savefig('output/amp_time.png')
        # plt.close()

    def export_stats(self):
        threshold = (0.5 + np.sin(np.pi/3))
        idx = np.where(self.data_beads[:, 1, :, 2] > threshold)#/2))
        # print(idx)
        frames_nuc = idx[0]
        beads_nuc = idx[1]

        return (self.it, frames_nuc[0], beads_nuc[0])
    

def job(base_folder, n, v, it):
    sim = Simulation(base_folder, n, v, it)

    # to read data from cluster and save it locally
    # print("Reading data from cluster")
    # sim.read_data_cluster()
    # print("Saving data locally")
    # sim.save_data_locally()

    # study momentum transfer
    print("Reading local data")
    sim.load_local_data()
    # sim.plot_transfer_momentuum()
    print("Computing angular momentum")
    sim.plot_angular_momentum_over_time()

def job_plot_time_amplitude(base_folder, n, v, it):
    # plt.figure(figsize=(10,10))
    # for it in range(1,2):


    sim = Simulation(base_folder, n, v, it)

    # to read data from cluster and save it locally
    # print("Reading data from cluster")
    # sim.read_data_cluster()
    # print("Saving data locally")
    # sim.save_data_locally()

    # study momentum transfer
    print("Reading local data")
    sim.load_local_data()
    # sim.plot_transfer_momentuum()
    print("Computing angular momentum")
    return sim.plot_time_amplitude() # to do over all sim then



    # plt.ylabel('Max amplitude of rolling average of PAM between two zeros (rad/tau)')
    # plt.xlabel('# of frames between two zeros of rolling average of PAM')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    # plt.colorbar()
    # plt.xlim([0, 1500])
    # plt.ylim([-0.05, 2.00])
    # plt.title(f'{n} beads, {v} rpm, 100 its, 1000 fps')

    # import pickle
    # with open('fig.pickle', 'wb') as f: # should be 'wb' rather than 'w'
    #     pickle.dump(plt.gcf(), f) 

    # plt.savefig('output/amp_time.png')
    # # only the first nuc bead
    # plt.close()

def assemble_plot(res, v):
    plt.figure(figsize=(10,10))

    its = []
    times = []
    amps = []
    z_amps = []
    i_idx = []
    i_frames_idx = []
    times_nuc = []
    amps_nuc = []
    spins = []
    spins_nuc = []

    pam_lim = 10#0.05

    for it, times1, amps1, z_amps1, i_idx1, i_frames_idx1, times_nuc1, amps_nuc1, spins1, spins_nuc1 in res:
        print("Assemblying plot for", it)

        its.append(it)
        times.extend(times1)
        amps.extend(amps1)
        z_amps.extend(z_amps1)
        i_idx.extend(i_idx1)
        i_frames_idx.extend(i_frames_idx1)
        times_nuc.extend(times_nuc1)
        amps_nuc.extend(amps_nuc1)
        spins.extend(spins1)
        spins_nuc.extend(spins_nuc1)

    times = np.array(times)
    amps = np.array(amps)
    z_amps = np.array(z_amps)
    i_idx = np.array(i_idx)
    i_frames_idx = np.array(i_frames_idx)
    spins = np.array(spins)
    times_nuc = np.array(times_nuc)
    amps_nuc = np.array(amps_nuc)
    spins_nuc = np.array(spins_nuc)

    filter = (amps < pam_lim) & (amps > 0.01)
    times = times[filter]
    amps = amps[filter]
    z_amps = z_amps[filter]
    i_idx = i_idx[filter]
    i_frames_idx = i_frames_idx[filter]
    spins = spins[filter]

    filter_nuc = amps_nuc < pam_lim
    times_nuc = times_nuc[filter_nuc]
    amps_nuc = amps_nuc[filter_nuc]
    spins_nuc = spins_nuc[filter_nuc]


    order = np.argsort(z_amps)
    print(len(times_nuc), len(amps_nuc))

    colors = np.array(["green"]*len(z_amps))
    limit = 0.93
    colors[z_amps[order] > limit] = "blue"
    plt.scatter(times[order][z_amps[order] <= limit] , amps[order][z_amps[order] <= limit], label='Other events', zorder=0, c="green") #c=z_amps[order], cmap='viridis', s=10, vmin=0.5, vmax=1.5) # plot( 'o', color='blue',  alpha=0.05,    
    plt.scatter(times[order][z_amps[order] > limit], amps[order][z_amps[order] > limit], label='Attempted nucleation events', zorder=0, c="blue") #c=z_amps[order], cmap='viridis', s=10, vmin=0.5, vmax=1.5) # plot( 'o', color='blue',  alpha=0.05,    


    data_saved = {
        "Other events": (times[order][z_amps[order] <= limit].tolist(), amps[order][z_amps[order] <= limit].tolist()),
        "Attempted nucleation events": (times[order][z_amps[order] > limit].tolist(), amps[order][z_amps[order] > limit].tolist()),
        "Nucleation events": (times_nuc.tolist(), amps_nuc.tolist())
    }
    import json
    with open('data.json', 'w') as fp:
        json.dump(data_saved, fp)

    # plt.scatter(times[order], amps[order], label='Other beads', zorder=0, c=list(colors)) #c=z_amps[order], cmap='viridis', s=10, vmin=0.5, vmax=1.5) # plot( 'o', color='blue',  alpha=0.05,    
    
    # uncomment to show nuc events
    # plt.plot(times_nuc, amps_nuc, '+', color='red', label=f'Nucleation events', zorder=20) # {self.it}')
    plt.scatter(times_nuc, amps_nuc, zorder=0, c="blue")
   
    # plt.scatter(times[order], spins[order], label='Other events', zorder=0, c=z_amps[order], cmap='viridis', s=10) # plot( 'o', color='blue',  alpha=0.05,    
    # plt.plot(times_nuc, spins_nuc, '+', color='red', label=f'Nuc event', zorder=20, markersize = 5) # {self.it}')
   
    # annotate with bead number
    # plt.plot(times_nuc, amps_nuc, '+', color='red', label=f'Nuc bead', zorder=10) # {self.it}')
    
    # for it, times1, amps1, z_amps1, i_idx1, i_frames_idx1, times_nuc1, amps_nuc1, spins1, spins_nuc1 in res:
    #     if len(times_nuc) > 0:
    #         plt.annotate(f"{it}", (times_nuc1[0], amps_nuc1[0]), color='black', fontsize=10)

    # for time, amp, z, i_id, i_frame in zip(times, amps, z_amps, i_idx, i_frames_idx):
    #     if z > 1 or time > 225 or amp < -0.1404:
    #         plt.annotate(f"{i_id}-{i_frame}", (time, amp), color='black', fontsize=10)
    
    # else:
    #     print("No nucleation event found", self.it)
    print(len(times))
    # for time, amp, z, i_id in zip(times, amps, z_amps, i_idx):
    #     if z > 1:
    #         plt.annotate(f"{it} {i_id}", (time, amp), color='black', fontsize=10, zorder=10)


    plt.xlim([0, 250])
    plt.ylim([0, 5.00])
    plt.xlabel('# of frames between two zeros of rolling average of PAM')

    # plt.ylabel('Max amplitude of angular momentum around z axis (rad/tau)')
    cb = plt.colorbar()
    plt.ylabel('Max amplitude of savgol filtering of PAM between two zeros (rad/tau)')    
    # plt.ylabel('Max amplitude of angular momentum differences with neighbors around z axis (rad/tau)')
    # cb.ax.set_ylabel("Max PAM")#Max z pos (bead diameter)")
    cb.ax.set_ylabel("Max z pos (bead diameter)")


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title(f'107 beads, {v} rpm, 100 its, 1000 fps')

    print("Pickling figure")
    import pickle
    with open('fig.pickle', 'wb') as f: # should be 'wb' rather than 'w'
        pickle.dump(plt.gcf(), f) 

    print("Saving figure")
    plt.savefig(f'output/amp_time_{v}.png')
    # only the first nuc bead
    plt.close()

def test_zeros():
    rolling_avg = np.array([1,0,2,4,6,0,-2,-5,0,2,1,0,5,7,2,0,1])
    threshold = 1
    zpos_short = np.array( [0,0,0,0,2,2, 2, 0,0,0,0,0,0,1,0,0,0])

    zero_indices = np.where(rolling_avg == 0)[0]
    print(zero_indices)
    # Calculate index differences
    index_diff = np.diff(zero_indices)

    # convert index to time at some point

    # extract max between every two zero
    # Shift zero_indices by 1 to get the starting indinoces for each interval
    starting_indices = zero_indices[:-1] + 1

    # Calculate the maximum value within each interval
    max_values = np.maximum.reduceat(rolling_avg, starting_indices)

    print(index_diff)
    print(max_values)

    zpos_startings = zpos_short[zero_indices] > threshold

    z_chg = (zpos_startings[:-1] == False) & (zpos_startings[1:] == True)
    print(z_chg)

def job_stats(base_folder, n, v, it):
    # plt.figure(figsize=(10,10))
    # for it in range(1,2):


    sim = Simulation(base_folder, n, v, it)

    # to read data from cluster and save it locally
    # print("Reading data from cluster")
    # sim.read_data_cluster()
    # print("Saving data locally")
    # sim.save_data_locally()

    # study momentum transfer
    print("Reading local data")
    sim.load_local_data()
    print("Exporting stats")
    return sim.export_stats()

def assemble_stats(res, v):
    print("Assembling stats")
    import pandas as pd

    pd.DataFrame(res, columns=['it', 'frame_nuc', 'bead_nuc']).to_csv(f'output/stats_{v}.csv')

if __name__ == "__main__":
    base_folder = "/n/holyscratch01/shared/adjellouli/simulations_reorganized/mechanisms/nucleation_mechanism_1000fps/d_amp=1.600"
    base_folder = "/n/holyscratch01/shared/adjellouli/simulations_reorganized/mechanisms/nucleation_mechanism_1000fps_extended/d_amp=1.600"
    base_folder2 = "/n/holyscratch01/shared/adjellouli/simulations_reorganized/mechanisms/nucleation_mechanism_1000fps_240rpm/d_amp=1.600"


    # test_zeros()

    # job(base_folder, 107, 205, 0)
    # exit()

    # job(base_folder, 107, 205, 1)
    # job(base_folder, 107, 205, 3)


    # job_plot_time_amplitude(base_folder, 107, 205, 10)


    import multiprocessing as mp

    # with mp.get_context('spawn').Pool() as p:
    #     res = []
    #     for it in range(100):#3):
    #         for v in [240]: #, 240]:#[205, 240]:
    #             res.append(p.apply_async(job_stats, args=(base_folder2, 107, v, it)))
    #     res = [r.get() for r in res]
    #     assemble_stats(res, 240)

    # exit()

    
    # with mp.get_context('spawn').Pool() as p:
    #     res = []
    #     for it in range(100):#range(100):#3):
    #         for v in [240]: #, 240]:#[205, 240]:
    #             res.append(p.apply_async(job, args=(base_folder2, 107, v, it)))
    #     [r.get() for r in res]
    
    with mp.get_context('spawn').Pool() as p:
        res = []
        for it in range(100):#3):
            for v in [240]: #, 240]:#[205, 240]:
                res.append(p.apply_async(job_plot_time_amplitude, args=(base_folder2, 107, v, it)))
        res = [r.get() for r in res]
        assemble_plot(res, 240)

    # with mp.get_context('spawn').Pool() as p:
        # res = []
        # for it in range(100):#3):
        #     for v in [205]: #, 240]:#[205, 240]:
        #         res.append(p.apply_async(job_plot_time_amplitude, args=(base_folder, 107, v, it)))
        # res = [r.get() for r in res]
        # assemble_plot(res, 205)


           

    # to start interactive session on the cluster
    # salloc -p test -t 0-08:00 --mem 32000 -c 4

# rsync -ah --progress  ALL/ /n/holylabs/LABS/bertoldi_lab/Lab/projects/shaking_beads_data/nucleation_mechanism/