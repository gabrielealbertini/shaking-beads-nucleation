
"""
read_binary.py

Read output binary files from simulations (on the cluster)
"""

import numpy as np
import struct
import glob
from natsort import natsorted
import os

# Hard-coded parameters (for unit conversions)
tau = 0.02545
dbead = 0.00635

def read_bins(basename: str, i0=0, chunk=100) -> tuple[np.ndarray,int,int]:
    """
    Read simulation binary files
    There might be several binary files for one simulation, because each file has a maximum size of around 100 Mo

    Format of a bin file:
        * Two int headers for n and the size of the header of each frame
        * The successive frames:
            - Its header: time (tau), xdish (d), ydish, dishangle, dishangularspeed, dishvx, dishvy, ntops (if correct) and slip displacement (set in special cases)
            - The data for each bead (to iterate thanks to n in the header): x, y, z, 4 quaternions for the rotation, vx, vy, vz, angularvelocityx, angularvelocityy, angularvelocityz
    
    Args:
        basename: str
            the prefix path of the binary files, it can be a folder path (ending with /) or a file path
        i0: int
            the index of the first file to read
        chunk: int
            the maximum number of files to read
    Returns:
        - the numpy array containing the data from simulation (shape = (number of frames, size of the frame header + 13 * number of beads))
        - the size of the frame header
        - the number of frames of the simulation
    """

    # looking for the data files and ordering them
    files = glob.glob(basename + '*.bin')
    files = natsorted(files)
    data = []
    nb_of_frames = 0

    # iterating through files
    for file in files[i0:i0+chunk]:
        print('Reading', os.path.basename(file))
        with open(file, 'rb') as readfile:
            binary_value = readfile.read(4)
            n = struct.unpack('i', binary_value)[0]
            binary_value = readfile.read(4)
            header_size = struct.unpack('i', binary_value)[0]
            size = header_size + n * 13
            try:
                # read frames until the end of the file (throw an Exception when the file is finished)
                while True:
                    infos = np.empty(size + 1)
                    for i in range(size):
                        binary_value = readfile.read(8)
                        infos[i] = struct.unpack('d', binary_value)[0]
                    infos[-1] = nb_of_frames
                    data.append(infos)
                    nb_of_frames += 1
            except Exception:
                pass

    return np.array(data), header_size, nb_of_frames