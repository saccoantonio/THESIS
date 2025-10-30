import os                              
import numpy as np                     
from tqdm import tqdm # type: ignore             
from ase import Atoms   
from ase.io import read               
from ase.neighborlist import NeighborList, natural_cutoffs
import numpy as np
import matplotlib as plt

plt.style.use(["science", "nature"])
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "figure.figsize": (8.0, 6.2),
        "axes.linewidth": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "grid.alpha": 0.25,
    }
)


def compute_cn(atoms, cutoff_factor=1.2):
    """Calculates the coordination number for each atom in an ASE Atoms object.
    
    atoms: ASE Atoms object
    cutoff_factor: multiplicative factor for the natural cutoffs (default 1.2)
    """

    n_atoms = len(atoms)                             # Total number of atoms in the Atoms object
    atoms.set_chemical_symbols(["Au"] * n_atoms)     # Set all atomic symbols to "Au" (needed for natural_cutoffs)
    
    # Compute natural cutoffs for each atom and scale by cutoff_factor
    cutoffs = natural_cutoffs(atoms, mult=cutoff_factor)
    # natural_cutoffs returns an array of cutoff radii for each atom
    # 'cutoff_factor' scales the natural values (e.g., 1.2 = 20% larger)

    # Create a NeighborList:
    # - cutoffs: search radius for each atom
    # - self_interaction=False: do not count the atom itself
    # - bothways=True: neighbors are recorded symmetrically (i is neighbor of j and j of i)
    nl = NeighborList(cutoffs=cutoffs, self_interaction=False, bothways=True)

    nl.update(atoms)                                 # Build the neighbor list for the current atomic configuration

    cn = np.zeros(n_atoms, dtype=int)                # Array to store coordination numbers for each atom

    for i in range(n_atoms):                         # Loop over all atoms
        indices, offsets = nl.get_neighbors(i)      # Get neighbor indices and cell offsets (periodicity)
        cn[i] = len(indices)                        # Coordination number = number of neighbors

    return cn                                       # Return array of coordination numbers
                                  

def plot_cn_xyz_binned(atoms, cn, bins, filename):
    """Plot the average coordination number (CN) versus x, y, z coordinates using bins.
    
    atoms: ASE Atoms object
    cn: array of coordination numbers for each atom
    bins: number of bins or array of bin edges
    filename: string used for the plot title
    """

    pos = atoms.get_positions()  # Get atomic positions as a numpy array

    # Create a figure with 3 subplots for x, y, z coordinates
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("cn profiles of " + filename)  # Overall title
    labels = ['x', 'y', 'z']

    for i, ax in enumerate(axes):
        coord = pos[:, i]  # Select coordinate array (x, y, or z)
        
        # Weighted histogram: sum of CNs in each bin
        bin_means, bin_edges = np.histogram(coord, bins=bins, weights=cn)
        # Count of atoms in each bin
        bin_counts, _ = np.histogram(coord, bins=bin_edges)
        # Compute average CN per bin, avoid division by zero
        bin_avg = bin_means / np.maximum(bin_counts, 1)
        # Compute bin centers for plotting
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax.plot(bin_centers, bin_avg, color='red', lw=2)  # Plot mean CN vs coordinate
        ax.set_xlabel(f"{labels[i]} [Å]")               # Label x-axis
        ax.set_ylabel("Mean CN")                        # Label y-axis
        ax.grid(True)                                   # Add grid for readability

    plt.tight_layout()  # Adjust spacing to prevent overlap
    plt.show()          # Display the plot


def mean_cn_in_z_range(atoms, cn, z_min, z_max):
    """Calculate the mean coordination number (CN) for atoms with z-coordinate 
    between z_min and z_max.
    
    atoms: ASE Atoms object
    cn: array of coordination numbers for each atom
    z_min, z_max: lower and upper bounds for z-coordinate (Å)
    """

    z = atoms.get_positions()[:, 2]  # Extract z-coordinates of all atoms

    # Create a boolean mask selecting atoms within the z range
    mask = (z >= z_min) & (z <= z_max)
    cn_selected = cn[mask]  # Apply mask to CN array to select relevant atoms

    mean_cn = cn_selected.mean()  # Compute mean CN in the selected z range
    print(f"Mean CN between z = {z_min:.2f} Å and z = {z_max:.2f} Å: {mean_cn:.3f}")

    return mean_cn  # Return the mean CN


def save_file_with_cn(atoms, cn, out_file):
    """Save an .xyz file with an extra column containing the coordination number (CN) for each atom.
    
    atoms: ASE Atoms object
    cn: array of coordination numbers
    out_file: output filename (string)
    """

    n_atoms = len(atoms)                     # Total number of atoms
    pos = atoms.get_positions()              # Atomic positions (Nx3 array)
    symbols = atoms.get_chemical_symbols()  # List of chemical symbols for each atom

    # Open the output file for writing
    with open(out_file, "w") as f:
        f.write(f"{n_atoms}\n")                     # First line: number of atoms
        f.write("XYZ file with Coordination Number\n")  # Second line: comment/header

        # Loop over atoms and write symbol, coordinates, and CN
        for s, (x, y, z), c in zip(symbols, pos, cn):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f} {c}\n")
            # Format coordinates with 6 decimal places, append CN

    print(f"File saved to: {out_file}")  # Print confirmation


# ====================================================================================


def analyze_cn_from_file(filename, cutoff_factor, bins, out_file):
    """Read a data file, compute coordination numbers (CN), print statistics,
    plot CN profiles, and optionally save an XYZ file with CN.

    filename: input data file
    cutoff_factor: multiplicative factor for CN calculation
    bins: number of bins or array of bin edges for plotting
    out_file: if True, save an XYZ file with CN appended
    """

    # Read atomic configuration from LAMMPS data file
    atoms = read(filename, format="lammps-data", atom_style='atomic')

    # Compute coordination numbers
    cn = compute_cn(atoms, cutoff_factor=cutoff_factor)

    # Print basic statistics
    print(f"Number of atoms: {len(atoms)}")
    print(f"Mean CN: {cn.mean():.3f}")
    print(f"CN min/max: {cn.min()} / {cn.max()}")
    # Compute mean CN for atoms between two slabs (z = 8.1 Å to 71.5 Å)
    print(f"Mean CN between slabs: {mean_cn_in_z_range(atoms, cn, 8.1, 71.5)}")

    # Plot CN profiles along x, y, z
    plot_cn_xyz_binned(atoms, cn, bins, filename)

    # Optionally save an XYZ file with CN column
    if out_file == True:
        outname = filename.replace("data", "xyzcn")  # Replace extension for output
        save_file_with_cn(atoms, cn, outname)


def process_large_dump(path, infile, out_dir, outfile, cutoff_factor=1.2, skip=1):
    """Processes a large LAMMPS dump file frame by frame, computes coordination numbers (CN),
    and writes a new dump file with CN appended to each atom.
    
    path: directory containing the input dump
    infile: input dump filename
    out_dir: directory to save output
    outfile: output filename
    cutoff_factor: multiplicative factor for CN calculation
    skip: process every 'skip' frame (default 1 = all frames)
    """

    # --- Create output directory if it doesn't exist ---
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)                        # Make the output directory
        print(f"Created output directory: {out_dir}")

    infile_path = os.path.join(path, infile)        # Full path to input file
    outfile_path = os.path.join(out_dir, outfile)   # Full path to output file

    print(f"Reading from: {infile_path} & Writing to: {outfile_path}\n")

    # --- Count number of frames ---
    n_frames = 0
    with open(infile_path, "r") as fcount:
        for line in fcount:
            if "ITEM: TIMESTEP" in line:            # Each LAMMPS frame starts with this line
                n_frames += 1                       # Increment frame count

    # --- Process file with tqdm progress bar ---
    with open(infile_path, "r") as fin, open(outfile_path, "w") as fout:
        frame = 0
        # tqdm shows progress bar with known total = n_frames
        with tqdm(total=n_frames, desc="Processing LAMMPS dump", unit="frame") as pbar:
            while True:
                line = fin.readline()               # Read a line from input file
                if not line:
                    break                           # End of file -> exit loop

                if "ITEM: TIMESTEP" in line:        # Start of a new LAMMPS frame
                    timestep = int(fin.readline().strip())  # Read the timestep number
                    fout.write("ITEM: TIMESTEP\n")
                    fout.write(f"{timestep}\n")            # Write header and timestep to output

                    # --- Number of atoms ---
                    fin.readline()                         # Skip "ITEM: NUMBER OF ATOMS" line
                    n_atoms = int(fin.readline().strip())  # Read number of atoms
                    fout.write("ITEM: NUMBER OF ATOMS\n")
                    fout.write(f"{n_atoms}\n")             # Write number of atoms to output

                    # --- Box bounds ---
                    box_header = fin.readline()            # Read box bounds header
                    fout.write(box_header)                 # Write header to output
                    box = []
                    for _ in range(3):                    # Read three lines defining x, y, z bounds
                        bounds = list(map(float, fin.readline().split()))
                        box.append(bounds)                # Store bounds
                        fout.write(f"{bounds[0]} {bounds[1]}\n")  # Write bounds to output

                    # --- Atom section header ---
                    atoms_header = fin.readline().strip()  # Read atom data header
                    fout.write(atoms_header + " cn\n")     # Write header adding a "cn" column

                    # --- Atom data ---
                    data_lines = [fin.readline() for _ in range(n_atoms)]
                    # Read n_atoms lines containing atomic data

                    data = np.array([list(map(float, l.split())) for l in data_lines])
                    # Convert lines to a numeric array
                    pos = data[:, 1:4]                    # Assume columns 1,2,3 are x, y, z
                    box_lengths = [b[1] - b[0] for b in box]
                    # Compute cell lengths from bounds
                    atoms = Atoms("Au" * n_atoms, positions=pos, cell=box_lengths, pbc=True)
                    # Create ASE Atoms object:
                    # - symbols: "Au" repeated n_atoms
                    # - positions: extracted positions
                    # - cell: x, y, z lengths
                    # - pbc=True: enable periodic boundary conditions

                    if frame % skip == 0:
                        cn = compute_cn(atoms, cutoff_factor)  # Compute CN every 'skip' frames
                    else:
                        cn = np.zeros(n_atoms)                 # If skipped, CN = 0

                    for i in range(n_atoms):
                        vals = " ".join(f"{v:g}" for v in data[i])
                        # Recompose original atom line in compact format
                        fout.write(f"{vals} {int(cn[i])}\n")
                        # Write line with CN appended

                    frame += 1
                    pbar.update(1)  # Update progress bar after processing the frame


# by Antonio Sacco & commented by ChatGPT