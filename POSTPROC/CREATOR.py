from ase.lattice.cubic import FaceCenteredCubic
import numpy as np
from ase.io import write
from scipy.spatial import cKDTree


def create_bulk(n):
    """Creates a bulk FCC gold (Au) crystal of size n x n x n unit cells with periodic boundary conditions."""

    # Parameters
    symbol = "Au"  # Atomic symbol for gold
    a0 = 4.078     # Lattice constant of Au FCC in Ångstroms

    # Create a bulk FCC structure of gold
    bulk = FaceCenteredCubic(
        directions=[[1, 0, 0],   # Define orientation of the lattice vectors
                    [0, 1, 0],
                    [0, 0, 1]],
        size=(n, n, n),           # Number of unit cells in x, y, z directions
        symbol=symbol,            # Element type
        pbc=True                   # Enable periodic boundary conditions
    )

    final_config = bulk         # Copy the bulk structure to final configuration
    final_config.set_cell(bulk.get_cell())  # Ensure the cell vectors are explicitly set
    final_config.set_pbc([True, True, True])  # Make sure periodic boundary conditions are on in all directions

    #print(f"Numero di atomi: {len(final_config)}")  # Optional: print the number of atoms
    return final_config        # Return the final bulk configuration


def create_vacancies(perc):
    """Creates vacancies in the central region of a bulk FCC gold (Au) crystal.
    perc: percentage of atoms to remove from the central region.
    """

    a0 = 4.078  # Lattice constant of Au FCC in Ångstroms
    bulk = create_bulk(20)  # Generate a 20x20x20 bulk FCC gold crystal
    pos = bulk.get_positions()  # Get the atomic positions as an array

    zmin = 2.01*a0  # Minimum z-coordinate for central region (exclude bottom slab)
    zmax = pos[:, 2].max() - zmin  # Maximum z-coordinate for central region (exclude top slab)
    
    # Get indices of atoms in the top slab
    upslab_indices = [i for i, atom in enumerate(bulk) if atom.z > zmax]
    # Get indices of atoms in the bottom slab
    lowslab_indices = [i for i, atom in enumerate(bulk) if atom.z < zmin]
    #slabs = bulk[upslab_indices] + bulk[lowslab_indices]  # Optional: the excluded slabs

    all_slab_indices = set(upslab_indices + lowslab_indices)  # Combine top and bottom slab indices
    # Get indices of atoms in the central region (excluding top and bottom slabs)
    central_indices = [i for i in range(len(bulk)) if i not in all_slab_indices]
    n_central = len(central_indices)  # Number of atoms in central region

    # Determine how many atoms to remove based on the given percentage
    n_remove = int(perc/100 * n_central)
    # Randomly select atoms to remove from the central region without replacement
    remove_from_central = np.random.choice(central_indices, n_remove, replace=False)
    
    # Final indices are all atoms minus those selected for removal
    final_indices = list(set(range(len(bulk))) - set(remove_from_central))
    final_system = bulk[final_indices]  # Create the final system with vacancies
    
    # Save the structure to a LAMMPS data file
    write(f'vacancies{perc}.data', final_system, format='lammps-data', atom_style='atomic')

    # Optional debug prints
    #print("Number of atoms in the central region before removal:", n_central)
    #print("Number of atoms removed:", len(remove_from_central), "i.e.,", len(remove_from_central)/n_central*100, "%")
    #print(f"Cell size: {final_system.get_cell()}")


def create_holes(perc, radius):
    """
    Rimuove "buchi" sferici fino a rimuovere perc% degli atomi della regione centrale.
    - perc: percentuale (0-100) di atomi centrali da rimuovere
    - radius: raggio della sfera (Å)
    - pbc: tuple booleana (px,py,pz) per abilitare PBC in ciascuna direzione
    - cell_bulk_size: parametro passato a create_bulk(...) (lo lascio come argomento per test)
    Restituisce l'oggetto ASE Atoms finale.
    """
    a0 = 4.078
    bulk = create_bulk(20)   # usa la tua factory
    pos = bulk.get_positions()
    cell = np.asarray(bulk.get_cell())   # 3x3 array

    # --- definizione slab superiore/inferiore (zona che NON tocchiamo) ---
    zmin = 2.01 * a0
    zmax = pos[:, 2].max() - zmin
    upslab = [i for i in range(len(bulk)) if pos[i, 2] > zmax]
    lowslab = [i for i in range(len(bulk)) if pos[i, 2] < zmin]
    all_slab = set(upslab + lowslab)

    # --- atomi della regione centrale ---
    central_indices = [i for i in range(len(bulk)) if i not in all_slab]
    if len(central_indices) == 0:
        raise RuntimeError("Nessun atomo centrale trovato: controlla zmin/zmax.")
    central_pos = pos[central_indices]
    n_central = len(central_indices)
    n_target = int(round(perc / 100.0 * n_central))
    if n_target <= 0:
        print("n_target = 0 (percentuale troppo piccola). Non rimuovo nulla.")
        return bulk

    print(f"Atomi totali: {len(bulk)}, atomi centrali: {n_central}, target da rimuovere: {n_target}")

    # --- costruiamo i vettori di shift (solo per le direzioni con PBC attive) ---
    pbc=(True, True, True)
    ranges = [(-1, 0, 1) if pbc[i] else (0,) for i in range(3)]
    shifts = np.array([[i, j, k] for i in ranges[0] for j in ranges[1] for k in ranges[2]])
    n_shifts = len(shifts)
    disp = shifts @ cell   # n_shifts x 3

    # --- repliche delle posizioni centrali e mappa indici (globali) ---
    # replicas shape: (n_shifts * n_central, 3)
    replicas = (central_pos[None, :, :] + disp[:, None, :]).reshape(-1, 3)
    # mapping: for each replica entry, the corresponding global index in the original cell
    mapping = np.tile(central_indices, n_shifts)

    print(f"Numero di repliche costruite: {replicas.shape[0]} (shifts={n_shifts})")

    # --- KDTree sulle repliche (serve per PBC) ---
    tree = cKDTree(replicas)

    # --- itero i possibili centri in ordine casuale (no replacement) ---
    order = np.random.permutation(n_central)
    to_remove = set()
    for local_center_idx in order:
        if len(to_remove) >= n_target:
            break
        center = central_pos[local_center_idx]
        # trova tutti i punti (repliche) entro `radius` dal centro
        neighbors = tree.query_ball_point(center, radius)
        if not neighbors:
            continue
        # converti indici delle repliche in indici globali e aggiungi al set
        bad_globals = {mapping[i] for i in neighbors}
        to_remove.update(bad_globals)

    # --- costruzione sistema finale e salvataggio ---
    final_indices = [i for i in range(len(bulk)) if i not in to_remove]
    final_system = bulk[final_indices]

    outname = f'holes_{perc:.1f}perc_R{radius:.1f}.data'
    write(outname, final_system, format='lammps-data', atom_style='atomic')
    print(f"Salvato {outname} — rimossi {len(to_remove)} atomi ({100*len(to_remove)/n_central:.1f}% della zona centrale)")


def create_void(radius):
    """Creates a hemispherical void in the central region of a bulk FCC gold (Au) crystal.
    radius: radius of the hemispherical void in Ångstroms.
    """

    a0 = 4.078  # Lattice constant of Au FCC in Ångstroms
    bulk = create_bulk(20)  # Generate a 20x20x20 bulk FCC gold crystal
    pos = bulk.get_positions()  # Get atomic positions as a numpy array

    # Define top and bottom slabs to exclude from the central region
    zmin = 2.01 * a0
    zmax = pos[:, 2].max() - zmin
    upslab_indices = [i for i, atom in enumerate(bulk) if atom.z > zmax]  # Atoms in top slab
    lowslab_indices = [i for i, atom in enumerate(bulk) if atom.z < zmin]  # Atoms in bottom slab

    # Identify central region atoms (excluding top and bottom slabs)
    all_slab_indices = set(upslab_indices + lowslab_indices)
    central_indices = [i for i in range(len(bulk)) if i not in all_slab_indices]

    # ---- Define hemisphere ----
    # Hemisphere center: middle of the cell in x, y, z
    box = bulk.get_cell().array  # Get the cell vectors as a 3x3 array
    center_x = box[0,0] / 2       # Center in x
    center_y = box[1,1] / 2       # Center in y
    center_z = box[2,2] / 2       # Center in z (now middle of the cell)
    center = np.array([center_x, center_y, center_z])  # Hemisphere center coordinates

    # Select central atoms inside the hemisphere
    remove_from_central = []
    for i in central_indices:
        r_vec = pos[i] - center  # Vector from center to atom
        dist = np.linalg.norm(r_vec)  # Euclidean distance
        if dist < radius and pos[i][2] >= center_z:  # Condition for being in the hemisphere
            remove_from_central.append(i)  # Mark atom for removal

    # Final system: all atoms minus those inside the hemisphere
    final_indices = list(set(range(len(bulk))) - set(remove_from_central))
    final_system = bulk[final_indices]

    # Recompute central indices and number of central atoms (for statistics)
    all_slab_indices = set(upslab_indices + lowslab_indices)
    central_indices = [i for i in range(len(bulk)) if i not in all_slab_indices]
    n_central = len(central_indices)

    # Save the final system to a LAMMPS data file
    write(f'void_radius{radius}.data', final_system, format='lammps-data', atom_style='atomic')

    # Print number of atoms removed and percentage relative to central region
    print(f"Number of atoms removed: {len(remove_from_central)}", 
          "i.e.,", len(remove_from_central)/n_central*100, "%")
    #print(f"Number of final atoms: {len(final_system)}")


def create_pillar(perc):
    """Creates a central pillar in a bulk FCC gold (Au) crystal, leaving perc% void in the central region.
    perc: percentage of empty space (void) in the central region.
    """

    a0 = 4.078  # Lattice constant of Au FCC in Ångstroms
    bulk = create_bulk(20)  # Generate a 20x20x20 bulk FCC gold crystal
    positions = bulk.get_positions()  # Get atomic positions as a numpy array

    # Geometric centers in x and y
    x_center = 0.5 * (positions[:, 0].min() + positions[:, 0].max())
    y_center = 0.5 * (positions[:, 1].min() + positions[:, 1].max())
    zmin = 2.01 * a0  # Minimum z-coordinate for central region (exclude bottom slab)
    zmax = positions[:, 2].max() - zmin  # Maximum z-coordinate (exclude top slab)

    # Compute target area for the pillar in x-y plane
    S_total = positions[:, 0].max() * positions[:, 1].max()  # Total cross-sectional area
    S_target = (1 - perc/100) * S_total  # Area to fill with pillar (void is perc%)
    dx = dy = np.sqrt(S_target)  # Pillar dimensions in x and y

    # Identify atoms in upper and lower slabs to keep
    upper_indices = [i for i, atom in enumerate(bulk) if atom.position[2] > zmax]
    lower_indices = [i for i, atom in enumerate(bulk) if atom.position[2] < zmin]
    slab_indices = set(upper_indices + lower_indices)

    # Select atoms that form the pillar in the central region
    column_indices = [
        i for i, atom in enumerate(bulk)
            if i not in slab_indices and  # Not in top/bottom slab
               abs(atom.position[0] - x_center) < dx / 2 and  # Within pillar x-range
               abs(atom.position[1] - y_center) < dy / 2 and  # Within pillar y-range
               zmin < atom.position[2] < zmax                  # Within central z-range
    ]

    # Combine upper slab, pillar, and lower slab atoms
    final_config = bulk[upper_indices] + bulk[column_indices] + bulk[lower_indices]

    # Set correct cell and periodic boundary conditions
    final_config.set_cell(bulk.get_cell())
    final_config.set_pbc([True, True, True])
    #print(f"Number of atoms in the final configuration: {len(final_config)}")

    # Save the configuration to a LAMMPS data file
    write('pillar'+str(perc)+'.data', final_config, format='lammps-data', atom_style='atomic')


def create_column(perc):
    """Creates a cylindrical column in the central region of a bulk FCC gold (Au) crystal.
    perc: percentage of empty space (void) in the central region.
    """

    a0 = 4.078  # Lattice constant of Au FCC in Ångstroms
    bulk = create_bulk(20)  # Generate a 20x20x20 bulk FCC gold crystal
    positions = bulk.get_positions()  # Get atomic positions as a numpy array

    # Geometric centers in x and y
    x_center = 0.5 * (positions[:, 0].min() + positions[:, 0].max())
    y_center = 0.5 * (positions[:, 1].min() + positions[:, 1].max())
    zmin = 2.01 * a0  # Minimum z-coordinate for central region (exclude bottom slab)
    zmax = positions[:, 2].max() - zmin  # Maximum z-coordinate (exclude top slab)

    # Compute target cross-sectional area for the column
    S_total = positions[:, 0].max() * positions[:, 1].max()  # Total area in x-y plane
    S_target = (1 - perc/100) * S_total  # Area to fill with column (void is perc%)
    radius = np.sqrt(S_target / np.pi)   # Compute radius of the cylindrical column

    # Identify atoms in upper and lower slabs to keep
    upper_indices = [i for i, atom in enumerate(bulk) if atom.position[2] > zmax]
    lower_indices = [i for i, atom in enumerate(bulk) if atom.position[2] < zmin]
    slab_indices = set(upper_indices + lower_indices)

    # Select atoms that form the cylindrical column in the central region
    column_indices = [
        i for i, atom in enumerate(bulk)
        if i not in slab_indices and              # Not in top/bottom slab
           (zmin < atom.z < zmax) and            # Within central z-range
           np.sqrt((atom.x - x_center)**2 + (atom.y - y_center)**2) < radius  # Within column radius
    ]

    # Combine upper slab, column, and lower slab atoms
    final_config = bulk[upper_indices] + bulk[column_indices] + bulk[lower_indices]

    # Set correct cell and periodic boundary conditions
    final_config.set_cell(bulk.get_cell())
    final_config.set_pbc([True, True, True])
    #print(f"Number of atoms in the final configuration: {len(final_config)}")

    # Save the configuration to a LAMMPS data file
    write('column'+str(perc)+'.data', final_config, format='lammps-data', atom_style='atomic')


# by Antonio Sacco & commented by ChatGPT
