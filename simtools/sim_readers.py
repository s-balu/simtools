import gc
import os
import glob
import numpy as np
import h5py
from scipy.spatial import KDTree
import time
import warnings
from pathos.multiprocessing import ProcessingPool as Pool

from simtools.quantities import hubble_parameter
from simtools.utils import vector_norm, recenter_coordinates


class GadgetBox:

    def __init__(self, unit_length_in_cm=None, unit_mass_in_g=None,
                 unit_velocity_in_cm_per_s=None):

        # Units for format 1/2 data. Later overwritten if data is in format 3.
        self.unit_length_in_cm = unit_length_in_cm
        self.unit_mass_in_g = unit_mass_in_g
        self.unit_velocity_in_cm_per_s = unit_velocity_in_cm_per_s

    def reorder_subfiles(self, filename, subfiles):

        subfile_err_msg = "Multiple files consistent with '{}' that " \
                          "aren't sub-files. Enter a more specific " \
                          "filename or wildcard.".format(filename)
        try:
            splt = np.array([f.split(".") for f in subfiles])
        except ValueError:
            raise ValueError(subfile_err_msg)
        ii, found_subfiles = 0, False
        while (not found_subfiles) and (ii < len(splt[0])):
            found_subfiles = np.all([x.isdigit() for x in splt[:, ii]])
            ii += 1
        if found_subfiles:
            subnums = [int(x) for x in splt[:, ii - 1]]
            return np.array(subfiles)[np.argsort(subnums)]
        else:
            raise ValueError(subfile_err_msg)

    def read_parameters(self, datafile, file_format):

        if file_format == 3:
            if 'Header' in datafile:
                if 'BoxSize' in datafile['Header'].attrs:
                    self.box_size = datafile['Header'].attrs['BoxSize']
                if 'Redshift' in datafile['Header'].attrs:
                    self.redshift = datafile['Header'].attrs['Redshift']
                    self.scale_factor = 1 / (1 + self.redshift)
                if 'Time' in datafile['Header'].attrs:
                    self.time = datafile['Header'].attrs['Time']
                if 'NSample' in datafile['Header'].attrs:
                    self.nsample = datafile['Header'].attrs['NSample']
                elif 'NumPart_Total' in datafile['Header'].attrs:
                    self.nsample = np.cbrt(datafile['Header'].attrs['NumPart_Total'][1])
                else:
                    self.nsample = 1
                if 'Omega0' in datafile['Header'].attrs:
                    self.Omega0 = datafile['Header'].attrs['Omega0']
                if 'OmegaBaryon' in datafile['Header'].attrs:
                    self.OmegaBaryon = datafile['Header'].attrs['OmegaBaryon']
                if 'OmegaLambda' in datafile['Header'].attrs:
                    self.OmegaLambda = datafile['Header'].attrs['OmegaLambda']
                if 'HubbleParam' in datafile['Header'].attrs:
                    self.h = datafile['Header'].attrs['HubbleParam']
                if 'Hubble' in datafile['Header'].attrs:
                    self.hubble = datafile['Header'].attrs['Hubble']
                if 'UnitLength_in_cm' in datafile['Header'].attrs:
                    self.unit_length_in_cm = datafile['Header'].attrs[
                        'UnitLength_in_cm']
                if 'UnitMass_in_g' in datafile['Header'].attrs:
                    self.unit_mass_in_g = datafile['Header'].attrs[
                        'UnitMass_in_g']
                if 'UnitVelocity_in_cm_per_s' in datafile['Header'].attrs:
                    self.unit_velocity_in_cm_per_s = datafile['Header'].attrs[
                        'UnitVelocity_in_cm_per_s']
            if 'Parameters' in datafile:
                if 'Time' in datafile['Parameters'].attrs:
                    self.time = datafile['Parameters'].attrs['Time']
                if 'ComovingIntegrationOn' in datafile['Parameters'].attrs:
                    if datafile['Parameters'].attrs[
                            'ComovingIntegrationOn'] == 1:
                        self.scale_factor = self.time
                if 'NSample' in datafile['Parameters'].attrs:
                    self.nsample = datafile['Parameters'].attrs['NSample']
                if 'Omega0' in datafile['Parameters'].attrs:
                    self.Omega0 = datafile['Parameters'].attrs['Omega0']
                if 'OmegaBaryon' in datafile['Parameters'].attrs:
                    self.OmegaBaryon = datafile['Parameters'].attrs[
                        'OmegaBaryon']
                if 'OmegaLambda' in datafile['Parameters'].attrs:
                    self.OmegaLambda = datafile['Parameters'].attrs[
                        'OmegaLambda']
                if 'HubbleParam' in datafile['Parameters'].attrs:
                    self.h = datafile['Parameters'].attrs['HubbleParam']
                if 'Hubble' in datafile['Parameters'].attrs:
                    self.hubble = datafile['Parameters'].attrs['Hubble']
                if 'UnitLength_in_cm' in datafile['Parameters'].attrs:
                    self.unit_length_in_cm = datafile['Parameters'].attrs[
                        'UnitLength_in_cm']
                if 'UnitMass_in_g' in datafile['Parameters'].attrs:
                    self.unit_mass_in_g = datafile['Parameters'].attrs[
                        'UnitMass_in_g']
                if 'UnitVelocity_in_cm_per_s' in datafile['Parameters'].attrs:
                    self.unit_velocity_in_cm_per_s = datafile[
                        'Parameters'].attrs['UnitVelocity_in_cm_per_s']

        else:
            offset = 0
            offset += 16
            offset += 4

            datafile.seek(offset, os.SEEK_SET)
            self.number_of_particles_this_file_by_type = np.fromfile(
                datafile, dtype=np.int32, count=6)
            self.number_of_particles_this_file = \
                self.number_of_particles_this_file_by_type[self.particle_type]

            offset += 24
            datafile.seek(offset, os.SEEK_SET)
            self.mass_table = np.fromfile(datafile, dtype=np.float64, count=6)

            offset += 48
            datafile.seek(offset, os.SEEK_SET)
            self.scale_factor = np.fromfile(
                datafile, dtype=np.float64, count=1)[0]

            offset += 8
            datafile.seek(offset, os.SEEK_SET)
            self.redshift = np.fromfile(datafile, dtype=np.float64, count=1)[0]

            offset += 8
            offset += 4  # FlagSfr
            offset += 4  # FlagFeedback
            datafile.seek(offset, os.SEEK_SET)
            self.number_of_particles_by_type = np.fromfile(
                datafile, dtype=np.int32, count=6)
            self.number_of_particles = self.number_of_particles_by_type[
                self.particle_type]

            offset += 24
            offset += 4  # FlagCooling
            datafile.seek(offset, os.SEEK_SET)
            self.number_of_files = np.fromfile(
                datafile, dtype=np.int32, count=1)[0]

            offset += 4
            datafile.seek(offset, os.SEEK_SET)
            self.box_size = np.fromfile(datafile, dtype=np.float64, count=1)[0]

            offset += 8
            datafile.seek(offset, os.SEEK_SET)
            self.Omega0 = np.fromfile(datafile, dtype=np.float64, count=1)[0]

            offset += 8
            datafile.seek(offset, os.SEEK_SET)
            self.OmegaLambda = np.fromfile(
                datafile, dtype=np.float64, count=1)[0]

            offset += 8
            datafile.seek(offset, os.SEEK_SET)
            self.h = np.fromfile(datafile, dtype=np.float64, count=1)[0]

        self.cm_per_kpc = 3.085678e21
        self.g_per_1e10Msun = 1.989e43
        self.cmps_per_kmps = 1.0e5
        if self.unit_length_in_cm is None:
            warnings.warn('No value for `UnitLength_in_cm` found!'
                          ' Assuming GADGET-4 default of {}.'.format(
                            self.cm_per_kpc))
            self.unit_length_in_cm = self.cm_per_kpc
        if self.unit_mass_in_g is None:
            warnings.warn('No value for `UnitMass_in_g` found!'
                          ' Assuming GADGET-4 default of {}.'.format(
                            self.g_per_1e10Msun))
            self.unit_mass_in_g = self.g_per_1e10Msun
        if self.unit_velocity_in_cm_per_s is None:
            warnings.warn('No value for `UnitVelocity_in_cm_per_s` found!'
                          ' Assuming GADGET-4 default of {}.'.format(
                            self.cmps_per_kmps))
            self.unit_velocity_in_cm_per_s = self.cmps_per_kmps
        self.unit_time_in_s = self.unit_length_in_cm / \
            self.unit_velocity_in_cm_per_s
        length_norm = self.unit_length_in_cm / self.cm_per_kpc
        mass_norm = self.unit_mass_in_g / self.g_per_1e10Msun
        velocity_norm = self.unit_velocity_in_cm_per_s / self.cmps_per_kmps
        self.gravitational_constant = 43009.1727 * mass_norm / \
            (length_norm * velocity_norm**2)

        if not hasattr(self, 'Omega0'):
            self.Omega0 = 0
        if not hasattr(self, 'OmegaBaryon'):
            self.OmegaBaryon = 0
        if not hasattr(self, 'OmegaLambda'):
            self.OmegaLambda = 0
        if not hasattr(self, 'h'):
            warnings.warn('No value for `HubbleParam` found!'
                          ' Assuming GADGET-4 default of 0.7.')
            self.h = 0.7
        if not hasattr(self, 'hubble'):
            self.hubble = 0.1 * length_norm / velocity_norm
            warnings.warn('No value for `Hubble` found!'
                          ' Using Hubble={}.'.format(self.hubble))

        self.hubble_constant = self.h * self.hubble
        if hasattr(self, 'redshift'):
            self.hubble_parameter = hubble_parameter(
                self.redshift, self.hubble_constant,
                self.Omega0, self.OmegaLambda, 0)

        if hasattr(self, 'hubble_parameter'):
            self.critical_density = 3 * (self.hubble_parameter / self.h)**2 / \
                (8 * np.pi * self.gravitational_constant)

        return


class GadgetSnapshot(GadgetBox):

    def __init__(self, path, snapshot_filename, snapshot_number,
                 particle_type=None, load_ids=False, load_coords=False,
                 load_vels=False, load_masses=False, region_positions=None,
                 region_radii=None, use_kdtree=False, buffer=0.0, read_mode=1,
                 unit_length_in_cm=None, unit_mass_in_g=None,
                 unit_velocity_in_cm_per_s=None, snapshot_format=None,
                 npool=None, verbose=True):

        super().__init__(unit_length_in_cm, unit_mass_in_g,
                         unit_velocity_in_cm_per_s)

        self.snapshot_path = path
        self.snapshot_filename = snapshot_filename
        self.snapshot_number = snapshot_number
        self.particle_type = particle_type
        self.region_positions = region_positions
        self.region_radii = region_radii
        self.use_kdtree = use_kdtree
        self.read_mode = read_mode
        self.npool = npool
        self.buffer = buffer

        if snapshot_format is None:
            if snapshot_filename.split('.')[-1] == 'hdf5':
                self.snapshot_format = 3
            else:
                self.snapshot_format = 2
        else:
            self.snapshot_format = snapshot_format

        snapshot_files = glob.glob(path + '/{}'.format(
            snapshot_filename.format('%03d' % snapshot_number)))

        nsnap = len(snapshot_files)
        if nsnap > 0:
            self.has_snap = True
            if nsnap > 1:
                snapshot_files = self.reorder_subfiles(
                    snapshot_filename, snapshot_files)
            if verbose:
                print('Found {} snapshot file(s) for snapshot {} in directory'
                      ' {}'.format(nsnap, snapshot_number, path))
                start = time.time()
            self.read_snapshot(
                snapshot_files, load_ids, load_coords, load_vels, load_masses,
                region_positions, region_radii, read_mode)
            if verbose:
                print("...Loaded PartType{} data in {} seconds\n".format(particle_type,
                    round(time.time() - start, 4)))

        else:
            self.has_snap = False
            warnings.warn('No snapshot files found!')

    def read_snapshot(self, filenames, load_ids, load_coords, load_vels,
                      load_masses, region_positions, region_radii, read_mode):

        def read_binary_snapshot(fnames):

            if self.number_of_particles_by_type[self.particle_type] == 0:
                self.ids = np.array([])
                self.coordinates = np.array([])
                self.velocities = np.array([])
                self.masses = np.array([])
                if self.particle_type == 0:
                    self.us_all = np.array([])
                    self.rhos_all = np.array([])
                if region_positions is not None:
                    self.region_offsets = np.array([])

                return

            coords_all, vels_all, ids_all, masses_all = [], [], [], []
            if self.particle_type == 0:
                us_all, rhos_all = [], []

            idx_with_mass = np.where(self.mass_table == 0)[0]
            if self.particle_type in idx_with_mass:
                masses_from_table = False
            else:
                masses_from_table = True

            for f in fnames:

                with open(f, 'rb') as snap:

                    self.read_parameters(snap, self.snapshot_format)

                    npart_total = np.sum(
                        self.number_of_particles_this_file_by_type)
                    npart = self.number_of_particles_this_file_by_type[
                        self.particle_type]
                    ptype_offset = np.sum(np.array(
                        self.number_of_particles_this_file_by_type[
                            :self.particle_type + 1])) - npart

                    if npart == 0:
                        continue

                    npart_by_type_in_mass_block = \
                        self.number_of_particles_this_file_by_type[
                            idx_with_mass]
                    npart_total_in_mass_block = np.sum(
                        npart_by_type_in_mass_block)
                    if not masses_from_table:
                        ptype_ind_in_mass_block = np.where(
                            idx_with_mass == self.particle_type)[0][0]
                        ptype_offset_in_mass_block = np.sum(np.array(
                            npart_by_type_in_mass_block[
                                :ptype_ind_in_mass_block + 1])) - npart

                    offset = 264  # includes 2 x 4 byte buffers
                    offset += 16
                    offset += 4  # 1st 4 byte buffer
                    offset += 16
                    offset += 3 * ptype_offset * 4
                    snap.seek(offset, os.SEEK_SET)
                    coords = np.fromfile(
                        snap, dtype=np.float32, count=3 * npart).reshape(
                        npart, 3)
                    coords_all.append(coords)
                    offset -= 3 * ptype_offset * 4

                    # Increment beyond the POS block
                    offset += 3 * npart_total * 4
                    offset += 4  # 2nd 4 byte buffer
                    offset += 4  # 1st 4 byte buffer
                    offset += 16
                    offset += 3 * ptype_offset * 4
                    snap.seek(offset, os.SEEK_SET)
                    vels = np.fromfile(
                        snap, dtype=np.float32, count=3 * npart).reshape(
                        npart, 3)
                    vels_all.append(vels)
                    offset -= 3 * ptype_offset * 4

                    # Increment beyond the VEL block
                    offset += 3 * npart_total * 4
                    offset += 4  # 2nd 4 byte buffer
                    offset += 4  # 1st 4 byte buffer
                    offset += 16
                    offset += ptype_offset * 4
                    snap.seek(offset, os.SEEK_SET)
                    ids = np.fromfile(snap, dtype=np.uint32, count=npart)
                    ids_all.append(ids)
                    offset -= ptype_offset * 4

                    # Increment beyond the IDS block
                    offset += npart_total * 4
                    offset += 4  # 2nd 4 byte buffer
                    offset += 4  # 1st 4 byte buffer
                    offset += 16
                    if not masses_from_table:
                        offset += ptype_offset_in_mass_block * 4
                        snap.seek(offset, os.SEEK_SET)
                        masses = np.fromfile(
                            snap, dtype=np.float32, count=npart)
                        masses_all.append(masses)
                        offset -= ptype_offset_in_mass_block * 4

                    if self.particle_type == 0:
                        # Increment beyond the mass block
                        offset += npart_total_in_mass_block * 4
                        offset += 4  # 2nd 4 byte buffer
                        offset += 4  # 1st 4 byte buffer
                        offset += 16
                        snap.seek(offset, os.SEEK_SET)
                        u = np.fromfile(snap, dtype=np.float32, count=npart)
                        us_all.append(u)

                        # Increment beyond the u block
                        offset += npart * 4
                        offset += 4  # 2nd 4 byte buffer
                        offset += 4  # 1st 4 byte buffer
                        offset += 16
                        snap.seek(offset, os.SEEK_SET)
                        rho = np.fromfile(snap, dtype=np.float32, count=npart)
                        rhos_all.append(rho)

            if region_positions is not None:
                coords = np.concatenate(coords_all)
                r = vector_norm(coords - region_positions[0])
                inds = np.argwhere(r < region_radii[0]).flatten()

                self.ids = np.concatenate(ids_all)[inds]
                self.coordinates = np.concatenate(coords_all)[inds]
                self.velocities = np.concatenate(vels_all)[inds]
                if masses_from_table:
                    self.masses = self.mass_table[self.particle_type]
                else:
                    masses = np.concatenate(masses_all)[inds]
                    if np.all(masses == masses[0]):
                        self.masses = masses[0]
                    else:
                        self.masses = masses
                if self.particle_type == 0:
                    self.internal_energy = np.concatenate(us_all)[inds]
                    self.density = np.concatenate(rhos_all)[inds]
                self.region_offsets = np.array([0])
            else:
                self.ids = np.concatenate(ids_all)
                self.coordinates = np.concatenate(coords_all)
                self.velocities = np.concatenate(vels_all)
                if masses_from_table:
                    self.masses = self.mass_table[self.particle_type]
                else:
                    masses = np.concatenate(masses_all)
                    if np.all(masses == masses[0]):
                        self.masses = masses[0]
                    else:
                        self.masses = masses
                if self.particle_type == 0:
                    self.internal_energy = np.concatenate(us_all)
                    self.density = np.concatenate(rhos_all)

            self.velocities *= np.sqrt(1 + self.redshift)

        def read_hdf5_snapshot(fnames):

            def read_files(ii):

                f = fnames[ii]
                with h5py.File(f, 'r') as snap:
                    snappt = snap['PartType{}'.format(self.particle_type)]

                    if region_positions is not None:
                        coords = snappt['Coordinates'][()]
                        num_part = len(coords)
                        if self.use_kdtree:
                            kdtree = KDTree(
                                coords, boxsize=self.box_size*(1+self.buffer))
                            region_inds = kdtree.query_ball_point(
                                region_positions, region_radii)
                        else:
                            region_inds = []
                            for pos, rad in zip(
                                    region_positions, region_radii):
                                r = vector_norm(
                                    recenter_coordinates(
                                        coords - pos, self.box_size))
                                region_inds.append(
                                    np.argwhere(r < rad).flatten())
                        region_lens = [len(inds) for inds in region_inds]
                        region_inds = np.hstack(region_inds).astype(int)
                        if len(region_inds) == 0:
                            nc = len(region_positions)
                            metallicities = None
                            formation_times = None
                            if 'Metallicity' in list(snappt):
                                metallicities = [np.array([])] * nc
                            if 'StellarFormationTime' in list(snappt):
                                formation_times = [np.array([])] * nc
                            return [np.array([], dtype=np.uint64)]*nc, \
                                [np.array([]).reshape(0, 3)]*nc, \
                                [np.array([]).reshape(0, 3)]*nc, \
                                [np.array([])]*nc, \
                                metallicities, \
                                formation_times
                        coords = coords[region_inds]
                        gc.collect()
                        coords = np.split(coords, np.cumsum(region_lens))[:-1]
                        if read_mode == 2:
                            region_inds_unique, inv = np.unique(
                                region_inds, return_inverse=True)
                            region_inds_bool = np.zeros(
                                num_part, dtype=bool)
                            region_inds_bool[region_inds_unique] = True
                            region_inds_bool_vec = np.zeros(
                                (num_part, 3), dtype=bool)
                            region_inds_bool_vec[region_inds_unique, :] = True
                    else:
                        region_inds = None

                    if load_ids:
                        if region_inds is None:
                            ids = snappt['ParticleIDs'][()]
                        else:
                            if read_mode == 1:
                                ids = snappt['ParticleIDs'][()]
                                ids = ids[region_inds]
                                gc.collect()
                            elif read_mode == 2:
                                ids = snappt['ParticleIDs']
                                ids = ids[region_inds_bool][inv]
                            ids = np.split(ids, np.cumsum(region_lens))[:-1]
                    else:
                        ids = None

                    if load_coords:
                        if region_inds is None:
                            coords = snappt['Coordinates'][()]
                    else:
                        coords = None

                    if load_vels:
                        if region_inds is None:
                            vels = snappt['Velocities'][()]
                        else:
                            if read_mode == 1:
                                vels = snappt['Velocities'][()]
                                vels = vels[region_inds]
                                gc.collect()
                            elif read_mode == 2:
                                vels = snappt['Velocities']
                                vels = vels[region_inds_bool_vec].reshape(
                                    len(region_inds_unique), 3)
                                vels = vels[inv]
                            vels = np.split(vels, np.cumsum(region_lens))[:-1]
                    else:
                        vels = None

                    if load_masses:
                        if 'Masses' in list(snappt):
                            if region_inds is None:
                                masses = snappt['Masses'][()]
                            else:
                                if read_mode == 1:
                                    masses = snappt['Masses'][()]
                                    masses = masses[region_inds]
                                    gc.collect()
                                elif read_mode == 2:
                                    masses = snappt['Masses']
                                    masses = masses[region_inds_bool][inv]
                                masses = np.split(
                                    masses, np.cumsum(region_lens))[:-1]
                        else:
                            masses = (snap['Header'].attrs['MassTable'])[
                                self.particle_type]
                    else:
                        masses = None

                    if 'Metallicity' in list(snappt):
                        if region_inds is None:
                            metallicities = snappt['Metallicity'][()]
                        else:
                            if read_mode == 1:
                                metallicities = snappt['Metallicity'][()]
                                metallicities = metallicities[region_inds]
                                gc.collect()
                            elif read_mode == 2:
                                metallicities = snappt['Metallicity']
                                metallicities = metallicities[
                                    region_inds_bool][inv]
                            metallicities = np.split(
                                metallicities, np.cumsum(region_lens))[:-1]
                    else:
                        metallicities = None

                    if 'StellarFormationTime' in list(snappt):
                        if region_inds is None:
                            formation_times = snappt[
                                'StellarFormationTime'][()]
                        else:
                            if read_mode == 1:
                                formation_times = snappt[
                                    'StellarFormationTime'][()]
                                formation_times = formation_times[region_inds]
                                gc.collect()
                            elif read_mode == 2:
                                formation_times = snappt[
                                    'StellarFormationTime']
                                formation_times = formation_times[
                                    region_inds_bool][inv]
                            formation_times = np.split(
                                formation_times, np.cumsum(region_lens))[:-1]
                    else:
                        formation_times = None

                return ids, coords, vels, masses, metallicities, \
                    formation_times

            if self.npool is None or self.npool == 1:
                snapdata = []
                for fi in range(len(fnames)):
                    snapdata.append(read_files(fi))
            else:
                print('Starting multiprocessing pool with {} processes'.format(
                    self.npool))
                snapdata = Pool(self.npool).map(
                    read_files, np.arange(len(fnames)))

            def stack(index):
                if region_positions is None:
                    return np.concatenate([x[index] for x in snapdata]), None
                else:
                    regions = []
                    for ri in range(len(region_positions)):
                        regions.append(
                            np.concatenate([x[index][ri] for x in snapdata]))
                    lens = [0] + [len(region) for region in regions]
                    return np.concatenate(regions), np.cumsum(lens)

            region_offsets = None
            if load_ids:
                self.ids, region_offsets = stack(0)
            if load_coords:
                self.coordinates, region_offsets = stack(1)
            if load_vels:
                self.velocities, region_offsets = stack(2)
                self.velocities *= np.sqrt(1 + self.redshift)
            if load_masses:
                if (not isinstance(snapdata[0][3], np.ndarray)) and \
                        (not isinstance(snapdata[0][3], list)):
                    self.masses = snapdata[0][3]
                else:
                    self.masses, region_offsets = stack(3)
            if snapdata[0][4] is not None:
                self.metallicities, region_offsets = stack(4)
            if snapdata[0][5] is not None:
                self.formation_times, region_offsets = stack(5)
            if region_offsets is not None:
                self.region_offsets = region_offsets[:-1]
                self.region_slices = [
                    slice(*x) for x in list(
                        zip(region_offsets[:-1], region_offsets[1:]))]

        if self.snapshot_format == 3:
            with h5py.File(filenames[0], 'r') as snap:
                self.read_parameters(snap, self.snapshot_format)
        else:
            with open(filenames[0], 'rb') as snap:
                self.read_parameters(snap, self.snapshot_format)

        if not np.any(
                np.array([load_ids, load_coords, load_vels, load_masses])):
            return

        if region_positions is not None:
            region_positions = np.atleast_2d(region_positions)
            region_radii = np.atleast_1d(region_radii)

        if self.snapshot_format == 3:
            read_hdf5_snapshot(filenames)
        else:
            read_binary_snapshot(filenames)

        return


class GadgetCatalogue(GadgetBox):

    def __init__(self, path, catalogue_filename, snapshot_number,
                 particle_type=None, load_halo_data=True,
                 unit_length_in_cm=None, unit_mass_in_g=None,
                 unit_velocity_in_cm_per_s=None, catalogue_format=None,
                 verbose=True):

        super().__init__(unit_length_in_cm, unit_mass_in_g,
                         unit_velocity_in_cm_per_s)

        self.catalogue_path = path
        self.catalogue_filename = catalogue_filename
        self.snapshot_number = snapshot_number
        self.particle_type = particle_type

        if catalogue_format is None:
            if catalogue_filename.split('.')[-1] == 'hdf5':
                self.catalogue_format = 3
            else:
                self.catalogue_format = 2
        else:
            self.catalogue_format = catalogue_format

        catalogue_files = np.sort(glob.glob(path + '/{}'.format(
            catalogue_filename.format('%03d' % snapshot_number))))
        ncat = len(catalogue_files)
        if ncat > 0:
            self.has_cat = True
            if ncat > 1:  # order files by sub-file number
                catalogue_files = self.reorder_subfiles(
                    catalogue_filename, catalogue_files)
            if verbose:
                print('Found {} halo catalogue file(s) for snapshot {} in '
                      'directory {}'.format(ncat, snapshot_number, path))
                start = time.time()
            self.group, self.halo = self.read_halos(
                catalogue_files, load_halo_data)
            if verbose:
                print("...Loaded in {} seconds\n".format(
                    round(time.time() - start, 4)))
        if ncat == 0 or self.group is None:
            self.has_cat = False
            warnings.warn('No catalogue files found!')

    def read_halos(self, filenames, load_halo_data):

        with h5py.File(filenames[0], 'r') as halo_cat:
            if not hasattr(self, 'redshift'):
                self.read_parameters(halo_cat, self.catalogue_format)
            self.number_of_groups = halo_cat['Header'].attrs['Ngroups_Total']
            self.number_of_halos = halo_cat['Header'].attrs['Nsubhalos_Total']

        if not load_halo_data:
            return {}, {}

        group = {}
        halo = {}

        group_keys_float = ['R_200crit', 'R_500crit', 'M_200crit', 'M_500crit',
                            'V_200crit', 'V_500crit', 'A_200crit', 'A_500crit',
                            'R_200mean', 'M_200mean', 'V_200mean', 'A_200mean',
                            'R_200tophat', 'M_200tophat', 'V_200tophat',
                            'A_200tophat', 'mass']
        group_keys_int = ['number_of_particles', 'offset', 'first_subhalo',
                          'number_of_subhalos']
        group_keys_vec3 = ['position_of_minimum_potential', 'center_of_mass',
                           'velocity']
        halo_keys_float = ['mass', 'halfmass_radius']
        halo_keys_int = ['number_of_particles', 'offset', 'ID_most_bound',
                         'group_number', 'rank_in_group']
        halo_keys_vec3 = ['position_of_minimum_potential', 'center_of_mass',
                          'velocity']
        for gkey in group_keys_float:
            group[gkey] = np.empty(self.number_of_groups)
        for gkey in group_keys_int:
            group[gkey] = np.empty(self.number_of_groups, dtype=np.int64)
        for gkey in group_keys_vec3:
            group[gkey] = np.empty((self.number_of_groups, 3))
        for hkey in halo_keys_float:
            halo[hkey] = np.empty(self.number_of_halos)
        for hkey in halo_keys_int:
            halo[hkey] = np.empty(
                self.number_of_halos, dtype=np.int64)
        for hkey in halo_keys_vec3:
            halo[hkey] = np.empty((self.number_of_halos, 3))

        gidx, hidx = 0, 0
        for filename in filenames:

            with h5py.File(filename, 'r') as halo_cat:

                ngroups = int(halo_cat['Header'].attrs['Ngroups_ThisFile'])
                nhalos = int(halo_cat['Header'].attrs['Nsubhalos_ThisFile'])
                gslice = slice(gidx, gidx + ngroups)
                gidx += ngroups
                hslice = slice(hidx, hidx + nhalos)
                hidx += nhalos

                if ngroups == 0:
                    continue

                config_options = list(halo_cat['Config'].attrs)

                group['mass'][gslice] = halo_cat['Group']['GroupMassType'][()][
                    :, self.particle_type]
                group['position_of_minimum_potential'][gslice] = halo_cat[
                    'Group']['GroupPos'][()]
                if 'GroupCM' in list(halo_cat['Group'].keys()):
                    group['center_of_mass'][gslice] = halo_cat['Group'][
                        'GroupCM'][()]
                else:
                    group['center_of_mass'][gslice] = np.nan
                group['velocity'][gslice] = halo_cat['Group']['GroupVel'][()] / \
                    self.scale_factor**2
                group['number_of_particles'][gslice] = halo_cat['Group'][
                    'GroupLenType'][()][:, self.particle_type]
                group['offset'][gslice] = halo_cat['Group']['GroupOffsetType'][()][
                    :, self.particle_type]

                for ref in [('Crit', 'crit'), ('Mean', 'mean'),
                            ('TopHat', 'tophat')]:
                    R_200 = halo_cat['Group']['Group_R_{}200'.format(ref[0])][()]
                    M_200 = halo_cat['Group']['Group_M_{}200'.format(ref[0])][()]
                    group['R_200{}'.format(ref[1])][gslice] = R_200
                    group['M_200{}'.format(ref[1])][gslice] = M_200

                    np.seterr(divide='ignore', invalid='ignore')
                    V_200 = np.sqrt(self.gravitational_constant * M_200 / R_200)
                    group['V_200{}'.format(ref[1])][gslice] = V_200
                    group['A_200{}'.format(ref[1])][gslice] = V_200**2 / R_200
                    np.seterr(divide='warn', invalid='warn')

                R_500 = halo_cat['Group']['Group_R_Crit500'][()]
                M_500 = halo_cat['Group']['Group_M_Crit500'][()]
                group['R_500crit'][gslice] = R_500
                group['M_500crit'][gslice] = M_500
                V_500 = np.sqrt(self.gravitational_constant * M_500 / R_500)
                group['V_500crit'][gslice] = V_500
                group['A_500crit'][gslice] = V_500 ** 2 / R_500

                if 'SUBFIND' or 'SUBFIND_HBT' in config_options:

                    group['first_subhalo'][gslice] = halo_cat['Group'][
                        'GroupFirstSub'][()]
                    group['number_of_subhalos'][gslice] = halo_cat['Group'][
                        'GroupNsubs'][()]

                    halo['mass'][hslice] = halo_cat[
                        'Subhalo']['SubhaloMassType'][()][:, self.particle_type]
                    halo['center_of_mass'][hslice] = halo_cat['Subhalo'][
                        'SubhaloCM'][()]
                    halo['position_of_minimum_potential'][hslice] = halo_cat[
                        'Subhalo']['SubhaloPos'][()]
                    halo['velocity'][hslice] = halo_cat['Subhalo'][
                        'SubhaloVel'][()] / self.scale_factor
                    halo['halfmass_radius'][hslice] = halo_cat['Subhalo'][
                        'SubhaloHalfmassRadType'][()][:, self.particle_type]
                    halo['number_of_particles'][hslice] = halo_cat[
                        'Subhalo']['SubhaloLenType'][()][:, self.particle_type]
                    halo['offset'][hslice] = halo_cat[
                        'Subhalo']['SubhaloOffsetType'][()][:, self.particle_type]
                    halo['ID_most_bound'][hslice] = halo_cat['Subhalo'][
                        'SubhaloIDMostbound'][()]
                    halo['group_number'][hslice] = halo_cat['Subhalo'][
                        'SubhaloGroupNr'][()]
                    halo['rank_in_group'][hslice] = halo_cat['Subhalo'][
                        'SubhaloRankInGr'][()]

        if np.all(np.isnan(group['center_of_mass'].flatten())):
            group['center_of_mass'] = None

        if gidx == 0:
            return {}, {}
        else:
            return group, halo


class VelociraptorCatalogue:

    def __init__(self, path, catalogue_filename, snapshot_number,
                 particle_type=None, thidv=int(1e12), load_halo_data=True,
                 load_halo_particle_ids=False, verbose=True):

        self.catalogue_path = path
        self.catalogue_filename = catalogue_filename
        self.snapshot_number = snapshot_number
        self.particle_type = particle_type
        self.thidv = thidv

        catalogue_files = glob.glob(
            path + '/{}'.format(catalogue_filename.format(
                '%03d' % snapshot_number, 'catalog_groups')))
        ncat = len(catalogue_files)
        if ncat > 0:
            self.has_cat = True
            if ncat > 1:
                subnums = [int(catfile.split('.')[-1]) for catfile in
                           catalogue_files]
                catalogue_files = np.array(catalogue_files)[
                    np.argsort(subnums)]
            if verbose:
                print('Found {} halo catalogue file(s) for snapshot {} in '
                      'directory {}'.format(ncat, snapshot_number, path))
                start = time.time()
            self.halo = self.read_halos(
                catalogue_files, load_halo_data, load_halo_particle_ids)
            if verbose:
                print("...Loaded in {} seconds\n".format(
                    round(time.time() - start, 4)))
        else:
            self.has_cat = False
            warnings.warn('No catalogue files found!')

    def read_params(self):

        param_data = []
        for fext in ['siminfo', 'units']:
            fname = glob.glob(self.catalogue_path + '/*{}*{}'.format(
                self.snapshot_number, fext))[0]
            data = []
            with open(fname, 'r') as fdata:
                for line in fdata:
                    s = line.split()
                    if len(s) > 2:
                        data.append([s[0], s[2]])
            param_data.append(data)

        siminfo, units = param_data
        siminfo, units = np.array(siminfo), np.array(units)
        siminfo = dict(zip(siminfo[:, 0], siminfo[:, 1].astype(float)))
        units = dict(zip(units[:, 0], units[:, 1].astype(float)))

        if siminfo['Cosmological_Sim'] == 1:
            self.scale_factor = siminfo['ScaleFactor']
            self.redshift = (1 / self.scale_factor) - 1
        self.Omega0 = siminfo['Omega_m']
        self.OmegaBaryon = siminfo['Omega_b']
        self.OmegaDM = siminfo['Omega_cdm']
        self.OmegaLambda = siminfo['Omega_Lambda']
        self.h = siminfo['h_val']
        self.hubble = siminfo['Hubble_unit']
        self.hubble_constant = self.h * self.hubble
        if hasattr(self, 'redshift'):
            self.hubble_parameter = hubble_parameter(
                self.redshift, self.hubble_constant,
                self.Omega0, self.OmegaLambda, 0)
        self.unit_length_in_kpc = units['Length_unit_to_kpc']
        self.unit_mass_in_solar_masses = units['Mass_unit_to_solarmass']
        self.unit_velocity_in_km_per_s = units['Velocity_unit_to_kms']

        self.gravitational_constant = siminfo['Gravity']

        if hasattr(self, 'hubble_parameter'):
            self.critical_density = 3 * (self.hubble_parameter / self.h)**2 / \
                (8 * np.pi * self.gravitational_constant)

    def read_halos(self, catalogue_files, load_halo_data,
                   load_halo_particle_ids):

        self.read_params()

        if not load_halo_data:
            return {}

        halo = {}

        catfile = catalogue_files[0]
        with h5py.File(catfile, 'r') as halo_cat:
            nhalos = halo_cat['Total_num_of_groups'][()][0]

        halo_keys_float = ['R_200crit', 'R_200mean', 'R_BN98', 'M_200crit',
                           'M_200mean', 'M_BN98', 'M_FOF', 'M_exclusive',
                           'halfmass_radius']
        halo_keys_int = ['halo_ID', 'ID_most_bound_particle', 'offset',
                         'number_of_particles', 'parent_halo_ID',
                         'rank_in_parent', 'number_of_subhalos',
                         'structure_type']
        halo_keys_vec3 = ['center_of_mass', 'position_of_most_bound_particle',
                          'position_of_minimum_potential',
                          'velocity_of_center_of_mass',
                          'velocity_of_most_bound_particle',
                          'velocity_of_minimum_potential']
        for hkey in halo_keys_float:
            halo[hkey] = np.empty(nhalos)
        for hkey in halo_keys_int:
            halo[hkey] = np.empty(nhalos, dtype=np.int64)
        for hkey in halo_keys_vec3:
            halo[hkey] = np.empty((nhalos, 3))
        if load_halo_particle_ids:
            halo['particle_IDs'] = []

        hidx = 0
        for catfile in catalogue_files:

            with h5py.File(catfile, 'r') as cat_groups:

                nhalos = int(cat_groups['Num_of_groups'][()][0])
                hslice = slice(hidx, hidx + nhalos)
                hidx += nhalos

                halo['number_of_subhalos'][hslice] = cat_groups[
                    'Number_of_substructures_in_halo'][()]

                hoffset = cat_groups['Offset'][()]

                parents = cat_groups['Parent_halo_ID'][()]
                halo['parent_halo_ID'][hslice] = parents
                rank = np.zeros(len(parents), dtype=np.int32)
                counts = np.unique(
                    parents[parents > -1], return_counts=True)[1]
                rankidx = len(np.argwhere(parents == -1))
                for k, c in enumerate(counts):
                    rank[rankidx:rankidx+c] = np.arange(1, c+1)
                    rankidx += c
                halo['rank_in_parent'][hslice] = rank

            catfile_particles = catfile.replace(
                'catalog_groups', 'catalog_particles')
            with h5py.File(catfile_particles, 'r') as cat_part:
                npart = cat_part['Num_of_particles_in_groups'][()][0]
                if len(hoffset) == 1:
                    halo['number_of_particles'][hslice] = np.array([npart])
                else:
                    hlen = hoffset[1:] - hoffset[:-1]
                    halo['number_of_particles'][hslice] = np.append(
                        hlen, npart - hoffset[-1])
                if load_halo_particle_ids:
                    halo['particle_IDs'].append(cat_part['Particle_IDs'][()])

            catfile_props = catfile.replace(
                'catalog_groups', 'properties')
            with h5py.File(catfile_props, 'r') as cat_props:
                haloids = cat_props['ID'][()]
                halo['halo_ID'][hslice] = haloids
                halo['structure_type'][hslice] = cat_props[
                    'Structuretype'][()]
                R_200 = cat_props['R_200crit'][()] * self.h / self.scale_factor
                M_200 = cat_props['Mass_200crit'][()] * self.h
                halo['R_200crit'][hslice] = R_200
                halo['M_200crit'][hslice] = M_200
                halo['R_200mean'][hslice] = cat_props['R_200mean'][()] * \
                    self.h / self.scale_factor
                halo['M_200mean'][hslice] = cat_props['Mass_200mean'][()] * \
                    self.h
                halo['R_BN98'][hslice] = cat_props['R_BN98'][()] * self.h / \
                    self.scale_factor
                halo['M_BN98'][hslice] = cat_props['Mass_BN98'][()] * \
                    self.h
                halo['M_FOF'][hslice] = cat_props['Mass_FOF'][()] * self.h
                halo['M_exclusive'][hslice] = cat_props['Mass_tot'][()] * \
                    self.h
                cmx, cmy, cmz = \
                    cat_props['Xc'][()] * self.h / self.scale_factor, \
                    cat_props['Yc'][()] * self.h / self.scale_factor, \
                    cat_props['Zc'][()] * self.h / self.scale_factor
                halo['center_of_mass'][hslice] = np.vstack((cmx, cmy, cmz)).T
                mbpx, mbpy, mbpz = \
                    cat_props['Xcmbp'][()] * self.h / self.scale_factor, \
                    cat_props['Ycmbp'][()] * self.h / self.scale_factor, \
                    cat_props['Zcmbp'][()] * self.h / self.scale_factor
                halo['position_of_most_bound_particle'][hslice] = np.vstack(
                    (mbpx, mbpy, mbpz)).T
                mpx, mpy, mpz = \
                    cat_props['Xcminpot'][()] * self.h / self.scale_factor, \
                    cat_props['Ycminpot'][()] * self.h / self.scale_factor, \
                    cat_props['Zcminpot'][()] * self.h / self.scale_factor
                halo['position_of_minimum_potential'][hslice] = np.vstack(
                    (mpx, mpy, mpz)).T
                velcmx, velcmy, velcmz = cat_props['VXc'][()], \
                    cat_props['VYc'][()], cat_props['VZc'][()]
                halo['velocity_of_center_of_mass'][hslice] = np.vstack(
                    (velcmx, velcmy, velcmz)).T
                velmbpx, velmbpy, velmbpz = cat_props['VXcmbp'][()], \
                    cat_props['VYcmbp'][()], cat_props['VZcmbp'][()]
                halo['velocity_of_most_bound_particle'][hslice] = np.vstack(
                    (velmbpx, velmbpy, velmbpz)).T
                velmpx, velmpy, velmpz = cat_props['VXcminpot'][()], \
                    cat_props['VYcminpot'][()], cat_props['VZcminpot'][()]
                halo['velocity_of_minimum_potential'][hslice] = np.vstack(
                    (velmpx, velmpy, velmpz)).T
                halo['ID_most_bound_particle'][hslice] = cat_props['ID_mbp'][
                    ()]
                halo['halfmass_radius'][hslice] = cat_props['R_HalfMass'][()] \
                    * self.h

                self.snapshot_number = int(haloids[0] / self.thidv)

        if load_halo_particle_ids:
            halo['particle_IDs'] = np.hstack(halo['particle_IDs'])
            halo['offset'][:] = np.array(
                [0] + list(np.cumsum(halo['number_of_particles']))[:-1])

        return halo


class AHFCatalogue:

    def __init__(self, path, catalogue_filename, snapshot_number,
                 particle_type=None, load_halo_data=True,
                 load_halo_particle_ids=False, verbose=True):

        self.catalogue_path = path
        self.catalogue_filename = catalogue_filename
        self.snapshot_number = snapshot_number
        self.particle_type = particle_type

        catalogue_files = glob.glob(path + '/{}'.format(
            catalogue_filename.format('%03d' % snapshot_number, 'halos')))
        ncat = len(catalogue_files)
        if ncat > 0:
            self.has_cat = True
            if verbose:
                print('Found {} halo catalogue file(s) for snapshot {} in '
                      'directory {}'.format(ncat, snapshot_number, path))
                start = time.time()
            self.halo = self.read_halos(
                catalogue_files, load_halo_data, load_halo_particle_ids)
            if verbose:
                print("...Loaded in {} seconds\n".format(
                    round(time.time() - start, 4)))
        if ncat == 0 or self.halo is None:
            self.has_cat = False
            warnings.warn('No catalogue files found!')

    def read_halos(self, filenames, load_halo_data, load_halo_particle_ids):

        if not load_halo_data:
            return {}

        halo = {}

        linecounts = []
        for filename in filenames:
            with open(filename, 'r') as f:
                linecounts.append(sum(1 for _ in f)-1)
        self.number_of_halos = np.sum(linecounts)
        if self.number_of_halos == 0:  # empty catalogue
            return
        else:
            keys_int64 = ['halo_ID', 'host_ID']
            keys_int = ['number_of_subhalos', 'number_of_particles']
            keys_float = ['virial_mass', 'virial_radius',
                          'most_bound_particle_offset',
                          'center_of_mass_offset']
            keys_vec = ['position_of_density_peak', 'velocity',
                        'angular_momentum']
            for key in keys_int64:
                halo[key] = np.empty(self.number_of_halos, dtype=np.int64)
            for key in keys_int:
                halo[key] = np.empty(self.number_of_halos, dtype=np.int32)
            for key in keys_float:
                halo[key] = np.empty(self.number_of_halos, dtype=np.float32)
            for key in keys_vec:
                halo[key] = np.empty(
                    (self.number_of_halos, 3), dtype=np.float32)
        halo['particle_IDs'] = []

        inds = np.insert(np.cumsum(linecounts), 0, 0)
        slices = [slice(start, end) for start, end in zip(inds[:-1], inds[1:])]
        for filename, sl in zip(filenames, slices):

            halo['halo_ID'][sl] = np.loadtxt(
                filename, dtype=np.uint64, skiprows=0, usecols=0)
            halo['host_ID'][sl] = np.loadtxt(
                filename, dtype=np.uint64, skiprows=0, usecols=1)

            data = np.atleast_2d(np.loadtxt(filename, skiprows=0))[:, 2:]
            halo['number_of_subhalos'][sl] = data[:, 0].astype(np.int32)
            halo['virial_mass'][sl] = data[:, 1]
            halo['number_of_particles'][sl] = data[:, 2].astype(np.int32)
            halo['position_of_density_peak'][sl] = data[:, 3:6]
            halo['velocity'][sl] = data[:, 6:9]
            halo['virial_radius'][sl] = data[:, 9]
            halo['most_bound_particle_offset'][sl] = data[:, 12]
            halo['center_of_mass_offset'][sl] = data[:, 13]
            halo['angular_momentum'][sl] = data[:, 19:22]

            if load_halo_particle_ids:
                filename = filename.replace('halos', 'particles')
                particle_ids = np.loadtxt(filename, dtype=np.uint64,
                                          skiprows=1, usecols=0)
                particle_types = np.loadtxt(filename, dtype=np.uint64,
                                            skiprows=1, usecols=1)
                for i, n in enumerate(range(self.number_of_halos-1), 1):
                    npart = halo['number_of_particles'][n]
                    start = np.sum(
                        np.array(halo['number_of_particles'][:n+1])) - npart

                    ptypes = particle_types[start+i:start+npart+i]
                    ptype_inds = np.argwhere(
                        ptypes == self.particle_type).flatten()
                    pids = (particle_ids[start+i:start+npart+i])[ptype_inds]
                    halo['particle_IDs'].append(pids)

        return halo
