import numpy as np
from scipy.signal import savgol_filter
from healpy.pixelfunc import npix2nside, ang2pix
from warnings import warn

from simtools.utils import vector_norm, simple_derivative, churazov_smooth
from simtools.quantities import radial_velocity, azimuthal_velocity, \
    velocity_dispersion


def bin_halo(coords, radial_bins, radius_limits=None, center=False,
             log_bins=True, n_angular_bins=1):

    if isinstance(radial_bins, int):

        if radius_limits is None:
            r = vector_norm(coords)
            if log_bins:
                r = r[np.argwhere(r > 0.0).flatten()]
            r1, r2 = min(r), max(r)
            radius_limits = (r1, r2)

        if center:
            if radius_limits is None:
                raise ValueError(
                    "Must supply radius limits when `center` is True.")
            r1, r2 = radius_limits
            if log_bins:
                log_bin_width = (np.log10(r2) - np.log10(r1)) / (
                            radial_bins - 1)
                r1_ = 10**(np.log10(r1) - log_bin_width / 2)
                r2_ = 10**(np.log10(r2) + log_bin_width / 2)
                radius_limits = (r1_, r2_)
            else:
                bin_width = (r2 - r1) / (radial_bins - 1)
                r1_ = r1 - bin_width / 2
                r2_ = r2 + bin_width / 2
                radius_limits = (r1_, r2_)

        n_radial_bins = radial_bins
        if log_bins:
            radial_bins = 10**np.linspace(
                *np.log10(radius_limits), radial_bins+1)
        else:
            radial_bins = np.linspace(*radius_limits, radial_bins+1)
    else:

        if center:
            warn("Bin centering is not applied when `radial_bins` is provided "
                 "as an array.")
        if radius_limits is not None:
            warn("Provided `radius_limits` are ignored when `radial_bins` is "
                 "provided as an array.")

        n_radial_bins = len(radial_bins) - 1
        radius_limits = (min(radial_bins), max(radial_bins))

    r = vector_norm(coords)
    inside_rlims = np.argwhere(
        (r >= radius_limits[0]) & (r <= radius_limits[1])).flatten()
    r = r[inside_rlims]

    if n_angular_bins > 1:
        x, y, z = np.copy(coords)[inside_rlims].T
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        pixels = ang2pix(npix2nside(n_angular_bins), theta, phi)

    binds = []
    for na in range(n_angular_bins):

        if n_angular_bins > 1:
            inside_angle = np.argwhere(pixels == na).flatten()
            ra = r[inside_angle]
            rad_digits = np.digitize(ra, radial_bins)-1
        else:
            inside_angle = np.arange(len(r))
            rad_digits = np.digitize(r, radial_bins)-1

        binds_angle = []
        for nr in range(n_radial_bins):

            inside_shell = np.argwhere(rad_digits == nr).flatten()
            binds_angle.append(inside_rlims[inside_angle[inside_shell]])

        binds.append(binds_angle)

    if n_angular_bins == 1:
        binds = binds[0]

    log_radial_bins = np.log10(radial_bins)
    logrx_offset = log_radial_bins + \
        (log_radial_bins[1] - log_radial_bins[0]) / 2
    rcenters = 10**logrx_offset[:-1]

    return binds, radial_bins, rcenters


def calc_density_profile(masses, coords=None, radial_bins=None,
                         radius_limits=None, center=None, log_bins=None,
                         n_angular_bins=None, binned_halo=None):

    if binned_halo is None:
        binds, redges, rcenters = bin_halo(
            coords, radial_bins, radius_limits, center, log_bins,
            n_angular_bins)
    else:
        binds, redges, rcenters = binned_halo

    uses_angular_binning = isinstance(binds[0], list)
    n_angular_bins = len(binds) if uses_angular_binning else 1

    shell_vols = (4 * np.pi / 3) * (redges[1:]**3 - redges[:-1]**3)
    bin_vols = shell_vols / n_angular_bins

    def calc_profile(binds_subset):
        if isinstance(masses, np.ndarray):
            bin_masses = np.array(
                [np.sum(masses[inds]) for inds in binds_subset])
        else:
            bin_masses = np.array(
                [masses*len(inds) for inds in binds_subset])

        return bin_masses / bin_vols

    if uses_angular_binning:
        density_profiles = []
        for binds_angle in binds:
            density_profiles.append(calc_profile(binds_angle))
        return rcenters, np.median(np.array(density_profiles), axis=0)
    else:
        return rcenters, calc_profile(binds)


def calc_log_density_slope_profile(density_profile, r=None, window_length=1,
                                   apply_filter=False, handle_edges=False,
                                   width=None, **savgol_kwargs):

    if apply_filter:

        dsp = savgol_filter(
            np.log10(density_profile), window_length=window_length, deriv=1,
            **savgol_kwargs)

        if handle_edges:
            nedge = int((window_length - 1) / 2)
            where_zero = np.argwhere(density_profile == 0).flatten()
            edge_inner = np.argwhere(density_profile != 0.0).flatten()[0]
            if len(where_zero) != 0:
                where_zero_after_nonzero = where_zero[where_zero > edge_inner]
                if len(where_zero_after_nonzero) > 0:
                    edge_outer = where_zero_after_nonzero[0]-1
                else:
                    edge_outer = len(density_profile) - 1
            else:
                edge_outer = len(density_profile) - 1
            filter_edge_left = slice(edge_inner, edge_inner+nedge+1)
            filter_edge_right = slice(edge_outer-nedge, edge_outer+1)

            r_edge_left = r[filter_edge_left]
            r_edge_right = r[filter_edge_right]
            profile_edge_left = density_profile[filter_edge_left]
            profile_edge_right = density_profile[filter_edge_right]

            if width is None:
                width = dict(savgol_kwargs)['delta']
            smoothed_left = churazov_smooth(
                r_edge_left, profile_edge_left, width)
            smoothed_right = churazov_smooth(
                r_edge_right, profile_edge_right, width)

            dsp_edge_left = simple_derivative(
                np.log10(r[filter_edge_left]), np.array(smoothed_left), 1)
            dsp_edge_right = simple_derivative(
                np.log10(r[filter_edge_right]), np.array(smoothed_right), 1)
            dsp[slice(edge_inner, edge_inner+nedge)] = dsp_edge_left
            dsp[slice(edge_outer-nedge, edge_outer)] = dsp_edge_right

        return dsp

    else:

        return simple_derivative(
            np.log10(r), np.log10(density_profile), window_length)


def calc_mass_profile(masses, coords=None, radii=None,
                      radius_limits=None, log_bins=True, binned_halo=None):

    if binned_halo is None:
        if isinstance(radii, int):
            if log_bins:
                radii = 10**np.linspace(*np.log10(radius_limits), radii)
            else:
                radii = np.linspace(*radius_limits, radii)
        binds, redges, _ = bin_halo(coords, np.insert(radii, 0, 0.0))
    else:
        binds, redges, _ = binned_halo

    if isinstance(masses, np.ndarray):
        bin_masses = np.array([np.sum(masses[inds]) for inds in binds])
    else:
        bin_masses = np.array([masses*len(inds) for inds in binds])

    return redges[1:], np.cumsum(bin_masses)


def calc_circular_velocity_profile(masses, gravitational_constant, coords=None,
                                   radii=None, radius_limits=None,
                                   log_bins=None, binned_halo=None):

    r, mass_profile = calc_mass_profile(
        masses, coords, radii, radius_limits, log_bins,
        binned_halo)

    return r, np.sqrt(gravitational_constant * mass_profile / r)


def calc_radial_velocity_profile(vels, coords, radial_bins=None,
                                 radius_limits=None, center=None,
                                 log_bins=None, n_angular_bins=None,
                                 binned_halo=None):

    if binned_halo is None:
        binds, redges, rcenters = bin_halo(
            coords, radial_bins, radius_limits, center, log_bins,
            n_angular_bins)
    else:
        binds, redges, rcenters = binned_halo

    def calc_profile(binds_subset):
        return np.array([
            np.mean(radial_velocity(coords[inds], vels[inds])[0]) for inds in
            binds_subset])

    if isinstance(binds[np.argwhere(
            np.array([len(x) for x in binds]) > 0).flat[0]][0], np.ndarray):
        radial_velocity_profiles = np.array(
            [calc_profile(binds_angle) for binds_angle in binds])
        return rcenters, np.median(radial_velocity_profiles, axis=0)
    else:
        return rcenters, calc_profile(binds)


def calc_azimuthal_velocity_profile(vels, coords, radial_bins=None,
                                    radius_limits=None, center=None,
                                    log_bins=None, n_angular_bins=None,
                                    binned_halo=None):

    if binned_halo is None:
        binds, redges, rcenters = bin_halo(
            coords, radial_bins, radius_limits, center, log_bins,
            n_angular_bins)
    else:
        binds, redges, rcenters = binned_halo

    def calc_profile(binds_subset):
        return np.array([
            np.mean(azimuthal_velocity(coords[inds], vels[inds])[0]) for inds
            in binds_subset])

    if isinstance(binds[np.argwhere(
            np.array([len(x) for x in binds]) > 0).flat[0]][0], np.ndarray):
        azimuthal_velocity_profiles = np.array(
            [calc_profile(binds_angle) for binds_angle in binds])
        return rcenters, np.median(
            np.array(azimuthal_velocity_profiles), axis=0)
    else:
        return rcenters, calc_profile(binds)


def calc_velocity_dispersion_profile(vels, masses=None, coords=None,
                                     radial_bins=None, radius_limits=None,
                                     center=None, log_bins=None,
                                     binned_halo=None):

    if binned_halo is None:
        binds, redges, rcenters = bin_halo(
            coords, radial_bins, radius_limits, center, log_bins,
            n_angular_bins=1)
    else:
        binds, redges, rcenters = binned_halo

    profile, m = [], None
    for inds in binds:
        if masses is not None and hasattr(masses, "__len__"):
            m = masses[inds]
        profile.append(velocity_dispersion(vels[inds], m))

    return rcenters, np.array(profile)
