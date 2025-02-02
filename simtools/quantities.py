import numpy as np
from simtools.utils import vector_norm
from scipy.interpolate import interp1d


def Omega(z, Omega_m, Omega_Lambda, Omega_k=0):

    Omz = Omega_m * (1 + z)**3
    Okz = Omega_k * (1 + z)**2
    OLz = Omega_Lambda
    Esq = Omz + Okz + OLz

    Om = np.array([Omz, OLz, Okz])
    return Om / Esq


def overdensity_BN98(z, Omega_m):

    x = Omega(z, Omega_m, 1-Omega_m, 0)[0] - 1

    return 18 * np.pi**2 + 82 * x - 39 * x**2


def hubble_parameter(z, H0, Omega_m, Omega_Lambda, Omega_k=0):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 +
                        Omega_k * (1 + z)**2 +
                        Omega_Lambda)


def add_hubble_flow(velocity, position, z, H0, **Omega):
    return velocity + position * hubble_parameter(z, H0, **Omega)


def radial_velocity(coords, vels, return_radii=False):
    rads, rhat = vector_norm(
        coords, return_magnitude=True, return_unit_vectors=True)
    if return_radii:
        return np.einsum('...i,...i', vels, rhat), rhat, rads
    else:
        return np.einsum('...i,...i', vels, rhat), rhat


def azimuthal_velocity(coords, vels):
    vr, rhat = radial_velocity(coords, vels)
    vaz = vels - vr[:, np.newaxis] * rhat
    return vector_norm(vaz, return_magnitude=True, return_unit_vectors=True)


def velocity_dispersion(vels, masses=None):

    vels_ = vels - np.mean(vels, axis=0)
    if masses is None or not hasattr(masses, "__len__"):
        return np.sqrt(np.mean(np.einsum('...i,...i', vels_, vels_)))
    else:
        return np.sqrt(
            np.mean(masses * np.einsum('...i,...i', vels_, vels_)) *
            len(masses) / np.sum(masses))


def calc_specific_angular_momentum(x, v, npart):
    return np.sum(np.cross(x, v), axis=0) / npart


def estimate_overdensity_mass_and_radius_from_profile(mass_profile, r,
                                                      overdensities):

    overdensity_profile = mass_profile / (4 * np.pi * r**3 / 3)
    fm = interp1d(overdensity_profile, mass_profile)
    fr = interp1d(overdensity_profile, r)

    masses, radii = [], []
    for overdensity in overdensities:
        masses.append(fm(overdensity))
        radii.append(fr(overdensity))

    return np.array(masses), np.array(radii)
