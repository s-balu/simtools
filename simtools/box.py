import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sphviewer.tools import QuickView

from simtools.utils import pretty_print


class Snapshot:

    def __init__(self, snap_reader=None, snap_options=None, snap_obj=None):
        """
        """

        if snap_obj is None:
            if snap_reader is not None:
                snap_obj = snap_reader(**snap_options)
            else:
                raise ValueError('Must provide a snapshot reader and options, '
                                 'or a snapshot object.')

        for attr in dir(snap_obj):
            if '__' not in attr:
                setattr(self, attr, getattr(snap_obj, attr))

    def plot_box(self, projection='xy', center=(0, 0, 0), extent=None,
                 length_unit=1e3, sphviewer=False, bins=(1000, 1000),
                 cmap=None, log=True, title=None, figsize=(6, 6), dpi=300,
                 save=False, savefile=None, return_fig=False):
        """
        """

        #length_norm = self.unit_length / (length_unit * self.cm_per_kpc)
        length_norm = self.unit_length_in_cm / (self.cm_per_kpc)

        #width = self.box_size * self.scale_phys / 2
        width = self.box_size * length_norm / 2
        coords = self.coords - np.array(center)  # - np.array([width]*3)
        if type(extent) == float or type(extent) == int:
            extent = [extent, extent]
        if extent is None:
            extent = [width, width]
        else:
            if type(bins) == int:
                bins = [bins, bins]
            bins1 = np.linspace(-extent[0], extent[0], bins[0])
            bins2 = np.linspace(-extent[1], extent[1], bins[1])
            bins = [bins1, bins2]
        order = []
        for p in projection:
            order.append(
                0 if p == 'x' else 1 if p == 'y' else 2 if p == 'z' else 3)
        order.append(list(set([0, 1, 2]) - set(order))[0])
        coords[:, [0, 1, 2]] = coords[:, order]
        f_box = plt.subplots(figsize=figsize, dpi=dpi)
        if cmap is None:
            cmaps = [plt.cm.magma, plt.cm.inferno, plt.cm.twilight_shifted,
                     plt.cm.twilight_shifted, plt.cm.cividis,
                     plt.cm.twilight_shifted]
            cmap = cmaps[self.particle_type]
            cmap.set_bad('k', 1)
        if sphviewer:
            qv_parallel = QuickView(coords*length_norm, r='infinity',
                                    plot=False, x=0, y=0, z=0,
                                    extent=np.array([-extent[0], extent[0],
                                                     -extent[1], extent[1]]) *
                                    length_norm)
            norm = mpl.colors.LogNorm() if log else mpl.colors.Normalize()
            f_box[1].imshow(qv_parallel.get_image(),
                            extent=qv_parallel.get_extent(), cmap=cmap,
                            origin='lower', norm=norm)
        else:
            norm = mpl.colors.LogNorm() if log else mpl.colors.Normalize()
            f_box[1].hist2d(coords[:, 0]*length_norm,
                            coords[:, 1]*length_norm, bins=bins,
                            cmap=cmap, norm=norm)
        f_box[1].set_xlabel(r'${}\,\,(h^{{-1}}\,\,{{\rm Mpc}})$'.format(
            projection[0]))
        f_box[1].set_ylabel(r'${}\,\,(h^{{-1}}\,\,{{\rm Mpc}})$'.format(
            projection[1]))
        if self.nsample == 1:
            npartstr = self.number_of_particles
        else:
            npartstr = '{}^3'.format(self.nsample)
        metadata = [r'${}\,\,{{\rm particles}}$'.format(npartstr),
                    r'${{\rm Box\,\,size}}={}\,\,{{\rm Mpc}}$'.format(
                        self.box_size*length_norm),
                    r'$z = {}$'.format(abs(round(self.redshift, 3)))]
        xlims, ylims = f_box[1].get_xlim(), f_box[1].get_ylim()
        xdiff, ydiff = xlims[1] - xlims[0], ylims[1] - ylims[0]
        for mi, m in enumerate(metadata):
            f_box[1].text(xlims[0]+0.03*xdiff, ylims[0]+0.03*(mi+1)*ydiff,
                          m, color='white')
        f_box[0].suptitle(title)
        f_box[0].tight_layout()
        if save:
            f_box[0].savefig(savefile, dpi=500)
        elif return_fig:
            return f_box
        else:
            f_box[0].show()

    def box_info(self):
        """
        Print some of the simulation parameters.

        """

        pretty_print([round(self.redshift, 3),
                      self.box_size*self.unit_length_in_cm/(1e3*self.cm_per_kpc),
                      '{}^3'.format(int(self.nsample)),
                      self.Omega0,
                      self.OmegaBaryon,
                      1-self.Omega0],
                     ['Redshift',
                      'Box size (Mpc)',
                      'Number of particles',
                      'Omega_0',
                      'Omega_Baryon',
                      'Omega_Lambda'],
                     'SIMULATION PARAMETERS')

        return


class Catalogue:

    def __init__(self, cat_reader=None, cat_options=None, cat_obj=None):
        """
        """

        if cat_obj is None:
            if cat_reader is not None:
                cat_obj = cat_reader(**cat_options)
            else:
                raise ValueError('Must provide a catalogue reader and options,'
                                 ' or a snapshot object.')

        for attr in dir(cat_obj):
            if '__' not in attr:
                setattr(self, attr, getattr(cat_obj, attr))

    def calc_mass_function(self, halo, nbins=20):
        """
        """

        mass_func_vol, self.mass_bins = np.histogram(np.log10(
            halo['Mass']), bins=nbins)
        self.mass_function = mass_func_vol / self.box_size**3

    def plot_mass_function(self, halo, nbins=20, title=None, save=False,
                           savefile=None):
        """
        """

        if not hasattr(self, 'mass_function'):
            self.calc_mass_function(halo, nbins=nbins)

        lin_mass = 10**(self.mass_bins +
                        (self.mass_bins[1] - self.mass_bins[0]) / 2)

        f_mass = plt.subplots(figsize=(5, 5), dpi=300)
        f_mass[1].plot(lin_mass[:-1], self.mass_function, linewidth=4)
        f_mass[1].set_xscale('log')
        f_mass[1].set_yscale('log')
        f_mass[1].set_xlabel(r'$M_{\rm halo}\,\,(10^{10}\,\,{\rm M}_\odot)$')
        f_mass[1].set_ylabel(
            r'${\rm d}n/{\rm d}M\,\,({\rm Mpc}^{-3}/10^{10}\,\,{\rm M}_\odot)$'
            )
        f_mass[0].suptitle(title)
        f_mass[0].tight_layout()
        if save:
            f_mass[0].savefig(savefile, dpi=300)
        else:
            f_mass[0].show()
