"""
Physics models and computational core for pyMKM.

This subpackage contains the track-structure and specific-energy machinery used
by the classical MKM, stochastic MKM (SMK), and MCF-MKM calculations.

Modules
-------

- :mod:`particle_track`: implements
  :class:`~pymkm.physics.particle_track.ParticleTrack` for analytical radial
  dose distributions using the supported track-structure models.
- :mod:`specific_energy`: provides
  :class:`~pymkm.physics.specific_energy.SpecificEnergy` for single-event and
  dose-averaged specific-energy calculations in cylindrical sensitive regions.
"""

from .particle_track import ParticleTrack
from .specific_energy import SpecificEnergy

__all__ = ["ParticleTrack", "SpecificEnergy"]
