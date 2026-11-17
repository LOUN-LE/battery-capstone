"""Separate plated/dead lithium without mislabelling reversible Li as lost forever.

PyBaMM26.8 base_plating defines the reported plating capacity as the sum of
plated and dead lithium concentrations times the ELECTRODE volume and F/3600.
Active-material fraction must not enter this conversion.
"""
from __future__ import annotations
import math


def positive(value, name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError('Positive finite '+name+' required')
    return float(value)


def audit_plating_capacity(plated_mol_m3, dead_mol_m3, reported_total_Ah, *,
                           electrode_area_m2, electrode_thickness_m, faraday_C_per_mol):
    import numpy as np
    area=positive(electrode_area_m2,'electrode area in m2')
    thickness=positive(electrode_thickness_m,'electrode thickness in m')
    faraday=positive(faraday_C_per_mol,'Faraday constant in C/mol')
    plated,dead,reported=[np.asarray(v,dtype=float) for v in (plated_mol_m3,dead_mol_m3,reported_total_Ah)]
    if not plated.size or plated.shape!=dead.shape or plated.shape!=reported.shape:
        raise ValueError('Paired nonempty plating/dead/reported samples required')
    if not all(np.all(np.isfinite(v)) for v in (plated,dead,reported)):
        raise ValueError('Finite lithium inventory samples required')
    conversion=area*thickness*faraday/3600
    plated_Ah=plated*conversion
    dead_Ah=dead*conversion
    residual=plated_Ah+dead_Ah-reported
    tolerance=1e-10+1e-8*np.abs(reported)
    return dict(plated_Ah=plated_Ah,dead_Ah=dead_Ah,reported_total_Ah=reported,
                residual_Ah=residual,max_abs_residual_Ah=float(np.max(np.abs(residual))),
                capacity_identity_pass=bool(np.all(np.abs(residual)<=tolerance)),
                electrode_volume_m3=area*thickness,conversion_Ah_per_mol_m3=conversion,
                conversion_scope='bulk electrode volume; no active-material-volume multiplier',
                interpretation='plated lithium is a potentially strippable inventory; dead lithium is a separate model state; neither identifies an experimental degradation mechanism')
