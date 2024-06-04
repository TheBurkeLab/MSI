# This file is part of Cantera. See License.txt in the top-level directory or
# at https://cantera.org/license.txt for license and copyright information.
from textwrap import dedent

from .. import Solution as _Solution, PureFluid as _PureFluid, CanteraError
from .. import (Heptane as _Heptane, Water as _Water, Hfc134a as _Hfc134a,
                CarbonDioxide as _CarbonDioxide, Hydrogen as _Hydrogen,
                Methane as _Methane, Nitrogen as _Nitrogen, Oxygen as _Oxygen)
from pint import get_application_registry

__all__ = ("units", "Q_", "Solution", "PureFluid", "Heptane", "CarbonDioxide",
           "Hfc134a", "Hydrogen", "Methane", "Nitrogen", "Oxygen", "Water",
           "CanteraError")

units = get_application_registry()
Q_ = units.Quantity


def copy_doc(method):
    """Decorator to copy docstrings from related methods in upstream classes.

    This decorator will copy the docstring from the same named method in the upstream
    class, either `Solution` or `PureFluid`. The docstring in the method being
    decorated is appended to the upstream documentation.
    """
    doc = getattr(method, "__doc__", None) or ""
    if isinstance(method, property):
        method = method.fget
        if not doc:
            doc = getattr(method, "__doc__", None) or ""
    original_klass = method.__qualname__.split(".")[0]
    klass = {"Solution": _Solution, "PureFluid": _PureFluid}[original_klass]
    original_method = getattr(klass, method.__name__)
    original_doc = dedent(getattr(original_method, "__doc__", ""))
    method.__doc__ = f"{original_doc}\n{doc}"
    return method


class Solution:
    """
    This implementation of `Solution <cantera.with_units.Solution>` operates with
    units by using the `pint` library to convert between unit systems. All properties
    are assigned units in the standard MKS system that Cantera uses, substituting kmol
    instead of mol. Each property is an instance of the `pint.Quantity` class.

    Similarly, properties must be instances of `pint.Quantity` classes when they are
    used for assignment to set the state. The properties may have any units, so long
    as the dimensions for the quantity are consistent. For example, temperatures can
    be provided in K, degC, degF, or degR; conversion will be done internally to
    Cantera's consistent unit system.

    See the `pint documentation <https://pint.readthedocs.io>`__ for more information
    about using pint's ``Quantity`` classes.

    **Note:** This class is experimental. It only implements methods from `ThermoPhase`.
    Methods from other classes are not yet supported. If you are interested in contributing
    to this feature, please chime in on our enhancements issue:
    `<https://github.com/Cantera/enhancements/issues/174>`__.
    """
    def __init__(self, infile="", name="", *, yaml=None):
        self.__dict__["_phase"] = _Solution(infile, name, yaml=yaml)

    def __getattr__(self, name):
        return getattr(self._phase, name)

    def __setattr__(self, name, value):
        if name in dir(self):
            object.__setattr__(self, name, value)
        else:
            setattr(self._phase, name, value)

    @property
    def basis_units(self):
        """The units associated with the mass/molar basis of this phase."""
        if self._phase.basis == "mass":
            return "kg"
        else:
            return "kmol"

    @property
    @copy_doc
    def X(self):
        """If an array is used for setting, the units must be dimensionless."""
        X = self._phase.X
        return Q_(X, "dimensionless")

    @X.setter
    def X(self, value):
        if value is not None:
            try:
                X = value.to("dimensionless").magnitude
            except AttributeError:
                X = value
        else:
            X = self.X.magnitude
        self._phase.X = X

    @property
    @copy_doc
    def Y(self):
        """If an array is used for setting, the units must be dimensionless."""
        Y = self._phase.Y
        return Q_(Y, "dimensionless")

    @Y.setter
    def Y(self, value):
        if value is not None:
            try:
                Y = value.to("dimensionless").magnitude
            except AttributeError:
                Y = value
        else:
            Y = self.Y.magnitude
        self._phase.Y = Y

    @property
    @copy_doc
    def density_mass(self):
        return Q_(self._phase.density_mass, "kg/m**3")

    @property
    @copy_doc
    def density_mole(self):
        return Q_(self._phase.density_mole, "kmol/m**3")

    @property
    @copy_doc
    def enthalpy_mass(self):
        return Q_(self._phase.enthalpy_mass, "J/kg")

    @property
    @copy_doc
    def enthalpy_mole(self):
        return Q_(self._phase.enthalpy_mole, "J/kmol")

    @property
    @copy_doc
    def entropy_mass(self):
        return Q_(self._phase.entropy_mass, "J/kg/K")

    @property
    @copy_doc
    def entropy_mole(self):
        return Q_(self._phase.entropy_mole, "J/kmol/K")

    @property
    @copy_doc
    def int_energy_mass(self):
        return Q_(self._phase.int_energy_mass, "J/kg")

    @property
    @copy_doc
    def int_energy_mole(self):
        return Q_(self._phase.int_energy_mole, "J/kmol")

    @property
    @copy_doc
    def volume_mass(self):
        return Q_(self._phase.volume_mass, "m**3/kg")

    @property
    @copy_doc
    def volume_mole(self):
        return Q_(self._phase.volume_mole, "m**3/kmol")

    @property
    @copy_doc
    def gibbs_mass(self):
        return Q_(self._phase.gibbs_mass, "J/kg")

    @property
    @copy_doc
    def gibbs_mole(self):
        return Q_(self._phase.gibbs_mole, "J/kmol")

    @property
    @copy_doc
    def cp_mass(self):
        return Q_(self._phase.cp_mass, "J/kg/K")

    @property
    @copy_doc
    def cp_mole(self):
        return Q_(self._phase.cp_mole, "J/kmol/K")

    @property
    @copy_doc
    def cv_mass(self):
        return Q_(self._phase.cv_mass, "J/kg/K")

    @property
    @copy_doc
    def cv_mole(self):
        return Q_(self._phase.cv_mole, "J/kmol/K")

    @property
    @copy_doc
    def P(self):
        return Q_(self._phase.P, "Pa")

    @property
    @copy_doc
    def P_sat(self):
        return Q_(self._phase.P_sat, "Pa")

    @property
    @copy_doc
    def T(self):
        return Q_(self._phase.T, "K")

    @property
    @copy_doc
    def T_sat(self):
        return Q_(self._phase.T_sat, "K")

    @property
    @copy_doc
    def atomic_weight(self):
        return Q_(self._phase.atomic_weight, "kg/kmol")

    @property
    @copy_doc
    def chemical_potentials(self):
        return Q_(self._phase.chemical_potentials, "J/kmol")

    @property
    @copy_doc
    def concentrations(self):
        return Q_(self._phase.concentrations, "kmol/m**3")

    @property
    @copy_doc
    def critical_pressure(self):
        return Q_(self._phase.critical_pressure, "Pa")

    @property
    @copy_doc
    def critical_temperature(self):
        return Q_(self._phase.critical_temperature, "K")

    @property
    @copy_doc
    def critical_density(self):
        return Q_(self._phase.critical_density, self.basis_units + "/m**3")

    @property
    @copy_doc
    def electric_potential(self):
        return Q_(self._phase.electric_potential, "V")

    @property
    @copy_doc
    def electrochemical_potentials(self):
        return Q_(self._phase.electrochemical_potentials, "J/kmol")

    @property
    @copy_doc
    def isothermal_compressibility(self):
        return Q_(self._phase.isothermal_compressibility, "1/Pa")

    @property
    @copy_doc
    def sound_speed(self):
        return Q_(self._phase.sound_speed, "m/s")

    @property
    @copy_doc
    def max_temp(self):
        return Q_(self._phase.max_temp, "K")

    @property
    @copy_doc
    def mean_molecular_weight(self):
        return Q_(self._phase.mean_molecular_weight, "kg/kmol")

    @property
    @copy_doc
    def min_temp(self):
        return Q_(self._phase.min_temp, "K")

    @property
    @copy_doc
    def molecular_weights(self):
        return Q_(self._phase.molecular_weights, "kg/kmol")

    @property
    @copy_doc
    def partial_molar_cp(self):
        return Q_(self._phase.partial_molar_cp, "J/kmol/K")

    @property
    @copy_doc
    def partial_molar_enthalpies(self):
        return Q_(self._phase.partial_molar_enthalpies, "J/kmol")

    @property
    @copy_doc
    def partial_molar_entropies(self):
        return Q_(self._phase.partial_molar_entropies, "J/kmol/K")

    @property
    @copy_doc
    def partial_molar_int_energies(self):
        return Q_(self._phase.partial_molar_int_energies, "J/kmol")

    @property
    @copy_doc
    def partial_molar_volumes(self):
        return Q_(self._phase.partial_molar_volumes, "m**3/kmol")

    @property
    @copy_doc
    def reference_pressure(self):
        return Q_(self._phase.reference_pressure, "Pa")

    @property
    @copy_doc
    def thermal_expansion_coeff(self):
        return Q_(self._phase.thermal_expansion_coeff, "1/K")

    @property
    @copy_doc
    def cp(self):
        return Q_(self._phase.cp, "J/K/" + self.basis_units)

    @property
    @copy_doc
    def cv(self):
        return Q_(self._phase.cv, "J/K/" + self.basis_units)

    @property
    @copy_doc
    def density(self):
        return Q_(self._phase.density, self.basis_units + "/m**3")

    @property
    @copy_doc
    def h(self):
        return Q_(self._phase.h, "J/" + self.basis_units)

    @property
    @copy_doc
    def s(self):
        return Q_(self._phase.s, "J/K/" + self.basis_units)

    @property
    @copy_doc
    def g(self):
        return Q_(self._phase.g, "J/" + self.basis_units)

    @property
    @copy_doc
    def u(self):
        return Q_(self._phase.u, "J/" + self.basis_units)

    @property
    @copy_doc
    def v(self):
        return Q_(self._phase.v, "m**3/" + self.basis_units)

    @property
    @copy_doc
    def TP(self):
        T, P = self._phase.TP
        return Q_(T, "K"), Q_(P, "Pa")

    @TP.setter
    def TP(self, value):
        T = value[0] if value[0] is not None else self.T
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((T, "K"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.TP = T.magnitude, P.magnitude

    @property
    @copy_doc
    def DP(self):
        density, P = self._phase.DP
        return Q_(density, self.basis_units + "/m**3"), Q_(P, "Pa")

    @DP.setter
    def DP(self, value):
        density = value[0] if value[0] is not None else self.density
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((density, self.basis_units + "/m**3"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.DP = density.magnitude, P.magnitude

    @property
    @copy_doc
    def HP(self):
        h, P = self._phase.HP
        return Q_(h, "J/" + self.basis_units), Q_(P, "Pa")

    @HP.setter
    def HP(self, value):
        h = value[0] if value[0] is not None else self.h
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((h, "J/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.HP = h.magnitude, P.magnitude

    @property
    @copy_doc
    def SP(self):
        s, P = self._phase.SP
        return Q_(s, "J/K/" + self.basis_units), Q_(P, "Pa")

    @SP.setter
    def SP(self, value):
        s = value[0] if value[0] is not None else self.s
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((s, "J/K/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.SP = s.magnitude, P.magnitude

    @property
    @copy_doc
    def SV(self):
        s, v = self._phase.SV
        return Q_(s, "J/K/" + self.basis_units), Q_(v, "m**3/" + self.basis_units)

    @SV.setter
    def SV(self, value):
        s = value[0] if value[0] is not None else self.s
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((s, "J/K/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.SV = s.magnitude, v.magnitude

    @property
    @copy_doc
    def TD(self):
        T, density = self._phase.TD
        return Q_(T, "K"), Q_(density, self.basis_units + "/m**3")

    @TD.setter
    def TD(self, value):
        T = value[0] if value[0] is not None else self.T
        density = value[1] if value[1] is not None else self.density
        for val, unit in ((T, "K"), (density, self.basis_units + "/m**3")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.TD = T.magnitude, density.magnitude

    @property
    @copy_doc
    def UV(self):
        u, v = self._phase.UV
        return Q_(u, "J/" + self.basis_units), Q_(v, "m**3/" + self.basis_units)

    @UV.setter
    def UV(self, value):
        u = value[0] if value[0] is not None else self.u
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((u, "J/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.UV = u.magnitude, v.magnitude

    @property
    @copy_doc
    def TPX(self):
        T, P, X = self._phase.TPX
        return Q_(T, "K"), Q_(P, "Pa"), Q_(X, "dimensionless")

    @TPX.setter
    def TPX(self, value):
        T = value[0] if value[0] is not None else self.T
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((T, "K"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.TPX = T.magnitude, P.magnitude, X

    @property
    @copy_doc
    def TPY(self):
        T, P, Y = self._phase.TPY
        return Q_(T, "K"), Q_(P, "Pa"), Q_(Y, "dimensionless")

    @TPY.setter
    def TPY(self, value):
        T = value[0] if value[0] is not None else self.T
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((T, "K"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.TPY = T.magnitude, P.magnitude, Y

    @property
    @copy_doc
    def DPX(self):
        density, P, X = self._phase.DPX
        return Q_(density, self.basis_units + "/m**3"), Q_(P, "Pa"), Q_(X, "dimensionless")

    @DPX.setter
    def DPX(self, value):
        density = value[0] if value[0] is not None else self.density
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((density, self.basis_units + "/m**3"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.DPX = density.magnitude, P.magnitude, X

    @property
    @copy_doc
    def DPY(self):
        density, P, Y = self._phase.DPY
        return Q_(density, self.basis_units + "/m**3"), Q_(P, "Pa"), Q_(Y, "dimensionless")

    @DPY.setter
    def DPY(self, value):
        density = value[0] if value[0] is not None else self.density
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((density, self.basis_units + "/m**3"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.DPY = density.magnitude, P.magnitude, Y

    @property
    @copy_doc
    def HPX(self):
        h, P, X = self._phase.HPX
        return Q_(h, "J/" + self.basis_units), Q_(P, "Pa"), Q_(X, "dimensionless")

    @HPX.setter
    def HPX(self, value):
        h = value[0] if value[0] is not None else self.h
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((h, "J/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.HPX = h.magnitude, P.magnitude, X

    @property
    @copy_doc
    def HPY(self):
        h, P, Y = self._phase.HPY
        return Q_(h, "J/" + self.basis_units), Q_(P, "Pa"), Q_(Y, "dimensionless")

    @HPY.setter
    def HPY(self, value):
        h = value[0] if value[0] is not None else self.h
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((h, "J/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.HPY = h.magnitude, P.magnitude, Y

    @property
    @copy_doc
    def SPX(self):
        s, P, X = self._phase.SPX
        return Q_(s, "J/K/" + self.basis_units), Q_(P, "Pa"), Q_(X, "dimensionless")

    @SPX.setter
    def SPX(self, value):
        s = value[0] if value[0] is not None else self.s
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((s, "J/K/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.SPX = s.magnitude, P.magnitude, X

    @property
    @copy_doc
    def SPY(self):
        s, P, Y = self._phase.SPY
        return Q_(s, "J/K/" + self.basis_units), Q_(P, "Pa"), Q_(Y, "dimensionless")

    @SPY.setter
    def SPY(self, value):
        s = value[0] if value[0] is not None else self.s
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((s, "J/K/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.SPY = s.magnitude, P.magnitude, Y

    @property
    @copy_doc
    def SVX(self):
        s, v, X = self._phase.SVX
        return Q_(s, "J/K/" + self.basis_units), Q_(v, "m**3/" + self.basis_units), Q_(X, "dimensionless")

    @SVX.setter
    def SVX(self, value):
        s = value[0] if value[0] is not None else self.s
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((s, "J/K/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.SVX = s.magnitude, v.magnitude, X

    @property
    @copy_doc
    def SVY(self):
        s, v, Y = self._phase.SVY
        return Q_(s, "J/K/" + self.basis_units), Q_(v, "m**3/" + self.basis_units), Q_(Y, "dimensionless")

    @SVY.setter
    def SVY(self, value):
        s = value[0] if value[0] is not None else self.s
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((s, "J/K/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.SVY = s.magnitude, v.magnitude, Y

    @property
    @copy_doc
    def TDX(self):
        T, density, X = self._phase.TDX
        return Q_(T, "K"), Q_(density, self.basis_units + "/m**3"), Q_(X, "dimensionless")

    @TDX.setter
    def TDX(self, value):
        T = value[0] if value[0] is not None else self.T
        density = value[1] if value[1] is not None else self.density
        for val, unit in ((T, "K"), (density, self.basis_units + "/m**3")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.TDX = T.magnitude, density.magnitude, X

    @property
    @copy_doc
    def TDY(self):
        T, density, Y = self._phase.TDY
        return Q_(T, "K"), Q_(density, self.basis_units + "/m**3"), Q_(Y, "dimensionless")

    @TDY.setter
    def TDY(self, value):
        T = value[0] if value[0] is not None else self.T
        density = value[1] if value[1] is not None else self.density
        for val, unit in ((T, "K"), (density, self.basis_units + "/m**3")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.TDY = T.magnitude, density.magnitude, Y

    @property
    @copy_doc
    def UVX(self):
        u, v, X = self._phase.UVX
        return Q_(u, "J/" + self.basis_units), Q_(v, "m**3/" + self.basis_units), Q_(X, "dimensionless")

    @UVX.setter
    def UVX(self, value):
        u = value[0] if value[0] is not None else self.u
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((u, "J/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.UVX = u.magnitude, v.magnitude, X

    @property
    @copy_doc
    def UVY(self):
        u, v, Y = self._phase.UVY
        return Q_(u, "J/" + self.basis_units), Q_(v, "m**3/" + self.basis_units), Q_(Y, "dimensionless")

    @UVY.setter
    def UVY(self, value):
        u = value[0] if value[0] is not None else self.u
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((u, "J/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.UVY = u.magnitude, v.magnitude, Y

Solution.__doc__ = f"{Solution.__doc__}\n{_Solution.__doc__}"

class PureFluid:
    """
    This implementation of `PureFluid <cantera.with_units.PureFluid>` operates with
    units by using the `pint` library to convert between unit systems. All properties
    are assigned units in the standard MKS system that Cantera uses, substituting kmol
    instead of mol. Each property is an instance of the `pint.Quantity` class.

    Similarly, properties must be instances of `pint.Quantity` classes when they are
    used for assignment to set the state. The properties may have any units, so long
    as the dimensions for the quantity are consistent. For example, temperatures can
    be provided in K, degC, degF, or degR; conversion will be done internally to
    Cantera's consistent unit system.

    See the `pint documentation <https://pint.readthedocs.io>`__ for more information
    about using pint's ``Quantity`` classes.
    """
    def __init__(self, infile, name="", *, yaml=None, **kwargs):
        self.__dict__["_phase"] = _PureFluid(infile, name, yaml=yaml, **kwargs)

    def __getattr__(self, name):
        return getattr(self._phase, name)

    def __setattr__(self, name, value):
        if name in dir(self):
            object.__setattr__(self, name, value)
        else:
            setattr(self._phase, name, value)

    @property
    def basis_units(self):
        """The units associated with the mass/molar basis of this phase."""
        if self._phase.basis == "mass":
            return "kg"
        else:
            return "kmol"

    @property
    @copy_doc
    def X(self):
        """If an array is used for setting, the units must be dimensionless."""
        X = self._phase.X
        return Q_(X, "dimensionless")

    @X.setter
    def X(self, value):
        if value is not None:
            try:
                X = value.to("dimensionless").magnitude
            except AttributeError:
                X = value
        else:
            X = self.X.magnitude
        self._phase.X = X

    @property
    @copy_doc
    def Y(self):
        """If an array is used for setting, the units must be dimensionless."""
        Y = self._phase.Y
        return Q_(Y, "dimensionless")

    @Y.setter
    def Y(self, value):
        if value is not None:
            try:
                Y = value.to("dimensionless").magnitude
            except AttributeError:
                Y = value
        else:
            Y = self.Y.magnitude
        self._phase.Y = Y

    @property
    @copy_doc
    def Q(self):
        """Must be set using a quantity with dimensionless units."""
        Q = self._phase.Q
        return Q_(Q, "dimensionless")

    @Q.setter
    def Q(self, value):
        if value is not None:
            try:
                Q = value.to("dimensionless").magnitude
            except AttributeError as e:
                if "'to'" in str(e):
                    raise CanteraError(
                        f"Value {value!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        else:
            Q = self.Q.magnitude
        self._phase.Q = Q

    @property
    @copy_doc
    def density_mass(self):
        return Q_(self._phase.density_mass, "kg/m**3")

    @property
    @copy_doc
    def density_mole(self):
        return Q_(self._phase.density_mole, "kmol/m**3")

    @property
    @copy_doc
    def enthalpy_mass(self):
        return Q_(self._phase.enthalpy_mass, "J/kg")

    @property
    @copy_doc
    def enthalpy_mole(self):
        return Q_(self._phase.enthalpy_mole, "J/kmol")

    @property
    @copy_doc
    def entropy_mass(self):
        return Q_(self._phase.entropy_mass, "J/kg/K")

    @property
    @copy_doc
    def entropy_mole(self):
        return Q_(self._phase.entropy_mole, "J/kmol/K")

    @property
    @copy_doc
    def int_energy_mass(self):
        return Q_(self._phase.int_energy_mass, "J/kg")

    @property
    @copy_doc
    def int_energy_mole(self):
        return Q_(self._phase.int_energy_mole, "J/kmol")

    @property
    @copy_doc
    def volume_mass(self):
        return Q_(self._phase.volume_mass, "m**3/kg")

    @property
    @copy_doc
    def volume_mole(self):
        return Q_(self._phase.volume_mole, "m**3/kmol")

    @property
    @copy_doc
    def gibbs_mass(self):
        return Q_(self._phase.gibbs_mass, "J/kg")

    @property
    @copy_doc
    def gibbs_mole(self):
        return Q_(self._phase.gibbs_mole, "J/kmol")

    @property
    @copy_doc
    def cp_mass(self):
        return Q_(self._phase.cp_mass, "J/kg/K")

    @property
    @copy_doc
    def cp_mole(self):
        return Q_(self._phase.cp_mole, "J/kmol/K")

    @property
    @copy_doc
    def cv_mass(self):
        return Q_(self._phase.cv_mass, "J/kg/K")

    @property
    @copy_doc
    def cv_mole(self):
        return Q_(self._phase.cv_mole, "J/kmol/K")

    @property
    @copy_doc
    def P(self):
        return Q_(self._phase.P, "Pa")

    @property
    @copy_doc
    def P_sat(self):
        return Q_(self._phase.P_sat, "Pa")

    @property
    @copy_doc
    def T(self):
        return Q_(self._phase.T, "K")

    @property
    @copy_doc
    def T_sat(self):
        return Q_(self._phase.T_sat, "K")

    @property
    @copy_doc
    def atomic_weight(self):
        return Q_(self._phase.atomic_weight, "kg/kmol")

    @property
    @copy_doc
    def chemical_potentials(self):
        return Q_(self._phase.chemical_potentials, "J/kmol")

    @property
    @copy_doc
    def concentrations(self):
        return Q_(self._phase.concentrations, "kmol/m**3")

    @property
    @copy_doc
    def critical_pressure(self):
        return Q_(self._phase.critical_pressure, "Pa")

    @property
    @copy_doc
    def critical_temperature(self):
        return Q_(self._phase.critical_temperature, "K")

    @property
    @copy_doc
    def critical_density(self):
        return Q_(self._phase.critical_density, self.basis_units + "/m**3")

    @property
    @copy_doc
    def electric_potential(self):
        return Q_(self._phase.electric_potential, "V")

    @property
    @copy_doc
    def electrochemical_potentials(self):
        return Q_(self._phase.electrochemical_potentials, "J/kmol")

    @property
    @copy_doc
    def isothermal_compressibility(self):
        return Q_(self._phase.isothermal_compressibility, "1/Pa")

    @property
    @copy_doc
    def sound_speed(self):
        return Q_(self._phase.sound_speed, "m/s")

    @property
    @copy_doc
    def max_temp(self):
        return Q_(self._phase.max_temp, "K")

    @property
    @copy_doc
    def mean_molecular_weight(self):
        return Q_(self._phase.mean_molecular_weight, "kg/kmol")

    @property
    @copy_doc
    def min_temp(self):
        return Q_(self._phase.min_temp, "K")

    @property
    @copy_doc
    def molecular_weights(self):
        return Q_(self._phase.molecular_weights, "kg/kmol")

    @property
    @copy_doc
    def partial_molar_cp(self):
        return Q_(self._phase.partial_molar_cp, "J/kmol/K")

    @property
    @copy_doc
    def partial_molar_enthalpies(self):
        return Q_(self._phase.partial_molar_enthalpies, "J/kmol")

    @property
    @copy_doc
    def partial_molar_entropies(self):
        return Q_(self._phase.partial_molar_entropies, "J/kmol/K")

    @property
    @copy_doc
    def partial_molar_int_energies(self):
        return Q_(self._phase.partial_molar_int_energies, "J/kmol")

    @property
    @copy_doc
    def partial_molar_volumes(self):
        return Q_(self._phase.partial_molar_volumes, "m**3/kmol")

    @property
    @copy_doc
    def reference_pressure(self):
        return Q_(self._phase.reference_pressure, "Pa")

    @property
    @copy_doc
    def thermal_expansion_coeff(self):
        return Q_(self._phase.thermal_expansion_coeff, "1/K")

    @property
    @copy_doc
    def cp(self):
        return Q_(self._phase.cp, "J/K/" + self.basis_units)

    @property
    @copy_doc
    def cv(self):
        return Q_(self._phase.cv, "J/K/" + self.basis_units)

    @property
    @copy_doc
    def density(self):
        return Q_(self._phase.density, self.basis_units + "/m**3")

    @property
    @copy_doc
    def h(self):
        return Q_(self._phase.h, "J/" + self.basis_units)

    @property
    @copy_doc
    def s(self):
        return Q_(self._phase.s, "J/K/" + self.basis_units)

    @property
    @copy_doc
    def g(self):
        return Q_(self._phase.g, "J/" + self.basis_units)

    @property
    @copy_doc
    def u(self):
        return Q_(self._phase.u, "J/" + self.basis_units)

    @property
    @copy_doc
    def v(self):
        return Q_(self._phase.v, "m**3/" + self.basis_units)

    @property
    @copy_doc
    def TP(self):
        T, P = self._phase.TP
        return Q_(T, "K"), Q_(P, "Pa")

    @TP.setter
    def TP(self, value):
        T = value[0] if value[0] is not None else self.T
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((T, "K"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.TP = T.magnitude, P.magnitude

    @property
    @copy_doc
    def DP(self):
        density, P = self._phase.DP
        return Q_(density, self.basis_units + "/m**3"), Q_(P, "Pa")

    @DP.setter
    def DP(self, value):
        density = value[0] if value[0] is not None else self.density
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((density, self.basis_units + "/m**3"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.DP = density.magnitude, P.magnitude

    @property
    @copy_doc
    def HP(self):
        h, P = self._phase.HP
        return Q_(h, "J/" + self.basis_units), Q_(P, "Pa")

    @HP.setter
    def HP(self, value):
        h = value[0] if value[0] is not None else self.h
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((h, "J/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.HP = h.magnitude, P.magnitude

    @property
    @copy_doc
    def SP(self):
        s, P = self._phase.SP
        return Q_(s, "J/K/" + self.basis_units), Q_(P, "Pa")

    @SP.setter
    def SP(self, value):
        s = value[0] if value[0] is not None else self.s
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((s, "J/K/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.SP = s.magnitude, P.magnitude

    @property
    @copy_doc
    def SV(self):
        s, v = self._phase.SV
        return Q_(s, "J/K/" + self.basis_units), Q_(v, "m**3/" + self.basis_units)

    @SV.setter
    def SV(self, value):
        s = value[0] if value[0] is not None else self.s
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((s, "J/K/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.SV = s.magnitude, v.magnitude

    @property
    @copy_doc
    def TD(self):
        T, density = self._phase.TD
        return Q_(T, "K"), Q_(density, self.basis_units + "/m**3")

    @TD.setter
    def TD(self, value):
        T = value[0] if value[0] is not None else self.T
        density = value[1] if value[1] is not None else self.density
        for val, unit in ((T, "K"), (density, self.basis_units + "/m**3")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.TD = T.magnitude, density.magnitude

    @property
    @copy_doc
    def UV(self):
        u, v = self._phase.UV
        return Q_(u, "J/" + self.basis_units), Q_(v, "m**3/" + self.basis_units)

    @UV.setter
    def UV(self, value):
        u = value[0] if value[0] is not None else self.u
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((u, "J/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.UV = u.magnitude, v.magnitude

    @property
    @copy_doc
    def TPX(self):
        T, P, X = self._phase.TPX
        return Q_(T, "K"), Q_(P, "Pa"), Q_(X, "dimensionless")

    @TPX.setter
    def TPX(self, value):
        T = value[0] if value[0] is not None else self.T
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((T, "K"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.TPX = T.magnitude, P.magnitude, X

    @property
    @copy_doc
    def TPY(self):
        T, P, Y = self._phase.TPY
        return Q_(T, "K"), Q_(P, "Pa"), Q_(Y, "dimensionless")

    @TPY.setter
    def TPY(self, value):
        T = value[0] if value[0] is not None else self.T
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((T, "K"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.TPY = T.magnitude, P.magnitude, Y

    @property
    @copy_doc
    def DPX(self):
        density, P, X = self._phase.DPX
        return Q_(density, self.basis_units + "/m**3"), Q_(P, "Pa"), Q_(X, "dimensionless")

    @DPX.setter
    def DPX(self, value):
        density = value[0] if value[0] is not None else self.density
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((density, self.basis_units + "/m**3"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.DPX = density.magnitude, P.magnitude, X

    @property
    @copy_doc
    def DPY(self):
        density, P, Y = self._phase.DPY
        return Q_(density, self.basis_units + "/m**3"), Q_(P, "Pa"), Q_(Y, "dimensionless")

    @DPY.setter
    def DPY(self, value):
        density = value[0] if value[0] is not None else self.density
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((density, self.basis_units + "/m**3"), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.DPY = density.magnitude, P.magnitude, Y

    @property
    @copy_doc
    def HPX(self):
        h, P, X = self._phase.HPX
        return Q_(h, "J/" + self.basis_units), Q_(P, "Pa"), Q_(X, "dimensionless")

    @HPX.setter
    def HPX(self, value):
        h = value[0] if value[0] is not None else self.h
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((h, "J/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.HPX = h.magnitude, P.magnitude, X

    @property
    @copy_doc
    def HPY(self):
        h, P, Y = self._phase.HPY
        return Q_(h, "J/" + self.basis_units), Q_(P, "Pa"), Q_(Y, "dimensionless")

    @HPY.setter
    def HPY(self, value):
        h = value[0] if value[0] is not None else self.h
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((h, "J/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.HPY = h.magnitude, P.magnitude, Y

    @property
    @copy_doc
    def SPX(self):
        s, P, X = self._phase.SPX
        return Q_(s, "J/K/" + self.basis_units), Q_(P, "Pa"), Q_(X, "dimensionless")

    @SPX.setter
    def SPX(self, value):
        s = value[0] if value[0] is not None else self.s
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((s, "J/K/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.SPX = s.magnitude, P.magnitude, X

    @property
    @copy_doc
    def SPY(self):
        s, P, Y = self._phase.SPY
        return Q_(s, "J/K/" + self.basis_units), Q_(P, "Pa"), Q_(Y, "dimensionless")

    @SPY.setter
    def SPY(self, value):
        s = value[0] if value[0] is not None else self.s
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((s, "J/K/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.SPY = s.magnitude, P.magnitude, Y

    @property
    @copy_doc
    def SVX(self):
        s, v, X = self._phase.SVX
        return Q_(s, "J/K/" + self.basis_units), Q_(v, "m**3/" + self.basis_units), Q_(X, "dimensionless")

    @SVX.setter
    def SVX(self, value):
        s = value[0] if value[0] is not None else self.s
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((s, "J/K/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.SVX = s.magnitude, v.magnitude, X

    @property
    @copy_doc
    def SVY(self):
        s, v, Y = self._phase.SVY
        return Q_(s, "J/K/" + self.basis_units), Q_(v, "m**3/" + self.basis_units), Q_(Y, "dimensionless")

    @SVY.setter
    def SVY(self, value):
        s = value[0] if value[0] is not None else self.s
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((s, "J/K/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.SVY = s.magnitude, v.magnitude, Y

    @property
    @copy_doc
    def TDX(self):
        T, density, X = self._phase.TDX
        return Q_(T, "K"), Q_(density, self.basis_units + "/m**3"), Q_(X, "dimensionless")

    @TDX.setter
    def TDX(self, value):
        T = value[0] if value[0] is not None else self.T
        density = value[1] if value[1] is not None else self.density
        for val, unit in ((T, "K"), (density, self.basis_units + "/m**3")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.TDX = T.magnitude, density.magnitude, X

    @property
    @copy_doc
    def TDY(self):
        T, density, Y = self._phase.TDY
        return Q_(T, "K"), Q_(density, self.basis_units + "/m**3"), Q_(Y, "dimensionless")

    @TDY.setter
    def TDY(self, value):
        T = value[0] if value[0] is not None else self.T
        density = value[1] if value[1] is not None else self.density
        for val, unit in ((T, "K"), (density, self.basis_units + "/m**3")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.TDY = T.magnitude, density.magnitude, Y

    @property
    @copy_doc
    def UVX(self):
        u, v, X = self._phase.UVX
        return Q_(u, "J/" + self.basis_units), Q_(v, "m**3/" + self.basis_units), Q_(X, "dimensionless")

    @UVX.setter
    def UVX(self, value):
        u = value[0] if value[0] is not None else self.u
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((u, "J/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                X = value[2].to("dimensionless").magnitude
            except AttributeError:
                X = value[2]
        else:
            X = self.X.magnitude
        self._phase.UVX = u.magnitude, v.magnitude, X

    @property
    @copy_doc
    def UVY(self):
        u, v, Y = self._phase.UVY
        return Q_(u, "J/" + self.basis_units), Q_(v, "m**3/" + self.basis_units), Q_(Y, "dimensionless")

    @UVY.setter
    def UVY(self, value):
        u = value[0] if value[0] is not None else self.u
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((u, "J/" + self.basis_units), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        if value[2] is not None:
            try:
                Y = value[2].to("dimensionless").magnitude
            except AttributeError:
                Y = value[2]
        else:
            Y = self.Y.magnitude
        self._phase.UVY = u.magnitude, v.magnitude, Y

    @property
    @copy_doc
    def TPQ(self):
        T, P, Q = self._phase.TPQ
        return Q_(T, "K"), Q_(P, "Pa"), Q_(Q, "dimensionless")

    @TPQ.setter
    def TPQ(self, value):
        T = value[0] if value[0] is not None else self.T
        P = value[1] if value[1] is not None else self.P
        Q = value[2] if value[2] is not None else self.Q
        for val, unit in ((T, "K"), (P, "Pa"), (Q, "dimensionless")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.TPQ = T.magnitude, P.magnitude, Q.magnitude

    @property
    @copy_doc
    def PQ(self):
        P, Q = self._phase.PQ
        return Q_(P, "Pa"), Q_(Q, "dimensionless")

    @PQ.setter
    def PQ(self, value):
        P = value[0] if value[0] is not None else self.P
        Q = value[1] if value[1] is not None else self.Q
        for val, unit in ((P, "Pa"), (Q, "dimensionless")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.PQ = P.magnitude, Q.magnitude

    @property
    @copy_doc
    def TQ(self):
        T, Q = self._phase.TQ
        return Q_(T, "K"), Q_(Q, "dimensionless")

    @TQ.setter
    def TQ(self, value):
        T = value[0] if value[0] is not None else self.T
        Q = value[1] if value[1] is not None else self.Q
        for val, unit in ((T, "K"), (Q, "dimensionless")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.TQ = T.magnitude, Q.magnitude

    @property
    @copy_doc
    def PV(self):
        P, v = self._phase.PV
        return Q_(P, "Pa"), Q_(v, "m**3/" + self.basis_units)

    @PV.setter
    def PV(self, value):
        P = value[0] if value[0] is not None else self.P
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((P, "Pa"), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.PV = P.magnitude, v.magnitude

    @property
    @copy_doc
    def SH(self):
        s, h = self._phase.SH
        return Q_(s, "J/K/" + self.basis_units), Q_(h, "J/" + self.basis_units)

    @SH.setter
    def SH(self, value):
        s = value[0] if value[0] is not None else self.s
        h = value[1] if value[1] is not None else self.h
        for val, unit in ((s, "J/K/" + self.basis_units), (h, "J/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.SH = s.magnitude, h.magnitude

    @property
    @copy_doc
    def ST(self):
        s, T = self._phase.ST
        return Q_(s, "J/K/" + self.basis_units), Q_(T, "K")

    @ST.setter
    def ST(self, value):
        s = value[0] if value[0] is not None else self.s
        T = value[1] if value[1] is not None else self.T
        for val, unit in ((s, "J/K/" + self.basis_units), (T, "K")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.ST = s.magnitude, T.magnitude

    @property
    @copy_doc
    def TH(self):
        T, h = self._phase.TH
        return Q_(T, "K"), Q_(h, "J/" + self.basis_units)

    @TH.setter
    def TH(self, value):
        T = value[0] if value[0] is not None else self.T
        h = value[1] if value[1] is not None else self.h
        for val, unit in ((T, "K"), (h, "J/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.TH = T.magnitude, h.magnitude

    @property
    @copy_doc
    def TV(self):
        T, v = self._phase.TV
        return Q_(T, "K"), Q_(v, "m**3/" + self.basis_units)

    @TV.setter
    def TV(self, value):
        T = value[0] if value[0] is not None else self.T
        v = value[1] if value[1] is not None else self.v
        for val, unit in ((T, "K"), (v, "m**3/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.TV = T.magnitude, v.magnitude

    @property
    @copy_doc
    def UP(self):
        u, P = self._phase.UP
        return Q_(u, "J/" + self.basis_units), Q_(P, "Pa")

    @UP.setter
    def UP(self, value):
        u = value[0] if value[0] is not None else self.u
        P = value[1] if value[1] is not None else self.P
        for val, unit in ((u, "J/" + self.basis_units), (P, "Pa")):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.UP = u.magnitude, P.magnitude

    @property
    @copy_doc
    def VH(self):
        v, h = self._phase.VH
        return Q_(v, "m**3/" + self.basis_units), Q_(h, "J/" + self.basis_units)

    @VH.setter
    def VH(self, value):
        v = value[0] if value[0] is not None else self.v
        h = value[1] if value[1] is not None else self.h
        for val, unit in ((v, "m**3/" + self.basis_units), (h, "J/" + self.basis_units)):
            try:
                val.ito(unit)
            except AttributeError as e:
                if "'ito'" in str(e):
                    raise CanteraError(
                        f"Value {val!r} must be an instance of a pint.Quantity class"
                    ) from None
                else:
                    raise
        self._phase.VH = v.magnitude, h.magnitude

    @property
    @copy_doc
    def DPQ(self):
        D, P, Q = self._phase.DPQ
        return Q_(D, self.basis_units + "/m**3"), Q_(P, "Pa"), Q_(Q, "dimensionless")

    @property
    @copy_doc
    def HPQ(self):
        H, P, Q = self._phase.HPQ
        return Q_(H, "J/" + self.basis_units), Q_(P, "Pa"), Q_(Q, "dimensionless")

    @property
    @copy_doc
    def SPQ(self):
        S, P, Q = self._phase.SPQ
        return Q_(S, "J/K/" + self.basis_units), Q_(P, "Pa"), Q_(Q, "dimensionless")

    @property
    @copy_doc
    def SVQ(self):
        S, V, Q = self._phase.SVQ
        return Q_(S, "J/K/" + self.basis_units), Q_(V, "m**3/" + self.basis_units), Q_(Q, "dimensionless")

    @property
    @copy_doc
    def TDQ(self):
        T, D, Q = self._phase.TDQ
        return Q_(T, "K"), Q_(D, self.basis_units + "/m**3"), Q_(Q, "dimensionless")

    @property
    @copy_doc
    def UVQ(self):
        U, V, Q = self._phase.UVQ
        return Q_(U, "J/" + self.basis_units), Q_(V, "m**3/" + self.basis_units), Q_(Q, "dimensionless")


PureFluid.__doc__ = f"{PureFluid.__doc__}\n{_PureFluid.__doc__}"


def Heptane():
    return PureFluid("liquidvapor.yaml", "heptane")

Heptane.__doc__ = _Heptane.__doc__


def CarbonDioxide():
    return PureFluid("liquidvapor.yaml", "carbon-dioxide")


CarbonDioxide.__doc__ = _CarbonDioxide.__doc__


def Hfc134a():
    return PureFluid("liquidvapor.yaml", "HFC-134a")


Hfc134a.__doc__ = _Hfc134a.__doc__


def Hydrogen():
    return PureFluid("liquidvapor.yaml", "hydrogen")


Hydrogen.__doc__ = _Hydrogen.__doc__


def Methane():
    return PureFluid("liquidvapor.yaml", "methane")


Methane.__doc__ = _Methane.__doc__


def Nitrogen():
    return PureFluid("liquidvapor.yaml", "nitrogen")


Nitrogen.__doc__ = _Nitrogen.__doc__


def Oxygen():
    return PureFluid("liquidvapor.yaml", "oxygen")


Oxygen.__doc__ = _Oxygen.__doc__


def Water(backend="Reynolds"):
    return PureFluid("liquidvapor.yaml", "water", backend=backend)

Water.__doc__ = _Water.__doc__
