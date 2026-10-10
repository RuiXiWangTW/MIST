"""Minimal spectrum record used by compact-cache construction."""

from __future__ import annotations


class Spectra:
    def __init__(
        self,
        *,
        spectra_name: str,
        spectra_formula: str,
        instrument: str,
    ) -> None:
        self.spectra_name = spectra_name
        self.formula = spectra_formula
        self.instrument = instrument

    def get_instrument(self) -> str:
        return self.instrument

    def get_spec_name(self) -> str:
        return self.spectra_name

    def get_spectra_formula(self) -> str:
        return self.formula
