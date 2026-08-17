"""The unit contract: one coherent unit system, declared and enforced.

A track file carrying mm positions with m/s velocities parses perfectly and is
numerically indistinguishable from a correct file. The declared header is the
only thing that can catch it, so when a particle file declares units they are
enforced as strictly as the geometry and kinematics headers -- and the resolved
units are echoed in the startup banner either way.
"""
import numpy as np
import pytest

from ccmplus.io_ledm import assemble, read_particle_file, run_banner
from ccmplus.tests.test_ledm_roi import _write_case


UNIT_LINE = "# units: length=mm, velocity=mm/s\n"


def _add_particle_units(tmp, line=UNIT_LINE):
    """Prepend a '# units:' header to every particle file in the case."""
    for pf in sorted((tmp / "particles").glob("particles_*.dat")):
        pf.write_text(line + pf.read_text(encoding="utf-8"), encoding="utf-8")


class TestParticleUnitHeader:
    def test_units_are_parsed_when_present(self, tmp_path):
        case = tmp_path / "case"
        _write_case(case)
        _add_particle_units(case)
        part = read_particle_file(next((case / "particles").glob("*.dat")))
        assert part["units"] == {"length": "mm", "velocity": "mm/s"}

    def test_header_is_optional(self, tmp_path):
        case = tmp_path / "case"
        _write_case(case)
        part = read_particle_file(next((case / "particles").glob("*.dat")))
        assert part["units"] == {}
        # And the run still assembles: the header is optional, not required.
        assert assemble(case / "params.txt").meta["n_particle_units_declared"] == 0

    def test_matching_units_pass(self, tmp_path):
        case = tmp_path / "case"
        _write_case(case)
        _add_particle_units(case)
        run = assemble(case / "params.txt")
        assert run.meta["n_particle_units_declared"] == len(run.frames)

    def test_mismatched_velocity_unit_is_rejected(self, tmp_path):
        """mm positions + m/s velocities: the silent-corruption case."""
        case = tmp_path / "case"
        _write_case(case)
        _add_particle_units(case, "# units: length=mm, velocity=m/s\n")
        with pytest.raises(ValueError, match="particle velocity unit"):
            assemble(case / "params.txt")

    def test_mismatched_length_unit_is_rejected(self, tmp_path):
        case = tmp_path / "case"
        _write_case(case)
        _add_particle_units(case, "# units: length=m, velocity=mm/s\n")
        with pytest.raises(ValueError, match="particle length unit"):
            assemble(case / "params.txt")


class TestStartupBanner:
    def test_banner_reports_units_geometry_grid_and_options(self, tmp_path):
        case = tmp_path / "case"
        _write_case(case)
        run = assemble(case / "params.txt")
        text = "\n".join(run_banner(run))

        # Units, spelled out.
        assert "length=mm" in text
        assert "time=s" in text
        assert "velocity=mm/s" in text
        assert "angular_velocity=rad/s" in text
        # Geometry.
        assert "sphere" in text
        assert "euler_seq=ZYX" in text
        # Grid + track count.
        assert f"{run.meta['n_nodes']:,} nodes" in text
        assert f"{run.meta['n_particles_kept']:,} kept" in text
        # Active LE-DM options.
        assert "boundary_constraints=on" in text
        assert "kernel=wide" in text
        assert "lambda_c=" in text
        assert "div_tol=" in text

    def test_banner_says_when_units_are_undeclared(self, tmp_path):
        case = tmp_path / "case"
        _write_case(case)
        text = "\n".join(run_banner(assemble(case / "params.txt")))
        assert "declare no units" in text

    def test_banner_confirms_a_checked_unit_declaration(self, tmp_path):
        case = tmp_path / "case"
        _write_case(case)
        _add_particle_units(case)
        text = "\n".join(run_banner(assemble(case / "params.txt")))
        assert "checked" in text
