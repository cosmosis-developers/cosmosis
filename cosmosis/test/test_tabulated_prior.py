import numpy as np
import pytest

from cosmosis.runtime.prior import TabulatedPDF


@pytest.mark.parametrize("density", [
    [-1.0, 2.0, 2.0],
    [2.0, -1.0, 2.0],
    [2.0, 2.0, -1.0],
    [1.0, -2.0, 1.0],
])
def test_tabulated_prior_rejects_negative_density(tmp_path, density):
    table = tmp_path / "prior.txt"
    np.savetxt(table, np.column_stack(([0.0, 1.0, 2.0], density)))

    with pytest.raises(ValueError, match="Negative probability"):
        TabulatedPDF(table)


def test_tabulated_prior_rejects_negative_integral(tmp_path):
    table = tmp_path / "prior.txt"
    # Positive densities still produce a negative integration interval
    # when the table's coordinates are not ordered.
    np.savetxt(table, [[0.0, 1.0], [2.0, 1.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="Negative probability") as error:
        TabulatedPDF(table)
    assert error.value.args[1] == table


def test_tabulated_prior_accepts_zero_density(tmp_path):
    table = tmp_path / "prior.txt"
    np.savetxt(table, [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

    prior = TabulatedPDF(table)

    assert prior(1.0) == pytest.approx(0.0)
    np.testing.assert_allclose(
        prior.denormalize_from_prior([0.0, 0.5, 1.0]), [0.0, 1.0, 2.0]
    )
