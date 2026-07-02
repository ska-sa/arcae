# Needed to load in the arrow c++ shared libraries
import pyarrow as pa  # noqa
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from arcae.lib.arrow_tables import Table

__version__ = "0.4.0-alpha.7"


def safe_multithreaded_writes() -> bool:
    """Returns True if this version of arcae supports safe
    multithreaded writes"""
    from arcae.lib.arrow_tables import safe_multithreaded_writes as safe_mt_writes

    return safe_mt_writes()


def table(
    filename: str,
    ninstances: int = 1,
    readonly: bool = True,
    lockoptions: Union[str, dict] = "auto",
) -> "Table":
    # Defer cython module import, to avoid conflicts between arcae casacore libraries
    # and python-casacore casacore libraries
    from arcae.lib.arrow_tables import Table

    return Table.from_filename(filename, ninstances, readonly, lockoptions)
