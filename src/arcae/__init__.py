# Needed to load in the arrow c++ shared libraries
import pyarrow as pa  # noqa
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from arcae.lib.arrow_tables import Table

__version__ = "0.5.3"


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
    cache_size: Union[int, dict, None] = None,
) -> "Table":
    """Open a CASA table.

    Parameters
    ----------
    filename:
        Path to the table.
    ninstances:
        Number of independent casacore table instances to open. Reads are
        multiplexed across them; each instance has its own storage-manager
        caches.
    readonly:
        Open the table read-only when True.
    lockoptions:
        casacore locking options.
    cache_size:
        Bound on the (otherwise unbounded) tiled storage-manager caches, in
        **MiB** per storage manager. ``0`` means unbounded (the casacore
        default). May be:

        * ``None`` (default) -- apply a bounded default of 128 MiB.
        * an ``int`` -- a single global default for every storage manager.
        * a ``dict`` combining three levels via prefixed keys, resolved with
          precedence ``column:`` > ``stman:`` > ``default``::

              {
                  "default": 128,             # unmatched columns
                  "stman:TiledData": 256,     # a whole storage manager
                  "column:WEIGHT_SPECTRUM": 32,  # a single column
              }

        A per-column cap physically targets that column's storage manager, so
        columns sharing a hypercolumn share the cap (last write wins). The
        worst-case resident cache is roughly
        ``ninstances x (number of tiled storage managers) x cache_size`` MiB.
    """
    # Defer cython module import, to avoid conflicts between arcae casacore libraries
    # and python-casacore casacore libraries
    from arcae.lib.arrow_tables import Table

    return Table.from_filename(filename, ninstances, readonly, lockoptions, cache_size)
