=======
History
=======

X.Y.Z (DD-MM-YYYY)
------------------
* Hoist ``build_selection`` local declarations to fix Cython 3.2 redeclaration errors (:pr:`229`)

0.5.4 (22-07-2026)
------------------
* Configure storage-manager cache sizes when opening a table (:pr:`222`)

0.5.3 (1-07-2026)
------------------
* Upgrade to casacore 3.8.1 (:pr:`218`)
* Remove extraneous whitespace in casacore patch (:pr:`215`)
* Introduce ``pytest != 9.1.0`` version restriction (:pr:`214`)

0.5.2 (12-06-2026)
------------------
* Build arcae as a static library with minimal symbol exports (:pr:`211`, :pr:`212`)
* Split PYARROW_LIBDIRS into a list before passing to target_link_directories (:pr:`209`, :pr:`210`)
* Pin pyarrow to 23.0.1 in C++ test cases (:pr:`204`)

0.5.1 (02-03-2026)
------------------
* Add an ``arcae.safe_multithreaded_writes`` function indicating whether writing
  from multiple threads is safe (:pr:`199`)

0.5.0 (27-02-2026)
------------------
* Upgrade to pyarrow 23.0.1 (:pr:`196`)
* Upgrade to vcpkg 2025.12.12 (:pr:`193`)
* Remove overlay port for unmaintained wcslib 8.3 (:pr:`193`)
* Fix casacore version in overlay port (:pr:`193`)
* Upgrade to macos-15-intel runners for x86_64 builds (:pr:`193`)
* Upgrade to cython \>= 3.2.0 (:pr:`193`)

0.3.3 (05-11-2025)
------------------
* Deprecate python 3.10 support (:pr:`187`)
* Support python 3.14 (:pr:`187`)
* Upgrade to pyarrow 22 (:pr:`187`)
* Update multithreaded benchmarks to use the newer API (:pr:`181`, :pr:`182`)

0.3.2 (17-09-2025)
------------------
* Address tar.extractall filter warning (:pr:`179`)
* Represent scalar column shapes as an array of uint8's composed of 1s (:pr:`179`)
* Defer to ColumnDesc.ndim if inference from row shapes fails for variably shaped columns (:pr:`179`)
* Run C++ test suite on macos (:pr:`178`)
* Default arrow allocation alignment if type size is not a power of 2 (:pr:`176`, :pr:`177`)
* Support missing column rows when result arrays are supplied during reads (:pr:`174`)
* Correct setting the shapes of missing column row rows during writes of
  variably shaped data (:pr:`174`)
* Fixes a data partitioning bug where the shape for a particular row was
  associated with a raw index into the result shape, instead of a memory index (:pr:`174`)
* Disable overly aggressive casacore logging of non-standard data types (:pr:`173`)

0.3.1 (10-09-2025)
------------------
* Replace deprecated appdirs with platformdirs (:pr:`172`)
* Build all wheels for an architecture with a single cibuildwheel call (:pr:`171`)
* Reintroduce vcpkg build artifact caching (:pr:`171`)
* Add a ``table.row_shapes`` method that returns column row shapes
  as an Arrow Array (:pr:`169`)
* Impute missing rows from result shape (:pr:`169`)

0.3.0 (21-06-2025)
------------------
* Upgrade to pyarrow 21 (:pr:`160`)
* Upgrade to pyarrow 20 (:pr:`157`)
* Support python 3.13 (:pr:`157`)

0.2.9 (10-06-2025)
------------------
* Avoid clipping selections against the result array shape for ignored rows for read operations (:pr:`154`)
* Add cfitsio mirrors (:pr:`153`)
* Disable use of github actions cache by vcpkg (:pr:`152`)
* Upgrade to C++20 (:pr:`150`)
* Upgrade to cibuildwheel 2.23.3 (:pr:`149`)
* Slight loop optimisation (:pr:`147`, :pr:`148`)
* Upgrade to vcpkg 2025.04.09 (:pr:`146`)

0.2.8 (24-03-2025)
------------------
* Upgrade to casacore 3.7.1 (:pr:`144`)
* Use cmake.version instead of deprecated cmake.mininum_version (:pr:`142`)
* Fix requires_python setting in pyproject.toml (:pr:`142`)

0.2.7 (28-01-2025)
------------------
* Link subtables against the Measurement Set during creation (:pr:`140`)
* Prevent print segmentation faults caused by the use of PrimitiveArrays (:pr:`139`)
* Update to vcpkg 2024.12.15 (:pr:`138`)
* Use github arm64 runners (:pr:`138`)
* Constrained mininum MACOSX_DEPLOYMENT_TARGET to 13.0 on Intel x86 and 14.0 on Darwin arm64 (:pr:`138`)

0.2.6 (10-09-2024)
------------------
* Disable tmate in continuous integration (:pr:`135`)
* Update to latest cibuildwheel (:pr:`131`, :pr:`133`)
* Use conda forge compilers on MACOSX and update MACOX_DEPLOYMENT_TARGET=12.0 (:pr:`131`, :pr:`133`)
* Avoid sorting sorted indices (:pr:`129`, :pr:`130`)

0.2.5 (04-10-2024)
------------------
* Support C++ merging of sorted NumPy partitions (:pr:`127`)
* Deprecate Python 3.9 support (:pr:`125`)
* Upgrade to casacore 3.6.1 (:pr:`124`)
* Build against NumPy 2 (:pr:`122`)
* Add pre-commit hooks, install formatters and linters (:pr:`121`)
* Support adding columns (:pr:`120`)

0.2.4 (05-09-2024)
------------------
* Disable OpenBLAS threading (:pr:`117`)
* Move Cython source to lib directory (:pr:`116`)
* Upgrade to pyarrow 16.1.0 (:pr:`115`)
* Code formatting nits (:pr:`114`)
* Multiplex CASA Table I/O over multiple table instances (:pr:`113`)
* Handle arrays and tables with no rows (:pr:`112`)
* Drop the Global Interpreter Lock (:pr:`111`)
* Remove FFTW3 and casacore apps from the casacore build (:pr:`110``)
* Add table name method (:pr:`109`)
* Re-implement partial support for selection over unconstrained columns
  that, in practice, have the same rank (:pr:`108`)
* Unpin manylinux_2_28_x86_64 image (:pr:`107`)
* Pin cython to less than 3.0.10 (:pr:`106`)
* Use casacore::RefRows for indexing the row dimension (:pr:`105`)
* Refactor arcae to use a finer-grained execution model (:pr:`101`)
* Pin manylinux_2_28 image to manylinux_2_28_x86_64:2024.07.02-0 (:pr:`102`)
* Restrict Numpy to less than 2.0.0 (:pr:`100`)
* Avoid stripping debug information (:pr:`96`)
* Set cmake build type to RelWithDebInfo (:pr:`96`)
* Avoid creating ColumnDesc objects in inner loops (:pr:`95`)
* Support Table arguments in TAQL queries (:pr:`93`)
* Upgrade to pyarrow 16.0.0 (:pr:`92`)
* Handle slice(None) in getcol index (:pr:`91`)

0.2.3 (04-08-2024)
------------------
* Remove unused utility code (:pr:`90`)
* Upgrade to pyarrow 15.0.2 (:pr:`89`)
* Fix python 3.9 typing (:pr:`87`, :pr:`88`)
* Support table lock options (:pr:`86`)
* Support complex indexing (:pr:`65`)
* Configure dependabot (:pr:`85`)
* Enable dependabot version updates (:pr:`84`)
* Upgrade to cibuildwheel 2.17 (:pr:`83`)
* Upgrade to pyarrow 15.0.0 (:pr:`76`)
* Build linux arm64 and macos arm64/x86_64 wheels (:pr:`76`)
* Upgrade vcpkg version to include wcslib 8.2.1 (:pr:`82`)
* Export compile_commands.json by default (:pr:`81`)
* Make export application dependencies optional (:pr:`80`)
* Fix ENV access within cmake files (:pr:`79`)

0.2.2 (10-11-2023)
------------------
* Upgrade to pyarrow 14.0.1 (:pr:`77`)
* Remove WITH_SOABI workaround (:pr:`75`)
* Add test case demonstrating the feasibility of parallel writes to
  Tiled Storage Manager Columns (:pr:`69`)
* Update README.rst to show that it is possible to convert
  unconstrained columns (ndim==-1) whose rows have the same rank (:pr:`74`)
* Guard against python-casacore imports (:pr:`73`)

0.2.1 (24-10-2023)
------------------
* Table Query Language Support (:pr:`71`)
* Set skip-existing=true when uploading distributables to TestPyPI (:pr:`68`)

0.2.0 (19-10-2023)
------------------
* Support multiple table objects reading from the same underlying table in multiple threads (:pr:`67`)
* Don't hold the GIL when calling GetResultValue (:pr:`66`)
* Add support for a C++ test suite (:pr:`64`)
* Use underscore for ColumnConvertVisitor member names (:pr:`62`)
* Migrate build system to scikit-build-core (:pr:`61`)
* Upgrade to Cython 3 and pyarrow 13.0.0 (:pr:`60`)
* Introduce a more canonical C++ project structure (:pr:`57`. :pr:`59`)
* Consistently use CamelCase throughout the C++ layer (:pr:`56`)
* Support getcol, tabledesc and getcoldesc (:pr:`55`, :pr:`58`)
* Enable initial OSX support in the build process (:pr:`54`)
* Add support for adding rows to a table (:pr:`53`)
* Create and use JSON Table Descriptors and Data Managers (:pr:`51`)
* Use ccache, if available (:pr:`50`)
* Use vcpkg's internal github actions binary caching (:pr:`49`)
* Generalise the opening and creation of Tables (:pr:`48`)
* Optimise storage and passing of TableProxy objects (:pr:`46`)
* Convert SAFE_TABLE_FUNCTOR from macro to template function (:pr:`45`)
* Fix `export CIBW_TEST_SKIP` (:pr:`42`)

0.1.0 (30-06-2023)
------------------
* First release
