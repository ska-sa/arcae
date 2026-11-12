#!/bin/sh

OS=$(uname -s)
ARCH=$(uname -m)

# The static monolith should export exactly one symbol: the module init
# entry point. Anything else means statically-linked dependencies (Arrow,
# casacore, the C++ runtime) are leaking and could clash with other
# extensions such as python-casacore.
check_exported_symbols() {
  SO=$(python -c "import arcae.lib.arrow_tables as m; print(m.__file__)")
  echo "Checking exported symbols of $SO"

  if [ "$OS" = "Darwin" ]; then
    # macOS mangles C symbols with a leading underscore; llvm-nm uses
    # -g (external only) and -U (defined only).
    EXPECTED="_PyInit_arrow_tables"
    SYMBOLS=$(nm -gU "$SO" | awk '{print $NF}' | grep . | sort -u)
  else
    # GNU nm needs -D to read the dynamic symbol table of a (possibly
    # stripped) shared object.
    EXPECTED="PyInit_arrow_tables"
    SYMBOLS=$(nm -D --defined-only "$SO" | awk '{print $NF}' | grep . | sort -u)
  fi

  if [ "$SYMBOLS" = "$EXPECTED" ]; then
    echo "OK: only $EXPECTED is exported"
  else
    echo "ERROR: unexpected exported symbols (expected only $EXPECTED):"
    echo "$SYMBOLS"
    exit 1
  fi
}

# On free-threaded CPython, importing an extension that isn't marked
# free-threading compatible silently re-enables the GIL (with a warning)
check_gil_disabled() {
  python -c "
import sys, sysconfig
if sysconfig.get_config_var('Py_GIL_DISABLED'):
    import arcae.lib.arrow_tables
    if sys._is_gil_enabled():
        sys.exit('ERROR: importing arcae re-enabled the GIL')
    print('OK: the GIL stays disabled after importing arcae')
"
}

if [ "$OS" = "Linux" ] || [ "$OS" = "Darwin" ]; then
  check_exported_symbols
fi

check_gil_disabled || exit 1

if [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
  # We can only rely on the python-casacore wheels for linux x86
  echo "Running complete test suite on $OS $ARCH"
  python -m pytest -s -vvv --pyargs arcae
else
  echo "Running sanity check test on $OS $ARCH"
  python -c "from arcae.testing import sanity; sanity()"
fi
