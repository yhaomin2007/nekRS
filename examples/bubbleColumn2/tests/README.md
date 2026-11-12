# Test scope

Active simplified case:

- `native_transport.py`: actual new compression, UL EXT and gas-drag kernels;
  actual gas-drag history helper; native scalar sumMakef; independently expected
  BDF/EXT RHS, changing coefficients, repeated samples and padding; source routing.
- `native_urst.py`: actual UDF launch wrapper against the native ten-argument
  kernel signature; typed stride/memory routing, and optional real native OKL
  execution with OCCA Serial when `OCCA_DIR` is set.
- `native_boundary.py`: actual active boundary code, uniform inlet values and
  shared inlet/wall rim precedence.

Other scripts cover the preserved historical implementation, including pressure
correctors, custom conservative transport, VM variants and extra filters. Their
UDF/configuration references use `bubbleColumn2Legacy.udf/.par`; they do not
claim those features are used by the active simplified case.

`method1_alignment.py` is a historical comparison, valid for its original
Method1 reference. Current Method1 has evolved separately, so that comparison
is skipped by default if its equation-source blob differs from the known
reference. To run against a frozen original reference, set
`METHOD1_LEGACY_REFERENCE=/path/to/original/bubbleColumn`.

Python, g++ and repository source headers/kernels are required. The historical
explicit-filter module test additionally requires OCCA (`OCCA_DIR`). These are
host/kernel checks, not full NekRS MPI/GPU integration or stability tests.
