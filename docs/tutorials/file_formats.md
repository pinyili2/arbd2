# ARBD File Formats

This tutorial explains the file formats used by ARBD.

## Coordinate Files

ARBD uses simple text-based coordinate files:

```
ATOM 0 argon 1.0 2.0 3.0
ATOM 1 argon 4.0 5.0 6.0
...
```

Each line specifies:
- ATOM keyword
- Atom index
- Atom type
- X, Y, Z coordinates

## Trajectory Files

ARBD outputs trajectories in DCD format, compatible with VMD and other analysis tools.

## Tabulated Potential Files

Tabulated potentials use two-column format:

```
# r    U(r)
0.0    100.0
0.1    90.0
...
```

The first column is the distance, the second is the potential value.

## Bond Files

Bond files define bonded interactions:

```
BOND ADD 0 1 bond-potential.dat
```

This defines a bond between atoms 0 and 1 using the potential in the specified file.

## APBS Grid Files

ARBD can use 3D potential grids in the APBS DX format for external potentials.

## Other File Formats

For more specialized formats, refer to the example files in the test directories.
