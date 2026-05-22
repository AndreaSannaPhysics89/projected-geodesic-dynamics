# Projected Geodesic Dynamics on a Sphere

Closed-form derivation and dual verification (symbolic + numerical) of the
apparent dynamics seen when a free geodesic on a sphere is observed through
its orthogonal projection onto a plane.

## Problem

A point moves freely along a great circle (a geodesic) of a sphere in
three-dimensional space, but an observer confined to the equatorial plane can
only see its orthogonal projection. What apparent dynamics does that observer
reconstruct from the projected motion alone, and what governs the relationship
between their clock and the motion's natural parameter?

## Approach

The core question is the time map dt/dλ — how the observer's time relates to
the affine parameter along the geodesic. Three candidate maps can be written
down, but only one is physically admissible. For that map I derive closed-form
expressions for:

- the apparent velocity,
- the apparent acceleration,
- a conserved invariant of the projected motion,

and show that a special-relativistic time-dilation factor emerges naturally
from the geometry.

Every closed form is reproduced independently in two ways:

- **symbolically**, in SymPy, with each algebraic identity verified to exactly zero;
- **numerically**, by direct integration, with agreement to machine precision (~10⁻¹⁵).

No result is accepted unless it survives both methods.

## Files

- `sphere_projection.py` — numerical study; compares the three time maps,
  computes the closed-form velocity / acceleration / invariant, cross-checks
  each against a numerical derivative, and generates the figures.
- `symbolic_projection_derivation.py` — symbolic derivation in SymPy, with
  every identity verified to zero.
- `derivation_note.pdf` — the full written derivation, with proofs.
- `project_overview.pdf` — one-page summary of the problem and what it demonstrates.

## Requirements

`numpy`, `scipy`, `sympy`, `matplotlib`.
