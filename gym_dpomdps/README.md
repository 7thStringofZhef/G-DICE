# gym-dpomdps

This repository contains gym environments for flat DPOMDPs which can be loaded
from the `.dpomdp` file format.

## Installation

This package is dependent on the
[rl_parsers](https://github.com/abaisero/rl_parsers/tree/feature_dpomdp) package, specifically the branch with a dpomdp parser.  Install
`rl_parsers` before proceeding.

## Contents

The main contents of the repository are the `DPOMDP` environment, and the
`MultiDPOMDP` wrapper.

### DPOMDP Environment

The DPOMDP environment receives a path to the `.dpomdp` file, and boolean flag
indicating whether the DPOMDP should be considered episodic or continuing.

NOTE:  the episodic version is only supported if the DPOMDP file format makes
use of the custom `reset` keyword (see
[rl_parsers](https://github.com/abaisero/rl_parsers) for details);  if this is
not the case, the two versions are equivalent (and both continuing).

All the DPOMDPs in the `dpomdps/` folder are registered under gym:
 * A continuing version under ID `DPOMDP-{name}-v0`; and
 * An episodic version under ID `DPOMDP-{name}-episodic-v0`.

### MultiPOMDP Wrapper

The MultiDPOMDP Wrapper allows to run multiple indipendent instances of the same
DPOMDP at the same time, and is more efficient that running each instance one
after the other.  The wrapper receives a standard DPOMDP environment and the
number of independent instances to run.  The resulting step function receives
an array of joint actions and returns arrays of observations (for each agent), rewards and dones.
