# G-DICE
## Basic Usage
The main script is generalGDICE.py. Running it as is will run GDICE on the 4x3 maze POMDP environment with some default parameters. You can alter the main script as follows:

1. Choose the environment in the first line. Any POMDP registered in gym-pomdps is accessible, though some (e.g., rocksample) may cause memory errors. Reference them by name (i.e., "POMDP-4x3-episodic-v0")
    * NOTE: Rocksample is too big to fit on a system with 32 GB of memory...
2. Create a controller distribution for the agent in the second line, specifying the number of nodes in the first argument.
3. Define your parameters with a GDICEParams object in the 3rd line. In the constructor, you can specify:
    1. Number of iterations
    2. Number of controller samples per iteration
    3. Number of simultations for each controller to run on environment
    4. Number of best samples to update with in each iteration
    5. The learning rate of the controller distribution
    6. A value threshold which additionally filters out samples below a certain value. By default, this is off (None)
4. Define a pool object in the 4th line if you want parallel processing.

## Dependencies
* Python 3
* Numpy
* Gym
* Andrea's repositories, rl_parsers and gym_pomdps. These are expected to be pip installed with their included script
    * https://github.ccs.neu.edu/abaisero/gym-pomdps
    * https://github.com/abaisero/rl_parsers

## Future work
* Create a grid search function to automatically work through environments to determine the best G-DICE parameters
* Extend to DPOMDPs with a parser
* Make the parallel approach more memory efficient
* Extend to continuous observation domains
* Apply to gym-minigrid