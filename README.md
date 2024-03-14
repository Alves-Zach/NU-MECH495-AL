# Hot to run code
### Main genetic algorithm:
To run creature generation run `python3 algorithm.py`

In `algorithm.py`, there are parameters that you can modify to run the simulation differently, such as



`numChildren` - Modifies the number of children per generation

`numGenerations` - Modifies the number of generations run throughout the simulation



Starting values such as:

`legSize` - The median leg size of the first generation of creatures

`speed` - The median speed of the legs of the first generation of creatures

`gear` - The median torque of the legs of the first generation of creatures

`numLeftLegs` - The number of left legs for the first generation of creatures

`numRightLegs` - The number of left legs for the first generation of creatures

### Testing a single creature at a time
To run one creature at a time run `python3 TestSingleCreature.py`

Prompts asking the user to enter the creatures parameters will come up on the terminal, after entering in each parameter, the creature will be simulated.

# Analysis of simulation
The starting parameters seemed to be fairly good at making the creatures move forward. Though clearly the creatures are evolving over time; multiple times during the simulation, generations of creatures got into "ruts" where both generational average and best performances would be significantly lower than previous generations. Though it can also be clearly seen that one great creature out of that generation can bring up the next generation significantly. One good example of this can be seen around generation 190 in the chart below.

![Average and Best distances per generation](Graphs/GenDist(300g10c).png)

Similarly, when the best child of a generation is significantly worse than any child in the previous generation puts the next generation into a new "rut".


This chart shows the features of each generation

![alt text](Graphs/BestFeatures(300g10c).png)

Due to the large number of parameters that are varied throughout each generation, a direct corilation from feature to performance is hard to find, meaning that the fitness landscape would be even more complicated. 