# PNS-Project
My finals project for my undergraduate degree.

Simulates the SPRINT based PNS attack shown in the paper by Ariel Ashkenazy, Yuval Idan, Dor Korn, Dror Fixler, Barak Dayan, Eliahu Cohen [[1]](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/qute.202300437) and outputs fidelity plots of both arms.

To run main both func and vis are required. The parameters of the simulation are:
- Integer \( N = 2 \) - The dimension of the Hilbert space.
- Integer lowest\_state = 0 - Determinants the lowest state permitted in the coherent state in the input.
- Float theta = 0 - Determinants the angle of the beam splitter (if 0, the identity operator is used).
- Boolean mid\_reset = False - Determinants whether or not we reset the atom in between the early and late time bins.
- Integer num\_of\_sprints = 1 - Determinants the number of times the state will be put through the SPRINT system (the atom is reset every time. If mid\_reset is set to true, at each pass through the atom, it will be reset between      the two time bins).
