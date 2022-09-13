# ------------------------------------------------------------------------
# The following Python code is implemented by Professor Terje Haukaas at
# the University of British Columbia in Vancouver, Canada. It is made
# freely available online at terje.civil.ubc.ca together with notes,
# examples, and additional Python code. Please be cautious when using
# this code; it may contain bugs and comes without warranty of any form.
# The good parts of this particular code is from an email that Professor
# Michael Scott at Oregon State sent me, which is gratefully acknowledged.
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# System of springs
#    |                                                               |
#    |-------(Spring 1)--------| DOF 1 >>> |-------(Spring 2)--------|
#    |                                                               |
#    |                                                               | DOF 2 >>>
#    |                                                               |
#    |----------------------------(Spring 3)-------------------------|
#    |                                                               |


# Basic spring responses
def spring1(u):
    force = 10.0 * np.arctan(u)
    stiffness = 10.0 / (1 + u*u)
    return force, stiffness

def spring2(u):
    force = 4.0 * np.arctan(0.5 * u)
    stiffness = 2.0 / (1 + 0.25 * u*u)
    return force, stiffness

def spring3(u):
    force = 7.0 * np.arctan(u)
    stiffness = 7.0 / (1 + u*u)
    return force, stiffness


# Plot the force-deformation relationships
u = np.linspace(0, 10.0, 50)
plt.ion()
fig, ax = plt.subplots()
ax.plot(u, (spring1(u))[0], 'k', label='Spring 1')
ax.plot(u, (spring2(u))[0], 'b', label='Spring 2')
ax.plot(u, (spring3(u))[0], 'r', label='Spring 3')
plt.xlabel("Displacement")
plt.ylabel("Force")
plt.title("Spring Responses")
legend = ax.legend(loc='lower right')
plt.show()
print('\n'"Press any key to continue...")
plt.waitforbuttonpress()


# Transformations from Basic to Final
Tbf1 = np.array([[ 1, 0]])
Tbf2 = np.array([[-1, 1]])
Tbf3 = np.array([[ 0, 1]])


# Analysis setup
tFinal = 1.0
numIncrements = 10
tol = 1e-3
maxIterations = 50
dt = tFinal / numIncrements


# Plot flag
plotResidual = False


# Select type of Newton-Raphson (1=Newton-Raphson) (maxIterations=Modified NR)
stiffnessCalcFrequency = 1


# Applied load
Fref = np.array([0, 10])


# Initial trial displacements
uf = np.array([0.0, 0.0])


# Loop over load steps [*** INCREMENTS ***]
t = 0
loadArray = [0]
disp1Array = [0]
disp2Array = [0]
for n in range (numIncrements):

    # Update pseudo time
    t += dt

    # Keep the user posted
    print('\n'"************ Starting load increment at t = %.2f ************" % t)

    # Set load factor lambda
    lam = t

    # Set current load level
    Ff = lam * Fref

    # Store load level
    loadArray.append(Ff[1])

    # State determination, starting with Basic displacements
    ub1 = np.dot(Tbf1, uf)
    ub2 = np.dot(Tbf2, uf)
    ub3 = np.dot(Tbf3, uf)

    # Basic forces
    Fb1 = (spring1(ub1))[0]
    Fb2 = (spring2(ub2))[0]
    Fb3 = (spring3(ub3))[0]

    # Final force vector
    tildeFf = np.transpose(Tbf1).dot(Fb1) \
              + np.transpose(Tbf2).dot(Fb2) \
              + np.transpose(Tbf3).dot(Fb3)

    # Residual vectdor and its norm
    Rf = tildeFf - Ff
    resNorm = np.linalg.norm(Rf)

    # Newton-Raphson loop [*** ITERATIONS ***]
    m = stiffnessCalcFrequency
    i = 0
    iArray = [str(i)]
    resArray = [resNorm]
    while resNorm > tol and i < maxIterations:

        # Keep the user posted
        print("Starting iteration", i+1)

        # Check if user wants Modified Newton-Raphson
        if m == stiffnessCalcFrequency:

            m = 0

            # Basic stiffnesses
            Kb1 = (spring1(ub1))[1]
            Kb2 = (spring2(ub2))[1]
            Kb3 = (spring3(ub3))[1]

            # Final stiffness matrix
            Kf = np.transpose(Tbf1 * Kb1).dot(Tbf1) \
                 + np.transpose(Tbf2 * Kb2).dot(Tbf2) \
                 + np.transpose(Tbf3 * Kb3).dot(Tbf3)

        # Solve for the displacement increment
        duf = np.linalg.solve(Kf, -Rf)

        # New trial displacements
        uf = uf + duf

        # Store displacement responses
        if i == 0:
            disp1Array.append(uf[0])
            disp2Array.append(uf[1])
        else:
            disp1Array[len(disp1Array)-1] = uf[0]
            disp2Array[len(disp2Array)-1] = uf[1]

        # State determination, starting with Basic displacements
        ub1 = np.dot(Tbf1, uf)
        ub2 = np.dot(Tbf2, uf)
        ub3 = np.dot(Tbf3, uf)

        # Basic forces
        Fb1 = (spring1(ub1))[0]
        Fb2 = (spring2(ub2))[0]
        Fb3 = (spring3(ub3))[0]

        # Final force vector
        tildeFf = np.dot(Tbf1.transpose(), Fb1) \
                  + np.dot(Tbf2.transpose(), Fb2) \
                  + np.dot(Tbf3.transpose(), Fb3)

        # Residual vectdor and its norm
        Rf = tildeFf - Ff
        resNorm = np.linalg.norm(Rf)

        # Increment counters
        m += 1
        i += 1

        # Store the i and the norm of the residual for plotting purposes
        iArray.append(str(i))
        resArray.append(resNorm)

    # Plot the evolution of the residual, if that is asked for
    if plotResidual:
        plt.ion()
        plt.figure(2)
        plt.clf()
        plt.semilogy(iArray, resArray, 'bs-', label='DOF 1')
        plt.xlabel("Iteration")
        plt.ylabel("Residual Norm")
        plt.title("Residual Norms at t=%.2f" % t)
        plt.show()
        print('\n'"Press any key to continue...")
        plt.waitforbuttonpress()

# Response plot
plt.ion()
fig, ax = plt.subplots()
ax.plot(disp1Array, loadArray, 'bs-', label='DOF 1')
ax.plot(disp2Array, loadArray, 'ks-', label='DOF 2')
plt.xlabel("Displacement")
plt.ylabel("Load")
plt.title("Load-Displacement Curves")
legend = ax.legend(loc='lower right')
plt.show()
print('\n'"Press any key to continue...")
plt.waitforbuttonpress()