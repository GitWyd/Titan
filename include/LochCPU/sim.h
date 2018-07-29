//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_SIM_H
#define LOCH_SIM_H

#include "spring.h"
#include "mass.h"
#include "object.h"
#include "vec.h"

#include <algorithm>
#include <list>
#include <vector>
#include <set>

static double G = 9.81;

class Simulation {
public:
    Simulation() = default;

    ~Simulation();

    Mass * createMass(); // create objects
    Mass * createMass(const Vec & position); // create objects

    Spring * createSpring();
    Spring * createSpring(Mass * m1, Mass * m2, double k = 1.0, double len = 1.0);

    Plane * createPlane(const Vec & abc, double d ); // creates half-space ax + by + cz < d
    Ball * createBall(const Vec & center, double r ); // creates ball with radius r at position center

    Cube * createCube(const Vec & center, double side_length); // creates cube
    Lattice * createLattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);
    Beam * createBeam(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

        void setSpringConstant(double k);
    void defaultRestLength();
    void setMass(double m);
    void setMassDeltaT(double dt);

    Mass * getMass(int i) { return masses[i]; }
    Spring * getSpring(int i) { return springs[i]; }
    ContainerObject * getObject(int i) { return objs[i]; }

    void setBreakpoint(double time);

    void run(); // should set dt to min(mass dt) if not 0, resets everything
    void resume(); // same as above but w/out reset

    double time() { return T; }

    void printPositions();
    void printForces();

    double dt; // set to 0 by default, when run is called will be set to min(mass dt) unless previously set
    double T; // global simulation time

    int RUNNING;

    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
    std::vector<Constraint *> constraints;
    std::vector<ContainerObject *> objs;

    Mass * mass_arr;
    Spring * spring_arr;

#ifdef GRAPHICS
    void clearScreen();
    void renderScreen();
    void updateBuffers();
    void generateBuffers();
    void draw();
#endif

    std::set<double> bpts; // list of breakpoints

    void computeForces();

    Mass * massToArray();
    Spring * springToArray();
    void toArray();

    void massFromArray();
    void springFromArray();
    void fromArray();
};

#endif //LOCH_SIM_H