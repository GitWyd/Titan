//
// Created by Jacob Austin on 5/17/18.
//

#ifndef TITAN_MASS_H
#define TITAN_MASS_H

#include "vec.h"
#include "object.h"

namespace titan {

class Mass;
struct CUDA_MASS;

class Mass {
public:
    Mass(const Vec & position, double mass = 0.1, bool fixed = false, double radius = 0.0, double stiffness = 1000,
         double maximum_magnet_force = 0.0, double magnet_scale_factor = 0.0);

    //Properties
    double m; // mass in kg
    double T; // local time
    Vec pos; // position in m
    Vec vel; // velocity in m/s

    // Magnet Mass related properties
    double rad; // magnet_sphere radius
    double stiffness; // spring constant of the magnet shell
    double max_mag_force; // maximum pull force excerted by the magnet
    double mag_scale_factor; // scales susceptibility to magnetic flux
    bool isMagnetic() {return (bool) round(max_mag_force);}
    void setExternalForce(const Vec & v) { extern_force = v; }
    Vec acceleration() { return acc; }

    // occupancy grid idx
    int og_idx; // 1D idx of 2D occupancy grid

#ifdef CONSTRAINTS
    void addConstraint(CONSTRAINT_TYPE type, const Vec & vec, double num);
    void clearConstraints(CONSTRAINT_TYPE type);
    void clearConstraints();

    void setDrag(double C);
    void fix();
    void unfix();
#endif
    
#ifdef GRAPHICS
    Vec color;
#endif

private:
    Vec acc; // acceleration in m/s^2
    Vec extern_force; // force in kg m / s^2

    bool valid;
    int ref_count;

    void decrementRefCount();

    CUDA_MASS * arrayptr; //Pointer to struct version for GPU cudaMemAlloc

#ifdef RK2
    Vec __rk2_backup_pos;
    Vec __rk2_backup_vel;
#endif

    Mass();
    void operator=(CUDA_MASS & mass);

    friend class Simulation;
    friend class Spring;
    friend struct CUDA_SPRING;
    friend struct CUDA_MASS;
    friend class Container;
    friend class Lattice;
    friend class Beam;
    friend class Cube;
    friend class RobotLink;

#ifdef CONSTRAINTS
    LOCAL_CONSTRAINTS constraints;

#endif

};
//Struct with CPU Spring properties used for optimal memory allocation on GPU memory
struct CUDA_MASS {
    CUDA_MASS() = default;
    CUDA_MASS(Mass & mass);

    double m; // mass in kg
    double T; // local time

    // Magnet Mass related properties
    double rad; // magnet_sphere radius
    double stiffness; // spring constant of the magnet shell
    double max_mag_force; // maximum pull force exerted by the magnet
    double mag_scale_factor; // scales susceptibility to magnetic flux

    // occupancy grid idx
    int og_idx; // 1D idx of 2D occupancy grid

    Vec pos; // position in m
    Vec vel; // velocity in m/s
    Vec acc; // acceleration in m/s^2
    Vec force; // vector to accumulate external forces
    Vec extern_force; // external force applied every time step

#ifdef RK2
    Vec __rk2_backup_pos;
    Vec __rk2_backup_vel;
#endif

#ifdef GRAPHICS
    Vec color;
#endif

    bool valid;

#ifdef CONSTRAINTS
    CUDA_LOCAL_CONSTRAINTS constraints;
#endif

};

} // namespace titan

#endif //TITAN_MASS_H
