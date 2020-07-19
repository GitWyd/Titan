//
// Created by Jacob Austin on 5/17/18.
//
#include "mass.h"

namespace titan {

Mass::Mass() {
    m = 1.0;
    T = 0;
    valid = true;
    arrayptr = nullptr;
    ref_count = 0;

#ifdef GRAPHICS
    color = Vec(1.0, 0.2, 0.2);
#endif
} // constructor TODO fix timing

void Mass::operator=(CUDA_MASS & mass) {
    m = mass.m;
    T = mass.T;
    pos = mass.pos;
    vel = mass.vel;
    valid = mass.valid;
    rad = mass.rad; // magnet_sphere radius
    stiffness = mass.stiffness;
    max_mag_force = mass.max_mag_force; // maximum pull force excerted by the magnet
    mag_scale_factor = mass.mag_scale_factor; // scales susceptibility to magnetic flux

    constraints.fixed = mass.constraints.fixed;

    acc = mass.acc;
    extern_force = mass.extern_force;

    ref_count = this -> ref_count;
    arrayptr = this -> arrayptr;

#ifdef CONSTRAINTS
    constraints = this -> constraints;
#endif

#ifdef GRAPHICS
    color = mass.color;
#endif
}

Mass::Mass(const Vec & position, double mass, bool fixed, double radius, double mag_k,
           double maximum_magnet_force, double magnet_scale_factor) {
    m = mass;
    pos = position;

    rad = radius;
    stiffness = mag_k;
    max_mag_force = maximum_magnet_force;
    mag_scale_factor = magnet_scale_factor;

    constraints.fixed = fixed;

    T = 0;
    
    valid = true;
    arrayptr = nullptr;
    ref_count = 0;

#ifdef GRAPHICS
    color = Vec(1.0, 0.2, 0.2);
#endif
}

CUDA_MASS::CUDA_MASS(Mass &mass) {
    m = mass.m;
    T = mass.T;
    
    pos = mass.pos;
    vel = mass.vel;
    extern_force = mass.extern_force;

    rad = mass.rad;
    stiffness = mass.stiffness; // spring constant of the magnet shell
    max_mag_force = mass.max_mag_force;
    mag_scale_factor = mass.mag_scale_factor;

    constraints.fixed = mass.constraints.fixed;

    valid = true;

#ifdef CONSTRAINTS
    constraints = CUDA_LOCAL_CONSTRAINTS(mass.constraints);
#endif

#ifdef GRAPHICS
    color = mass.color;
#endif
}

#ifdef CONSTRAINTS

void Mass::addConstraint(CONSTRAINT_TYPE type, const Vec & vec, double num) { // TODO make this more efficient
    if (type == 0) {
        this -> constraints.constraint_plane.push_back(CudaConstraintPlane(vec, num));
        this -> constraints.num_constraint_planes++;
        this -> constraints.constraint_plane_ptr = thrust::raw_pointer_cast(constraints.constraint_plane.data());
    } else if (type == 1) {
        this -> constraints.contact_plane.push_back(CudaContactPlane(vec, num));
        this -> constraints.num_contact_planes++;
        this -> constraints.contact_plane_ptr = thrust::raw_pointer_cast(constraints.contact_plane.data());
    } else if (type == 2) {
        this -> constraints.ball.push_back(CudaBall(vec, num));
        this -> constraints.num_balls++;
        this -> constraints.ball_ptr = thrust::raw_pointer_cast(constraints.ball.data());
    } else if (type == 3) {
        this -> constraints.direction.push_back(CudaDirection(vec, num));
        this -> constraints.num_directions++;
        this -> constraints.direction_ptr = thrust::raw_pointer_cast(constraints.direction.data());
    }
}

void Mass::clearConstraints(CONSTRAINT_TYPE type) {
    if (type == 0) {
        this -> constraints.constraint_plane.clear();
        this -> constraints.constraint_plane.shrink_to_fit();
        this -> constraints.num_constraint_planes = 0;
    } else if (type == 1) {
        this -> constraints.contact_plane.clear();
        this -> constraints.contact_plane.shrink_to_fit();
        this -> constraints.num_contact_planes = 0;
    } else if (type == 2) {
        this -> constraints.ball.clear();
        this -> constraints.ball.shrink_to_fit();
        this -> constraints.num_balls = 0;
    } else if (type == 3) {
        this -> constraints.direction.clear();
        this -> constraints.direction.shrink_to_fit();
        this -> constraints.num_directions = 0;
    }
}

void Mass::clearConstraints() {
    clearConstraints(CONSTRAINT_PLANE);
    clearConstraints(CONTACT_PLANE);
    clearConstraints(DIRECTION);
    clearConstraints(BALL);
}

void Mass::fix() {
    this -> constraints.fixed = true;
}
void Mass::unfix() {
    this -> constraints.fixed = false;
}

void Mass::setDrag(double C) {
    this -> constraints.drag_coefficient = C;
}


#endif

void Mass::decrementRefCount() {
    if (--ref_count == 0) {

        if (arrayptr) {
            cudaFree(arrayptr);
        }

        delete this;
    }
}

} // namespace titan