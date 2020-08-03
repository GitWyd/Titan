//
// Created by Jacob Austin on 5/21/18.
// object.cu defines constraint objects like planes and balls that allow the users
// to enforce limitations on movements of objects within the scene.
// Generally, an object defines the applyForce method that determines whether to apply a force
// to a mass, for example a normal force pushing the mass out of a constaint object or
// a frictional force.

#include "object.h"
#include <cmath>
#include "sim.h"

#ifdef GRAPHICS
#define GLM_FORCE_PURE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp> // for rotation
#endif

namespace titan {

#ifdef GRAPHICS
const Vec RED(1.0, 0.2, 0.2);
const Vec GREEN(0.2, 1.0, 0.2);
const Vec BLUE(0.2, 0.2, 1.0);
const Vec PURPLE(0.5, 0.2, 0.5);

#endif

__device__ const double NORMAL = 20000; // normal force coefficient for contact constaints

#ifdef CONSTRAINTS
void Container::addConstraint(CONSTRAINT_TYPE type, const Vec & v, double d) {
    for (Mass * m : masses) {
        m -> addConstraint(type, v, d);
    }
}

void Container::clearConstraints() {
    for (Mass * m : masses) {
        m -> clearConstraints();
    }
}

#endif

CUDA_CALLABLE_MEMBER CudaBall::CudaBall(const Vec & center, double radius) {
    _center = center;
    _radius = radius;
}

CUDA_CALLABLE_MEMBER CudaBall::CudaBall(const Ball & b) {
    _center = b._center;
    _radius = b._radius;
}

CUDA_CALLABLE_MEMBER void CudaBall::applyForce(CUDA_MASS * m) {
    double dist = (m -> pos - _center).norm();
    m -> force += (dist <= _radius) ? NORMAL * (m -> pos - _center) / dist : Vec(0, 0, 0);
}

CUDA_CALLABLE_MEMBER CudaContactPlane::CudaContactPlane(const Vec & normal, double offset) {
    _normal = normal / normal.norm();
    _offset = offset;
    _FRICTION_S = 0.0;
    _FRICTION_K = 0.0;
}

CudaContactPlane::CudaContactPlane(const ContactPlane & p) {
    _normal = p._normal;
    _offset = p._offset;

    _FRICTION_S = p._FRICTION_S;
    _FRICTION_K = p._FRICTION_K;
}

CUDA_CALLABLE_MEMBER void CudaContactPlane::applyForce(CUDA_MASS * m) {
    //    m -> force += (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // TODO fix this for the host

    double disp = dot(m -> pos, _normal) - _offset; // displacement into the plane
    Vec f_normal = dot(m -> force, _normal) * _normal; // normal force

    if (disp < 0 && (_FRICTION_S > 0 || _FRICTION_K > 0)) { // if inside the plane
        Vec v_perp = m -> vel - dot(m -> vel, _normal) * _normal; // perpendicular velocity
        double v_norm = v_perp.norm();

        if (v_norm > 1e-16) { // kinetic friction domain
            double friction_mag = _FRICTION_K * f_normal.norm();
            m->force -= v_perp * friction_mag / v_norm;
        } else { // static friction
            Vec f_perp = m -> force - f_normal; // perpendicular force
	        if (_FRICTION_S * f_normal.norm() > f_perp.norm()) {
                m -> force -= f_perp;
	        } // else { // kinetic domain again
            //     double friction_mag = _FRICTION_K * f_normal.norm();
            //     m->force -= v_perp * friction_mag / v_norm;
	        // }
        }
    }

    // now apply the offset force to push the object out of the plane.
    // if (disp < 0) {
    //     m -> pos[2] = 0;
    //     m -> vel = m -> vel - 2 * dot(m -> vel, _normal) * _normal;
    //     m -> force -= f_normal;
    // }

    Vec contact = (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // displacement force
    m -> force += contact;
}

CUDA_CALLABLE_MEMBER CudaConstraintPlane::CudaConstraintPlane(const Vec & normal, double friction) {
    assert(normal.norm() != 0.0);

    _normal = normal / normal.norm();
    _friction = friction;
}

CUDA_CALLABLE_MEMBER void CudaConstraintPlane::applyForce(CUDA_MASS * m) {
    double normal_force = dot(m -> force, _normal);
    m -> force += - _normal * normal_force; // constraint force
    double v_norm = m -> vel.norm();

    if (v_norm >= 1e-16) {
        m -> vel += - _normal * dot(m -> vel, _normal); // constraint velocity
        m -> force += - _friction * normal_force * m -> vel / v_norm; // apply friction force
    }
}

CUDA_CALLABLE_MEMBER CudaDirection::CudaDirection(const Vec & tangent, double friction) {
    assert(tangent.norm() != 0.0);

    _tangent = tangent / tangent.norm();
    _friction = friction;
}

CUDA_CALLABLE_MEMBER void CudaDirection::applyForce(CUDA_MASS * m) {
    Vec normal_force = m -> force - dot(m -> force, _tangent) * _tangent;
    m -> force += - normal_force;

    if (m -> vel.norm() >= 1e-16) {
        m -> vel = _tangent * dot(m -> vel, _tangent);
        m -> force += - normal_force.norm() * _friction * _tangent;
    }
}

void Container::setMassValues(double m) { // set masses for all Mass objects
    for (Mass * mass : masses) {
        mass -> m += m;
    }
}

void Container::setSpringConstants(double k) {
    for (Spring * spring : springs) {
        spring -> _k = k;
    }
}

void Container::setRestLengths(double len) { // set masses for all Mass objects
    for (Spring * spring : springs) {
        spring -> _rest = len;
    }
}

void Container::add(Mass * m) {
    masses.push_back(m);
}

void Container::add(Spring * s) {
    springs.push_back(s);
}

void Container::add(Container * c) {
    for (Mass * m : c -> masses) {
        masses.push_back(m);
    }

    for (Spring * s : c -> springs) {
        springs.push_back(s);
    }
}

Cube::Cube(const Vec & center, double side_length) {
    _center = center;
    _side_length = side_length;

    for (int i = 0; i < 8; i++) {
        masses.push_back(new Mass(side_length * (Vec(i & 1, (i >> 1) & 1, (i >> 2) & 1) - Vec(0.5, 0.5, 0.5)) + center));
    }

    for (int i = 0; i < 8; i++) { // add the appropriate springs
        for (int j = i + 1; j < 8; j++) {
            springs.push_back(new Spring(masses[i], masses[j]));
        }
    }

    for (Spring * s : springs) {
        s -> setRestLength((s -> _right -> pos - s -> _left -> pos).norm());
    }
}

void Container::translate(const Vec & displ) {
    for (Mass * m : masses) {
        m -> pos += displ;
    }
}

void Container::rotate(const Vec & axis, double angle) {
    Vec com(0, 0, 0);

    double total_mass = 0;

    for (Mass * m : masses) {
        com += m -> m * m -> pos;
        total_mass += m -> m;
    }

    com = com / total_mass; // center of mass as centroid
    Vec temp_axis = axis / axis.norm();

    for (Mass * m : masses) {
        Vec temp = m -> pos - com; // subtract off center of mass
        Vec y = temp - dot(temp, temp_axis) * temp_axis; // project onto the given axis and find offset (y coordinate)

        if (y.norm() < 0.0001) { // if on the axis, don't do anything
            continue;
        }

        Vec planar(-sin(angle) * y.norm(), cos(angle) * y.norm(), 0); // coordinate in xy space
        Vec spatial = planar[0] * cross(temp_axis, y / y.norm()) + y / y.norm() * planar[1] + dot(temp, temp_axis) * temp_axis + com; // return to 3D space, then to COM space, then to absolute space

        m -> pos = spatial; // update position
    }
}

Lattice::Lattice(const Vec & center, const Vec & dims, int nx, int ny, int nz) {
    _center = center;
    _dims = dims;
    this -> nx = nx;
    this -> ny = ny;
    this -> nz = nz;

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                masses.push_back(new Mass(Vec((nx > 1) ? (double) i / (nx - 1.0) - 0.5 : 0, (ny > 1) ? j / (ny - 1.0) - 0.5 : 0, (nz > 1) ? k / (nz - 1.0) - 0.5 : 0) * dims + center));
            }
        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                for (int l = 0; l < ((i != nx - 1) ? 2 : 1); l++) {
                    for (int m = 0; m < ((j != ny - 1) ? 2 : 1); m++) {
                        for (int n = 0; n < ((k != nz - 1) ? 2 : 1); n++) {
                            if (l != 0 || m != 0 || n != 0) {
                                springs.push_back(new Spring(masses[k + j * nz + i * ny * nz],
                                                             masses[(k + n) + (j + m) * nz + (i + l) * ny * nz]));
                            }
                        }
                    }
                }

                if (k != nz - 1) {
                    if (j != ny - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz], // get the full triangle
                                                     masses[k + (j + 1) * nz + i * ny * nz]));
                    }

                    if (i != nx - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz],
                                                     masses[k + j * nz + (i + 1) * ny * nz]));
                    }

                    if (j != ny - 1 && i != nx - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz],
                                                     masses[k + (j + 1) * nz + (i + 1) * ny * nz]));
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + (i + 1) * ny * nz],
                                                     masses[k + (j + 1) * nz + i * ny * nz]));
                        springs.push_back(new Spring(masses[(k + 1) + (j + 1) * nz + i * ny * nz],
                                                     masses[k + j * nz + (i + 1) * ny * nz]));
                    }
                }

                if (j != ny - 1 && i != nx - 1) {
                    springs.push_back(new Spring(masses[k + (j + 1) * nz + i * ny * nz],
                                                 masses[k + j * nz + (i + 1) * ny * nz]));
                }
            }
        }
    }

    for (Spring * s : springs) {
        s -> setRestLength((s -> _right -> pos - s -> _left -> pos).norm());
    }
}

#ifdef CONSTRAINTS
Beam::Beam(const Vec & center, const Vec & dims, int nx, int ny, int nz) {
    _center = center;
    _dims = dims;
    this -> nx = nx;
    this -> ny = ny;
    this -> nz = nz;

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
	            masses.push_back(new Mass(Vec((nx > 1) ? (double) i / (nx - 1.0) - 0.5 : 0, (ny > 1) ? j / (ny - 1.0) - 0.5 : 0, (nz > 1) ? k / (nz - 1.0) - 0.5 : 0) * dims + center));
                if (i == 0) {
                    masses[masses.size() - 1] -> constraints.fixed = true;
                }
            }
        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                for (int l = 0; l < ((i != nx - 1) ? 2 : 1); l++) {
                    for (int m = 0; m < ((j != ny - 1) ? 2 : 1); m++) {
                        for (int n = 0; n < ((k != nz - 1) ? 2 : 1); n++) {
                            if (l != 0 || m != 0 || n != 0) {
                                springs.push_back(new Spring(masses[k + j * nz + i * ny * nz],
                                                             masses[(k + n) + (j + m) * nz + (i + l) * ny * nz]));
                            }
                        }
                    }
                }

                if (k != nz - 1) {
                    if (j != ny - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz], // get the full triangle
                                                     masses[k + (j + 1) * nz + i * ny * nz]));
                    }

                    if (i != nx - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz],
                                                     masses[k + j * nz + (i + 1) * ny * nz]));
                    }

                    if (j != ny - 1 && i != nx - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz],
                                                     masses[k + (j + 1) * nz + (i + 1) * ny * nz]));
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + (i + 1) * ny * nz],
                                                     masses[k + (j + 1) * nz + i * ny * nz]));
                        springs.push_back(new Spring(masses[(k + 1) + (j + 1) * nz + i * ny * nz],
                                                     masses[k + j * nz + (i + 1) * ny * nz]));
                    }
                }

                if (j != ny - 1 && i != nx - 1) {
                    springs.push_back(new Spring(masses[k + (j + 1) * nz + i * ny * nz],
                                                 masses[k + j * nz + (i + 1) * ny * nz]));
                }
            }
        }
    }

    for (Spring * s : springs) {
        s -> setRestLength((s -> _right -> pos - s -> _left -> pos).norm());
    }
}
#endif
/*
 * Robot Link Class Functions
 */
    RobotLink::RobotLink(const Vec &pos1, const Vec &pos2, double mass, double max_exp_length, double min_exp_length,
                         double expansion_rate, double k, double magnetic_force, double radius) {
        max_length = max_exp_length; // expanded link length (magnet center to magnet center)
        min_length = min_exp_length; // contracted length of link
        k_link = k; // link stiffness
        max_mag_force = magnetic_force; // magnetic force of connector
        // shell stiffness
        ml = new Mass(pos1, mass, false, radius, 5000.0, max_mag_force, 1.0);
        mr = new Mass(pos2, mass, false, radius, 5000.0, max_mag_force, 1.0);
        s = new Spring(ml, mr, k_link, min_length, PASSIVE_SOFT, 0.0, max_length,
                min_length, expansion_rate);
        s->_rest = min_length;
        // add masses and springs to respective vectors
        masses.push_back(ml);
        masses.push_back(mr);
        springs.push_back(s);
    }
    /*
     * ToDo: Implement class functions
     */
    bool RobotLink::expand() {
        if (max_length <= s->_rest){
            s->_type = PASSIVE_SOFT;
            return false;
        } else {
            s->_type = ACTUATED_EXPAND;
            this->attach(); // the expanding link is always in attachment mode
            return true;
        }
    }

    bool RobotLink::contract() {
        if (min_length >= s->_rest){
            s->_type = PASSIVE_SOFT;
            return false;
        } else {
            s->_type = ACTUATED_EXPAND;
            return true;
        }
    }

    // removes magnet force from masses
    bool RobotLink::detach() {
        if (!this->contract()){
            if (ml->isMagnetic()){
                ml->max_mag_force = 0.0;
            }
            if (mr->isMagnetic()){
                mr->max_mag_force = 0.0;
            }
            return true;
        }
        return false;
    }
    // adds magnet force to masses
    bool RobotLink::attach() {
        if (!ml->isMagnetic()){
            ml->max_mag_force = max_mag_force;
        }
        if (!mr->isMagnetic()){
            mr->max_mag_force = max_mag_force;
        }
        return false;
    }

    void RobotLink::setExpansionRate(double exp_rate) {
        this->exp_rate = exp_rate;
        s->_rate = exp_rate;
    }

    void RobotLink::setRobotMass(double mass) {
        ml->m = mass/2;
        mr->m = mass/2;
    }

    void RobotLink::setColor(Vec c) {
        ml->color = c;
        mr->color = c;
    }

    void RobotLink::setStiffness(double k) {
        k_link = k;
        s->_k = k;
    }

// Robot::Robot(const Vec & center, const cppn& encoding, double side_length,  double omega, double k_soft, double k_stiff){
//     _center = center;
//     _side_length = side_length;
//     _omega = omega;
//     _k_soft = k_soft;
//     _k_stiff = k_stiff;
//     _encoding = encoding;
    
//     int RobotDim = encoding.size(); // number of cubes per side
//     Vec dims(side_length,side_length,side_length);
//     // keep trace of number of cubes that each mass is connected to 
//     std::vector<std::vector<std::vector<int>>> mass_conn(RobotDim+1, std::vector<std::vector<int>>(RobotDim+1,std::vector<int>(RobotDim+1,0)));
    
//     std::vector<std::vector<std::vector<Mass *>>> _masses(RobotDim+1, std::vector<std::vector<Mass *>>(RobotDim+1,std::vector<Mass *>(RobotDim+1,nullptr)));
  
//     // store number of cubes that should be connected to each mass
//     for (int i = 0; i < RobotDim+1; i++) {
//       for (int j = 0; j < RobotDim+1; j++) {
// 	for (int k = 0; k < RobotDim+1; k++) {
// 	  // if index mode RobotDim+1 is 0, then it is on the edge
// 	  int i_edge = (i % (RobotDim)) ? 0:1; 
// 	  int j_edge = (j % (RobotDim)) ? 0:1;
// 	  int k_edge = (k % (RobotDim)) ? 0:1;

	
// 	  if (i_edge + j_edge + k_edge ==0){
// 	    mass_conn[i][j][k] = 8; //corner
// 	  }else if (i_edge+j_edge+k_edge ==3){
// 	    mass_conn[i][j][k] = 1; //corner
// 	  }else if (i_edge+j_edge+k_edge ==2){
// 	    mass_conn[i][j][k] = 2; //edge
// 	  }else{
// 	    mass_conn[i][j][k] = 4; //surface
// 	  }	
// 	}
//       }
//     }

//     // Remove appropriate masses
//     for (int i = 0; i < RobotDim; i++) {
//       for (int j = 0; j < RobotDim; j++) {
// 	for (int k = 0; k < RobotDim; k++) {
	
// 	  int exist = encoding[i][j][k][0];

// 	  if (!exist){
// 	    // subtract connectedness of each mass for the cube
// 	    mass_conn[i][j][k] -= 1;
// 	    mass_conn[i][j][k+1] -= 1;
// 	    mass_conn[i][j+1][k] -= 1;
// 	    mass_conn[i][j+1][k+1] -= 1;
// 	    mass_conn[i+1][j][k] -= 1;
// 	    mass_conn[i+1][j][k+1] -= 1;
// 	    mass_conn[i+1][j+1][k] -= 1;
// 	    mass_conn[i+1][j+1][k+1] -= 1;
// 	  }
// 	}
//       }
//     }
  
//     // create masses
//     for (int i = 0; i < RobotDim+1; i++) {
//         for (int j = 0; j < RobotDim+1; j++) {
//             for (int k = 0; k < RobotDim + 1; k++) {
//                 if (mass_conn[i][j][k] > 0){
//                     Mass * m;
//                     if (RobotDim == 1) {
//                     m = new Mass(Vec(i-0.5, j-0.5, k-0.5) * dims + _center);
//                     } else {
//                         m = new Mass(Vec(i / (RobotDim - 1.0) - 0.5,
//                                 j / (RobotDim - 1.0) - 0.5,
//                                 k / (RobotDim - 1.0) - 0.5) * dims + _center);
//                     }

// #ifdef GRAPHICS
//                     m -> color = Vec(0,0,0);
// #endif

//                     masses.push_back(m);
//                     _masses[i][j][k] = m;
//                 }
//             }
//         }
//     }


//     // create springs
//     for (int i = 0; i < RobotDim; i++) {
//         for (int j = 0; j < RobotDim; j++) {
// 	        for (int k = 0; k < RobotDim; k++) {
	
//             int exist = encoding[i][j][k][0];

//             if (exist) {
//                 int type = encoding[i][j][k][1];
            
//                 for(int l=0; l<8; l++) {
//                     int l_x = (l<4)? 0:1;
//                     int l_y = (l<2)? 0:(l<4)?1:(l<6)?0:1;
//                     int l_z = (l%2)? 1:0;
                
//                     for (int m=l+1; m<8; m++) {
//                         int r_x = (m<4)? 0:1;
//                         int r_y = (m<2)? 0:(m<4)?1:(m<6)?0:1;
//                         int r_z = (m%2)? 1:0;

//                         Spring * spr = new Spring(_masses[i+l_x][j+l_y][k+l_z],
//                                     _masses[i+r_x][j+r_y][k+r_z]);
                        
//                         spr -> _type = type;
//                         spr -> _omega = omega;

//                         if (type==0) { // green, contract then expand
//                             spr -> _k = k_soft;

// #ifdef GRAPHICS
//                             _masses[i+l_x][j+l_y][k+l_z]->color += GREEN/16;
//                             _masses[i+r_x][j+r_y][k+r_z]->color += GREEN/16;
// #endif
//                         } else if (type==1) { // red, expand then contract
//                             spr -> _k = k_soft;
// #ifdef GRAPHICS
//                             _masses[i+l_x][j+l_y][k+l_z]->color += RED/16;
//                             _masses[i+r_x][j+r_y][k+r_z]->color += RED/16;
// #endif
                
//                         } else if (type==2) { // passive soft
//                             spr -> _k = k_soft;
// #ifdef GRAPHICS
//                             _masses[i+l_x][j+l_y][k+l_z]->color += BLUE/16;
//                             _masses[i+r_x][j+r_y][k+r_z]->color += BLUE/16;
// #endif
//                         } else { // passive stiff
//                             spr -> _k = k_stiff;
// #ifdef GRAPHICS
//                             _masses[i+l_x][j+l_y][k+l_z]->color += PURPLE/16;
//                             _masses[i+r_x][j+r_y][k+r_z]->color += PURPLE/16;
// #endif
//                         }

// 		                springs.push_back(spr);
//                         }
//                     }
//                 }
//             }
//         }
//     }

    
//     for (Spring * s : springs) {
//         s -> setRestLength((s -> _right -> pos - s -> _left -> pos).norm());
//     }
// }

#ifdef CONSTRAINTS

void Container::fix() {
    for (Mass * mass : masses) {
        mass -> constraints.fixed = true;
    }
}

LOCAL_CONSTRAINTS::LOCAL_CONSTRAINTS() {
//    constraint_plane = thrust::device_vector<CudaConstraintPlane>(1);
//    contact_plane = thrust::device_vector<CudaContactPlane>(1);
//    ball = thrust::device_vector<CudaBall>(1);
//    direction = thrust::device_vector<CudaDirection>(1);

//    contact_plane_ptr = thrust::raw_pointer_cast(contact_plane.data()); // TODO make sure this is safe
//    constraint_plane_ptr = thrust::raw_pointer_cast(constraint_plane.data());
//    ball_ptr = thrust::raw_pointer_cast(ball.data());
//    direction_ptr = thrust::raw_pointer_cast(direction.data());

    num_contact_planes = 0;
    num_constraint_planes = 0;
    num_balls = 0;
    num_directions = 0;

    drag_coefficient = 0;
    fixed = false;
}

CUDA_LOCAL_CONSTRAINTS::CUDA_LOCAL_CONSTRAINTS(LOCAL_CONSTRAINTS & c) {
    contact_plane = c.contact_plane_ptr;
    constraint_plane = c.constraint_plane_ptr;
    ball = c.ball_ptr;
    direction = c.direction_ptr;

    num_contact_planes = c.num_contact_planes;
    num_constraint_planes = c.num_constraint_planes;
    num_balls = c.num_balls;
    num_directions = c.num_directions;

    fixed = c.fixed;
    drag_coefficient = c.drag_coefficient;
}

#endif

#ifdef GRAPHICS

void Ball::normalize(GLfloat * v) {
    GLfloat norm = sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2],2)) / _radius;

    for (int i = 0; i < 3; i++) {
        v[i] /= norm;
    }
}

void Ball::writeTriangle(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3) {
    for (int j = 0; j < 3; j++) {
        arr[j] = v1[j] + _center[j];
    }

    arr += 3;

    for (int j = 0; j < 3; j++) {
        arr[j] = v2[j] + _center[j];
    }

    arr += 3;

    for (int j = 0; j < 3; j++) {
        arr[j] = v3[j] + _center[j];
    }
}

void Ball::subdivide(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3, int depth) {
    GLfloat v12[3], v23[3], v31[3];

    if (depth == 0) {
        writeTriangle(arr, v1, v2, v3);
        return;
    }

    for (int i = 0; i < 3; i++) {
        v12[i] = v1[i]+v2[i];
        v23[i] = v2[i]+v3[i];
        v31[i] = v3[i]+v1[i];
    }

    normalize(v12);
    normalize(v23);
    normalize(v31);

    subdivide(arr, v1, v12, v31, depth - 1);
    arr += 3 * 3 * (int) pow(4, depth - 1);
    subdivide(arr, v2, v23, v12, depth - 1);
    arr += 3 * 3 * (int) pow(4, depth - 1);
    subdivide(arr, v3, v31, v23, depth - 1);
    arr += 3 * 3 * (int) pow(4, depth - 1);
    subdivide(arr, v12, v23, v31, depth - 1);
}


void Ball::generateBuffers() {
    glm::vec3 color = {0.22f, 0.71f, 0.0f};

    GLfloat * vertex_data = new GLfloat[20 * 3 * 3 * (int) pow(4, depth)]; // times 4 for subdivision

    GLfloat X = (GLfloat) _radius * .525731112119133606;
    GLfloat Z = (GLfloat) _radius * .850650808352039932;

    static GLfloat vdata[12][3] = {
            {-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},
            {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
            {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}
    };
    static GLuint tindices[20][3] = {
            {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
            {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
            {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
            {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} };

    for (int i = 0; i < 20; i++) {
        subdivide(&vertex_data[3 * 3 * (int) pow(4, depth) * i], vdata[tindices[i][0]], vdata[tindices[i][1]], vdata[tindices[i][2]], depth);
    }

    glGenBuffers(1, &vertices); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, 20 * 3 * 3 * (int) pow(4, depth) * sizeof(GLfloat), vertex_data, GL_STATIC_DRAW);

    GLfloat * color_data = new GLfloat[20 * 3 * 3 * (int) pow(4, depth)]; // TODO constant length array

    for (int i = 0; i < 20 * 3 * (int) pow(4, depth); i++) {
        color_data[3*i] = color[0];
        color_data[3*i + 1] = color[1];
        color_data[3*i + 2] = color[2];
    }

    glGenBuffers(1, &colors);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, 20 * 3 * 3 * (int) pow(4, depth) * sizeof(GLfloat), color_data, GL_STATIC_DRAW);

    delete [] color_data;
    delete [] vertex_data;

    _initialized = true;
}

void Ball::draw() {
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertices);

    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
    );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
    );

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 20 * 3 * (int) pow(4, depth)); // 12*3 indices starting at 0 -> 12 triangles

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}

#endif

#ifdef GRAPHICS
/*
 * Contact Plane Shader (Source: boxiXia)
 */
void ContactPlane::generateBuffers() {
    const int radius = 10; // radius [unit] of the plane
    // 10*10*4*6 = 2400
    // total 15*15*4*6=5400 points

    // define color
    glm::vec3 c1 = glm::vec3(0.729f, 0.78f, 0.655f);
    glm::vec3 c2 = glm::vec3(0.533f, 0.62f, 0.506f);
    // refer to: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/
    std::vector<GLfloat> vertex_data;
    std::vector<GLfloat> color_data;

    GLfloat s = 0.1;// scale
    for (int i = -radius; i < radius; i++)
    {
        for (int j = -radius; j < radius; j++)
        {
            GLfloat x = i*s;
            GLfloat y = j*s;
            vertex_data.insert(vertex_data.end(), {
                    x,y,0,
                    x+s,y+s,0,
                    x+s,y,0,
                    x,y,0,
                    x,y+s,0,
                    x+s,y+s,0});//2 triangles of a quad
            // pick one color
            glm::vec3 c = (i + j) % 2 == 0? c1: c2;
            color_data.insert(color_data.end(), {
                    c[0],c[1],c[2],
                    c[0],c[1],c[2],
                    c[0],c[1],c[2],
                    c[0],c[1],c[2],
                    c[0],c[1],c[2],
                    c[0],c[1],c[2]});
        }
    }

    glm::vec3 glm_normal = glm::vec3(_normal[0], _normal[1], _normal[2]);
    auto quat_rot = glm::rotation(glm::vec3(0, 0, 1), glm_normal);

    glm::vec3 glm_offset = (float)_offset*glm_normal;

    #pragma omp parallel for
    for (size_t i = 0; i < vertex_data.size()/3; i++)
    {
        glm::vec3 v(vertex_data[3 * i], vertex_data[3 * i+1], vertex_data[3 * i+2]);
        v = glm::rotate(quat_rot, v) + glm_offset;
        vertex_data[3 * i] = v[0];
        vertex_data[3 * i+1] = v[1];
        vertex_data[3 * i+2] = v[2];
    }

    glGenBuffers(1, &vertices); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)* vertex_data.size(), vertex_data.data(), GL_STATIC_DRAW);


    glGenBuffers(1, &colors);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * color_data.size(), color_data.data(), GL_STATIC_DRAW);

    _initialized = true;
}

void ContactPlane::draw() {
    // 1st attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertices);

    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
    );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
    );

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 10*10*4*6); // number of vertices

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}
#endif

} // namespace titan