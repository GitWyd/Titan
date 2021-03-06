#ifndef TITAN_OBJECT_H
#define TITAN_OBJECT_H

#include <vector>
#include "vec.h"

#ifdef GRAPHICS

#define GLM_FORCE_PURE

// Include GLEW
#include <GL/glew.h>

// Include GLFW
// TODO add SDL2 instead
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#endif

#ifdef CONSTRAINTS
#include <thrust/device_vector.h>
#endif

namespace titan {

struct CUDA_MASS;
class Spring;
class Mass;

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif

class Constraint { // constraint like plane or sphere which applies force to masses
public:
    virtual ~Constraint() = default;

#ifdef GRAPHICS
    bool _initialized;
    virtual void generateBuffers() = 0;
    virtual void draw() = 0;
#endif
};

struct Ball : public Constraint {
    Ball(const Vec & center, double radius) {
        _center = center;
        _radius = radius;

#ifdef GRAPHICS
        _initialized = false;
#endif

    }

    double _radius;
    Vec _center;

#ifdef GRAPHICS
    ~Ball() {
        glDeleteBuffers(1, &vertices);
        glDeleteBuffers(1, &colors);
    }

    void generateBuffers();
    void draw();

    void subdivide(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3, int depth);
    void writeTriangle(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3);
    void normalize(GLfloat * v);

    int depth = 2;

    GLuint vertices;
    GLuint colors;
#endif
};

struct CudaBall {
    CudaBall() = default;
    CUDA_CALLABLE_MEMBER CudaBall(const Vec & center, double radius);
    CUDA_CALLABLE_MEMBER CudaBall(const Ball & b);

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m);

    double _radius;
    Vec _center;
};

struct ContactPlane : public Constraint {
    ContactPlane(const Vec & normal, double offset) {
        _normal = normal / normal.norm();
        _offset = offset;

        _FRICTION_K = 0.0;
        _FRICTION_S = 0.0;

#ifdef GRAPHICS
        _initialized = false;
#endif
    }

    Vec _normal;
    double _offset;

    double _FRICTION_K;
    double _FRICTION_S;

#ifdef GRAPHICS
    ~ContactPlane() {
        glDeleteBuffers(1, &vertices);
        glDeleteBuffers(1, &colors);
    }

    void generateBuffers();
    void draw();

    GLuint vertices;
    GLuint colors;
#endif
};

struct CudaContactPlane {
    CudaContactPlane() = default;
    CUDA_CALLABLE_MEMBER CudaContactPlane(const Vec & normal, double offset);
    CudaContactPlane(const ContactPlane & p);

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m);

    Vec _normal;
    double _offset;

    double _FRICTION_K;
    double _FRICTION_S;
};


struct CudaConstraintPlane {
    CudaConstraintPlane() = default;

    CUDA_CALLABLE_MEMBER CudaConstraintPlane(const Vec & normal, double friction);

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m);

    Vec _normal;
    double _friction;
};

struct CudaDirection {
    CudaDirection() = default;

    CUDA_CALLABLE_MEMBER CudaDirection(const Vec & tangent, double friction);

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m);

    Vec _tangent;
    double _friction;
};

struct CUDA_GLOBAL_CONSTRAINTS {
    CudaContactPlane * d_planes;
    CudaBall * d_balls;

    int num_planes;
    int num_balls;
};


#ifdef CONSTRAINTS
struct LOCAL_CONSTRAINTS {
    LOCAL_CONSTRAINTS();

    thrust::device_vector<CudaContactPlane> contact_plane;
    thrust::device_vector<CudaConstraintPlane> constraint_plane;
    thrust::device_vector<CudaBall> ball;
    thrust::device_vector<CudaDirection> direction;

    CudaContactPlane * contact_plane_ptr;
    CudaConstraintPlane * constraint_plane_ptr;
    CudaBall * ball_ptr;
    CudaDirection * direction_ptr;

    int num_contact_planes;
    int num_balls;
    int num_constraint_planes;
    int num_directions; // if this is greater than 1, just make it fixed

    double drag_coefficient;
    bool fixed; // move here from the class itself;
};

struct CUDA_LOCAL_CONSTRAINTS {
    CUDA_LOCAL_CONSTRAINTS() = default;

    CUDA_LOCAL_CONSTRAINTS(LOCAL_CONSTRAINTS & c);

    CudaContactPlane * contact_plane;
    CudaConstraintPlane * constraint_plane;
    CudaBall * ball;
    CudaDirection * direction;

    int drag_coefficient;
    bool fixed; // move here from the class itself;

    int num_contact_planes;
    int num_balls;
    int num_constraint_planes;
    int num_directions; // if this is greater than 1, just make it fixed
};

#endif

#ifdef CONSTRAINTS
enum CONSTRAINT_TYPE {
    CONSTRAINT_PLANE, CONTACT_PLANE, BALL, DIRECTION
};
#endif

class Container { // contains and manipulates groups of masses and springs
public:
    virtual ~Container() {};
    void translate(const Vec & displ); // translate all masses by fixed amount
    // rotate all masses around a fixed axis by a specified angle with respect to the center of mass.
    // in radians
    void rotate(const Vec & axis, double angle); 
    void setMassValues(double m); // set masses for all Mass objects
    void setSpringConstants(double k); // set k for all Spring objects
    void setRestLengths(double len); // set masses for all Mass objects

#ifdef CONSTRAINTS
    void addConstraint(CONSTRAINT_TYPE type, const Vec & v, double d);
    void clearConstraints();
#endif

    void fix();

    void add(Mass * m);
    void add(Spring * s);
    void add(Container * c);

    // we can have more of these
    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
};

class Cube : public Container {
public:
    ~Cube() {};

    Cube(const Vec & center, double side_length = 1.0);

    double _side_length;
    Vec _center;
};

class Lattice : public Container {
public:
    ~Lattice() {};

    Lattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    int nx, ny, nz;
    Vec _center, _dims;
};

class Beam : public Container {
public:
    ~Beam() {};

    Beam(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    int nx, ny, nz;
    Vec _center, _dims;
};

/*
 * Magnet Truss Robot Link
 */
class RobotLink : public Container {
public:
    RobotLink(const Vec & pos1, const Vec & pos2, double mass, double max_exp_length, double min_exp_length,
            double expansion_rate, double k, double magnetic_force, double radius = 0.015);
    ~RobotLink() {}

    Mass * ml; // left mass
    Mass * mr; // right mass
    Spring * s; // spring actuator

    double exp_rate = 0.009; // [m/s] 2 * 4.5 mm/s (Actuonix L12I 210:1)
    double max_length; // expanded link length (magnet center to magnet center)
    double min_length; // contracted length of link
    double k_link; // link stiffness
    double max_mag_force; // magnetic force of connector

    /*
     * expand()
     * returns True if link still can expand more, and false otherwise
     * puts robot link into expansion mode unless it is already at full length
     */
    bool expand();
    /*
     * contract()
     * returns True if link still can contract more, and false otherwise
     * puts robot link into contraction mode unless it is already at full length
     */
    bool contract();
    /*
     * setLength(double length)
     * returns True if link is still in the process of approaching length, and false otherwise
     * puts robot link at length unless it is already at length
     */
    bool setLength(double length);
    bool detach();
    bool attach();
    void setExpansionRate(double exp_rate);
    void setRobotMass(double mass);
    void setColor(Vec c);
    void setStiffness(double k);
};

// typedef std::vector<std::vector<std::vector<std::vector<int>>>> cppn;
// class Robot : public Container {
// public:
//     ~Robot() {};
    
//     // with known encoding
//     Robot(const Vec & center, const cppn& encoding, double side_length,  double omega=1.0, double k_soft=2e3, double k_stiff=2e5);
    
//     Vec _center;
//     cppn _encoding;
//     double _side_length;
//     double _omega;
//     double _k_soft;
//     double _k_stiff;
//     /* Vec _com; */
//     /* double _score; */
// };

} // namespace titan

#endif //TITAN_OBJECT_H
