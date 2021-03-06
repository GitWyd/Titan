//
// Created by Jacob Austin on 5/17/18.
//

#ifndef TITAN_SPRING_H
#define TITAN_SPRING_H

#include "mass.h"
#include "vec.h"

namespace titan {

class Mass;
struct CUDA_SPRING;
struct CUDA_MASS;
// ToDo: Remove PASSIVE_STIFF has no effect at all, remove
enum SpringType {PASSIVE_SOFT, PASSIVE_STIFF, ACTIVE_CONTRACT_THEN_EXPAND, ACTIVE_EXPAND_THEN_CONTRACT,
        ACTUATED_EXPAND, ACTUATED_CONTRACT};

class Spring {
public:
    Spring() : _left(nullptr), _right(nullptr), _k(10000.0), _rest(1.0), _type(PASSIVE_SOFT),
    _omega(0.0), _damping(0.0), arrayptr(nullptr) {}

    Spring(Mass * left, Mass * right) : _left(left), _right(right), _k(10000.0), _type(PASSIVE_SOFT), 
    _omega(0.0), _damping(0.0), arrayptr(nullptr) 
    {
        this -> defaultLength();
    }

    Spring(Mass * left, Mass * right, double k, double rest_length) : _left(left), _right(right), 
    _k(k), _rest(rest_length), _type(PASSIVE_SOFT), _omega(0.0), _damping(0.0) {}

    // full actuator constructor
    Spring(Mass * left, Mass * right, double k, double rest_length, SpringType type, double omega, double max_length,
            double min_length, double expansion_rate) :
            _left(left), _right(right), _k(k), _rest(rest_length), _type(type), _omega(omega), _damping(0.0),
            _l_max(max_length), _l_min(min_length), _rate(expansion_rate){};

    void update(const CUDA_SPRING & spr);
    void setRestLength(double rest_length) { _rest = rest_length; } //sets Rest length
    void defaultLength(); //sets rest length
    void changeType(SpringType type, double omega) { _type = type; _omega = omega;}
    void addDamping(double constant) { _damping = constant; }

    void setLeft(Mass * left); // sets left mass (attaches spring to mass 1)
    void setRight(Mass * right);

    void setMasses(Mass * left, Mass * right) { _left = left; _right = right; } //sets both right and left masses

    Mass * _left; // pointer to left mass object // private
    Mass * _right; // pointer to right mass object

    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)

    SpringType _type; // 0-3, for oscillating springs
    double _omega; // frequency of oscillation
    double _damping; // damping on the masses.
    // Actuator
    double _l_max; // maximum actuator length
    double _l_min; // minimum actuator length
    double _rate; // expansion rate [m/s]

private:
    CUDA_SPRING *arrayptr; //Pointer to struct version for GPU cudaMalloc

    friend class Simulation;
    friend struct CUDA_SPRING;
    friend class Container;
    friend class Lattice;
    friend class Cube;
    friend class Beam;
    friend class RobotLink; // is this necessary
};

struct CUDA_SPRING {
  CUDA_SPRING() {};
  CUDA_SPRING(const Spring & s);
  
  CUDA_SPRING(const Spring & s, CUDA_MASS * left, CUDA_MASS * right);
  
  CUDA_MASS * _left; // pointer to left mass object
  CUDA_MASS * _right; // pointer to right mass object
  
  double _k; // spring constant (N/m)
  double _rest; // spring rest length (meters)

  // Breathing
  SpringType _type;
  double _omega;
  double _damping;
  // Actuator
  double _l_max; // maximum actuator length
  double _l_min; // minimum actuator length
  double _rate; // expansion rate [m/s]
};

} // namespace titan

#endif //TITAN_SPRING_H
