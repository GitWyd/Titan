//
// Created by ron on 9/22/20.
//

#ifndef TITAN_GEOMETRY_H
#define TITAN_GEOMETRY_H
#include <vector>
#include <vec.h>
#include <glm/vec4.hpp>

namespace titan {
class Geometry;
struct CUDA_GEOMETRY;
/*
 * geometry class representing general shapes to be rendered
 */
class Geometry {
public:
    Geometry();
    virtual ~Geometry();
    virtual void ComputeVertices();
    virtual void ComputeIndices();
    virtual void SetColor(glm::vec4 const color);
private:
    unsigned int vertices_count;
    unsigned int indices_count;
    float vertices_size;
    float indices_size;
    std::vector<float> v_positions;
    std::vector<float> v_normals;
    std::vector<float> v_colors;
    std::vector<unsigned int> indices;
};
/*
 * GPU memory analog of geometry class to represent shape-geometries on GPU side
 */
struct CUDA_GEOMETRY{
    CUDA_GEOMETRY(Geometry & object);
    float ** v_positions;
    float ** v_normals;
    float ** v_colors;
    unsigned int ** indices;
    unsigned int vertices_count;
    unsigned int indices_count;
    float vertices_size;
    float indices_size;
};
}

#endif //TITAN_GEOMETRY_H
