//
// Created by Philippe Wyder on 8/7/20.
//

#ifndef TITAN_OG_H
#define TITAN_OG_H
#include "mass.h"
#include <unordered_set>
#include <cmath>
namespace titan{
class OccupancyGrid;
struct CUDA_OCCUPANCYGRID;

class OccupancyGrid {
private:
    const int size_x = 100;
    const int size_y = 100;
    const int size_b = 10; // bucket size
    const double cell_dimension_x = .25;
    const double cell_dimension_y = .25;
    const int center_x = size_x / 2;
    const int center_y = size_y / 2;

    int ref_count;
    void decrementRefCount();

    CUDA_OCCUPANCYGRID * arrayptr; // ptr to destruct version for GPU cudaMemAlloc
    const int EOL = -1; // end of list symbol
public:
    std::vector<std::unordered_set<Mass *>> grid;
    OccupancyGrid();
    void update(Mass ** masses, int nr_masses);
    void insert(Mass *);
    bool isInCell(int x, int y);
    int idx2DTo1D(int x, int y);
};

struct CUDA_OCCUPANCYGRID{
    CUDA_OCCUPANCYGRID() = default;
    CUDA_OCCUPANCYGRID(OccupancyGrid & og);
    Mass ** grid;
    const int size_x;
    const int size_y;
    const int size_b; // bucket size
    const int EOL = -1; // end of list symbol
};
}
#endif //TITAN_OG_H
