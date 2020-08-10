//
// Created by ron on 8/7/20.
//

#include "og.h"
namespace titan {
OccupancyGrid::OccupancyGrid() {
    arrayptr = nullptr;
    ref_count = 0;
    // initialize with empty unordered sets
    for (int i = 0; i < size_x*size_y*size_b; i++){
        grid.push_back(std::unordered_set<Mass *>());
    }
}

    void OccupancyGrid::decrementRefCount() { ref_count--; }
    void OccupancyGrid::update(Mass **masses, int nr_masses) {
        for (int i = 0; i < nr_masses; ++i){
        }

    }

    void OccupancyGrid::insert(Mass * m) {
        Vec tmp;
        int x_val, y_val;
        Vec scale = Vec(size_x, size_y, 1);
        tmp = m->pos/scale;
        x_val = int(std::floor(tmp[0])+center_x);
        y_val = int(std::floor(tmp[1])+center_x);
        x_val = (x_val > (size_x-1) ? size_x-1 : x_val);
        y_val = (y_val > (size_y-1) ? size_y-1 : y_val);

    }

    bool OccupancyGrid::isInCell(int x, int y) {
        return false;
    }
    int OccupancyGrid::idx2DTo1D(int x, int y) {
        int idx_1D = x*size_x+y;
    }

}