#include "kernels.h"

__global__
void compute_one_iter_kernel(
        Point<double>* points, Point<double>* newPoints, Point<double>* velocities, std::uint32_t* edgeLengths,
        std::uint32_t* edgesStartIndex, std::uint32_t* edgesCount,  std::uint32_t* edgesPerPoint, std::uint32_t points_count,
        double vertexRepulsion, double vertexMass,double edgeCompulsion,
        double slowdown, double timeQuantum, double velocityFact)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // position
    double px = points[idx].x;
    double py = points[idx].y;
    // force
    double fx = 0.0;
    double fy = 0.0;

    for(std::uint32_t i = 0; i < idx; ++i) {
        if (i == idx) continue;// this point does not affect itself

        double dx = px - points[i].x;
        double dy = py - points[i].y;
        double sqLen = dx*dx + dy*dy;
        sqLen = sqLen > (double)0.0001 ? sqLen: (double)0.0001;
        double fact = vertexRepulsion / (sqLen * (double)std::sqrt(sqLen));

        fx += dx * fact;
        fy += dy * fact;
    }

    // compute compulsive forces
    for(std::uint32_t i = edgesStartIndex[idx]; i < edgesStartIndex[idx] + edgesCount[idx]; i++) {
        std::uint32_t pointIdx = edgesPerPoint[i];
        std::uint32_t len = edgeLengths[pointIdx];

        double dx = points[pointIdx].x - px;
        double dy = points[pointIdx].y - py;
        double fact = std::sqrt(dx*dx + dy*dy) * edgeCompulsion / (double)(len);

        fx += dx * fact;
        fy += dy * fact;
    }

    // update velocity and position
    velocities[idx].x = (velocities[idx].x + fx * velocityFact) * slowdown;
    velocities[idx].y = (velocities[idx].y + fy * velocityFact) * slowdown;
    newPoints[idx].x = px + velocities[idx].x * timeQuantum;
    newPoints[idx].y = py + velocities[idx].y * timeQuantum;
}

void compute_one_iter_wrapper(
        Point<double>* points, Point<double>* newPoints, Point<double>* velocities, std::uint32_t* edgeLengths,
        std::uint32_t* edgesStartIndex, std::uint32_t* edgesCount,  std::uint32_t* edgesPerPoint, std::uint32_t points_count,
        double vertexRepulsion, double vertexMass,double edgeCompulsion,
        double slowdown, double timeQuantum, double velocityFact)
{
    compute_one_iter_kernel<<<points_count / 64, 64>>>(
            points, newPoints, velocities, edgeLengths,
            edgesStartIndex, edgesCount, edgesPerPoint, points_count,
            vertexRepulsion, vertexMass, edgeCompulsion,
            slowdown, timeQuantum, velocityFact);
}