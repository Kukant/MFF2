#ifndef CUDA_POTENTIAL_IMPLEMENTATION_HPP
#define CUDA_POTENTIAL_IMPLEMENTATION_HPP

#include "kernels.h"

#include <interface.hpp>
#include <data.hpp>

#include <cuda_runtime.h>


/*
 * Final implementation of the tested program.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class ProgramPotential : public IProgramPotential<F, IDX_T, LEN_T>
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

private:
    point_t* mVelocities;
    point_t* mPoints;
    point_t* mNewPoints;
    index_t* mEdgeLengths;
    index_t* mEdgesStartIndex;
    index_t* mEdgesCount;
    index_t* mEdgesPerPoint;
    index_t* mEdgesPerPointMapper;

    index_t points_count;
    bool first_call = true;
    double velocityFact;
    ModelParameters<real_t> mParams;

public:
	virtual void initialize(index_t points, const std::vector<edge_t>& edges, const std::vector<length_t> &lengths, index_t iterations)
	{
        points_count = points;
        mParams = this->mParams;
        velocityFact = mParams.timeQuantum / mParams.vertexMass;

        std::vector<index_t> edge_counts(points_count);
        std::vector<index_t> edges_start_idx(points_count);

        for(auto&& edge: edges) {
            edge_counts[edge.p1]++;
            edge_counts[edge.p2]++;
        }

        std::uint32_t sum = 0;
        for(std::uint32_t i = 0; i < points_count; i++) {
            edges_start_idx[i] = sum;
            sum += edge_counts[i];
            edge_counts[i] = 0; // counts will be filled in again during filling edges_per_point
        }

        std::vector<index_t> edges_per_point(sum); // sum == 2*edges.size()

        for(std::uint32_t i = 0; i < edges.size(); i++)
        {
            index_t p1 = edges[i].p1;
            index_t p2 = edges[i].p2;
            edges_per_point[edges_start_idx[p1] + edge_counts[p1]] = p2;
            edges_per_point[edges_start_idx[p2] + edge_counts[p2]] = p1;
            edge_counts[p1]++;
            edge_counts[p2]++;
        }


        CUCH(cudaSetDevice(0));

        cudaMalloc((void**)&mVelocities, points * sizeof(point_t));
        cudaMalloc((void**)&mPoints, points * sizeof(point_t));
        cudaMalloc((void**)&mNewPoints, points * sizeof(point_t));
        cudaMalloc((void**)&mEdgeLengths, edges.size() * sizeof(length_t));
        cudaMalloc((void**)&mEdgesStartIndex, points * sizeof(index_t));
        cudaMalloc((void**)&mEdgesCount, points * sizeof(index_t));
        cudaMalloc((void**)&mEdgesPerPoint, sum * sizeof(index_t));

        cudaMemcpy(mEdgeLengths, &lengths[0], edges.size() * sizeof(index_t), cudaMemcpyHostToDevice);
        cudaMemcpy(mEdgesStartIndex, &edges_start_idx[0], points * sizeof(index_t), cudaMemcpyHostToDevice);
        cudaMemcpy(mEdgesCount, &edge_counts[0], points * sizeof(index_t), cudaMemcpyHostToDevice);
        cudaMemcpy(mEdgesPerPoint, &edges_per_point[0], sum * sizeof(index_t), cudaMemcpyHostToDevice);
	}


	virtual void iteration(std::vector<point_t> &points)
	{
        if(first_call) {
            cudaMemcpy(mPoints, &points[0], points_count * sizeof(point_t), cudaMemcpyHostToDevice);
            first_call = false;
        }

        compute_one_iter_wrapper(
                mPoints, mNewPoints, mVelocities, mEdgeLengths,
                mEdgesStartIndex, mEdgesCount, mEdgesPerPoint, points_count,
                mParams.vertexRepulsion, mParams.vertexMass, mParams.edgeCompulsion,
                mParams.slowdown, mParams.timeQuantum, velocityFact
        );

        cudaMemcpy(&points[0], mNewPoints, points_count * sizeof(point_t), cudaMemcpyDeviceToHost);

        std::swap(mPoints, mNewPoints);
	}


    virtual void getVelocities(std::vector<point_t> &velocities)
    {
        velocities.clear();
        velocities.resize(points_count);
        cudaMemcpy(&velocities[0], mVelocities, points_count * sizeof(point_t), cudaMemcpyDeviceToHost);
    }

    ~ProgramPotential()
    {
        cudaFree(mVelocities);
        cudaFree(mPoints);
        cudaFree(mNewPoints);
        cudaFree(mEdgeLengths);
        cudaFree(mEdgesStartIndex);
        cudaFree(mEdgesCount);
        cudaFree(mEdgesPerPoint);
    }
};


#endif
