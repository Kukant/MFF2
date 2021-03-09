#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>
#include <cmath>
#include <iostream>

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/concurrent_vector.h"
#include "tbb/enumerable_thread_specific.h"

using namespace tbb;
using namespace std;

template<typename POINT = point_t, typename ASGN = uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG>
{
private:
    typedef typename POINT::coord_t coord_t;

    struct local_state {
        size_t changes;
        vector<POINT> sums;
        vector<size_t> counts;
        size_t k;

        explicit local_state(size_t _k) {
            k = _k;
            sums.resize(k);
            counts.resize(k);
            changes = 0;
            reset();
        }

        void add_point(POINT p, size_t i) {
            sums[i].x += p.x;
            sums[i].y += p.y;

            counts[i]++;
        }

        void reset() {
            for (size_t i = 0; i < k; i++) {
                sums[i].x = sums[i].y = counts[i] = 0;
            }
            changes = 0;
        }

        void merge(local_state other) {
            changes += other.changes;
            for (size_t i = 0; i < k; i++) {
                sums[i].x += other.sums[i].x;
                sums[i].y += other.sums[i].y;
                counts[i] += other.counts[i];
            }
        }
    };

    typedef enumerable_thread_specific<local_state> local_state_thread_t;


    static coord_t distance(const POINT &point, const POINT &centroid)
    {
        int64_t dx = (int64_t)point.x - (int64_t)centroid.x;
        int64_t dy = (int64_t)point.y - (int64_t)centroid.y;
        return (coord_t)(dx*dx + dy*dy);
    }

    static size_t getNearestCluster(const POINT &point, const vector<POINT> &centroids)
    {
        coord_t minDist = distance(point, centroids[0]);
        size_t nearest = 0;
        coord_t dist;

        for (size_t i = 1; i < centroids.size(); i++) {
            dist = distance(point, centroids[i]);
            if (dist < minDist) {
                minDist = dist;
                nearest = i;
            }
        }

        return nearest;
    }

public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void init(size_t points, size_t k, size_t iters) {}

	/*
	 * \brief Perform the clustering and return the cluster centroids and point assignment
	 *		yielded by the last iteration.
	 * \note First k points are taken as initial centroids for first iteration.
	 * \param points Vector with input points.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 * \param centroids Vector where the final cluster centroids should be stored.
	 * \param assignments Vector where the final assignment of the points should be stored.
	 *		The indices should correspond to point indices in 'points' vector.
	 */
	virtual void compute(const vector<POINT> &points, size_t k, size_t iters,
		vector<POINT> &centroids, vector<ASGN> &assignments)
    {
        centroids.resize(k);
        assignments.resize(points.size());

        local_state main_state(k);
        local_state_thread_t states([&] {return k; });


        for (size_t i = 0; i < k; i++)
            centroids[i] = points[i];


        main_state.changes = 1;

        // Run the k-means refinements
        while (main_state.changes && iters > 0) {
            --iters;
            main_state.reset();

            parallel_for(blocked_range<size_t>(0, points.size()), [&](blocked_range<size_t>& r) {

                local_state& ls = states.local();

                for (size_t i = r.begin(); i != r.end(); i++)
                {
                    size_t nearest = getNearestCluster(points[i], centroids);
                    if (assignments[i] != nearest) {
                        ls.changes++;
                        assignments[i] = nearest;
                    }

                    ls.add_point(points[i], nearest);
                }

            });

            for (auto i = states.begin(); i != states.end(); i++)
            {
                local_state& ls = *i;
                main_state.merge(ls);
                ls.reset();
            }

            for (size_t i = 0; i < k; i++)
            {
                auto counts = main_state.counts[i];
                if (counts != 0)
                {
                    centroids[i].x = main_state.sums[i].x / (int64_t) counts;
                    centroids[i].y = main_state.sums[i].y / (int64_t) counts;
                }
            }
        }
    }
};


#endif
