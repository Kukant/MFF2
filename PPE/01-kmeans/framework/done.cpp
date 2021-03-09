#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>

#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/enumerable_thread_specific.h>



template<typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG>
{
private:

    // subresult holding sum and count of points in one chunk
    struct sum_count
    {
        sum_count() : sum(), count(0)
        {
            sum.x = 0;
            sum.y = 0;
        }

        void add(const POINT& p)
        {
            sum.x += p.x;
            sum.y += p.y;
            ++count;
        }

        void operator+=(const sum_count& other)
        {
            sum.x += other.sum.x;
            sum.y += other.sum.y;
            count += other.count;
        }

        void reset()
        {
            sum.x = 0;
            sum.y = 0;
            count = 0;
        }

        POINT sum;
        size_t count;
    };

    // thread local storage
    struct state
    {
        std::vector<sum_count> accumulator;
        size_t change;

        state(size_t k) : accumulator(k), change(0) {}

    private:
        state(const state& other);
        void operator=(const state& other);
    };

    typedef typename POINT::coord_t coord;
    typedef tbb::enumerable_thread_specific<state> state_type;

    size_t get_nearest_cluster(const POINT& point, const std::vector<POINT>& centroids)
    {
        coord min = distance(point, centroids[0]);
        size_t nearest = 0;
        coord dist;

        for (size_t i = 1; i < centroids.size(); ++i)
        {
            dist = distance(point, centroids[i]);

            if (dist < min)
            {
                min = dist;
                nearest = i;
            }
        }

        return nearest;
    }

    coord distance(const POINT& point, const POINT& centroid)
    {
        std::int64_t dx = (std::int64_t)point.x - (std::int64_t)centroid.x;
        std::int64_t dy = (std::int64_t)point.y - (std::int64_t)centroid.y;
        return (coord)(dx * dx + dy * dy);
    }

public:
    /*
     * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
     * \param points Number of points being clustered.
     * \param k Number of clusters.
     * \param iters Number of refining iterations.
     */
    virtual void init(std::size_t points, std::size_t k, std::size_t iters)
    {

    }


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
    virtual void compute(const std::vector<POINT> &points, std::size_t k, std::size_t iters,
                         std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
    {

        centroids.resize(k);
        assignments.resize(points.size());

        state_type task_state([&] {return k; });
        state global_state(k);

        // first centroids
        for (size_t i = 0; i < k; ++i)
        {
            centroids[i] = points[i];
        }

        // extracted first iteration of cluster assigning
        tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()), [&](tbb::blocked_range<size_t>& r) {

            state& s = task_state.local();

            for (size_t i = r.begin(); i != r.end(); ++i)
            {
                size_t nearest = get_nearest_cluster(points[i], centroids);
                assignments[i] = (ASGN)nearest;
                s.accumulator[nearest].add(points[i]);
            }

        });

        global_state.change = 1; // if result did not change, computation can stop

        while(global_state.change != 0 && iters > 0)
        {
            --iters;


            // collect subresluts
            for (auto i = task_state.begin(); i != task_state.end(); ++i)
            {
                state& s = *i;

                for (size_t j = 0; j < k; ++j)
                {
                    global_state.accumulator[j] += s.accumulator[j];
                    s.accumulator[j].reset();
                }
            }


            // find centroids
            for (size_t i = 0; i < k; ++i)
            {
                if (global_state.accumulator[i].count != 0)
                {
                    centroids[i].x = global_state.accumulator[i].sum.x / (std::int64_t) global_state.accumulator[i].count;
                    centroids[i].y = global_state.accumulator[i].sum.y / (std::int64_t) global_state.accumulator[i].count;
                    global_state.accumulator[i].reset();
                }
            }

            if (iters == 0) break; // because of reversed order in while loop we might want to stop before creating new clusters

            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()), [&](tbb::blocked_range<size_t>& r) {

                state& s = task_state.local();

                for (auto i = r.begin(); i != r.end(); ++i)
                {
                    size_t nearest = get_nearest_cluster(points[i], centroids);

                    if (nearest != assignments[i])
                    {
                        assignments[i] = (ASGN)nearest;
                        ++s.change;
                    }

                    s.accumulator[nearest].add(points[i]);
                }

            });

            // check for changes in clusters
            global_state.change = 0;

            for (auto i = task_state.begin(); i != task_state.end(); ++i)
            {
                state& s = *i;
                global_state.change += s.change;
                s.change = 0;
            }
        }
    }
};


#endif