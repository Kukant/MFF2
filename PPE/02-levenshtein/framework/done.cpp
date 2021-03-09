#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>

#include <iostream>
#include <algorithm>
#include <omp.h>

template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG>
{
private:

    int M; // horizontal dimension
    int N; // vertical dimension
    int s; // sub-matrix dimension
    int p_limit; // iteration limit used in for cycles

    size_t* h; // spans horizontal dimension, h[j] = x[i,j] for largest i so far
    size_t* v; // spans vertical dimension, v[i] = x[i,j] for largest j so far
    size_t* d; // spans vertical dimension, d[i] = x[i - 1,j] for largest j so far

    // fill arrays with initial values
    void init_vectors(DIST vertical, DIST horizontal)
    {
        // array is aligned to cache line size
        h = static_cast<size_t*>(aligned_alloc(64, horizontal*sizeof(*h)));
        for(size_t i = 0; i < horizontal; ++i)
        {
            h[i] = i + 1;
        }

        d = static_cast<size_t*>(aligned_alloc(64, vertical*sizeof(*d)));
        v = static_cast<size_t*>(aligned_alloc(64, vertical*sizeof(*v)));
        for(size_t i = 0; i < vertical; ++i)
        {
            v[i] = i + 1;
            d[i] = i;
        }
    }

    // this runs in parallel for matrices on diagonal
    // I think this is the bottle neck but I don't see where it is
    // (i,j) is top left corner of sub-matrix, str1 and str2 are input strings
    inline void evaluate_stripe(int i, int j, const std::vector<C>& str1, const std::vector<C>& str2)
    {
        size_t value;

        // rows
        for(int k = i; k < i + s; ++k)
        {
            // collumns
            for(int l = j; l < j + s; ++l)
            {
                // compute x[k,l]
                value = std::min(d[k] + ((str1[k] == str2[l]) ? 0 : 1), std::min(v[k] + 1, h[l] + 1));

                // update arrays
                // I don't think false sharing happens here
                // tasks running in parallel have disjoint memory accesses and dimension of sub-matrix is 256
                // 256 is multiple of 64 (cache line size) => two tasks should always operate on different cache lines
                d[k] = h[l];
                v[k] = value;
                h[l] = value;
            }
        }
    }

public:
    /*
     * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
     * \param len1, len2 Lengths of first and second string respectively.
     */
    virtual void init(DIST len1, DIST len2)
    {
        // shorter is always vertical
        if(len1 > len2)
        {
            std::swap(len1, len2);
        }

        init_vectors(len1, len2);

        M = len2;
        N = len1;

        s = 256; // best performing value so far

        if(M == N)
        {
            // number of upper (and lower) diagonals in square matrix
            p_limit = M / s;
        }
        else
        {
            // number of all diagonals in general matrix
            p_limit = M / s + N / s - 2;
        }

    }

    // free alocated memory
    ~EditDistance()
    {
        free(h);
        free(v);
        free(d);
    }

    /*
     * \brief Compute the distance between two strings.
     * \param str1, str2 Strings to be compared.
     * \result The computed edit distance.
     */
    virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
    {
        // pointer to shared arrays, used for dependecy declaration between tasks
        // a variable must be used inside depend clause
        auto a = &v[0];
        auto b = &h[0];

        // counters
        int i, j;

        // both dimensions of the matrix are the same
        if(M == N)
        {
            #pragma omp parallel
            {
                #pragma omp single
                {
                    // iterate through upper diagonals
                    for(int p = 0; p < p_limit; ++p)
                    {
                        i = p*s;
                        j = 0;

                        // for each matrix on the diagonal run parallel task
                        for(int q = 0; q <= p; ++q)
                        {
                            #pragma omp task depend(inout: a[i:i+s-1], b[j:j+s-1])
                            {
                                evaluate_stripe(i, j, str1, str2);
                            }

                            i -= s;
                            j += s;
                        }
                    }

                    // same for lower diagonals
                    for(int p = p_limit - 2; p >= 0; --p)
                    {
                        i = (p_limit - 1) * s;
                        j = (p_limit - p - 1) * s;

                        for(int q = 0; q <= p; ++q)
                        {
                            #pragma omp task depend(inout: a[i:i+s-1], b[j:j+s-1])
                            {
                                evaluate_stripe(i, j, str1, str2);
                            }

                            i -= s;
                            j += s;
                        }
                    }
                }
            }
        }
        else // general case
        {
            const std::vector<C>* s1 = &str1;
            const std::vector<C>* s2 = &str2;

            // make sure smaller dimension is always vertical
            if(str1.size() > str2.size())
            {
                std::swap(s1, s2);
            }

            #pragma omp parallel
            {
                #pragma omp single
                {
                    // iterate through diagonals
                    for(int p = 0; p <= p_limit; ++p)
                    {
                        i = p*s;
                        j = 0;

                        for(int q = 0; q <= p; ++q)
                        {
                            // if matrix lays on the diagonal and is inside the N*M matrix run parallel task
                            if(i <= N && j <= M)
                            {
                                #pragma omp task depend(inout: a[i:i+s-1], b[j:j+s-1])
                                {
                                    evaluate_stripe(i, j, *s1, *s2);
                                }
                            }

                            i -= s;
                            j += s;
                        }
                    }
                }
            }
        }

        // bottom right corner is the result
        return h[M - 1];
    }
};

#endif