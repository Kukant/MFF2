#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>
#include <algorithm>
#include <omp.h>

using namespace std;

template<typename C = char, typename DIST = size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG>
{
public:
    virtual void init(DIST len1, DIST len2)
    {
        // make sure the matrix is wider than higher
        DIST h_len = len1 > len2 ? len1 : len2;
        DIST v_len = len1 < len2 ? len1 : len2;

        hor_buff.resize(h_len);
        ver_buff.resize(v_len);

        hor_chunks = h_len / chunk_size;
        ver_chunks = v_len / chunk_size;

        dependencies = new DIST *[hor_chunks + 1];
        for (DIST i = 0; i < hor_chunks + 1; i++)
            dependencies[i] = new DIST[ver_chunks + 1];

        // initialize the matrix
        for (size_t i = 0; i < hor_buff.size(); ++i)
            hor_buff[i] = i + 1;
        for (size_t i = 0; i < ver_buff.size(); ++i)
            ver_buff[i] = i + 1;

        // initialize corner values
        for (size_t i = 0; i < hor_chunks; ++i)
            dependencies[i][0] = i * chunk_size;
        for (size_t i = 0; i < ver_chunks; ++i)
            dependencies[0][i] = i * chunk_size;
    }

    virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
    {
        DIST **deps = dependencies;

        hor = str1.data();
        ver = str2.data();

        if (str1.size() < str2.size())
            std::swap(hor, ver);

        #pragma omp parallel shared(hor_buff, ver_buff)
        {
            #pragma omp single
            {
                #pragma omp task depend(out: deps[0][0])
                {
                    calc_submatrix(0, 0);
                }
                // calc rest of first row
                for (size_t i = 1; i < hor_chunks; i++) {
                    #pragma omp task depend(in: deps[i-1][0]) depend(out: deps[i][0])
                    {
                        calc_submatrix(i, 0);
                    }
                }
                // calc rest of first column
                for (size_t j = 1; j < ver_chunks; j++) {
                    #pragma omp task depend(in: deps[0][j-1]) depend(out: deps[0][j])
                    {
                        calc_submatrix(0, j);
                    }
                }
                // calc the rest
                for (size_t i = 1; i < hor_chunks; i++)
                {
                    for (size_t j = 1; j < ver_chunks; j++)
                    {
                        #pragma omp task depend(in: deps[i-1][j]) depend(in: deps[i][j-1]) depend(out: deps[i][j])
                        {
                            calc_submatrix(i, j);
                        }
                    }
                }
            }
        #pragma omp taskwait
        }

        return hor_buff.back();
    }

    ~EditDistance()
    {
        // delete allocated array
        for (size_t i = 0; i < hor_chunks + 1; ++i)
            delete[] dependencies[i];
        delete[] dependencies;
    }

private:
    DIST **dependencies;
    const C* ver;
    const C* hor;

    DIST ver_chunks;
    DIST hor_chunks;
    DIST chunk_size = 256;

    vector<DIST> ver_buff;
    vector<DIST> hor_buff;

    void calc_submatrix(size_t h_idx, size_t v_idx)
    {
        DIST last_hor_idx = (h_idx + 1) * chunk_size - 1;
        for (size_t i = v_idx * chunk_size; i < (v_idx + 1) * chunk_size; ++i) {
            DIST upper_left, left;
            upper_left = dependencies[h_idx][v_idx];
            left = ver_buff[i];

            for (size_t j = h_idx * chunk_size; j < (h_idx + 1) * chunk_size; ++j) {
                DIST upper = hor_buff[j];
                DIST sub_cost = ver[i] == hor[j] ? 0 : 1;
                DIST new_value = std::min<DIST>({left + 1, upper + 1, upper_left + sub_cost});
                hor_buff[j] = new_value;
                left = new_value;
                upper_left = upper;
            }
            ver_buff[i] = hor_buff[last_hor_idx];
        }

        dependencies[h_idx + 1][v_idx + 1] = hor_buff[last_hor_idx];
    }
};

#endif