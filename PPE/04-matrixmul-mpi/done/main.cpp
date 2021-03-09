#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <queue>
#include <mpi.h>

#include "utils.h"

#define PROCESSES 256
#define TERMINATE_FLAG 13

using namespace std;

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // arrays holding a submatrix, row = row block from A, collumn = collumn block from B, result = result block
    unique_ptr<float[]> row = make_unique<float[]>(size_policy::row_chunk() * size_policy::result_chunk());
    unique_ptr<float[]> collumn = make_unique<float[]>(size_policy::collumn_chunk() * size_policy::result_chunk());
    unique_ptr<float[]> result = make_unique<float[]>(size_policy::row_chunk() * size_policy::collumn_chunk());

    // rank 0 splits the work and sends data to other nodes
    if(rank == 0)
    {
        ifstream astream, bstream;
        ofstream result_stream;

        astream.open(argv[1], std::ios::binary | std::ios::in);
        bstream.open(argv[2], std::ios::binary | std::ios::in);
        result_stream.open(argv[3], std::ios::out | std::ios::binary);

        loader loader(astream, bstream);
        writer writer(result_stream, loader);

        queue<int> available_nodes; // 1..n
        queue<worker> workers;

        for(int i = 1; i < PROCESSES; ++i)
        {
            available_nodes.push(i);
        }

        bool working = true;

        while(working)
        {
            // create jobs
            while(available_nodes.size() > 0)
            {
                worker worker;
                uint32_t row_count;
                uint32_t collumn_count;

                // load data into arrays
                if (!loader.get_job(worker, row.get(), row_count, collumn.get(), collumn_count))
                {
                    working = false;
                    break;
                }

                // get free node
                int node = available_nodes.front();
                available_nodes.pop();

                // send data
                MPI_Send(row.get(), row_count, MPI_FLOAT, node, 0, MPI_COMM_WORLD);
                MPI_Send(&worker.row, 2, MPI_UINT32_T, node, 0, MPI_COMM_WORLD);

                MPI_Send(collumn.get(), collumn_count, MPI_FLOAT, node, 0, MPI_COMM_WORLD);
                MPI_Send(&worker.collumn, 2, MPI_UINT32_T, node, 0, MPI_COMM_WORLD);

                // node is now working
                worker.id = node;
                workers.push(worker);
            }

            // get results from workers
            while (workers.size() > 0)
            {
                worker current_worker = workers.front();
                workers.pop();

                uint32_t row_count = current_worker.row.end - current_worker.row.begin;
                uint32_t collumn_count = current_worker.collumn.end - current_worker.collumn.begin;

                uint32_t result_count = row_count * collumn_count;

                // get data
                MPI_Recv(result.get(), result_count, MPI_FLOAT, current_worker.id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                writer.process_result(current_worker, result.get());
                // node is free again
                available_nodes.push(current_worker.id);
            }
        }

        // tell other nodes to stop computing
        while (available_nodes.size() > 0)
        {
            int node = available_nodes.front();
            available_nodes.pop();

            MPI_Send(&node, 0, MPI_INT, node, TERMINATE_FLAG, MPI_COMM_WORLD);
        }

        writer.write();

        astream.close();
        bstream.close();
        result_stream.close();
    }
    else
    {
        chunk a_row;
        chunk b_collumn;

        uint32_t a_row_count, a_collumn_count;
        uint32_t b_row_count, b_collumn_count;
        int elements;

        MPI_Status status;

        while (true)
        {
            // get row block
            MPI_Recv(row.get(), size_policy::row_chunk() * size_policy::result_chunk(), MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TERMINATE_FLAG)
            {
                break;
            }

            fill(result.get(), result.get() + (size_policy::row_chunk() * size_policy::collumn_chunk()), 0.0f);

            // get number of elements
            MPI_Get_count(&status, MPI_FLOAT, &elements);
            // get range of the block
            MPI_Recv(&a_row, 2, MPI_UINT32_T, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            a_row_count = a_row.end - a_row.begin;
            a_collumn_count = elements / a_row_count;

            // same for collumn block
            MPI_Recv(collumn.get(), size_policy::collumn_chunk() * size_policy::result_chunk(), MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_FLOAT, &elements);
            MPI_Recv(&b_collumn, 2, MPI_UINT32_T, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            b_collumn_count = b_collumn.end - b_collumn.begin;
            b_row_count = elements / b_collumn_count;

            // multiplication
            for (uint32_t i = 0; i < a_row_count; ++i)
            {
                for (uint32_t j = 0; j < b_collumn_count; ++j)
                {
                    for (uint32_t k = 0; k < b_row_count; ++k)
                    {
                        result.get()[i * b_collumn_count + j] += row.get()[i*a_collumn_count + k] * collumn.get()[k*b_collumn_count + j];
                    }
                }
            }

            // send result to rank 0
            MPI_Send(result.get(), a_row_count * b_collumn_count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }


    MPI_Finalize();

}