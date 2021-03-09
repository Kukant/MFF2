#include <iostream>
#include <mpi.h>
#include <vector>
#include <queue>
#include <memory>
#include <unistd.h>

#include "write_utils.hpp"

using namespace std;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    if (argc != 4) {
        cerr << "There must be exactly 3 arguments" << endl;
        MPI_Finalize();
        return 1;
    }

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    unique_ptr<float[]> row = make_unique<float[]>(BLOCK_SIZE);
    unique_ptr<float[]> cols = make_unique<float[]>(BLOCK_SIZE);
    unique_ptr<float[]> result = make_unique<float[]>(BLOCK_SIZE);

    if(world_rank == 0)
    {
        ofstream rstream;
        ifstream astream, bstream;
        astream.open(argv[1], ios::binary | ios::in);
        bstream.open(argv[2], ios::binary | ios::in);
        rstream.open(argv[3], ios::binary | ios::out);

        matrix_reader reader(astream, bstream);
        result_writer writer(reader, rstream);
        // cout << "jobs: " << reader.jobs.size() << endl;
        while (!reader.jobs.empty()) {
            // send data to workers
            queue<sub_mult> running_jobs;
            int todo_jobs = reader.jobs.size();
            for (int worker_id = 1; worker_id < min(world_size, todo_jobs + 1); worker_id++) {
                sub_mult job = reader.get_job(row.get(), cols.get());
                job.worker_id = worker_id;
                uint32_t rows_count = job.a_row_end - job.a_row_start;
                uint32_t cols_count = job.b_col_end - job.b_col_start;
                uint32_t chunk_count = job.chunk_count;

                // cout<< worker_id <<  ". sending A " << endl;
                // send matrices
                MPI_Send(row.get(),
                         BLOCK_SIZE,
                        MPI_FLOAT, worker_id, 0, MPI_COMM_WORLD);

                // cout<< worker_id << ". sending B " << endl;
                MPI_Send(cols.get(),
                         BLOCK_SIZE,
                         MPI_FLOAT, worker_id, 0, MPI_COMM_WORLD);

                // cout<< worker_id << ". sending Additinal info  " << endl;

                // send additional info
                MPI_Send(&rows_count, 1, MPI_INT32_T, worker_id, 0, MPI_COMM_WORLD);
                MPI_Send(&cols_count, 1, MPI_INT32_T, worker_id, 0, MPI_COMM_WORLD);
                MPI_Send(&chunk_count, 1, MPI_INT32_T, worker_id, 0, MPI_COMM_WORLD);
                // cout<< worker_id << ". Additinal info sent  " << endl;
                running_jobs.push(job);
            }

            while (!running_jobs.empty()) {
                sub_mult job = running_jobs.front();
                running_jobs.pop();
                uint32_t result_count = (job.a_row_end - job.a_row_start) * (job.b_col_end - job.b_col_start);
                // cout<< "receiving " <<  result_count << endl;
                MPI_Recv(result.get(), result_count, MPI_FLOAT, job.worker_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // cout<< "received " <<  result_count << endl;
                writer.add_result(job, result.get());
                // cout<< "result added" << endl;
            }
        }

        for (int i = 1; i < world_size; i++) {
            // finish all workers
            // cout<< i <<". finsihing" << endl;
            MPI_Send(&i, 1, MPI_INT32_T, i, DONE_CODE, MPI_COMM_WORLD);
        }

        bstream.close();
        astream.close();
        writer.write();
    } else {
        // worker code
        // cout<< world_rank  << ".working " << endl;
        MPI_Status status;
        int32_t a_row_count, b_col_count, chunk_count;
        // receive block of rows from A

        MPI_Recv(row.get(), BLOCK_SIZE, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        // cout<< world_rank  << ".received initial A "  << endl;
        while (status.MPI_TAG != DONE_CODE) {
            // fill result with zeros
            // cout<< world_rank  << ".Filling with zeros " << endl;
            fill(result.get(), result.get() + BLOCK_SIZE, .0f);

            // receive rows from A, cols from B
            MPI_Recv(cols.get(), BLOCK_SIZE, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // cout<< world_rank  << ".Received B " << endl;

            MPI_Recv(&a_row_count, 1, MPI_INT32_T, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // cout<< world_rank  << ".1" << endl;
            MPI_Recv(&b_col_count, 1, MPI_INT32_T, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // cout<< world_rank  << ".2" << endl;
            MPI_Recv(&chunk_count, 1, MPI_INT32_T, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // cout<< world_rank  << ".couting" << endl;
            // multiplication
            for (uint32_t i = 0; i < a_row_count; ++i) {
                for (uint32_t j = 0; j < b_col_count; ++j) {
                    for (uint32_t k = 0; k < chunk_count; ++k) {
                        result.get()[i * chunk_count + j] += row.get()[i*chunk_count + k] * cols.get()[k*chunk_count + j];
                    }
                }
            }

            // cout<< world_rank  << ".counted" << endl;

            // cout<< world_rank  << ".Sending result " << a_row_count * b_col_count << endl;
            // send result back
            MPI_Send(result.get(), a_row_count * b_col_count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

            // get rows from A
            MPI_Recv(row.get(), BLOCK_SIZE, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}