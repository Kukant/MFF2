#include <iostream>
#include <mpi.h>
#include <fstream>
#include <vector>
#include <queue>
#include <memory>

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

    unique_ptr<float[]> row = make_unique<float[]>(BLOCK_SIZE);
    unique_ptr<float[]> cols = make_unique<float[]>(BLOCK_SIZE);
    unique_ptr<float[]> result = make_unique<float[]>(BLOCK_SIZE);

    if(world_rank == 0)
    {
        ofstream rstream;
        ifstream astream, bstream;
        astream.open(argv[1], std::ios::binary | std::ios::in);
        bstream.open(argv[2], std::ios::binary | std::ios::in);
        rstream.open(argv[3], std::ios::binary | std::ios::out);

        matrix_reader reader(astream, bstream);
        result_writer writer(reader, rstream);

        uint32_t job_assigments = reader.jobs.size() / (WORKERS_COUNT - 1);
        if (reader.jobs.size()  % WORKERS_COUNT)  job_assigments++;

        for (int i = 0; i < job_assigments; i++) {
            // send data to workers
            queue<sub_mult> running_jobs;
            int todo_jobs = reader.jobs.size();
            cout << "todo " << todo_jobs << endl;
            for (int worker_id = 1; worker_id < min(WORKERS_COUNT, todo_jobs); worker_id++) {
                cout << worker_id << ". starting worker "  << endl;
                sub_mult job = reader.get_job(row.get(), cols.get());
                cout << worker_id << ". job loaded " << endl;
                job.worker_id = worker_id;
                uint32_t rows_count = job.a_row_end - job.a_row_start;
                uint32_t cols_count = job.b_col_end - job.b_col_start;
                uint32_t chunk_count = job.chunk_count;

                cout << worker_id <<  ". sending A " << endl;
                // send matrices
                MPI_Send(row.get(),
                         BLOCK_SIZE,
                        MPI_FLOAT, worker_id, 0, MPI_COMM_WORLD);

                cout << worker_id << ". sending B " << endl;
                MPI_Send(cols.get(),
                         BLOCK_SIZE,
                         MPI_FLOAT, worker_id, 0, MPI_COMM_WORLD);

                cout << worker_id << ". sending Additinal info  " << endl;

                // send additional info
                MPI_Send(&rows_count, 1, MPI_INT32_T, worker_id, 0, MPI_COMM_WORLD);
                MPI_Send(&cols_count, 1, MPI_INT32_T, worker_id, 0, MPI_COMM_WORLD);
                MPI_Send(&chunk_count, 1, MPI_INT32_T, worker_id, 0, MPI_COMM_WORLD);
                cout << worker_id << ". Additinal info sent  " << endl;
                running_jobs.push(job);
            }

            while (!running_jobs.empty()) {
                sub_mult job = running_jobs.front();
                running_jobs.pop();
                uint32_t result_count = (job.a_row_end - job.a_row_start) * (job.b_col_end - job.b_col_start);
                cout << "receiving " <<  result_count << endl;
                MPI_Recv(result.get(), result_count, MPI_FLOAT, job.worker_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cout << "received " <<  result_count << endl;
                writer.add_result(job, result.get());
                cout << "result added" << endl;
            }
        }

        for (int i = 1; i < WORKERS_COUNT; i++) {
            // finish all workers
            cout << i <<". finsihing" << endl;
            MPI_Send(&i, 1, MPI_INT32_T, i, DONE_CODE, MPI_COMM_WORLD);
        }

    bstream.close();
    astream.close();
    writer.write();
}