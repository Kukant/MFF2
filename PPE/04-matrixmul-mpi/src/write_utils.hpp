#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <memory>
#include <unistd.h>

#define CHUNK_SIZE 512
#define BLOCK_SIZE CHUNK_SIZE * CHUNK_SIZE
#define DONE_CODE 424242

using namespace std;

struct sub_mult {
    uint32_t a_row_start,
            b_col_start,
            a_col_start,
            b_row_start,
            a_row_end,
            b_col_end,
            chunk_count,
            worker_id;

    sub_mult(uint32_t a_row_start,
             uint32_t b_col_start,
             uint32_t a_row_end,
             uint32_t b_col_end,
             uint32_t chunk_count,
             uint32_t b_row_start,
             uint32_t a_col_start
    ):
            a_row_start(a_row_start), b_col_start(b_col_start),
            a_row_end(a_row_end), b_col_end(b_col_end),
            a_col_start(a_col_start), b_row_start(b_row_start),
            chunk_count(chunk_count) {}

    void debug_print() {
        cout<< "sr:er:sc:ec:off "
            << a_row_start
            << ":" << a_row_end
            << ":-:" << b_col_start
            << ":" << b_col_end
            << ":" << chunk_count
            << ":" << a_col_start
            << endl;
    }
};

class matrix_reader {
    ifstream& astream;
    ifstream& bstream;
public:
    queue<sub_mult> jobs;
    uint32_t a_cols, b_cols, a_rows, b_rows;
    matrix_reader(ifstream& a, ifstream& b): astream(a), bstream(b) {
        // load matrix sizes
        astream.read(reinterpret_cast<char*>(&a_cols), sizeof(uint32_t));
        astream.read(reinterpret_cast<char*>(&a_rows), sizeof(uint32_t));
        bstream.read(reinterpret_cast<char*>(&b_cols), sizeof(uint32_t));
        bstream.read(reinterpret_cast<char*>(&b_rows), sizeof(uint32_t));

        //cout<< "A:[" << a_rows << ", " << a_cols << "]\n";
        //cout<< "B:[" << b_rows << ", " << b_cols << "]\n";

        uint32_t row_ranges = a_rows/CHUNK_SIZE;
        uint32_t col_ranges = b_cols/CHUNK_SIZE;
        uint32_t sub_jobs_count = a_cols/CHUNK_SIZE;

        for(uint32_t i = 0; i < row_ranges; i++) {
            uint32_t r_count = CHUNK_SIZE,
                    r_start = i*CHUNK_SIZE;
            for(uint32_t j = 0; j < col_ranges; j++) {
                uint32_t c_count = CHUNK_SIZE;
                uint32_t c_start = j*CHUNK_SIZE;

                for(uint32_t k = 0; k < sub_jobs_count; k++) {
                    uint32_t ch_count = CHUNK_SIZE;
                    uint32_t ch_offset = k * CHUNK_SIZE;

                    sub_mult ss(
                            r_start, c_start,
                            r_start + r_count, c_start + c_count, ch_count,
                            ch_offset, ch_offset);
                    jobs.emplace(ss);
                }
            }
        }
    }

    sub_mult get_job(float* row_buffer, float* col_buffer)
    {
        sub_mult job = jobs.front();
        jobs.pop();
        //job.debug_print();

        uint32_t offset = 8;
        uint32_t row_offset;
        uint32_t col_offset;
        uint32_t cols_to_move;
        uint32_t seek;
        // cout<< "loading A" << endl;
        // load rows
        for (uint32_t i = job.a_row_start; i < job.a_row_end; i++)
        {
            row_offset = i * a_cols;
            col_offset = job.a_col_start;
            seek = offset + (row_offset + col_offset) * sizeof(float);
            astream.seekg(seek);
            uint32_t start_write = (i - job.a_row_start) * job.chunk_count;
            //// cout<< "loading A-" << i <<  " " << seek <<  " " << start_write << endl;
            astream.read((char*)&row_buffer[start_write], sizeof(float) * job.chunk_count);
        }
        // cout<< "loading B" << endl;
        // load cols
        cols_to_move = job.b_col_end - job.b_col_start;
        for (uint32_t i = job.b_row_start; i < job.b_row_start + job.chunk_count; i++)
        {
            row_offset = i * b_cols;
            col_offset = job.b_col_start;
            seek = offset + (row_offset + col_offset) * sizeof(float);

            bstream.seekg(seek);
            uint32_t start_write = (i - job.b_row_start) * cols_to_move;
            bstream.read((char*)&col_buffer[start_write], sizeof(float) * cols_to_move);
        }

        // cout<< "A, B loaded" << endl;

        return job;
    }
};

class result_writer {
    ofstream& output;
    uint32_t r_cols, r_rows;
public:
    vector<float> write_result;
    result_writer(matrix_reader &reader, ofstream& o):
            output(o), r_cols(reader.b_cols), r_rows(reader.a_rows) {
        // cout<< "R[" << r_rows << ", " << r_cols << "]\n";
        write_result.resize(r_cols * r_rows);
    }

    void add_result(sub_mult job, float* res) {
        static float last_ch = 0;
        for (uint32_t i = job.a_row_start; i < job.a_row_end; i++) {
            for (uint32_t j = job.b_col_start; j < job.b_col_end; j++) {
                uint32_t res_i = i - job.a_row_start;
                uint32_t res_j = j - job.b_col_start;
                write_result[i * r_cols + j] += res[res_i * job.chunk_count + res_j];
            }
        }
    }

    void write() {
        //cout << "result: '" <<  write_result[1024*1024 -1] << "' " << endl;
        output.write((char*)&r_cols, sizeof(uint32_t));
        output.write((char*)&r_rows, sizeof(uint32_t));
        output.write((char*)write_result.data(), sizeof(float) * r_cols * r_rows);
    }
};
