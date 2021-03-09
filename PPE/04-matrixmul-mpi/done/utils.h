#include <fstream>
#include <vector>
#include <queue>
#include <iostream>
#include <memory>

struct matrix
{
    std::uint32_t rows, collumns;

    matrix(): rows(0), collumns(0) {}
};

struct chunk
{
    std::uint32_t begin, end;

    chunk(): begin(0), end(0) {}
    chunk(std::uint32_t b, std::uint32_t e): begin(b), end(e) {}
    chunk(const chunk& other): begin(other.begin), end(other.end) {}
};

struct job
{
    std::uint32_t   a_row_begin,
            b_row_begin,
            a_collumn_begin,
            b_collumn_begin,
            a_rows,
            b_collumns,
            chunk_size;

    job(std::uint32_t arb, std::uint32_t brb, std::uint32_t acb, std::uint32_t bcb, std::uint32_t ar, std::uint32_t bc, std::uint32_t cs):
            a_row_begin(arb), b_row_begin(brb), a_collumn_begin(acb), b_collumn_begin(bcb), a_rows(ar), b_collumns(bc), chunk_size(cs)
    {}
};

struct worker
{
    int id;
    chunk row;
    chunk collumn;

    worker(): id(-1) {}
    worker(const worker& other): id(other.id), row(other.row), collumn(other.collumn) {}
};

struct size_policy
{
    static std::uint32_t row_chunk()
    {
        return 512;
    }

    static std::uint32_t collumn_chunk()
    {
        return 512;
    }

    static std::uint32_t result_chunk()
    {
        return 512;
    }
};

template <typename T = size_policy>
class loader
{
    std::ifstream& input_A;
    std::ifstream& input_B;
    matrix A, B;
    std::uint32_t   row_chunk, // row is from A
    collumn_chunk, // collumn is from B
    result_chunk;

    std::queue<job> jobs;

    std::vector<chunk> split_to_chunks(std::uint32_t chunk_begin, std::uint32_t chunk_end, std::uint32_t chunk_size)
    {
        std::vector<chunk> chunks;

        while(chunk_begin + chunk_size <= chunk_end)
        {
            //std::cout << chunk_begin << " " << chunk_begin + chunk_size << std::endl;
            chunks.emplace_back(chunk(chunk_begin, chunk_begin + chunk_size));
            chunk_begin += chunk_size;
        }

        if(chunk_begin != chunk_end)
        {
            //std::cout << chunk_begin << " " << chunk_end << std::endl;
            chunks.emplace_back(chunk(chunk_begin, chunk_end));
        }

        return chunks;
    }

    void create_chunks()
    {
        std::vector<chunk> a_chunks = split_to_chunks(0, A.rows, row_chunk);
        std::vector<chunk> b_chunks = split_to_chunks(0, B.collumns, collumn_chunk);
        std::vector<chunk> result_chunks = split_to_chunks(0, A.collumns, result_chunk);

        for(auto&& a: a_chunks)
        {
            for(auto&& b: b_chunks)
            {
                for(auto&& r: result_chunks)
                {
                    jobs.push(job(a.begin, r.begin, r.begin, b.begin, a.end - a.begin, b.end - b.begin, r.end - r.begin));
                }
            }
        }
    }

public:

    loader(std::ifstream& a, std::ifstream& b): input_A(a), input_B(b), row_chunk(T::row_chunk()), collumn_chunk(T::collumn_chunk()), result_chunk(T::result_chunk())
    {
        input_A.read(reinterpret_cast<char*>(&A.collumns), sizeof(uint32_t));
        input_A.read(reinterpret_cast<char*>(&A.rows), sizeof(uint32_t));

        input_B.read(reinterpret_cast<char*>(&B.collumns), sizeof(uint32_t));
        input_B.read(reinterpret_cast<char*>(&B.rows), sizeof(uint32_t));

        create_chunks();
    }

    bool get_job(worker& worker, float* row_buffer, std::uint32_t& row_element_count, float* collumn_buffer, std::uint32_t& collumn_element_count)
    {
        if(jobs.size() == 0)
        {
            return false;
        }

        job current_job = jobs.front();
        jobs.pop();

        std::uint32_t offset = 8 + (current_job.a_row_begin * A.collumns + current_job.a_collumn_begin) * sizeof(float);
        std::uint32_t seek;

        for (std::uint32_t i = 0; i < current_job.a_rows; ++i)
        {
            seek = offset + i * A.collumns * sizeof(float);

            input_A.seekg(seek);
            input_A.read((char*)&row_buffer[i * current_job.chunk_size], sizeof(float) * current_job.chunk_size);
        }

        offset = 8 + (current_job.b_row_begin * B.collumns + current_job.b_collumn_begin) * sizeof(float);

        for (std::uint32_t i = 0; i < current_job.chunk_size; ++i)
        {
            seek = offset + i * B.collumns * sizeof(float);

            input_B.seekg(seek);
            input_B.read((char*)&collumn_buffer[i*current_job.b_collumns], sizeof(float) * current_job.b_collumns);
        }

        row_element_count = current_job.a_rows * current_job.chunk_size;
        collumn_element_count = current_job.b_collumns * current_job.chunk_size;

        worker.row.begin = current_job.a_row_begin;
        worker.row.end = current_job.a_row_begin + current_job.a_rows;

        worker.collumn.begin = current_job.b_collumn_begin;
        worker.collumn.end = current_job.b_collumn_begin + current_job.b_collumns;

        return true;
    }

    const matrix& get_A()
    {
        return A;
    }

    const matrix& get_B()
    {
        return B;
    }

};

template <typename T = size_policy>
class writer
{
    std::ofstream& output;
    std::unique_ptr<float[]> result;
    std::uint32_t result_rows;
    std::uint32_t result_collumns;

public:

    writer(std::ofstream& o, loader<T>& l): output(o)
    {
        result_rows = l.get_A().rows;
        result_collumns = l.get_B().collumns;
        result = std::make_unique<float[]>(result_rows * result_collumns);
    }

    void process_result(const worker& worker, float* result_chunk)
    {
        std::uint32_t rows = worker.row.end - worker.row.begin;
        std::uint32_t collumns = worker.collumn.end - worker.collumn.begin;

        float* result_place = &(result.get()[worker.row.begin * result_collumns + worker.collumn.begin]);

        for (std::uint32_t i = 0; i < rows; ++i)
        {
            for (std::uint32_t j = 0; j < collumns; ++j)
            {
                result_place[i * result_collumns + j] += result_chunk[i * collumns + j];
            }
        }
    }

    void write()
    {
        output.write((char*)&result_collumns, sizeof(std::uint32_t));
        output.write((char*)&result_rows, sizeof(std::uint32_t));

        output.write((char*)&(result.get()[0]), sizeof(float) * result_collumns * result_rows);
    }
};