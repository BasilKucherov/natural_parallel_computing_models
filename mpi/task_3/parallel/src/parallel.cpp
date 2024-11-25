#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>

namespace {
struct TLifeConfig {
  int n;
  int T;
};
} // namespace

int F(int *data, int i, int j, int n) {
  int state = data[i * (n + 2) + j];
  int s = -state;
  for (int ii = i - 1; ii <= i + 1; ii++)
    for (int jj = j - 1; jj <= j + 1; jj++)
      s += data[ii * (n + 2) + jj];
  if (state == 0 && s == 3)
    return 1;
  if (state == 1 && (s < 2 || s > 3))
    return 0;
  return state;
}

void UpdateData(int n, int *data, int *temp) {
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      temp[i * (n + 2) + j] = F(data, i, j, n);
}

void Init(int n, int *data, int *temp) {
  for (int i = 0; i < (n + 2) * (n + 2); i++)
    data[i] = temp[i] = 0;
  int n0 = 1 + n / 2;
  int m0 = 1 + n / 2;
  data[(n0 - 1) * (n + 2) + m0] = 1;
  data[n0 * (n + 2) + m0 + 1] = 1;
  for (int i = 0; i < 3; i++)
    data[(n0 + 1) * (n + 2) + m0 + i - 1] = 1;
}

void SetupBoundaries(int n, int *data) {
  for (int i = 0; i < n + 2; i++) {
    data[i * (n + 2) + 0] = data[i * (n + 2) + n];
    data[i * (n + 2) + n + 1] = data[i * (n + 2) + 1];
  }
  for (int j = 0; j < n + 2; j++) {
    data[0 * (n + 2) + j] = data[n * (n + 2) + j];
    data[(n + 1) * (n + 2) + j] = data[1 * (n + 2) + j];
  }
}

void DistributeData(int *data, int n, int p, MPI_Datatype mpi_block_type) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int N = p * n;

  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      for (int j = 0; j < p; j++) {
        MPI_Send(&data[(i * n * (N + 2) + j * n)], 1, mpi_block_type,
                 i * p + j + 1, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Recv(data, (n + 2) * (n + 2), MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void CreateBlockType(int N, int blockSize, MPI_Datatype *blockType) {
  MPI_Datatype temp_type;

  int count = blockSize;
  int blocklength = blockSize;
  int stride = N;

  MPI_Type_vector(count, blocklength, stride, MPI_INT, &temp_type);
  MPI_Type_commit(&temp_type);

  MPI_Type_create_resized(temp_type, 0, blockSize * sizeof(int), blockType);
  MPI_Type_commit(blockType);

  MPI_Type_free(&temp_type);
}

void CreateColType(int n, MPI_Datatype *colType) {
  MPI_Datatype temp_type;

  MPI_Type_vector(n, 1, n, MPI_INT, &temp_type);
  MPI_Type_commit(&temp_type);

  MPI_Type_create_resized(temp_type, 0, sizeof(int), colType);

  MPI_Type_commit(colType);
  MPI_Type_free(&temp_type);
}

void SetupBoundariesMPI(int *data, int n, int p, MPI_Datatype mpiColType) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int i = (rank - 1) / p;
  int j = (rank - 1) % p;

  int left = (j != 0) ? (rank - 1) : (rank + p - 1);
  int right = (j != p - 1) ? (rank + 1) : (rank - p + 1);
  int above = (i != 0) ? (rank - p) : (rank + p * (p - 1));
  int below = (i != p - 1) ? (rank + p) : (rank - p * (p - 1));

  MPI_Sendrecv(data + 1, 1, mpiColType, left, 0, data + n + 1, 1, mpiColType,
               right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(data + n, 1, mpiColType, right, 0, data, 1, mpiColType, left, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(data + n + 2, n + 2, MPI_INT, above, 0, data + (n + 2) * (n + 1),
               n + 2, MPI_INT, below, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(data + (n + 2) * n, n + 2, MPI_INT, below, 0, data, n + 2,
               MPI_INT, above, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void CollectData(int *data, int n, int p, MPI_Datatype mpi_block_type) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = p * n;

  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      for (int j = 0; j < p; j++) {
        MPI_Recv(&data[(i * n * (N + 2) + j * n)], 1, mpi_block_type,
                 i * p + j + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  } else {
    MPI_Send(data, (n + 2) * (n + 2), MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

void RunLife(int n, int T) {
  const auto startTime = std::chrono::high_resolution_clock::now();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int p = std::floor(std::sqrt(size - 1));
  int N = p * n;

  MPI_Datatype mpiBlockType;
  CreateBlockType(N + 2, n + 2, &mpiBlockType);

  MPI_Datatype mpiColType;
  CreateColType(n + 2, &mpiColType);

  int *data = nullptr;
  int *temp = nullptr;

  if (rank == 0) {
    data = new int[(N + 2) * (N + 2)];
    Init(N, data, data);
    SetupBoundaries(N, data);
  } else {
    data = new int[(n + 2) * (n + 2)];
    temp = new int[(n + 2) * (n + 2)];
  }

  DistributeData(data, n, p, mpiBlockType);
  MPI_Barrier(MPI_COMM_WORLD);

  for (int t = 0; t < T; t++) {
    if (rank != 0) {
      UpdateData(n, data, temp);
      std::swap(data, temp);
      SetupBoundariesMPI(data, n, p, mpiColType);
    }
  }

  CollectData(data, n, p, mpiBlockType);

  MPI_Barrier(MPI_COMM_WORLD);
  const auto endTime = std::chrono::high_resolution_clock::now();

  if (rank == 0) {
    std::ofstream f("output.dat");
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j <= N; j++)
        f << data[i * (N + 2) + j];
      f << std::endl;
    }
    f.close();

    std::ofstream fstat("stat.txt");
    fstat << "{" << std::endl;
    fstat << "  \"time\": "
          << std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                                   startTime)
                 .count()
          << "," << std::endl;
    fstat << "  \"n\": " << n << "," << std::endl;
    fstat << "  \"T\": " << T << "," << std::endl;
    fstat << "  \"P\": " << size << std::endl;
    fstat << "}" << std::endl;
    fstat.close();
  }

  delete[] data;
  delete[] temp;

  MPI_Type_free(&mpiBlockType);
  MPI_Type_free(&mpiColType);
}

void PrintUsage() { std::cout << "Usage: ./parallel <n> <T>" << std::endl; }

TLifeConfig ParseArgs(int argc, char **argv) {
  if (argc != 3) {
    PrintUsage();
    exit(1);
  }
  TLifeConfig config;

  config.n = std::stoi(argv[1]);
  config.T = std::stoi(argv[2]);

  if (config.n <= 0 || config.T <= 0) {
    PrintUsage();
    exit(1);
  }

  return config;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto config = ParseArgs(argc, argv);

  RunLife(config.n, config.T);

  MPI_Finalize();
  return 0;
}