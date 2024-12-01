#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <string>

namespace {
constexpr char OUTPUT_FILE[] = "parallel_output.txt";
constexpr char OUTPUT_STAT_FILE[] = "parallel_stat.txt";
constexpr char AXIOM[] = "a";

enum class RuleSetType { D0L_1, D0L_2, D0L_3, D0L_4 };
} // namespace

std::string
UpdateData(std::string data,
           const std::map<char, std::pair<std::string, double>> &R) {
  std::string buf = "";

  for (unsigned int i = 0; i < data.length(); i++) {
    if (R.find(data[i]) != R.end() &&
        std::rand() < R.at(data[i]).second * RAND_MAX) {
      buf += R.at(data[i]).first;
    } else {
      buf += data[i];
    }
  }

  return buf;
}

std::map<char, std::pair<std::string, double>>
GetRuleSet(const RuleSetType ruleSetType) {
  std::map<char, std::pair<std::string, double>> R;

  switch (ruleSetType) {
  case RuleSetType::D0L_1:
    R['a'] = {"b", 1.0};
    R['b'] = {"ab", 1.0};
    break;
  case RuleSetType::D0L_2:
    R['a'] = {"ab", 1.0};
    R['b'] = {"bc", 1.0};
    break;
  case RuleSetType::D0L_3:
    R['a'] = {"aa", 0.001};
    break;
  case RuleSetType::D0L_4:
    R['a'] = {"ab", 0.01};
    R['b'] = {"a", 0.01};
    break;
  }

  return R;
}

size_t GetCount(size_t l, size_t lNext) {
  if (l > lNext)
    return (l - lNext) / 2;
  return 0;
}

std::string AlignLoad(const std::string &data) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int prev = (rank == 0) ? MPI_PROC_NULL : (rank - 1);
  int next = (rank == size - 1) ? MPI_PROC_NULL : (rank + 1);

  std::string newData;
  size_t lNext = 0, lPrev = 0;
  size_t l = data.length();

  MPI_Sendrecv(&l, 1, MPI_INT, prev, 0, &lNext, 1, MPI_INT, next, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&l, 1, MPI_INT, next, 0, &lPrev, 1, MPI_INT, prev, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  auto sendNextSize = rank != (size - 1) ? GetCount(l, lNext) : 0;
  auto recvPrevSize = rank != 0 ? GetCount(lPrev, l) : 0;
  char *recvBuf = new char[recvPrevSize + 1];

  MPI_Sendrecv(&data[0] + data.length() - sendNextSize, sendNextSize, MPI_CHAR,
               next, 0, recvBuf, recvPrevSize, MPI_CHAR, prev, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  recvBuf[recvPrevSize] = 0;

  newData = std::string(recvBuf) + data.substr(0, data.length() - sendNextSize);

  MPI_Sendrecv(&l, 1, MPI_INT, prev, 0, &lNext, 1, MPI_INT, next, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&l, 1, MPI_INT, next, 0, &lPrev, 1, MPI_INT, prev, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  auto sendPrevSize = rank != 0 ? GetCount(l, lPrev) : 0;
  auto recvNextSize = rank != (size - 1) ? GetCount(lNext, l) : 0;

  delete[] recvBuf;
  recvBuf = new char[recvNextSize + 1];

  MPI_Sendrecv(&newData[0], sendPrevSize, MPI_CHAR, prev, 0, recvBuf,
               recvNextSize, MPI_CHAR, next, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  recvBuf[recvNextSize] = 0;
  newData = newData.substr(sendPrevSize, newData.size() - sendPrevSize) +
            std::string(recvBuf);

  return newData;
}

void RunLsystem(const int T, const int rebalancePeriod,
                const std::map<char, std::pair<std::string, double>> &R,
                const std::string &axiom) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string data = rank == (size / 2) ? axiom : "";
  std::vector<int> iterLengths;
  iterLengths.reserve(T / rebalancePeriod);

  for (int t = 0; t < T; t++) {
    if (rank == 0 && (t + 1) % 100 == 0) {
      std::cout << t << "  " << std::to_string(data.length()) << std::endl;
    }
    data = UpdateData(data, R);

    if ((t + 1) % rebalancePeriod == 0) {
      iterLengths.push_back(data.length());
      data = AlignLoad(data);
    }
  }

  std::string result;
  if (rank == 0) {
    std::map<int, std::vector<int>> processIterLengths;
    processIterLengths[0] = iterLengths;

    std::vector<int> lengths;
    lengths.resize(size);
    lengths[0] = data.length();
    int totalLength = lengths[0];
    int maxLength = lengths[0];
    for (int i = 1; i < size; i++) {
      MPI_Recv(&lengths[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      totalLength += lengths[i];

      if (lengths[i] > maxLength) {
        maxLength = lengths[i];
      }
    }

    result.resize(totalLength);
    result = "";
    result += data;

    auto currPoint = lengths[0];
    char *buf = new char[maxLength + 1];
    for (int i = 1; i < size; i++) {
      MPI_Recv(buf, lengths[i], MPI_CHAR, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      buf[lengths[i]] = 0;
      result += std::string(buf);
      currPoint += lengths[i];
    }
    delete[] buf;

    std::ofstream f(OUTPUT_FILE);
    f << result << std::endl;
    f.close();

    int *bufLengths = new int[T / rebalancePeriod];
    for (int i = 1; i < size; i++) {
      MPI_Recv(bufLengths, T / rebalancePeriod, MPI_INT, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      processIterLengths[i] =
          std::vector<int>(bufLengths, bufLengths + T / rebalancePeriod);
    }
    delete[] bufLengths;

    std::vector<int> totalIterLength(T / rebalancePeriod, 0);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < T / rebalancePeriod; j++) {
        totalIterLength[j] += processIterLengths[i][j];
      }
    }

    std::ofstream fStat(OUTPUT_STAT_FILE);
    fStat << "{";
    for (int i = 0; i < size; i++) {
      fStat << "\"" << i << "\": [";
      for (int j = 0; j < T / rebalancePeriod; j++) {
        fStat << ((double)processIterLengths[i][j]) / totalIterLength[j];
        if (j != (T / rebalancePeriod - 1)) {
          fStat << ", ";
        }
      }
      fStat << "]";
      if (i != size - 1) {
        fStat << ", ";
      }
    }
    fStat << "}" << std::endl;
    fStat.close();
  } else {
    auto dataLength = data.length();
    MPI_Send(&dataLength, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&data[0], dataLength, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&iterLengths[0], iterLengths.size(), MPI_INT, 0, 0,
             MPI_COMM_WORLD);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <T> <rebalancePeriod> <ruleSetType>"
              << std::endl;
    return 1;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::srand(std::time(NULL) + rank);

  int T = std::stoi(argv[1]);
  int rebalancePeriod = std::stoi(argv[2]);
  RuleSetType ruleSetType = static_cast<RuleSetType>(std::stoi(argv[3]));

  RunLsystem(T, rebalancePeriod, GetRuleSet(ruleSetType), AXIOM);

  MPI_Finalize();
  return 0;
}
