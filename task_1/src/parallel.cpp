#include <mpi.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

constexpr char DEFAULT_OUTPUT_FILENAME[] = "output.txt";
constexpr char DEFAULT_STAT_FILENAME[] = "stat.txt";

double FRand(double min, double max) {
  return min + (max - min) * (rand() / double(RAND_MAX));
}

int DoWalk(int leftBorder,
           int rightBorder,
           int start,
           double goRightProbability,
           int64_t& totalSteps) {
  int pos = start;
  while (pos > leftBorder && pos < rightBorder) {
    if (FRand(0, 1) < goRightProbability)
      pos += 1;
    else
      pos -= 1;
    totalSteps += 1;
  }
  return pos;
}

struct TMCResult {
  int64_t totalSteps;
  int64_t rightBorderReachedCount;
};

TMCResult RunMC(int leftBorder,
                int rightBorder,
                int start,
                double goRightProbability,
                int particlesNumber) {
  TMCResult mcResult{0, 0};

  for (int i = 0; i < particlesNumber; i++) {
    int newPos = DoWalk(leftBorder, rightBorder, start, goRightProbability,
                        mcResult.totalSteps);
    if (newPos == rightBorder)
      mcResult.rightBorderReachedCount += 1;
  }

  return mcResult;
}

void WriteStatToFile(const std::string& fileName,
                     int leftBorder,
                     int rightBorder,
                     int startPos,
                     double goRightProbability,
                     int particlesNumber,
                     double elapsedTime,
                     int parallelProcesses) {
  std::ofstream f(fileName);

  f << std::fixed << std::setprecision(8);
  f << leftBorder << " " << rightBorder << "\n";
  f << startPos << "\n";
  f << goRightProbability << "\n";
  f << particlesNumber << "\n";
  f << elapsedTime << "\n";
  f << parallelProcesses << "\n";

  f.close();
}

void WriteExperimentResultsToFile(const std::string& fileName,
                                  double rightReachedRatio,
                                  double avgStepsCount) {
  std::ofstream f(fileName);
  f << std::fixed << std::setprecision(8);
  f << rightReachedRatio << " " << avgStepsCount << "\n";
  f.close();
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  const int leftBorder = std::stoi(argv[1]);
  const int rightBorder = std::stoi(argv[2]);
  const int startPos = std::stoi(argv[3]);
  const double goRightProbability = std::stod(argv[4]);
  const int particlesNumber = std::stoi(argv[5]);

  const std::string outputFilename =
      argc >= 7 ? argv[6] : DEFAULT_OUTPUT_FILENAME;
  const std::string statFilename = argc >= 8 ? argv[7] : DEFAULT_STAT_FILENAME;

  const int localParticlesNumber =
      particlesNumber / worldSize +
      ((rank < (particlesNumber % worldSize)) ? 1 : 0);

  const auto localStartTime = MPI_Wtime();
  const auto localResult = RunMC(leftBorder, rightBorder, startPos,
                                 goRightProbability, localParticlesNumber);
  const auto localEndTime = MPI_Wtime();
  const double localElapsedTime = localEndTime - localStartTime;

  TMCResult globalResult;
  MPI_Reduce(&localResult.totalSteps, &globalResult.totalSteps, 1,
             MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&localResult.rightBorderReachedCount,
             &globalResult.rightBorderReachedCount, 1, MPI_LONG_LONG, MPI_SUM,
             0, MPI_COMM_WORLD);

  double maxElapsedTime;
  MPI_Reduce(&localElapsedTime, &maxElapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    const auto rightReachedRatio =
        (double)globalResult.rightBorderReachedCount / particlesNumber;
    const auto avgStepsCount =
        (double)globalResult.totalSteps / particlesNumber;

    WriteStatToFile(statFilename, leftBorder, rightBorder, startPos,
                    goRightProbability, particlesNumber, maxElapsedTime,
                    worldSize);
    WriteExperimentResultsToFile(outputFilename, rightReachedRatio,
                                 avgStepsCount);
  }

  MPI_Finalize();
  return 0;
}