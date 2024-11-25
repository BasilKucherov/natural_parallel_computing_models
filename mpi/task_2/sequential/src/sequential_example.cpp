#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

enum class EFunction { SPHERICAL, ROSENBROCK, RASTRIGIN };

namespace {
struct TGenerationStat {
  double bestError;
  double meanError;
  int generation;
};

struct TGAConfig {
  int genomeLength;
  int populationSize;
  int numberOfGenerations;
  std::pair<double, double> searchRange;
  double mutationFactor;
  EFunction function;
  std::string functionName;
};

std::map<std::string, EFunction> functionNameToEnum = {
    {"spherical", EFunction::SPHERICAL},
    {"rosenbrock", EFunction::ROSENBROCK},
    {"rastrigin", EFunction::RASTRIGIN}};
} // namespace

double frand() { return double(std::rand()) / RAND_MAX; }

double Eval(double *genome, int genomeLength,
            EFunction function = EFunction::SPHERICAL) {
  double sum = 0;

  switch (function) {
  case EFunction::SPHERICAL:
    for (int i = 0; i < genomeLength; i++)
      sum += std::pow(genome[i], 2);
    break;
  case EFunction::ROSENBROCK:
    for (int i = 0; i < genomeLength - 1; i++)
      sum += 100 * std::pow(std::pow(genome[i], 2) - genome[i + 1], 2) +
             std::pow(genome[i] - 1, 2);
    break;
  case EFunction::RASTRIGIN:
    for (int i = 0; i < genomeLength; i++)
      sum += std::pow(genome[i], 2) - 10 * std::cos(2 * M_PI * genome[i]) + 10;
    break;
  }

  return sum;
}

TGenerationStat EvalGeneration(double *population, int populationSize,
                               int genomeLength, EFunction function) {
  TGenerationStat generationStat;
  generationStat.generation = 0;

  generationStat.bestError = std::numeric_limits<double>::max();
  generationStat.meanError = 0;

  for (int k = 0; k < populationSize; k++) {
    const auto error =
        Eval(population + k * genomeLength, genomeLength, function);
    generationStat.meanError += error;
    if (error < generationStat.bestError) {
      generationStat.bestError = error;
    }
  }

  generationStat.meanError /= populationSize;

  return generationStat;
}

void Init(double *population, int populationSize, int genomeLength,
          std::pair<double, double> searchRange) {
  for (int k = 0; k < populationSize; k++)
    for (int i = 0; i < genomeLength; i++)
      population[k * genomeLength + i] =
          searchRange.first +
          frand() * (searchRange.second - searchRange.first);
}

void Shuffle(double *population, int populationSize, int genomeLength) {
  for (int k = 0; k < populationSize; k++) {
    int l = rand() % populationSize;
    for (int i = 0; i < genomeLength; i++)
      std::swap(population[k * genomeLength + i],
                population[l * genomeLength + i]);
  }
}

void Select(double *population, int populationSize, int genomeLength) {
  double pwin = 0.75;
  Shuffle(population, populationSize, genomeLength);
  for (int k = 0; k < populationSize / 2; k++) {
    int a = 2 * k;
    int b = 2 * k + 1;
    int fa = Eval(population + a * genomeLength, genomeLength);
    int fb = Eval(population + b * genomeLength, genomeLength);
    double p = frand();
    if (fa < fb && p < pwin || fa > fb && p > pwin)
      for (int i = 0; i < genomeLength; i++)
        population[b * genomeLength + i] = population[a * genomeLength + i];
    else
      for (int i = 0; i < genomeLength; i++)
        population[a * genomeLength + i] = population[b * genomeLength + i];
  }
}

void Crossover(double *population, int populationSize, int genomeLength) {
  Shuffle(population, populationSize, genomeLength);
  for (int k = 0; k < populationSize / 2; k++) {
    int a = 2 * k;
    int b = 2 * k + 1;
    int j = rand() % genomeLength;
    for (int i = j; i < genomeLength; i++)
      std::swap(population[a * genomeLength + i],
                population[b * genomeLength + i]);
  }
}

void Mutate(double *population, int populationSize, int genomeLength,
            double mutationFactor = 1) {
  double mutationProbability = 0.1;
  for (int k = 0; k < populationSize; k++)
    for (int i = 0; i < genomeLength; i++)
      if (frand() < mutationProbability)
        population[k * genomeLength + i] = population[k * genomeLength + i] +
                                           2 * mutationFactor * (frand() - 0.5);
}

void PrintBest(double *population, int populationSize, int genomeLength) {
  int k0 = -1;
  double f0 = std::numeric_limits<double>::max();
  for (int k = 0; k < populationSize; k++) {
    auto f = Eval(population + k * genomeLength, genomeLength);
    if (f < f0) {
      f0 = f;
      k0 = k;
    }
  }
  std::cout << f0 << ": ";
  for (int i = 0; i < genomeLength; i++)
    std::cout << population[k0 * genomeLength + i] << " ";
  std::cout << std::endl;
}

std::vector<TGenerationStat> RunGA(const TGAConfig &config,
                                   bool printBest = true) {
  std::vector<TGenerationStat> generationStats;
  generationStats.reserve(config.numberOfGenerations + 1);

  double *population = new double[config.genomeLength * config.populationSize];
  Init(population, config.populationSize, config.genomeLength,
       config.searchRange);

  generationStats.push_back(EvalGeneration(
      population, config.populationSize, config.genomeLength, config.function));
  generationStats.back().generation = 0;

  for (int t = 0; t < config.numberOfGenerations; t++) {
    Select(population, config.populationSize, config.genomeLength);
    Crossover(population, config.populationSize, config.genomeLength);
    Mutate(population, config.populationSize, config.genomeLength,
           config.mutationFactor);

    if (printBest)
      PrintBest(population, config.populationSize, config.genomeLength);

    generationStats.push_back(EvalGeneration(population, config.populationSize,
                                             config.genomeLength,
                                             config.function));
    generationStats.back().generation = t + 1;
  }

  delete[] population;
  return generationStats;
}

void PrintHelp(const std::string &programName) {
  std::cerr << "Usage: " << programName << " <genome length> <population size> "
            << "<number of generations> <function name>" << std::endl;
  std::cerr << "Function name: spherical, rosenbrock, rastrigin";
}

TGAConfig ParseConfig(int argc, char **argv) {
  if (argc != 5) {
    PrintHelp(argv[0]);
    throw std::invalid_argument("Invalid number of arguments");
  }

  TGAConfig config;
  try {
    config.genomeLength = std::stoi(argv[1]);
    config.populationSize = std::stoi(argv[2]);
    config.numberOfGenerations = std::stoi(argv[3]);
    config.functionName = argv[4];
    config.function = functionNameToEnum.at(config.functionName);
  } catch (const std::exception &e) {
    PrintHelp(argv[0]);
    throw std::invalid_argument(e.what());
  }

  config.searchRange = {-100, 100};
  config.mutationFactor = 1;

  return config;
}

void SaveGenerationStatsCSV(const std::vector<TGenerationStat> &generationStats,
                            const std::string &filename) {
  std::ofstream file(filename);
  file << "generation,bestError,meanError" << std::endl;
  for (const auto &stat : generationStats) {
    file << stat.generation << "," << stat.bestError << "," << stat.meanError
         << std::endl;
  }
}

int main(int argc, char **argv) {
  const auto config = ParseConfig(argc, argv);

  const auto generationStats = RunGA(config, false);
  SaveGenerationStatsCSV(generationStats, config.functionName + ".csv");
  return 0;
}