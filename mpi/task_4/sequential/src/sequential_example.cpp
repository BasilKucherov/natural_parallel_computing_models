#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace {
constexpr char OUTPUT_FILE[] = "sequential_output.txt";
constexpr char OUTPUT_STAT_FILE[] = "sequential_stat.txt";
constexpr char AXIOM[] = "a";

enum class RuleSetType { D0L_1, D0L_2, D0L_3, D0L_4 };
} // namespace

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

void RunLsystem(const int T,
                const std::map<char, std::pair<std::string, double>> &R,
                const std::string &axiom) {
  std::string data = axiom;

  for (int t = 0; t < T; t++) {
    data = UpdateData(data, R);
    std::cout << "t = " << t << ", length = " << data.length() << std::endl;
  }

  std::ofstream f(OUTPUT_FILE);
  f << data << std::endl;
  f.close();
}

int main(int argc, char **argv) {
  std::srand(std::time(nullptr));
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <T> <ruleSetType>" << std::endl;
    return 1;
  }

  int T = std::stoi(argv[1]);
  RuleSetType ruleSetType = static_cast<RuleSetType>(std::stoi(argv[2]));

  const auto ruleSet = GetRuleSet(ruleSetType);
  RunLsystem(T, ruleSet, AXIOM);

  return 0;
}
