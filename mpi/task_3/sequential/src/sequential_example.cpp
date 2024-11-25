#include <chrono>
#include <fstream>
#include <iostream>

using namespace std;

int f(int *data, int i, int j, int n) {
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

void update_data(int n, int *data, int *temp) {
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      temp[i * (n + 2) + j] = f(data, i, j, n);
}

void init(int n, int *data, int *temp) {
  for (int i = 0; i < (n + 2) * (n + 2); i++)
    data[i] = temp[i] = 0;
  int n0 = 1 + n / 2;
  int m0 = 1 + n / 2;
  data[(n0 - 1) * (n + 2) + m0] = 1;
  data[n0 * (n + 2) + m0 + 1] = 1;
  for (int i = 0; i < 3; i++)
    data[(n0 + 1) * (n + 2) + m0 + i - 1] = 1;
}

void setup_boundaries(int n, int *data) {
  for (int i = 0; i < n + 2; i++) {
    data[i * (n + 2) + 0] = data[i * (n + 2) + n];
    data[i * (n + 2) + n + 1] = data[i * (n + 2) + 1];
  }
  for (int j = 0; j < n + 2; j++) {
    data[0 * (n + 2) + j] = data[n * (n + 2) + j];
    data[(n + 1) * (n + 2) + j] = data[1 * (n + 2) + j];
  }
}

void run_life(int n, int T) {
  auto start = chrono::high_resolution_clock::now();
  int *data = new int[(n + 2) * (n + 2)];
  int *temp = new int[(n + 2) * (n + 2)];

  init(n, data, temp);

  for (int t = 0; t < T; t++) {
    setup_boundaries(n, data);
    update_data(n, data, temp);
    swap(data, temp);
  }

  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
  cout << "Time taken: " << duration.count() << " milliseconds" << endl;

  ofstream f("sequential_output.dat");
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++)
      f << data[i * (n + 2) + j];
    f << endl;
  }
  f.close();

  delete[] data;
  delete[] temp;
}

int main(int argc, char **argv) {
  int n = atoi(argv[1]);
  int T = atoi(argv[2]);

  run_life(n, T);

  return 0;
}