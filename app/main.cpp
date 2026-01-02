#include "Csv.h"
#include "NeuralNet.h"

static const int MAX_ITERATIONS = 1000;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: app process" << std::endl;
    return 1;
  }
  if (argv[1] == std::string("gen")) {
    genData(argv[2]);
    return 0;
  }
  if (argv[1] == std::string("train")) {
    std::vector<RowVector *> input_data, output_data;
    ReadCSV(std::string(argv[2]) + "-in", input_data);
    ReadCSV(std::string(argv[2]) + "-out", output_data);
    NeuralNetwork nn({2, 3, 1});
    if (argc == 4) {
      for (uint i = 0; i < std::stoi(argv[3]); i++) {
        Scalar mmse = nn.train(input_data, output_data);
        std::cout << "Iteration: " << i << "Total MSE: " << mmse << std::endl;
      }
    } else {
      for (uint i = 0; i < MAX_ITERATIONS; i++) {
        Scalar mmse = nn.train(input_data, output_data);
        std::cout << "Iteration: " << i << "Total MSE: " << mmse << std::endl;
      }
    }
    return 0;
  }
  return 0;
}
