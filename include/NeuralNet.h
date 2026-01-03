#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

void print_hello();

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NPMatrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;


// neural network implementation class!
class NeuralNetwork {
public:
  // constructor
  NeuralNetwork(std::vector<uint> topology,
                Scalar learningRate = Scalar(0.005));

  // function for forward propagation of data
  void propagateForward(Eigen::Ref<RowVector> input);

  // function for backward propagation of errors made by neurons
  void propagateBackward(Eigen::Ref<RowVector> output);

  // function to calculate errors made by neurons in each layer
  void calcErrors(Eigen::Ref<RowVector> output);

  // function to update the weights of connections
  void updateWeights();

  // function to train the neural network give an array of data points
  // void train(std::vector<RowVector*> data);
  Scalar train(std::vector<RowVector *> input_data,
               std::vector<RowVector *> output_data);
  // better training function
  Scalar trainMat(Eigen::Ref<const NPMatrix> input_data, Eigen::Ref<const NPMatrix> output_data);
  // storage objects for working of neural network
  /*
        use pointers when using std::vector<Class> as std::vector<Class>
     calls destructor of Class as soon as it is pushed back! when we use
     pointers it can't do that, besides it also makes our neural network
     class less heavy!! It would be nice if you can use smart pointers
     instead of usual ones like this
      */
  std::vector<uint> topology; // stores the topology of the network
  std::vector<RowVector *>
      neuronLayers; // stores the different layers of out network
  std::vector<RowVector *> cacheLayers; // stores the unactivated (activation fn
                                        // not yet applied) values of layers
  std::vector<RowVector *>
      deltas; // stores the error contribution of each neurons
  std::vector<Matrix *> weights; // the connection weights itself
  Scalar learningRate;
  // debug info
};
