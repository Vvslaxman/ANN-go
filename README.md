# Neural Network Training with Go

This project implements a simple neural network using the Go programming language. The network is trained using backpropagation with L2 regularization and can be used for classification tasks. The training and testing data are read from CSV files, and the performance metrics are printed and plotted.

## Features
- Neural network with one hidden layer
- Training using backpropagation with L2 regularization
- Accuracy, precision, recall, and F1-score calculations
- Plots for training accuracy and loss

## Requirements
- Go 1.14+
- `gonum` package
- `gonum/plot` package

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/neural-network-go.git
   cd neural-network-go
   ```

2. **Install dependencies:**

   ```sh
   go get -u gonum.org/v1/gonum/...
   go get -u gonum.org/v1/plot/...
   ```

## Usage

1. **Prepare the data:**
   - Place your training data in `data/train.csv`
   - Place your testing data in `data/test.csv`

   The CSV files should have the following format:

   ```
   feature1,feature2,feature3,feature4,label1,label2,label3
   ```

2. **Run the program:**

   ```sh
   go run main.go
   ```

3. **Check the output:**
   - The program will print the accuracy, precision, recall, and F1-score of the trained network.
   - Plots for training accuracy and loss will be saved as `accuracy_plot.png` and `loss_plot.png`.

## Code Structure

- `main.go`: The main file containing the neural network implementation, training, and evaluation code.
- `data/`: Directory to store training and testing CSV files.

## Functions

- **main()**: The main function that sets up the network configuration, trains the network, evaluates the performance, and plots the metrics.
- **newNetwork(config neuralNetConfig) *neuralNet**: Initializes a new neural network.
- **train(x, y *mat.Dense) ([]float64, []float64, error)**: Trains the neural network using backpropagation with L2 regularization.
- **calculateLoss(predictions, labels *mat.Dense) float64**: Calculates the mean squared error loss.
- **predict(x *mat.Dense) (*mat.Dense, error)**: Makes predictions using the trained network.
- **calculateAccuracy(predictions, labels *mat.Dense) float64**: Calculates the accuracy of the predictions.
- **calculatePrecision(predictions, labels *mat.Dense) float64**: Calculates the precision of the predictions.
- **calculateRecall(predictions, labels *mat.Dense) float64**: Calculates the recall of the predictions.
- **calculateF1Score(precision, recall float64) float64**: Calculates the F1-score.
- **sigmoid(x float64) float64**: Sigmoid activation function.
- **sigmoidPrime(x float64) float64**: Derivative of the sigmoid function.
- **sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error)**: Sums elements along a specified axis of a matrix.
- **makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense)**: Reads the CSV file and returns the inputs and labels matrices.
- **plotMetrics(accuracy, loss []float64)**: Plots the training accuracy and loss.

## License
This project is licensed under the MIT License.

## Contributing
Feel free to submit issues, fork the repository and send pull requests. Contributions are welcome!

## Acknowledgements
- The `gonum` and `gonum/plot` packages for numerical computations and plotting in Go.

## Contact
For any questions or suggestions, please open an issue or contact the repository owner.

---

This README provides an overview of the project, how to set it up, and the functionalities included in the code. Adjust the file paths and URLs as necessary to match your specific project setup.
