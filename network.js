class NeuralNetwork {
  // Create level for each neuron count provided
  constructor(neuronCounts) {
    this.levels = [];
    for (let i = 0; i < neuronCounts.length - 1; i++) {
      this.levels.push(new Level(
        neuronCounts[i], neuronCounts[i+1]
      ));
    }
  }

  // Using outputs of previous levels as inputs for the next
  static feedForward(givenInputs, network) {
    let outputs = Level.feedForward(givenInputs, network.levels[0]);
    for (let i = 1; i < network.levels.length; i++) {
      outputs = Level.feedForward(outputs, network.levels[i]);
    }
    // Returns final car movement result
    return outputs;
  }

  // Randomly mutate weights / biases by an amount (SIMPLE GENETIC ALGORITHM)
  static mutate(network, amount = 1) {
    network.levels.forEach(level => {
      for (let i = 0; i < level.biases.length; i++) {
        level.biases[i] = lerp(
          level.biases[i],
          Math.random() * 2 - 1,
          amount
        );
      }
      for (let i = 0; i < level.weights.length; i++) {
        for (let j = 0; j < level.weights[i].length; j++) {
          level.weights[i][j] = lerp(
            level.weights[i][j],
            Math.random() * 2 - 1,
            amount
          );
        }
      }
    })
  }
}

class Level {
  constructor(inputCount, outputCount) {
    this.inputs = new Array(inputCount);
    this.outputs = new Array(outputCount);
    this.biases = new Array(outputCount);

    this.weights = [];
    for (let i = 0; i < inputCount; i++) {
      this.weights[i] = new Array(outputCount);
    }

    Level.#randomize(this);
  }

  // Set each weights/biases randomly between -1 and 1
  // Static since we want to serialize
  static #randomize(level) {
    for (let i = 0; i < level.inputs.length; i++) {
      for (let j = 0; j < level.outputs.length; j++) {
        level.weights[i][j] = Math.random() * 2 - 1;
      } 
    }

    for (let i = 0; i < level.biases.length; i++) {
      level.biases[i] = Math.random() * 2 - 1;
    }
  }

  // NOTE: our feedforward algorithm only computes binary values for the hidden layer.
  // Better models will have all neurons firing at different levels, with only the output being binary.
  // They will also apply a function at the output of each layer (level.outputs[i])
  // such as sigmoid, ReLU, ... to control values. (PyTorch / TensorFlow stuff)
  static feedForward(givenInputs, level) {
    // Set inputs
    // (initially based on active car sensors, then on previous layer)
    for (let i = 0; i < level.inputs.length; i++) {
      level.inputs[i] = givenInputs[i];
    }
  
    // For each output, calculate a sum of (input value * input/output weight)
    for (let i = 0; i < level.outputs.length; i++) {
      let sum = 0;
      for (let j = 0; j < level.inputs.length; j++) {
        sum += level.inputs[j] * level.weights[j][i];
      }
      // If larger than the output neuron's bias, activate
      if (sum > level.biases[i]) {
        level.outputs[i] = 1;
      }
      // Otherwise, no activation
      else {
        level.outputs[i] = 0;
      }
    }

    return level.outputs;
  }
}