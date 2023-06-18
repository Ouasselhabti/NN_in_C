#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define eps 1e-3
#define learning_rate 1e-1
#define MAX_ITER 500000

float and_input[][3] = {
    {1, 1, 1},
    {1, 0, 0},
    {0, 1, 0},
    {0,0,0},
};

typedef struct
{
    float stored_value;
} Neuron;

typedef struct
{
    size_t ni;
    size_t ni1;
    float** weights;
    float* biases;
} Params;

typedef struct
{
    size_t size;
    Neuron* neurons;
    Params* params;
    float (*activation)(float param);
} Layer;

float sigmoid(float x)
{
    return (1.0 / (1.0 + expf(-x)));
}

float random_number(void)
{
    return (float)rand() / ((float)RAND_MAX);
}

void init_params(Params** params, size_t ni, size_t ni1)
{
    *params = (Params*)malloc(sizeof(Params));
    (*params)->ni = ni;
    (*params)->ni1 = ni1;
    (*params)->weights = (float**)malloc(ni1 * sizeof(float*));
    (*params)->biases = (float*)malloc(ni1 * sizeof(float));
    for (size_t i = 0; i < ni1; ++i)
    {
        (*params)->weights[i] = (float*)malloc(ni * sizeof(float));
        for (size_t j = 0; j < ni; ++j)
        {
            (*params)->weights[i][j] = random_number();
        }
        (*params)->biases[i] = random_number();
    }
}

void init_neuron(Neuron* neuron)
{
    neuron->stored_value = 0.0f;
}

void init_layer(Layer** layer, Neuron* neurons, size_t n_neurons, Params* params, float (*activation_fct)(float param))
{
    *layer = (Layer*)malloc(sizeof(Layer));
    (*layer)->size = n_neurons;
    (*layer)->neurons = neurons;
    (*layer)->params = params;
    (*layer)->activation = activation_fct;
}

void forward(Layer* layer1, Layer* layer2)
{
    size_t n1 = layer1->size;
    size_t n2 = layer2->size;
    float (*activation_ptr)(float param) = layer2->activation;
    Params* params = layer2->params;
    Neuron* neurons = layer2->neurons;
    for (size_t i = 0; i < n2; ++i)
    {
        for (size_t j = 0; j < n1; ++j)
        {
            Neuron neuronj = (layer1->neurons)[j];
            neurons[i].stored_value += (params->weights)[i][j] * neuronj.stored_value;
        }
        neurons[i].stored_value = activation_ptr(neurons[i].stored_value + (params->biases)[i]);
    }
}

int main(void)
{
    srand((unsigned)time(0));

    // Create the input layer
    size_t input_size = 2;
    Neuron* input_neurons = (Neuron*)malloc(input_size * sizeof(Neuron));
    for (size_t i = 0; i < input_size; ++i)
    {
        init_neuron(&input_neurons[i]);
    }
    Layer* input_layer;
    init_layer(&input_layer, input_neurons, input_size, NULL, NULL);

    // Create the hidden layer
    size_t hidden_size = 2;
    Params* hidden_params;
    init_params(&hidden_params, input_size, hidden_size);
    Neuron* hidden_neurons = (Neuron*)malloc(hidden_size * sizeof(Neuron));
    for (size_t i = 0; i < hidden_size; ++i)
    {
        init_neuron(&hidden_neurons[i]);
    }
    Layer* hidden_layer;
    init_layer(&hidden_layer, hidden_neurons, hidden_size, hidden_params, sigmoid);

    // Create the output layer
    size_t output_size = 1;
    Params* output_params;
    init_params(&output_params, hidden_size, output_size);
    Neuron* output_neurons = (Neuron*)malloc(output_size * sizeof(Neuron));
    for (size_t i = 0; i < output_size; ++i)
    {
        init_neuron(&output_neurons[i]);
    }
    Layer* output_layer;
    init_layer(&output_layer, output_neurons, output_size, output_params, sigmoid);

    // Train the network
    for (size_t iter = 0; iter < MAX_ITER; ++iter)
    {
        for (size_t sample_index = 0; sample_index < sizeof(and_input) / sizeof(and_input[0]); ++sample_index)
        {
            // Set the input layer values
            input_layer->neurons[0].stored_value = and_input[sample_index][0];
            input_layer->neurons[1].stored_value = and_input[sample_index][1];

            // Forward pass through the network
            forward(input_layer, hidden_layer);
            forward(hidden_layer, output_layer);

            // Compute the error and update the weights and biases
            float error = and_input[sample_index][2] - output_layer->neurons[0].stored_value;

            // Backpropagation for the output layer
            for (size_t i = 0; i < output_size; ++i)
            {
                float output = output_layer->neurons[i].stored_value;
                float delta = error * output * (1 - output);

                // Update the biases
                output_layer->params->biases[i] += learning_rate * delta;

                // Update the weights
                for (size_t j = 0; j < hidden_size; ++j)
                {
                    float hidden_output = hidden_layer->neurons[j].stored_value;
                    output_layer->params->weights[i][j] += learning_rate * delta * hidden_output;
                }
            }

            // Backpropagation for the hidden layer
            for (size_t i = 0; i < hidden_size; ++i)
            {
                float hidden_output = hidden_layer->neurons[i].stored_value;
                float sum = 0.0f;

                // Compute the sum of the weighted deltas from the output layer
                for (size_t j = 0; j < output_size; ++j)
                {
                    float delta = error * output_layer->params->weights[j][i];
                    sum += delta;
                }

                // Compute the delta for the hidden layer
                float delta = sum * hidden_output * (1 - hidden_output);

                // Update the biases
                hidden_layer->params->biases[i] += learning_rate * delta;

                // Update the weights
                for (size_t j = 0; j < input_size; ++j)
                {
                    float input = input_layer->neurons[j].stored_value;
                    hidden_layer->params->weights[i][j] += learning_rate * delta * input;
                }
            }
        }
    }

    // Test the network
    printf("Input\t\tOutput\n");
    for (size_t sample_index = 0; sample_index < sizeof(and_input) / sizeof(and_input[0]); ++sample_index)
    {
        // Set the input layer values
        input_layer->neurons[0].stored_value = and_input[sample_index][0];
        input_layer->neurons[1].stored_value = and_input[sample_index][1];

        // Forward pass through the network
        forward(input_layer, hidden_layer);
        forward(hidden_layer, output_layer);

        // Print the input and output values
        printf("%d AND %d\t->\t%.2f\n", (int)and_input[sample_index][0], (int)and_input[sample_index][1], output_layer->neurons[0].stored_value);
    }

    // Clean up
    free(input_neurons);
    free(hidden_neurons);
    free(output_neurons);
    free(input_layer);
    free(hidden_layer);
    free(output_layer);
    free(hidden_params->weights);
    free(hidden_params->biases);
    free(hidden_params);
    free(output_params->weights);
    free(output_params->biases);
    free(output_params);

    return 0;
}
