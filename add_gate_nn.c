#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define eps 1e-3
#define learning_rate 1e-1
#define number_input sizeof(and_input)/sizeof(and_input[0])
#define MAX_ITER 500000

float and_input[][3] =
{
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 1},
};

float sigmoid(float x)
{
  return (1.0 / (1.0 + expf(-x)));
}

float random_number(void)
{
  return (float)rand() / ((float)RAND_MAX);
}

float cost(float w1, float w2, float bias)
{
  float result = 0.0f;
  for (size_t i = 0; i < number_input; ++i)
  {
    float x1 = and_input[i][0];
    float x2 = and_input[i][1];
    float expected_y = and_input[i][2];
    float y = sigmoid(w1 * x1 + w2 * x2 + bias);
    float d = expected_y - y;
    result += d * d;
  }
  result /= number_input;
  return result;
}

void predict(float (*cost)(float, float, float), float *w1, float *w2, float* bias)
{
  for (size_t i = 0; i < MAX_ITER; ++i)
  {
    float cost_val = cost(*w1, *w2, *bias);
    float dw1 = (cost(*w1 + eps, *w2, *bias) - cost_val) / eps;
    float dw2 = (cost(*w1, *w2 + eps, *bias) - cost_val) / eps;
    float dBias = (cost(*w1, *w2, *bias + eps) - cost_val) /eps;
    *w1 -= learning_rate * dw1;
    *w2 -= learning_rate * dw2;
    *bias -= learning_rate * dBias;
  }
}

int main(void)
{
  srand((unsigned)time(0));
  float w1 = random_number() * 10.0f - 5.0f;
  float w2 = random_number() * 10.0f - 5.0f;
  float bias = random_number();
  printf("The cost value is: %f\n", cost(w1, w2, bias));
  printf("Let's train the model...\n");
  printf("---------------------\n");
  predict(&cost, &w1, &w2, &bias);
  printf("Cost function after training: %f\n", cost(w1, w2, bias));
  printf("The final value of w1 is: %f\n", w1);
  printf("The final value of w2 is: %f\n", w2);
  printf("The final value of the bias is: %f\n", bias);
  printf("AND gate predictions:\n");
  for (size_t i = 0; i < 2; i++)
  {
    for (size_t j = 0; j < 2; j++)
    {
      printf("%d OR %d => %f \n", i, j,sigmoid(i * w1 + j * w2 + bias));
    }
  }

  return 0;
}
