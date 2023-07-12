#ifndef NN_H
#define NN_H
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#ifndef MAX_ITER
#define MAX_ITER 100
#endif // MAX_ITER
#define M(m,i,j) (m.mat[(i)*(m.cols)+(j)])
#define PRINT_MAT(m) (print(m,#m))
#ifndef ASSERT
#include <assert.h>
#define ASSERT assert
#endif // ASSERT

typedef struct {
  size_t rows;
  size_t cols;
  float* mat;
} Mat;

typedef struct {
  // the input layer and output layer are part of layers (so weights[0] is in fact the input matrix)
  // the number of biases will be layers_number - 1 seen that the 0th layer (input) do not contain biases but just input (in weihgts0)
  size_t layers_number;
  size_t* layers_size;
  Mat* weights;
  Mat* biases;
  /* float (*fptr[])(float); // an array of activation functions */
} NN;

Mat alloc(size_t rows, size_t cols);
void add(Mat dest, Mat a, Mat b);
void add(Mat dest, Mat a, float b);
void mult(Mat dest, Mat a, Mat b);
void mult(Mat dest, Mat a, float b);
void mat_rand(Mat m, float a, float b);
void fill(Mat m, float x);
void print(Mat m, const char* name);
void fmap(Mat dest, Mat m, float(*fptr)(float));
void duplicate_row(Mat dest, Mat row);
void duplicate_col(Mat dest, Mat col);
void transpose(Mat dest, Mat m);
void mat_copy(Mat dest, Mat src);
float costf(Mat y, Mat predicted);
Mat elementary(size_t rows,size_t cols, size_t i, size_t j, float x);
void free_mat(Mat m);
NN NN_init(size_t layers_number, size_t* layers_size, Mat* input /* float (*fptr[])(float) */);
NN NN_init0(size_t layers_number, size_t* layers_size /* float (*fptr[])(float) */);
void NN_print(NN nn);
void NN_free(NN nn);
void NN_clear(NN nn);
Mat NN_forward(NN nn);
void NN_copy(NN nn, NN dest);
void NN_grad(NN nn, NN grad_holder, float epsilon);
void NN_backpropagation(NN nn, Mat expected, float epsilon);

#endif // NN_H

#ifdef NN_IMPLEM

Mat alloc(size_t rows, size_t cols)
{
  Mat mat;
  mat.rows = rows;
  mat.cols = cols;
  mat.mat = (float*)calloc(rows*cols,sizeof(*mat.mat));
  return mat;
}

void add(Mat dest, Mat a, Mat b)
{
  size_t rows = dest.rows;
  size_t cols = dest.cols;
  ASSERT(a.cols == cols && a.rows == rows);
  ASSERT(b.cols == a.cols && b.rows == a.rows);
  for (size_t i = 0; i < rows; i++)
  {
    for (size_t j = 0; j < cols; j++)
    {
      M(dest,i,j) = M(a,i,j) + M(b,i,j);
    }
  }
}

void add(Mat dest, Mat a, float b)
{
  for (size_t i = 0; i < dest.rows; i++)
  {
    for (size_t j = 0; j < dest.cols; j++)
    {
      M(dest,i,j)  = b + M(a,i,j);
    }
  }
}

void mult(Mat dest, Mat a, Mat b)
{
  size_t rows = dest.rows;
  size_t cols = dest.cols;
  size_t inner = a.cols;
  ASSERT(a.cols == b.rows);
  ASSERT(rows == a.rows && cols == b.cols);
  for (size_t i = 0; i < rows; i++)
  {
    for (size_t j = 0; j < cols; j++)
    {
      M(dest,i,j) = 0.0f;
      for (size_t k = 0; k < inner; k++)
      {
        M(dest,i,j) += M(a,i,k)*M(b,k,j);
      }
    }
  }
}

void mult(Mat dest, Mat a, float b)
{
  for (size_t i = 0; i < dest.rows; i++)
  {
    for (size_t j = 0; j < dest.cols; j++)
    {
      M(dest,i,j) = b*M(a,i,j);
    }
  }
}

void mat_rand(Mat m, float a, float b)
{
  size_t rows = m.rows;
  size_t cols = m.cols;
  for (size_t i = 0; i < rows; i++)
  {
    for (size_t j = 0; j < cols; j++)
    {
      float o = (float)rand() / (float)RAND_MAX;
      M(m,i,j) = a + (b-a)*o;
    }
  }
}

void fill(Mat m, float x)
{
  size_t rows = m.rows;
  size_t cols = m.cols;
  for (size_t i = 0; i < rows; i++)
  {
    for (size_t j = 0; j < cols; j++)
    {
      M(m,i,j) = x;
    }
  }
}

void print(Mat m, const char* name)
{
  size_t rows = m.rows;
  size_t cols = m.cols;
  printf("_____________________________\n \n");
  printf("%s = ", name);
  for (size_t i = 0; i < rows; i++)
  {
    printf("[");
    for (size_t j = 0; j < cols; j++)
    {
      printf("%f", M(m,i,j));
      if (j < cols-1)
      {
        printf(", ");
      }
    }
    printf("]\n");
  }
  printf("_____________________________\n");
}

void fmap(Mat dest, Mat m, float(*fptr)(float))
{
  ASSERT(dest.cols == m.cols && dest.rows == m.rows);
  size_t cols = dest.cols;
  size_t rows = dest.rows;
  for (size_t i = 0; i < rows; ++i)
  {
    for (size_t j = 0; j < cols; ++j)
    {
      M(dest,i,j) = (*fptr)(M(m,i,j));
    }
  }
}

void duplicate_row(Mat dest, Mat row)
{
  ASSERT(dest.cols == row.cols);
  for (size_t i = 0; i < dest.rows; ++i)
  {
    for (size_t j = 0; j < dest.cols; ++j)
    {
      M(dest,i,j) = M(row,0,j);
    }
  }
}

void duplicate_col(Mat dest, Mat col)
{
  ASSERT(dest.rows == col.rows);
  for (size_t i = 0; i < dest.rows; ++i)
  {
    for (size_t j = 0; j < dest.cols; ++j)
    {
      M(dest,i,j) = M(col,i,0);
    }
  }
}

void transpose(Mat dest, Mat m)
{
  ASSERT(dest.rows == m.cols && dest.cols == m.rows);
  for (size_t i = 0; i < dest.rows; ++i)
  {
    for (size_t j = 0; j < dest.cols; ++j)
    {
      M(dest,i,j) = M(m,j,i);
    }
  }
}

void mat_copy(Mat dest, Mat src)
{
  ASSERT(dest.rows == src.rows && dest.cols == src.cols);
  for (size_t i = 0; i < dest.rows; ++i)
  {
    for (size_t j = 0; j < dest.cols; ++j)
    {
      M(dest,i,j) = M(src,i,j);
    }
  }
}

float costf(Mat y, Mat expected)
{
  ASSERT(y.cols == expected.cols && y.rows == expected.rows);
  float result = 0.0f;
  for (size_t i = 0; i < y.rows; ++i)
  {
    for (size_t j = 0; j < y.cols; ++j) 
    {
      float diff = abs(M(y,i,j)-M(expected,i,j));
      result += diff*diff;
    }
  }
  return result/(y.cols*y.rows);
}

Mat elementary(size_t rows,size_t cols, size_t i, size_t j, float x)
{
  ASSERT(i < rows && j < cols);
  Mat m = alloc(rows,cols);
  M(m,i,j) = x;
  return m;
}

void free_mat(Mat m)
{
  m.cols = 0;
  m.rows = 0;
  free(m.mat);
}

NN NN_init(size_t layers_number, size_t* layers_size, Mat* input/* , float (*fptr[])(float) */)
{
  NN nn;
  nn.layers_number = layers_number;
  nn.layers_size = layers_size;
  nn.weights = (Mat*)malloc(layers_number*sizeof(Mat));
  nn.biases = (Mat*)malloc((layers_number-1)*sizeof(Mat));
  nn.weights[0] = alloc(1,layers_size[0]);
  if (input != NULL) {
    mat_copy(nn.weights[0],*input);
    ASSERT(input->rows == 1 && input->cols == layers_size[0]);
  } else {
    for (size_t i = 1; i < layers_number; ++i) {
      nn.weights[i] = alloc(layers_size[i-1],layers_size[i]);
      nn.biases[i-1] = alloc(1,layers_size[i]);
      mat_rand(nn.weights[i],0,1);
      mat_rand(nn.biases[i-1],0,1);
      /* nn.fptr[i] = fptr[i]; */
    }
  }
  return nn;
}

NN NN_init0(size_t layers_number, size_t* layers_size /* float (*fptr[])(float) */)
{
  NN nn;
  nn.layers_number = layers_number;
  nn.layers_size = layers_size;
  nn.weights = (Mat*)malloc(layers_number*sizeof(Mat));
  nn.biases = (Mat*)malloc((layers_number-1)*sizeof(Mat));
  for (size_t i = 1; i < layers_number; ++i) {
    nn.weights[i] = alloc(layers_size[i-1],layers_size[i]);
    nn.biases[i-1] = alloc(1,layers_size[i]);
    /* nn.fptr[i] = fptr[i]; */
  }
  nn.weights[0] = alloc(1,layers_size[0]);
  return nn;
}

void NN_print(NN nn)
{
  Mat* weights = nn.weights;
  Mat* biases = nn.biases;
  printf("--------- Input layer --------\n");
  PRINT_MAT(weights[0]);
  for (size_t i = 1; i < nn.layers_number; ++i) {
    (i < nn.layers_number-1)? printf("--------- Layer %lu---------\n",i) : printf("--------- Output layer ---------\n");
    printf("---WEIGHTS---\n");
    PRINT_MAT(weights[i]);
    printf("---BIASES---\n");
    PRINT_MAT(biases[i-1]);
  }
}

void NN_free(NN nn)
{
  for ( size_t i = 0; i < nn.layers_number; ++i){
    free_mat(nn.weights[i]);
    if (i < nn.layers_number - 1)
      free_mat(nn.biases[i]);
  }
  free_mat(nn.biases[nn.layers_number-2]);
  free(nn.weights);
  free(nn.biases);
}

Mat NN_forward(NN nn)
{
  size_t n = nn.layers_number;
  Mat input = alloc(nn.weights[0].rows,nn.weights[0].cols);
  mat_copy(input,nn.weights[0]);
  Mat output = alloc(1,1);
  for (size_t i = 1; i < n; ++i){
    free_mat(output);
    output = alloc(1,nn.layers_size[i]);
    mult(output,input,nn.weights[i]);
    add(output,output,nn.biases[i-1]);
    free_mat(input);
    input = alloc(output.rows,output.cols);
    mat_copy(input,output);
  }
  free_mat(output);
  return input;
}

void NN_copy(NN nn, NN* dest)
{
  size_t layers_number = nn.layers_number;
  dest->layers_number = layers_number;
  dest->layers_size = (size_t*)malloc(layers_number*sizeof(size_t));
  dest->weights = (Mat*)malloc((layers_number)*sizeof(Mat));
  dest->biases  = (Mat*)malloc((layers_number-1)*sizeof(Mat));
  memcpy(dest->layers_size, nn.layers_size, layers_number);
  for (size_t i = 0; i < layers_number; ++i) {
    (dest->weights)[i] = alloc( (nn.weights)[i].rows, (nn.weights)[i].cols );
    mat_copy((dest->weights)[i],(nn.weights)[i]);
    if (i>0) {
      (dest->biases)[i-1] = alloc(1, (nn.layers_size)[i]);
      mat_copy((dest->biases)[i-1],(nn.biases)[i-1]);
    }
  }
}

void NN_grad(NN nn, NN grad_holder,Mat expected,  float epsilon)
{
  ASSERT((nn.layers_size)[nn.layers_number - 1] == expected.cols);
  for (size_t i = 1; i < nn.layers_number; ++i) {
    //adding eps to ijk then calculating the cost then removing it , basicaly this is the idea
    for (size_t j = 0; j < (nn.weights)[i].rows; ++j){
      for (size_t k = 0; k < (nn.weights)[i].cols; ++k){
	Mat output1 = NN_forward(nn);
	float w1 = costf(output1, expected);
	
	M(nn.weights[i],j,k) += epsilon;
	
	//forward with the new nn
	Mat output2 = NN_forward(nn);
	//compute the cost
	float w2 = costf(output2, expected);
	float dw = w2-w1;
	M(grad_holder.weights[i],j,k) += dw/epsilon;

	//goes back to the initial NN
	M(nn.weights[i],j,k) -= epsilon;
	
	//Same thing for biases
	//free first output2
	free_mat(output2);
	M(nn.biases[i-1],j,k) += epsilon;
	output2 = NN_forward(nn);
	w2 = costf(output2, expected);
	dw = w2-w1;
	M(grad_holder.biases[i-1],j,k) += dw/epsilon;

	//Re goes back
	M(nn.biases[i-1],j,k) -= epsilon;
	M(nn.biases[i-1],j,k) += dw/epsilon;	
      }
    }
  }
}

void NN_clear(NN nn)
{
  for (size_t i = 1; i < nn.layers_number; ++i) {
    fill((nn.weights)[i],0.0f);
    fill((nn.biases)[i-1],0.0f);
  }
}

void NN_backpropagation(NN nn, Mat expected, float epsilon)
{
  NN holder = NN_init0(nn.layers_number, nn.layers_size);
  for (size_t i = 0; i < MAX_ITER; ++i) {
    NN_grad(nn,holder,expected, epsilon);
  }
}

#endif // NN_IMPLEM
