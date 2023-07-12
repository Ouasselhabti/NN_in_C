#define NN_IMPLEM
#include "nn_c.h"
#include <time.h>



int main(void)
{
  srand((unsigned)time(0));
  Mat m = alloc(5,5);
  print(m);
  return 0;
}
