#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <kmcuda.h>

// ./example /path/to/data <number of clusters>
int main(int argc, const char **argv) {
  assert(argc == 3);
  // we open the binary file with the data
  // [samples_size][features_size][samples_size x features_size]
  FILE *fin = fopen(argv[1], "rb");
  assert(fin);
  uint32_t samples_size = 400000, features_size = 2048;
  //assert(fread(&samples_size, sizeof(samples_size), 1, fin) == 1);
  //assert(fread(&features_size, sizeof(features_size), 1, fin) == 1);
  uint64_t total_size = ((uint64_t)samples_size) * features_size;
  float *samples = malloc(total_size * sizeof(float));
  assert(samples);
  int i,j,flag;
  for (i=0; i<samples_size; i++){
    for (j=0; j<features_size; j++)
      flag = fscanf(fin, "%f", &samples[i * features_size + j]);
    if(i%1000 == 0)    
      printf("%d", i);
  }
  for (i=0; i<10; i++){
    printf("\n%f\n", samples[i * features_size]);
  }
  //assert(fread(samples, sizeof(float), total_size, fin) == total_size);
  fclose(fin);
  printf("%d,%d\n", samples_size, features_size);//100, 9
  int clusters_size = atoi(argv[2]);
  // we will store cluster centers here

  float *centroids = malloc(clusters_size * features_size * sizeof(float));
  assert(centroids);
  // we will store assignments of every sample here
  uint32_t *assignments = malloc(((uint64_t)samples_size) * sizeof(uint32_t));
  assert(assignments);
  float average_distance;
  KMCUDAResult result = kmeans_cuda(
      kmcudaInitMethodPlusPlus, NULL,  // kmeans++ centroids initialization
      0.0001,                            // less than 1% of the samples are reassigned in the end
      0.1,                             // activate Yinyang refinement with 0.1 threshold
      kmcudaDistanceMetricL2,          // Euclidean distance
      samples_size, features_size, clusters_size,
      0xDEADBEEF,                      // random generator seed
      0,                               // use all available CUDA devices
      -1,                              // samples are supplied from host
      0,                               // not in float16x2 mode
      1,                               // moderate verbosity
      samples, centroids, assignments, &average_distance);
  free(samples);
  //free(centroids);
  //free(assignments);
  assert(result == kmcudaSuccess);
  printf("Average distance between a centroid and the corresponding "
         "cluster members: %f\n", average_distance);
  for(i=0; i<1000; i++){
    //for(j=0; j<features_size;j++){
    //  printf("%f ", centroids[i * features_size + j]);
    //}
    printf("%d %d\n", i, assignments[i]);
  }
  for(i=0; i<clusters_size; i++){
    printf("%d %f\n", i, centroids[i*features_size]);
  }
  free(centroids);
  free(assignments);
  return 0;
}
