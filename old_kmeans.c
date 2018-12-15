#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>

float *loadData(char *fileName, unsigned nbVec, unsigned dim) {
	FILE *fp = fopen(fileName, "r");
	if (!fp) {
	       printf("File not Found: %s\n", fileName);
       		exit(1);
	}

	float *tab = malloc(sizeof(float) * nbVec * dim);
	fread(tab, sizeof(float), nbVec * dim, fp);
	fclose(fp);
	
	return tab;
}

void writeClassinFloatFormat(unsigned char *data, unsigned nbelt, char *fileName) {
    FILE *fp = fopen(fileName, "w");
    if (!fp) {
        printf("Cannot create File: %s\n", fileName);
        exit(1);
    }

    for(unsigned i = 0; i < nbelt; ++i) {
        float f = data[i];
        fwrite(&f, sizeof(float), 1, fp);
    }
    fclose(fp);
}

double distance(float *vec1, float *vec2, unsigned dim) {
	double dist = 0;
	for(unsigned i = 0; i < dim; ++i, ++vec1, ++vec2) {
		double d = *vec1 - *vec2;
		dist += d * d;
	}

	return (sqrt(dist));
}

unsigned char classify(float *vec, float *means, unsigned dim, unsigned char K, double *e) {
	unsigned char min = 0;
	float dist, distMin = FLT_MAX;

	for(unsigned i = 0; i < K; ++i) {
		dist = distance(vec, means + i * dim, dim);
		if(dist < distMin) {
			distMin = dist;
			min = i;
		}
	}
	
	*e = distMin;
	return min;
}



unsigned char *Kmeans(float *data, unsigned nbVec, unsigned dim, unsigned char K, double minErr, unsigned maxIter)
{
	unsigned iter = 0;
	double e, diffErr = DBL_MAX, Err = DBL_MAX;

	float *means = malloc(sizeof(float) * dim * K);
	unsigned *card = malloc(sizeof(unsigned) * K);
	unsigned char* c = malloc(sizeof(unsigned char) * nbVec); 

	// Random init of c
	for(unsigned i = 0; i < nbVec; ++i)
		c[i] = rand() / (RAND_MAX + 1.) * K;

	for(unsigned i = 0; i < dim * K; ++i)
		means[i] = 0.;
    	for(unsigned i = 0; i < K; ++i)
        	card[i] = 0.;
    
	for(unsigned i = 0; i < nbVec; ++i) {
		for(unsigned j = 0; j < dim; ++j)
			means[c[i] * dim + j] += data[i * dim  + j];
		++card[c[i]];
	}
	for(unsigned i = 0; i < K; ++i)
		for(unsigned j = 0; j < dim; ++j)
			means[i * dim + j] /= card[i];

	while ((iter < maxIter) && (diffErr > minErr)) {
		diffErr = Err;
		// Classify data
		Err = 0.;
		for(unsigned i = 0; i < nbVec; ++i) {
			c[i] = classify(data + i * dim, means, dim, K, &e);
			Err += e;
		}

		// update Mean
		for(unsigned i = 0; i < dim * K; ++i)
			means[i] = 0.;
        	for(unsigned i = 0; i < K; ++i)
            		card[i] = 0.;
		for(unsigned i = 0; i < nbVec; ++i) {
			for(unsigned j = 0; j < dim; ++j)
				means[c[i] * dim + j] += data[i * dim  + j];
			++card[c[i]];
		}
		for(unsigned i = 0; i < K; ++i)	
			for(unsigned j = 0; j < dim; ++j)
				means[i * dim + j] /= card[i];
		++iter;
		Err /= nbVec;
		diffErr = fabs(diffErr - Err);
        	printf("Iteration: %d, Error: %f\n", iter, Err);
	}

	free(means);
	free(card);

	return c;
}

int main(int ac, char *av[])
{
	if (ac != 8) {
		printf("Usage :\n\t%s <K: int> <maxIter: int> <minErr: float> <dim: int> <nbvec:int> <datafile> <outputClassFile>\n", av[0]);
		exit(1);
	}
	unsigned maxIter = atoi(av[2]);
	double minErr = atof(av[3]);
	unsigned K = atoi(av[1]);
	unsigned dim = atoi(av[4]);
	unsigned nbVec = atoi(av[5]);

	printf("Start Kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n", av[6], K, dim, nbVec);

	float *tab = loadData(av[6], nbVec, dim);
	unsigned char * classif = Kmeans(tab, nbVec, dim, K, minErr, maxIter);
    	writeClassinFloatFormat(classif, nbVec, av[7]);
	free(tab);
	free(classif);

	return 0;
}
