#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <Eigen/Core>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include "SpectralClustering.h"


#define MAX_BUFF_SIZE  1000 


double eculidean_distance(std::vector<double> item1, std::vector<double> item2) {
	long double e_d = 0.0;
	if (item1.size() != item2.size()) {
		
		std::cerr << "ERROR ..." << std::endl;
		exit(0);
	}
	for (int i = 0; i < item1.size(); i++) {
		e_d += pow(item1[i] - item2[i], 2);
	}

	return sqrt(e_d);
}

std::vector< std::vector<double> > load_data(const std::string filename) {

	std::vector< std::vector<double> > data;

	std::ifstream in(filename.c_str(), std::ios::in);
	if(!in) {
		std::cerr << "Can't open file" << filename << std::endl;
	}

	char buff[MAX_BUFF_SIZE];
	int numSamples = 200;

	int lineno = 0;
	while(!in.eof()) {
		if(lineno == numSamples) {
			break;
		}
		in.getline(buff, MAX_BUFF_SIZE);
		std::vector<double> record;
		int index = 0;
		char * token = NULL;
		for(token = strtok(buff, ","); token; token = strtok(NULL, ",")) {
			if (index == 0) {
				index++;
				continue;
			}
			double value = atof(token);				
			//std :: cout << value << ", ";
			//std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(1) << value << ",";
			record.push_back(value);
			index++;
		}
		//std::cout << std::endl;
		data.push_back(record);
		lineno++;
		
	}
	in.close();
	return data;


}

int main() {
    //std::vector<int> items = {1,2,3,4,5,6,7,8,9,10};

    std::string filename("sampleEmbedding_200.txt");
    //std::string filename("iris.data");
    std::vector< std::vector<double> > items = load_data(filename);

    // generate similarity matrix
    unsigned int size = items.size();
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(size,size);
    std::cout << "------------" << std::endl;

    for (unsigned int i=0; i < size; i++) {
        for (unsigned int j=0; j < size; j++) {
            // generate similarity
            //int d = items[i] - items[j];
	    double d = eculidean_distance(items[i], items[j]);
            int similarity = exp(-d*d / 1);
            m(i,j) = similarity;
            m(j,i) = similarity;
        }
    }

    // the number of eigenvectors to consider. This should be near (but greater) than the number of clusters you expect. Fewer dimensions will speed up the clustering
    int numDims = size;

    // do eigenvalue decomposition
    SpectralClustering* c = new SpectralClustering(m, numDims);

    // whether to use auto-tuning spectral clustering or kmeans spectral clustering
    bool autotune = false;

    std::vector<std::vector<int> > clusters;
    if (autotune) {
        // auto-tuning clustering
        clusters = c->clusterRotate();
    } else {
        // how many clusters you want
        int numClusters = 10;
        clusters = c->clusterKmeans(numClusters);
    }

    // output clustered items
    // items are ordered according to distance from cluster centre
    for (unsigned int i=0; i < clusters.size(); i++) {
        std::cout << "Cluster " << i << ": " << "Item ";
        std::copy(clusters[i].begin(), clusters[i].end(), std::ostream_iterator<int>(std::cout, ", "));
        std::cout << std::endl;
    }
	std::map<int, int> sampleid_to_clusterid;
    for (unsigned int i = 0; i < clusters.size(); i++) {
    	for (unsigned int j = 0; j < clusters[i].size(); j++){
		sampleid_to_clusterid.insert(std::make_pair(clusters[i][j], i));
	}
    }
    
    std::string output_filename("sample_assignment.txt");

    std::ofstream out(output_filename.c_str(), std::ios::out);
    if(!out) {
    	std::cerr << "Can't open output file " << output_filename << std::endl;
    	return 0;
    }
    std::map<int, int>::iterator it;
    for(it = sampleid_to_clusterid.begin(); it != sampleid_to_clusterid.end(); it++) {
    	out << it->second << std::endl;
    }
    out.close();
   
}
