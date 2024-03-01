#include<iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include<algorithm>

using namespace std;

const string TRAIN_IMAGES_FILE = "../data/train-images-idx3-ubyte";
const string TRAIN_LABELS_FILE = "../data/train-labels-idx1-ubyte";
const string TEST_IMAGES_FILE = "../data/t10k-images-idx3-ubyte";
const string TEST_LABELS_FILE = "../data/t10k-labels-idx1-ubyte";


const int IMAGE_MAGIC_NUMBER = 2051;

const int LABEL_MAGIC_NUMBER = 2049;

const int NUM_TRAIN_IMAGES = 60000;
const int NUM_TEST_IMAGES = 10000;
const int IMAGE_ROWS = 28;
const int IMAGE_COLS = 28;
const int IMAGE_SIZE = IMAGE_ROWS * IMAGE_COLS;


double sigmoid(double x) {
    return (double)1.0 / ((double)1.0 + (double)exp(-x));
}

double sigmoid_derivative(double x) {
    return sigmoid(x)*((double)1-sigmoid(x));
}

// Function to read MNIST images from file with pixel values between 0 and 256
vector<vector<int> > readMNISTImages(const string& filename, int number_images) {
    vector<vector<int> > images;

    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        // Read magic number
        int magic_number = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number); // Swap bytes if necessary
        if (magic_number != IMAGE_MAGIC_NUMBER) {
            cerr << "Invalid magic number for images file" << endl;
            return images;
        }

        // Read number of images
        int num_images = 0;
        file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        num_images = __builtin_bswap32(num_images); // Swap bytes if necessary
        if (num_images != number_images) {
            cerr << "Invalid number of images" << endl;
            return images;
        }

        // Read image dimensions
        int num_rows = 0, num_cols = 0;
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
        num_rows = __builtin_bswap32(num_rows); // Swap bytes if necessary
        num_cols = __builtin_bswap32(num_cols); // Swap bytes if necessary
        if (num_rows != IMAGE_ROWS || num_cols != IMAGE_COLS) {
            cerr << "Invalid image dimensions" << endl;
            return images;
        }

        // Read images pixel by pixel
        printf("Num images %d\n", num_images);
        for (int i = 0; i < num_images; ++i) {
            vector<int> image(IMAGE_SIZE);
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                image[j] = static_cast<int>(pixel); // Convert unsigned char to int
            }
            images.push_back(image);
        }

        file.close();
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }

    return images;
}

// Function to read MNIST labels from file
vector<unsigned char> readMNISTLabels(const string& filename, int number_images) {
    vector<unsigned char> labels;

    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        // Read magic number
        int magic_number = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number); // Swap bytes if necessary
        if (magic_number != LABEL_MAGIC_NUMBER) {
            cerr << "Invalid magic number for labels file" << endl;
            return labels;
        }

        // Read number of labels
        int num_labels = 0;
        file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
        num_labels = __builtin_bswap32(num_labels); // Swap bytes if necessary
        if (num_labels != number_images) {
            cerr << "Invalid number of labels" << endl;
            return labels;
        }

        // Read labels
        unsigned char label = 0;
        for (int i = 0; i < num_labels; ++i) {
            file.read(reinterpret_cast<char*>(&label), sizeof(label));
            labels.push_back(label);
        }

        file.close();
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }

    return labels;
}

// Function to shuffle indices and select 'm' of them
void shuffleIndices(vector<int>& indices) {

    // Shuffle the indices
    mt19937 g(149);
    shuffle(indices.begin(), indices.end(), g);
}

int predict(vector<vector<double> > &image, vector<int>&sizes, vector<vector<vector<double> > > &weights, 
vector<vector<double> > &biases, int dense_layers, int nh, int ne, int nb, float alpha ){
    int L = sizes.size();

    // ACTIVATION LAYERS ----------------
    vector<vector<double> > activations;
    vector<vector<double> > zs;
    vector<double> activation;
    // set activation for first layer
    for(int i=0; i< IMAGE_ROWS; i++){
        for(int j=0; j<IMAGE_COLS; j++){
            // first Z layer
            activation.push_back(image[i][j]);
        }
    }
    activations.push_back(activation);
    activation.clear();

    // Forwards pass
    for(int layer = 0; layer < L-1; layer++){
        int forward_size = sizes[layer+1];
        int back_size = sizes[layer];
        vector<double> z(forward_size, 0.0);
        vector<double> a(forward_size, 0.0);
        for(int neuron = 0; neuron < forward_size; neuron++){
            double sum = 0.0;
            for(int previous_neuron = 0; previous_neuron < back_size; previous_neuron++ ){
                sum+= weights[layer][neuron][previous_neuron] * activations[layer][previous_neuron] + biases[layer][neuron];
            }
            a[neuron] = sigmoid(sum);
        }
        activations.push_back(a);
    }
    // get final result
    double max= -0.1;
    int ans = 0;

        //printf("Final Values are: ");
    for(int i=0;i < 10; i++){
        //cout<< a_final_layer[i]<<' ';
        if(activations[L-1][i] >= max){
            ans = i;
            max = activations[L-1][i];
        }
    }
    //cout<<'\n';
    return ans;

}

void simple_stochastic_gradient_descent(vector<int>& sizes, int dense_layers, int nh, int ne, int nb, double alpha ) {
    // load images

    // Read MNIST training images
    vector<vector<int> > training_images = readMNISTImages(TRAIN_IMAGES_FILE, NUM_TRAIN_IMAGES);
    vector<unsigned char> training_labels = readMNISTLabels(TRAIN_LABELS_FILE, NUM_TRAIN_IMAGES);
    vector<vector<int> > test_images = readMNISTImages(TEST_IMAGES_FILE, NUM_TEST_IMAGES);
    vector<unsigned char> test_labels = readMNISTLabels(TEST_LABELS_FILE, NUM_TEST_IMAGES);

    int tr_labels_size = training_labels.size();
    int te_labels_size = test_images.size();

    mt19937 twista(42);
    // Define the range for random numbers
    double min_value = -1;
    double max_value = 1;

    // Create a uniform distribution for integers within the range
    uniform_real_distribution<double> dist(min_value, max_value);
    int L = sizes.size();

    // Middle weight matrice 3D
    vector<vector<vector<double> > > weights(L-1);
    // BIASES ----------------
    vector<vector<double> > biases(L-1);

    for(int layer = 0; layer < L-1; layer++){
        int forward_size = sizes[layer+1];
        int back_size = sizes[layer];
        vector<vector<double> > middle_weight(forward_size, vector<double>(back_size, 0.0));

        vector<double> bias_layer(forward_size, 0.0);
        for(int i=0; i< forward_size; i++){
            bias_layer[i] = dist(twista);
            for(int j=0; j< back_size; j++){
                middle_weight[i][j] = dist(twista);
            }
        }
        weights[layer]  = middle_weight;
        biases[layer]   = bias_layer;

    }
        
    // for nh!=10 need to change above
    auto start_time = chrono::high_resolution_clock::now();

    // Everything is randomized by now initialized
    // Randomize what images to pick up
    vector<int> indices(tr_labels_size);
    for (int i = 0; i < tr_labels_size; ++i) {
        indices[i] = i;
    }

    shuffleIndices(indices);

    for(int epochs = 0; epochs < ne; epochs++){
        // choose m images

        printf("Epoch {%d}: Starting Epoch\n", epochs);
        shuffleIndices(indices);
        printf("Epoch {%d}: Finished Shuffling\n", epochs);
        int number_of_steps = tr_labels_size/nb;
        
        // set number of steps
        if(number_of_steps * nb != tr_labels_size){
            number_of_steps++;
        }

        printf("Epoch {%d}: Number of steps: {%d}\n", epochs, number_of_steps);
        
        // for mini batch in batch

        double cost_function = 0.0;
        // Create the batches
        for(int n_step = 0; n_step < number_of_steps; n_step++) {
            int start_of_batch  = n_step * nb;
            int end_of_batch    = min(tr_labels_size-1, start_of_batch + nb);

            // Dragging Sums
            vector<vector<vector<double> > > weight_dragging_sum (L-1);
            vector<vector<double> > bias_dragging_sum   (L-1);
            weight_dragging_sum.clear();
            bias_dragging_sum.clear();

            for(int layer = 0; layer < L-1; layer++){
                int forward_size = sizes[layer+1];
                int back_size = sizes[layer];
                vector<vector<double> > middle_weight(forward_size, vector<double>(back_size, 0.0));
                vector<double> bias_layer(forward_size, 0.0);
                weight_dragging_sum[layer]  = middle_weight;
                bias_dragging_sum[layer]   = bias_layer;
            }
            // Iterate over images in this batch
            for(int img_index = start_of_batch; img_index <= end_of_batch; img_index++){
                
                int img_id = indices[img_index];
                vector<vector<double> > image(IMAGE_ROWS, vector<double>(IMAGE_COLS, 0.0));
                vector<double> ground_truth_y(10,0.0);
                // hot vector encode
                for(int i=0;i<10;i++){
                    if( i == static_cast<int>(training_labels[img_id])){
                        ground_truth_y[i] = (double) 1.0;
                    } 
                }

                vector<vector<double> > activations;
                vector<vector<double> > zs;
                vector<double> activation;

                // set activation for first layer
                for(int i=0; i< IMAGE_ROWS; i++){
                    for(int j=0; j<IMAGE_COLS; j++){
                        image[i][j] = ((double)training_images[img_id][i*IMAGE_COLS + j] /(double)255.0);
                        // first Z layer
                        activation.push_back(image[i][j]);
                    }
                }
                activations.push_back(activation);
                activation.clear();

                for(int layer = 0; layer < L-1; layer++){
                    int forward_size = sizes[layer+1];
                    int back_size = sizes[layer];

                    vector<double> z(forward_size, 0.0);
                    vector<double> a(forward_size, 0.0);
                    
                    for(int neuron = 0; neuron < forward_size; neuron++){

                        double sum = 0.0;
                        for(int previous_neuron = 0; previous_neuron < back_size; previous_neuron++ ){
                            sum+= weights[layer][neuron][previous_neuron] * activations[layer][previous_neuron] + biases[layer][neuron];
                        }
                        z[neuron] = sum;
                        a[neuron] = sigmoid(sum);
                    }
                    zs.push_back(z);
                    activations.push_back(a);
                }


                // ===============================================================================
                // Find output error

                //printf("Epoch {%d}: Going to calculate Output Error!\n", epochs);
                vector<double> delta;
                for(int neuron=0; neuron < sizes[L-1]; neuron++){
                    double value = (activations[L-1][neuron] - ground_truth_y[neuron]) * sigmoid_derivative(zs[L-2][neuron]);
                    delta.push_back(value); 
                    for(int previous_neuron = 0; previous_neuron < sizes[L-2]; previous_neuron++){
                        weight_dragging_sum[L-2][neuron][previous_neuron] += delta[neuron] * activations[L-2][previous_neuron]; 
                    }
                    bias_dragging_sum[L-2][neuron] += delta[neuron];
                    cost_function += (activations[L-1][neuron] - ground_truth_y[neuron])*(activations[L-1][neuron] - ground_truth_y[neuron]);
                }

                //printf("Epoch {%d}: Output Error Calculated!\n", epochs);
                // ===============================================================================
                // Backprop error

                //printf("Epoch {%d}:Starting Backprop\n", epochs);
                vector<double> tmp_delta;
                // second last layer to first layer
                for(int layer = L-2; layer>=1; layer--) {

                    for(int neuron = 0; neuron < sizes[layer]; neuron++){
                        double sum = 0.0;
                        for(int forward_neuron = 0; forward_neuron<sizes[layer+1]; forward_neuron++){
                            sum += weights[layer][forward_neuron][neuron] * delta[forward_neuron] *sigmoid_derivative(zs[layer-1][neuron]);
                        }
                        tmp_delta.push_back(sum);
                        for(int previous_neuron = 0; previous_neuron < sizes[layer-1]; previous_neuron++){
                            weight_dragging_sum[layer-1][neuron][previous_neuron] += tmp_delta[neuron] * activations[layer-1][previous_neuron];
                        }
                        bias_dragging_sum[layer-1][neuron] += tmp_delta[neuron];
                    }
                    
                    delta = tmp_delta;
                    //tmp_delta.clear(); uncommenting this gives worse performance which doesnt make sense
                }
            
                    
            // ===============================================================================        
            
            // ===============================================================================    
            // inside an image loop
            }
            cost_function = (double)cost_function/((double)2.0 * nb);
            printf("Epoch {%d}: Cost function value: %f\n", epochs, cost_function);
            
       
            // inside batch loop
            // Grad descent and change weights and biases

            for(int layer = L-2; layer >=0; layer--){
                int forward_size = sizes[layer+1];
                int back_size = sizes[layer];

                for(int neuron = 0; neuron < forward_size; neuron++){

                    
                    for(int previous_neuron = 0; previous_neuron < back_size; previous_neuron++ ){
                        weights[layer][neuron][previous_neuron] -= (double)(alpha)/((double)nb) *(weight_dragging_sum[layer][neuron][previous_neuron]);
                    }
                        biases[layer][neuron] -= (double)(alpha)/((double)nb) *(bias_dragging_sum[layer][neuron]);
                    
                }
            }
            
        }
        
    }

    printf("Outside Epochs\n");

    auto end_time = chrono::high_resolution_clock::now();
    // Compute duration and print the result
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Execution time for training: " << duration.count() << " milliseconds" << std::endl;
    start_time = chrono::high_resolution_clock::now();
    printf("Epoch {%d}: Running Over Test Data Set\n", 1);

        int test_data_size = test_images.size();
        int success = 0;
        for(int img = 0; img < test_data_size; img++){
            int ground_truth_label = static_cast<int>(test_labels[img]);
            vector<vector<double> > image(IMAGE_ROWS, vector<double>(IMAGE_COLS, 0.0));
            for(int i = 0; i < IMAGE_ROWS; i++){
                for(int j=0; j< IMAGE_COLS; j++){
                    image[i][j] = ((double)test_images[img][i*IMAGE_COLS + j] /(double)255.0);
                }
            }

            int found_label = predict(image, sizes, weights, biases, dense_layers, nh, ne, nb, alpha);
            if(found_label == ground_truth_label){
                success++;
            }
            // if(img<5){
            //     if (!test_images.empty() && !test_labels.empty()) {
            //         for (int i = 0; i < IMAGE_ROWS; ++i) {
            //             for (int j = 0; j < IMAGE_COLS; ++j) {
            //                 if (image[i][j] > 0.5) {
            //                     cout << 'X' ;
            //                 } else {
            //                     cout <<  '.';
            //                 }
            //             }
            //             cout << endl;
            //         }
            //     } 
            // }   
            printf("Prediction is %d, actual: %d\n", found_label, ground_truth_label);
        }
        double accuracy = (double)success/(double)test_data_size;
        printf("\nEpoch {%d}: Success: {%d}, Images: {%d}, Accuracy is {%f}\n", 1, success, test_data_size, accuracy);
        printf("==============\n");
        end_time = chrono::high_resolution_clock::now();
        // Compute duration and print the result
        duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        cout << "Execution time for Inference: " << duration.count() << " milliseconds" << std::endl;

}


int main(int argc, char* argv[]){

    // Read args
    if(argc != 6){
        cout << "Wrong number of arguments other than 6, found " << argc <<'\n';
        return -1;
    }
    int dense_layers, nh, ne, nb;
    double alpha;

    dense_layers    = atoi(argv[1]);
    nh              = atoi(argv[2]);
    ne              = atoi(argv[3]);
    nb              = atoi(argv[4]);
    alpha           = atof(argv[5]);

    vector<int> sizes;
    sizes.push_back(784);
    for(int i=0;i<dense_layers;i++){
        sizes.push_back(nh);
    }
    sizes.push_back(10);
    // sizes = [784, 10, 10, 10]
    simple_stochastic_gradient_descent(sizes, dense_layers, nh, ne, nb, alpha);



    return 0;
}