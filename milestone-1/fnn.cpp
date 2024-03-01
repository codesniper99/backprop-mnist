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

int predict(vector<vector<double> > &image, vector<vector<double> > &first_layer_weights, 
    vector<vector<vector<double> > > &dense_weights, vector<vector<double> > &biases, int dense_layers, int nh, int ne, int nb, float alpha ){
    int L = dense_layers + 2;
    // ACTIVATION LAYERS ----------------

    // first layer
    vector<double> a_first_layer(IMAGE_SIZE, 0.0);
    
    // create hidden layers
    vector<vector<double> >a_hidden_layers(L, vector<double>(nh, 0.0));

    // final layer
    vector<double> a_final_layer(10, 0.0);

    // Z LAYERS ----------------

    // first Z layer
    vector<double> z_first_layer(IMAGE_SIZE, 0.0);
    
    // create hidden Z layers
    vector<vector<double> >z_hidden_layers(L, vector<double>(nh, 0.0));

    // final Z layer
    vector<double> z_final_layer(10, 0.0);

    for(int i=0; i< IMAGE_ROWS; i++){
        for(int j=0; j<IMAGE_COLS; j++){
            // first Z layer
            z_first_layer[i*IMAGE_COLS + j] = (image[i][j]);
            // first layer
            a_first_layer[i*IMAGE_COLS + j] = sigmoid(image[i][j]);
        }
    }

    // Forward Pass
        // First and middle layers
        for(int layer = 2; layer <= L-1; layer++){
            if(layer == 2){
                // get inputs from first layer
                
                for(int neuron = 0; neuron < nh; neuron++){
                    double sum = 0.0;
                    for(int previous_neuron = 0; previous_neuron < IMAGE_SIZE; previous_neuron++){
                        sum += first_layer_weights[neuron][previous_neuron] * a_first_layer[previous_neuron] + biases[layer][neuron];
                    }

                    z_hidden_layers[layer][neuron] = sum;

                    double activation = sigmoid(sum);
                    a_hidden_layers[layer][neuron] = activation;
                }
                
            } else {
                
                for(int neuron = 0; neuron < nh; neuron++){
                    // get input from last layer
                    double sum = 0.0;
                    for(int previous_neuron = 0; previous_neuron < nh; previous_neuron++){
                        sum += dense_weights[layer][neuron][previous_neuron] * a_hidden_layers[layer-1][previous_neuron] + biases[layer][neuron];
                    }
                    z_hidden_layers[layer][neuron] = sum;

                    double activation = sigmoid(sum);
                    a_hidden_layers[layer][neuron] = activation;
                }
            }

            
        }
        // Last layer
        
        for(int neuron = 0; neuron < nh; neuron++){
            // get input from last layer
            double sum = 0.0;
            for(int previous_neuron = 0; previous_neuron < nh; previous_neuron++){
                // layer considering is last hidden
                sum += dense_weights[L][neuron][previous_neuron] * a_hidden_layers[L-1][previous_neuron] + biases[L][neuron];
            }
            z_final_layer[neuron] = sum;

            double activation = sigmoid(sum);
            a_final_layer[neuron] = activation;
        }
    
    // get final result
    double max= -0.1;
    int ans = 0;

    //     printf("Final Values are: ");
    // for(int i=0;i < 10; i++){
    //     cout<< a_final_layer[i]<<' ';
    //     if(a_final_layer[i] >= max){
    //         ans = i;
    //         max = a_final_layer[i];
    //     }
    // }
    // cout<<'\n';
    return ans;

}

void simple_stochastic_gradient_descent(int dense_layers, int nh, int ne, int nb, double alpha ) {
    // load images

    // Read MNIST training images
    vector<vector<int> > training_images = readMNISTImages(TRAIN_IMAGES_FILE, NUM_TRAIN_IMAGES);
    // Read MNIST training labels
    vector<unsigned char> training_labels = readMNISTLabels(TRAIN_LABELS_FILE, NUM_TRAIN_IMAGES);

    // Read MNIST training images
    vector<vector<int> > test_images = readMNISTImages(TEST_IMAGES_FILE, NUM_TEST_IMAGES);
    // Read MNIST training labels
    vector<unsigned char> test_labels = readMNISTLabels(TEST_LABELS_FILE, NUM_TEST_IMAGES);

    int tr_labels_size = training_labels.size();
    int te_labels_size = test_images.size();

    // Print the first 3 images as an example
    if (!training_images.empty() && !training_labels.empty()) {
        for(int z = 0; z< 3; z++){
            cout << z << "th image:" << '\n';
            cout << "Label found is: " << static_cast<int>(training_labels[z]) << '\n';
            for (int i = 0; i < IMAGE_ROWS; ++i) {
                for (int j = 0; j < IMAGE_COLS; ++j) {
                    if (training_images[z][i * IMAGE_COLS + j] > 128) {
                        cout << 'X' ;
                    } else {
                        cout <<  '.';
                    }
                }
                cout << endl;
            }
        }
        
    }   
    cout << "Total Images in Training Set: " << tr_labels_size <<'\n'; 

    // Print the first 3 images as an example
    if (!test_images.empty() && !test_labels.empty()) {
        for(int z = 0; z< 3; z++){
            cout << z << "th image:" << '\n';
            cout << "Label found is: " << static_cast<int>(test_labels[z]) << '\n';
            for (int i = 0; i < IMAGE_ROWS; ++i) {
                for (int j = 0; j < IMAGE_COLS; ++j) {
                    if (test_images[z][i * IMAGE_COLS + j] > 128) {
                        cout << 'X' ;
                    } else {
                        cout <<  '.';
                    }
                }
                cout << endl;
            }
        }
        
    }   
    cout << "Total Images in Test Set: " << te_labels_size <<'\n'; 


    mt19937 twista(42);
    // Create and intialize Matrics
    // Define the range for random numbers
    double min_value = -0.5;
    double max_value = 0.5;

    // Create a uniform distribution for integers within the range
    uniform_real_distribution<double> dist(min_value, max_value);
    int L = dense_layers + 2;

    // ACTIVATION LAYERS ----------------

        // first layer
        vector<double> a_first_layer(IMAGE_SIZE, 0.0);
        
        // create hidden layers
        vector<vector<double> >a_hidden_layers(L, vector<double>(nh, 0.0));

        // final layer
        vector<double> a_final_layer(10, 0.0);

    // Z LAYERS ----------------

        // first Z layer
        vector<double> z_first_layer(IMAGE_SIZE, 0.0);
        
        // create hidden Z layers
        vector<vector<double> >z_hidden_layers(L, vector<double>(nh, 0.0));

        // final Z layer
        vector<double> z_final_layer(10, 0.0);


    // WEIGHTS ----------------
    // first weight layer 2D
    vector<vector<double> >  first_layer_weights(nh, vector<double>(IMAGE_SIZE, 0.0));
    
    // Middle weight matrice 3D
    vector<vector<vector<double> > > dense_weights(L+1, vector<vector<double> >(nh, vector<double>(nh, 0.0)));
    // if last weight comes then above size becomes dense - 1
        // Last weight layer
        // vector<vector<vector<double> > > last_layer_weights(1, vector<vector<double>>(10, vector<double>(nh, 0.0)));
        
    // BIASES ----------------
    // Middle biases
    vector<vector<double> > biases(L+1, vector<double>(nh, 0.0));
    
    // for nh!=10 need to change above
    auto start_time = std::chrono::high_resolution_clock::now();
    // ------------------------
    // RANDOMIZE WEIGHTS AND BIASES
        // First layer weight
        for(int neuron = 0; neuron < nh; neuron++){
                for(int previous_neuron = 0; previous_neuron < IMAGE_SIZE; previous_neuron++){
                double random_num = dist(twista);
                first_layer_weights[neuron][previous_neuron] = random_num;
                //cout << "Number = " << random_num<<' ';
            }
        }
        // Middle layer weights
        for(int layer = 3; layer <= L; layer++){
            for(int neuron = 0; neuron < nh; neuron++){
                for(int previous_neuron = 0; previous_neuron < nh; previous_neuron++){
                    double random_num = dist(twista);
                    dense_weights[layer][neuron][previous_neuron] = random_num;
                    //cout << "Number = " << random_num<<' ';
                }
            }
        }
        // Biases
        for(int layer = 2; layer <= L; layer++){
        for(int neuron = 0; neuron < nh; neuron++){
            biases[layer][neuron] = dist(twista);
        }
    }

    // Everything is randomized by now initialized
    // ------------------------
    // Randomize what images to pick up
    vector<int> indices(tr_labels_size);
    for (int i = 0; i < tr_labels_size; ++i) {
        indices[i] = i;
    }

    shuffleIndices(indices);
    // ------------------------
    // loop over epochs
    // first layer
    // printf("Before run weights, first layer\n");
    //     for(int i=0;i<nh;i++){
    //         for(int j=0;j<IMAGE_SIZE;j++){
    //             printf("w[%d,%d]: %f", i,j,first_layer_weights[i][j]);
    //         }
    //     }
    // for(int layer=2;layer<=L;layer++){
    //     printf(" %d layer weights", layer);
    //     for(int i=0;i<nh;i++){
    //         for(int j=0;j<nh;j++){
    //             printf("w[%d,%d]: %f", i,j,dense_weights[layer][i][j]);
    //         }
    //     }
    // }

    for(int epochs = 0; epochs < ne; epochs++){
        // choose m images

        printf("Epoch {%d}: Starting Epoch\n", epochs);
        shuffleIndices(indices);
        // for(int i=0;i<10;i++){
        //     cout<<indices[i]<<' ';
        // }
        // cout<<'\n';
        printf("Epoch {%d}: Finished Shuffling\n", epochs);
        int number_of_steps = tr_labels_size/nb;
        
        // set number of steps
        if(number_of_steps * nb != tr_labels_size){
            number_of_steps++;
        }

        printf("Epoch {%d}: Number of steps: {%d}\n", epochs, number_of_steps);
        
        // main Loop
        for(int n_step = 0; n_step < number_of_steps; n_step++) {
            int start_of_batch  = n_step * nb;
            int end_of_batch    = min(tr_labels_size-1, start_of_batch + nb);
            //printf("Epoch {%d}: Start and End {%d, %d}\n", epochs, start_of_batch, end_of_batch);
            vector<vector<vector<double> > > weight_dragging_sum (L+1, vector<vector<double> >(nh, vector<double> (nh, 0.0)));
            vector<vector<double> > first_layer_weight_dragging_sum (nh, vector<double> (IMAGE_SIZE, 0.0));
            vector<vector<double> > bias_dragging_sum   (L+1, vector<double>(nh, 0.0));
   
            // The delta for 
            vector<vector<double> > delta_layers (L+1, vector<double>(nh, 0.0));
   
            double cost_function = 0.0;
            // Dragging Sums
            
            // Loop over Batch of nb size and start Training
            for(int img_index = start_of_batch; img_index <= end_of_batch; img_index++){
                
                int img_id = indices[img_index];
                //printf("Picking image: {%d}", img_id);
                vector<vector<double> > image(IMAGE_ROWS, vector<double>(IMAGE_COLS, 0.0));

                vector<double> ground_truth_y(10,0.0);
                // hot vector encode
                for(int i=0;i<10;i++){
                    if( i == static_cast<int>(training_labels[img_id])){
                        ground_truth_y[i] = (double) 1.0;
                    } 
                }
                // set image value and z and a for first layer
                for(int i=0; i< IMAGE_ROWS; i++){
                    for(int j=0; j<IMAGE_COLS; j++){
                        image[i][j] = ((double)training_images[img_id][i*IMAGE_COLS + j] /(double)255.0);
                        // first Z layer
                        z_first_layer[i*IMAGE_COLS + j] = (image[i][j]);
                        // first layer
                        a_first_layer[i*IMAGE_COLS + j] = sigmoid(image[i][j]);
                    }
                }

                // Forward Pass
                // middle layers
                for(int layer = 2; layer <= L-1; layer++){
                    if(layer == 2){
                        // get inputs from first layer
                        for(int neuron = 0; neuron < nh; neuron++){
                            double sum = 0.0;
                            for(int previous_neuron = 0; previous_neuron < IMAGE_SIZE; previous_neuron++){
                                sum += first_layer_weights[neuron][previous_neuron] * a_first_layer[previous_neuron] + biases[layer][neuron];
                            }

                            z_hidden_layers[layer][neuron] = sum;

                            double activation = sigmoid(sum);
                            a_hidden_layers[layer][neuron] = activation;
                        }
                        
                    } else {
                        
                        for(int neuron = 0; neuron < nh; neuron++){
                            double sum = 0.0;
                            // get input from last layer
                            for(int previous_neuron = 0; previous_neuron < nh; previous_neuron++){
                                sum += dense_weights[layer][neuron][previous_neuron] * a_hidden_layers[layer-1][previous_neuron] + biases[layer][neuron];
                            }
                            z_hidden_layers[layer][neuron] = sum;

                            double activation = sigmoid(sum);
                            a_hidden_layers[layer][neuron] = activation;
                        }
                    }    
                }

                // Last layer
                
                for(int neuron = 0; neuron < nh; neuron++){
                    // get input from last layer
                    double sum = 0.0;
                    for(int previous_neuron = 0; previous_neuron < nh; previous_neuron++){
                        // layer considering is last hidden
                        sum += dense_weights[L][neuron][previous_neuron] * a_hidden_layers[L-1][previous_neuron] + biases[L][neuron];
                    }
                    z_final_layer[neuron] = sum;

                    double activation = sigmoid(sum);
                    a_final_layer[neuron] = activation;
                }
                // ===============================================================================
                // Find output error

                //printf("Epoch {%d}: Going to calculate Output Error!\n", epochs);
                
                for(int neuron=0; neuron< nh; neuron++){
                    delta_layers[L][neuron] = (a_final_layer[neuron] - ground_truth_y[neuron]) * sigmoid_derivative(z_final_layer[neuron]);
                    // if(img_index == 0)
                    //     printf("Epoch {%d}: Delta final:\nw[%d]: %f\n", epochs, neuron, delta_layers[L][neuron]);
                    for(int previous_neuron = 0; previous_neuron < nh; previous_neuron++){
                        weight_dragging_sum[L][neuron][previous_neuron] = delta_layers[L][neuron] * a_hidden_layers[L-1][previous_neuron]; 
                    }
                    bias_dragging_sum[L][neuron] += delta_layers[L][neuron];
                    cost_function += (a_final_layer[neuron] - ground_truth_y[neuron])*(a_final_layer[neuron] - ground_truth_y[neuron]);
                }

                //printf("Epoch {%d}: Output Error Calculated!\n", epochs);
                // ===============================================================================
                // Backprop error

                //printf("Epoch {%d}:Starting Backprop\n", epochs);
                
                // second last layer to first layer
                for(int layer = L-1; layer>=2; layer--) {
                    if(layer > 2){
                        for(int neuron = 0; neuron < nh; neuron++){
                            double sum = 0.0;
                            for(int forward_neuron =0; forward_neuron<nh; forward_neuron++){
                                sum += dense_weights[layer+1][forward_neuron][neuron] * delta_layers[layer+1][forward_neuron] *sigmoid_derivative(z_hidden_layers[layer][neuron]);
                            }
                            delta_layers[layer][neuron] = sum;
                            for(int previous_neuron = 0; previous_neuron < nh; previous_neuron++){
                                weight_dragging_sum[layer][neuron][previous_neuron] += delta_layers[layer][neuron] * a_hidden_layers[layer-1][previous_neuron];
                            }
                            bias_dragging_sum[layer][neuron] += delta_layers[layer][neuron];
                        }
                    } else {
                        for(int neuron = 0; neuron < nh; neuron++){
                            double sum = 0.0;
                            for(int forward_neuron = 0; forward_neuron<nh; forward_neuron++){
                                sum += dense_weights[layer+1][forward_neuron][neuron] * delta_layers[layer+1][forward_neuron] *sigmoid_derivative(z_hidden_layers[layer][neuron]);
                            }
                            delta_layers[layer][neuron] = sum;
                            for(int previous_neuron = 0; previous_neuron < IMAGE_SIZE; previous_neuron++){
                                first_layer_weight_dragging_sum[neuron][previous_neuron] += delta_layers[layer][neuron] * a_first_layer[previous_neuron];
                            }
                            bias_dragging_sum[layer][neuron] += delta_layers[layer][neuron];
                        }
                    }
                    
                }
            
                    
            cost_function = (double)cost_function/((double)2.0 * nb);
            //printf("Epoch {%d}: Cost function value: %f\n", epochs, cost_function);
            // ===============================================================================        
            
            // ===============================================================================    
            // inside an image loop
            }
        // for(int neuron = 0; neuron < 2; neuron++){
        //     for(int previous_neuron = 0; previous_neuron < 2; previous_neuron++){
        //     printf("\nEpoch[%d]: Weight dragging Sum[%d,%d]: %f\n",epochs, neuron, previous_neuron, weight_dragging_sum[L-1][neuron][previous_neuron] );
    
        //     }
        //     }
        // inside batch loop
            // Grad descent and change weights and biases

            // first layer weights

            for(int neuron = 0; neuron < nh; neuron++){
                for(int previous_neuron = 0; previous_neuron < IMAGE_SIZE; previous_neuron++){
                    first_layer_weights[neuron][previous_neuron] -= ((double)alpha/(double)nb)*(first_layer_weight_dragging_sum[neuron][previous_neuron]);
                    biases[2][neuron] -= ((double)alpha/(double)nb)*(bias_dragging_sum[2][neuron]);
                }
            }

            // remaining layer weight bias

            for(int layer = 3; layer <= L; layer++){
                for(int neuron=0; neuron<nh; neuron++){ //wlkj
                    for(int previous_neuron=0; previous_neuron< nh; previous_neuron++){
                        dense_weights[layer][neuron][previous_neuron] -= ((double)alpha/(double)nb)*(weight_dragging_sum[layer][neuron][previous_neuron]);
                        biases[layer][neuron] -= ((double)alpha/(double)nb)*(bias_dragging_sum[layer][neuron]);
                    }
                }
            }
        }
        

        
        // printf("Epoch {%d}:Last layer weights\n", epochs);
        // for(int i=0;i<nh;i++){
        //     for(int j=0;j<10;j++){
        //         printf("w[%d,%d]: %f", i,j,dense_weights[L][i][j]);
        //     }
        //     cout<<"\n==========\n";
        // }
        
        
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

            int found_label = predict(image, first_layer_weights, dense_weights, biases, dense_layers, nh, ne, nb, alpha);
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
             //printf("Prediction is %d, actual: %d\n", found_label, ground_truth_label);
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

    
    simple_stochastic_gradient_descent(dense_layers, nh, ne, nb, alpha);



    return 0;
}