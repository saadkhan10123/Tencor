// #include <iostream>
// #include <fstream>
// #include <vector>
// #include "opencv2/opencv.hpp"
// #include "opencv2/ml.hpp"

// // Function to read IDX3-UBYTE files (image data)
// std::vector<std::vector<unsigned char>> readIDX3UByteFile(const std::string& filename) {
//    std::ifstream file(filename, std::ios::binary);

//    if (!file) {
//        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
//        return {};
//    }

//    // Read the IDX3-UBYTE file header
//    char magicNumber[4];
//    char numImagesBytes[4];
//    char numRowsBytes[4];
//    char numColsBytes[4];

//    file.read(magicNumber, 4);
//    file.read(numImagesBytes, 4);
//    file.read(numRowsBytes, 4);
//    file.read(numColsBytes, 4);

//    // Convert the header information from big-endian to native endianness
//    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | 
//                    (static_cast<unsigned char>(numImagesBytes[1]) << 16) | 
//                    (static_cast<unsigned char>(numImagesBytes[2]) << 8) | 
//                    static_cast<unsigned char>(numImagesBytes[3]);
//    int numRows = (static_cast<unsigned char>(numRowsBytes[0]) << 24) | 
//                  (static_cast<unsigned char>(numRowsBytes[1]) << 16) | 
//                  (static_cast<unsigned char>(numRowsBytes[2]) << 8) | 
//                  static_cast<unsigned char>(numRowsBytes[3]);
//    int numCols = (static_cast<unsigned char>(numColsBytes[0]) << 24) | 
//                  (static_cast<unsigned char>(numColsBytes[1]) << 16) | 
//                  (static_cast<unsigned char>(numColsBytes[2]) << 8) | 
//                  static_cast<unsigned char>(numColsBytes[3]);

//    std::vector<std::vector<unsigned char>> images;

//    // Read the image data for each image
//    for (int i = 0; i < numImages; i++) {
//        std::vector<unsigned char> image(numRows * numCols);
//        file.read((char*)(image.data()), numRows * numCols);
//        images.push_back(image);
//    }

//    file.close();
//    return images;
// }

// // Function to read IDX1-UBYTE files (label data)
// std::vector<std::vector<unsigned char>> readLabelFile(const std::string& filename) {
//    std::ifstream file(filename, std::ios::binary);

//    if (!file) {
//        std::cerr << "Failed to open the label file." << std::endl;
//        return {};
//    }

//    char magicNumber[4];
//    char numImagesBytes[4];

//    file.read(magicNumber, 4);
//    file.read(numImagesBytes, 4);

//    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | 
//                    (static_cast<unsigned char>(numImagesBytes[1]) << 16) | 
//                    (static_cast<unsigned char>(numImagesBytes[2]) << 8) | 
//                    static_cast<unsigned char>(numImagesBytes[3]);

//    std::vector<std::vector<unsigned char>> labels;

//    // Read the label data for each image
//    for (int i = 0; i < numImages; i++) {
//        std::vector<unsigned char> label(1);
//        file.read((char*)(label.data()), 1);
//        labels.push_back(label);
//    }

//    file.close();
//    return labels;
// }

// int main1() {
//    // Define the file paths for the image and label data
//    std::string filename = "C:/Users/USMAN-PC/Desktop/Tencor/mnist/t10k-images.idx3-ubyte";
//    std::string label_filename = "C:/Users/USMAN-PC/Desktop/Tencor/mnist/t10k-images.idx3-ubyte";

//    // Read image and label data from files
//    std::vector<std::vector<unsigned char>> imagesFile = readIDX3UByteFile(filename);
//    std::vector<std::vector<unsigned char>> labelsFile = readLabelFile(label_filename);

//    // Loop to display the first 10 images (change the loop range to display more images)
//    // if display all replace 10 this with (int)imagesFile.size()
//    for (int imgCnt = 0; imgCnt < 10; imgCnt++) {
//        int rowCounter = 0;
//        int colCounter = 0;

//        // Create a 28x28 matrix to hold the image data
//        cv::Mat tempImg = cv::Mat::zeros(cv::Size(28, 28), CV_8UC1);

//        // Fill the matrix with pixel data from the current image
//        for (int i = 0; i < (int)imagesFile[imgCnt].size(); i++) {
//            tempImg.at<uchar>(cv::Point(colCounter++, rowCounter)) = (int)imagesFile[imgCnt][i];

//            // Move to the next row after every 28th pixel
//            if ((i + 1) % 28 == 0) {
//                rowCounter++;
//                colCounter = 0;
//            }
//        }

//        // Print the image matrix (28x28) for each of the first 10 images
//        std::cout << "Image " << imgCnt + 1 << " (Label: " << (int)labelsFile[imgCnt][0] << "):" << std::endl;

//        // Print each pixel value in the matrix
//        for (int i = 0; i < 28; i++) {
//            for (int j = 0; j < 28; j++) {
//                std::cout << (int)tempImg.at<uchar>(i, j) << " ";  // Print pixel value as integer
//            }
//            std::cout << std::endl;  // Move to the next line after each row of the image
//        }

//        // To visualize the image (this step is optional but useful to check visually)
//        cv::Mat resizedImg;
//        cv::resize(tempImg, resizedImg, cv::Size(560, 560), 0, 0, cv::INTER_LINEAR);  // Resize the image for better visibility

//        // Create a window with a custom size for displaying the image
//        cv::namedWindow("TempImg", cv::WINDOW_NORMAL);  // Allow resizing of the window
//        cv::resizeWindow("TempImg", 560, 560);  // Set the size of the display window

//        // Show the resized image
//        cv::imshow("TempImg", resizedImg);
//        cv::waitKey(0);  // Wait for a key press before showing the next image

//        std::cout << std::endl;  
//    }

//    return 0;  
// }


// // int main() {

// //     std::string filename = "C:/Users/USMAN-PC/Desktop/mnist/train-images.idx3-ubyte";
// //     std::string label_filename = "C:/Users/USMAN-PC/Desktop/mnist/train-labels.idx1-ubyte";

// //     std::vector<std::vector<unsigned char>> imagesFile = readIDX3UByteFile(filename);
// //     std::vector<std::vector<unsigned char>> labelsFile = readLabelFile(label_filename);
// //     std::vector<cv::Mat> imagesData;  // Store your images
// //     std::vector<int> labelsData;      // Corresponding labels

// //     for(int imgCnt=0; imgCnt<(int)imagesFile.size(); imgCnt++)
// //     {
// //         int rowCounter = 0;
// //         int colCounter = 0;

// //         cv::Mat tempImg = cv::Mat::zeros(cv::Size(28,28),CV_8UC1);
// //         for (int i = 0; i < (int)imagesFile[imgCnt].size(); i++) {

// //             tempImg.at<uchar>(cv::Point(colCounter++,rowCounter)) = (int)imagesFile[imgCnt][i];
           
// //             if ((i) % 28 == 0) {
// //                 rowCounter++;
// //                 colCounter= 0;
// //                 if(i == 756)
// //                     break;
// //             }
// //         }
// //         std::cout<<(int)labelsFile[imgCnt][0]<<std::endl;

// //         imagesData.push_back(tempImg);
// //         labelsData.push_back((int)labelsFile[imgCnt][0]);
// //     }

// // // to visualize each image ,n dataset  to check only
// //        cv::Mat resizedImg;
// //     cv::resize(tempImg, resizedImg, cv::Size(560, 560), 0, 0, cv::INTER_LINEAR);

// //     // Create a window with a custom size
// //     cv::namedWindow("TempImg", cv::WINDOW_NORMAL);  // Use cv::WINDOW_NORMAL to allow resizing
// //     cv::resizeWindow("TempImg", 560, 560);  // Set the size of the window

// //     // Show the resized image
// //     cv::imshow("TempImg", resizedImg);
// //     cv::waitKey(0);
// // //     }
// //     return 0;
// // }
