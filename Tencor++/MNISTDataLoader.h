#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include "Tensor.h"  

class MNISTDataLoader {
private:
    Tensor3<double> images;  // 3D tensor for storing images (numImages x numRows x numCols)
    Tensor1<double> labels;  // 1D tensor for storing labels (numLabels)

    // Helper function to reverse bytes for proper interpretation of data
    uint32_t reverseBytes(uint32_t value);

    // Function to load image data from the binary file
    void loadImages(const std::string& imageFile);

    // Function to load label data from the binary file
    void loadLabels(const std::string& labelFile);

public:
    // Constructor: loads both images and labels from specified files
    MNISTDataLoader(const std::string& imageFile, const std::string& labelFile);

    // Getter function for images
    Tensor3<double> getImages() const;

    // Getter function for labels
    Tensor1<double> getLabels() const;

    // Normalizes images (values between 0 and 255 to [0, 1])
    void normalizeImages();

    // Prints summary of the dataset (number of images, size, labels count)
    void printSummary() const;
};

// Implementation of MNISTDataLoader methods
uint32_t MNISTDataLoader::reverseBytes(uint32_t value) {
    return ((value & 0xFF) << 24) |
           ((value & 0xFF00) << 8) |
           ((value & 0xFF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
}

void MNISTDataLoader::loadImages(const std::string& imageFile) {
    std::ifstream file(imageFile, std::ios::binary);
    if (!file.is_open()) {
		std::cerr << "Unable to open image file: " << imageFile << std::endl;
        throw std::runtime_error("Unable to open image file: " + imageFile);
    }

    uint32_t magicNumber = 0;
    uint32_t numImages = 0;
    uint32_t numRows = 0;
    uint32_t numCols = 0;

    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

    magicNumber = reverseBytes(magicNumber);
    numImages = reverseBytes(numImages);
    numRows = reverseBytes(numRows);
    numCols = reverseBytes(numCols);

    if (magicNumber != 2051) {
		std::cerr << "Invalid magic number in image file" << std::endl;
        throw std::runtime_error("Invalid magic number in image file");
    }

    numImages = 50;

    images = Tensor3<double>({static_cast<int>(numImages), static_cast<int>(numRows), static_cast<int>(numCols)}, InitType::Default);
    for (uint32_t i = 0; i < numImages; ++i) {
        for (uint32_t r = 0; r < numRows; ++r) {
            for (uint32_t c = 0; c < numCols; ++c) {
                uint8_t pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                images({static_cast<int>(i), static_cast<int>(r), static_cast<int>(c)}) = static_cast<float>(pixel);
            }
        }
    }
    file.close();
}

void MNISTDataLoader::loadLabels(const std::string& labelFile) {
    std::ifstream file(labelFile, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open label file: " + labelFile);
    }

    uint32_t magicNumber = 0;
    uint32_t numLabels = 0;

    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

    magicNumber = reverseBytes(magicNumber);
    numLabels = reverseBytes(numLabels);

	numLabels = 50;

    if (magicNumber != 2049) {
		std::cerr << "Invalid magic number in label file" << magicNumber << std::endl;
        throw std::runtime_error("Invalid magic number in label file");
    }

    std::vector<int> labelsSize = { static_cast<int>(numLabels) };
    labels = Tensor1<double>( labelsSize, InitType::Default);
    for (uint32_t i = 0; i < numLabels; ++i) {
        uint8_t label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels({static_cast<int>(i)}) = (double)label;
    }
    file.close();
}

MNISTDataLoader::MNISTDataLoader(const std::string& imageFile, const std::string& labelFile) 
    : images({1, 1, 1}),  // Initialize with a valid shape to avoid default constructor issues
      labels({1})         // Initialize with a valid shape
{
    loadImages(imageFile);
    loadLabels(labelFile);
}

Tensor3<double> MNISTDataLoader::getImages() const {
    return images;
}

Tensor1<double> MNISTDataLoader::getLabels() const {
    return labels;
}

void MNISTDataLoader::normalizeImages() {
	images /= 255.0f;
}

void MNISTDataLoader::printSummary() const {
    std::cout << "Loaded " << images.getShape()[0] << " images, each of size "
              << images.getShape()[1] << "x" << images.getShape()[2] << " pixels." << std::endl;
    std::cout << "Loaded " << labels.getShape()[0] << " labels." << std::endl;
}
