# Pattern Recognition Project: k-Nearest Neighbors Digit Classifier

This is a foundational case study in **Statistical Pattern Recognition**, demonstrating the complete image classification pipeline using the classic **MNIST** dataset of handwritten digits ($\mathbf{0-9}$).

## Project Goal

The primary objective is to implement and evaluate the **k-Nearest Neighbors (k-NN)** algorithm—a core concept in pattern recognition—to classify high-dimensional image patterns. The project showcases how to train a model to distinguish between the ten classes of handwritten digits.

-----

## Project Pipeline and Methodology (k-NN)

The project is structured into $\mathbf{5}$ logical steps, directly correlating with the stages of the Pattern Recognition system:

1.  **Data Acquisition & Preprocessing:**
      * Loads $\mathbf{70,000}$ images from the MNIST dataset.
      * **Normalization:** Pixel values are scaled from $\mathbf{0}$ to $\mathbf{1}$.
2.  **Feature Extraction:**
      * The $\mathbf{28} \times \mathbf{28}$ image is **flattened** into a $\mathbf{784}$-dimensional **feature vector**. This vector is the numerical representation of the digit pattern.
3.  **Model Training:**
      * The $\mathbf{k}$-NN classifier is selected ($\mathbf{k=5}$). As a "lazy learner," training involves simply **storing** the $\mathbf{5,000}$ selected training feature vectors (a sample is used for rapid execution).
4.  **Classification & Decision:**
      * For the $\mathbf{10,000}$ unseen test images, the model calculates the $\mathbf{Euclidean Distance}$ to the stored training patterns.
      * The classification decision is made by a $\mathbf{majority\ vote}$ among the $\mathbf{5}$ nearest neighbors.
5.  **Evaluation:**
      * Performance is measured using **Accuracy** and a detailed **Confusion Matrix** to analyze errors and instances of **pattern misrecognition**.

-----

## Key Concepts Demonstrated

  * **Statistical Pattern Recognition:** Classification based on the proximity of patterns in a feature space.
  * **k-Nearest Neighbors (k-NN):** A non-parametric, distance-based classification algorithm.
  * **Feature Vector:** The representation of an input pattern as a high-dimensional vector.
  * **Evaluation:** Using the **Confusion Matrix** to gain insight into which patterns (digits) are most frequently misclassified.

-----

## Results

The implemented k-NN model consistently achieves an accuracy of approximately **$\mathbf{96\% \text{ to } 97\%}$** on the $\mathbf{10,000}$ test images, validating k-NN's effectiveness for this classification task.

## Requirements

The project uses common Python libraries:

```bash
pip install numpy scikit-learn matplotlib seaborn
```

## How to Run the Project

1.  Open the provided Python script (or upload the code to a new Google Colab notebook).
2.  Run the code cells sequentially. The output will display the model's accuracy, prediction time, a sample result, and the final Confusion Matrix.

## Author

\[Naman Kumar Sharma]
\[Pattern Recognition Technique]

-----
