# Double Path Neural Network

This project implements a neural network with a double pathway, inspired by the ventral and dorsal pathways in the human visual system. The model is designed to recognize objects in images and perceive space and motion through the use of optical flow.

## Code Structure

The code is organized into the following files:

1. **double_path_model.py:** Contains the definition of the double pathway neural network model.
2. **optical_flow.py:** Implements the function for calculating optical flow using the Lucas-Kanade method.
3. **train_and_test.py:** Contains the code for generating the example dataset, splitting into training and testing sets, and the training and evaluation process of the model.

## Dependencies

The project requires the following Python libraries:

- TensorFlow
- OpenCV
- NumPy
- scikit-learn

Install dependencies by running:

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

## Usage

1. Run `train_and_test.py` to generate the example dataset, split into training and testing sets, and train the model.
2. Modify the code according to the specific needs of your project.
3. Explore the training and evaluation results to gain insights into the model's performance.


## License

This project is licensed under the [MIT License](LICENSE).
```
