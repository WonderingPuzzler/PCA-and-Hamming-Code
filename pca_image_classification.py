# Simple PCA Image Classification 
# Import statements
from pca_algorithm import PCA
from sklearn import datasets
import random
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# PCA Classifier
class PCAImageClassifier(PCA):
    """
    A PCA-based image classifier using Python and the fetch_lfw_people dataset.
    The fetch_lfw_people dataset from sklearn contains grayscale face images of famous people that are 62x47 pixels.
    Each image is already flattened into a 1D array of 2914 pixel values.
    
    TERMINOLOGY:
    - Sample: A single observation in the dataset; in this context, a sample is one face image represented as a 1D array of pixel values. Each sample is a row in the data matrix X.
    - Feature: A specific measurable property or characteristic of the data being analyzed; in this context, features are pixel values from the face images. Each feature is a column in the data matrix X.
    - Images: The samples/observations (rows in data matrix X)
    - Pixel values: The features/variables (columns in data matrix X) 
    - Labels: The target classes - which person each image represents
    - Data matrix X: Shape (n_samples, n_features) where samples are rows, features are columns
    - Eigenfaces: The components derived from face images (eigenvectors of the covariance matrix)
    - Component matrix: The matrix of all components, shape (n_components, n_features)
    - Component: A new axis in the transformed feature space that captures maximum variance; components are oriented to capture the most significant patterns and variations in the data. In this context, components are derived from face images, and are only created after the PCA fitting process.
    - Component weights/features: The numerical coefficients in the component matrix that determine how much each original feature (pixel) contributes to each principal component. Each weight represents the importance of a specific pixel in defining a particular component.
    - Training data PCA: The training images transformed into PCA space, represented as a matrix where each row is a PCA-transformed image.

    """
    
    def __init__(self, n_components, k=9, variance_threshold=0.70):
        """
        Initialize the classifier by storing the number of components.
        
        Parameters:
        n_components (int): How many principal components to keep
        k (int): Number of nearest neighbors to use for classification (default=9)
        variance_threshold (float): Desired variance retention (0.0 to 1.0)

        Returns:
        None
        """
        super().__init__(n_components)
        self.set_k(k)  # Use setter for validation
        self.set_training_data_pca(None)  # Initialize with None using setter
        self.set_training_labels(None)  # Initialize with None using setter
        self.set_variance_threshold(variance_threshold)


    def get_k(self):
        """
        Get the number of neighbors (k) used in k-NN classification.

        Parameters:
        None
        
        Returns:
        k (int): Number of neighbors
        """
        return self.__k
    
    def get_training_data_pca(self):
        """
        Get the training data transformed into PCA space.

        Parameters:
        None
        
        Returns:
        training_data_pca (list of lists): Training images in PCA space
        """

        return self.__training_data_pca

    def get_training_labels(self):
        """
        Get the training labels corresponding to the PCA-transformed training data.

        Parameters:
        None

        Returns:
        training_labels (list): Labels for the training images
        """

        return self.__training_labels
    
    def get_variance_threshold(self):
        """
        Get the variance threshold for automatic component selection.

        Parameters:
        None
        
        Returns:
        variance_threshold (float): Desired variance retention ratio
        """
        return self.__variance_threshold
    
    def set_k(self, k):
        """
        Set the number of neighbors (k) used in k-NN classification.
        
        Parameters:
        k (int): Number of neighbors

        Returns:
        None
        """
        if not isinstance(k, int) or k <= 0: # Make sure k is a positive integer and not less than or equal to zero
            raise ValueError("k must be a positive integer.")
        
        self.__k = k

    def set_training_data_pca(self, training_data_pca):
        """
        Set the training data transformed into PCA space.
        
        Parameters:
        training_data_pca (list of lists): Training images in PCA space

        Returns:
        None
        """
        if training_data_pca is not None: # Validate input if not None

            if not isinstance(training_data_pca, list): # Make sure we received a list
                raise ValueError("Training data PCA must be a list or None.")
            
            if training_data_pca and not all(isinstance(sample, list) for sample in training_data_pca): # Check inner lists and see if all samples are lists and not empty
                raise ValueError("Training data PCA must be a list of lists.")
            
        self.__training_data_pca = training_data_pca

    def set_training_labels(self, training_labels):
        """
        Set the training labels corresponding to the PCA-transformed training data.

        Parameters:
        training_labels (list): Labels for the training images

        Returns:
        None
        """
        if training_labels is not None: # Validate input if not None

            if not isinstance(training_labels, list): # Make sure we received a list
                raise ValueError("Training labels must be a list or None.")
            
        self.__training_labels = training_labels

    def set_variance_threshold(self, variance_threshold):
        """
        Set the variance threshold for automatic component selection.
        
        Parameters:
        variance_threshold (float): Desired variance retention ratio (0.0 to 1.0)

        Returns:
        None
        """
        try:
            if not isinstance(variance_threshold, (int, float)): # Make sure it's a number
                raise ValueError("variance_threshold must be a number.")
            
            if not (0.0 <= variance_threshold <= 1.0): # Check range ( it must be between 0 and 1 )
                raise ValueError("variance_threshold must be between 0.0 and 1.0.")
            
        except ValueError as e:
            print(f"Input validation error for variance_threshold: {e}")
            return
            
        self.__variance_threshold = variance_threshold

    def __load_faces_dataset(self):
        """
        Load the fetch_lfw_people dataset from sklearn.

        Parameters:
        None
        
        Returns:
        images (list of lists): Each inner list contains pixel values for one face image (row format)
        labels (list): List of target labels (person identifiers)
        lfw_people: The original dataset object (for visualization)
        """

        lfw_people = datasets.fetch_lfw_people(min_faces_per_person=60) # Load dataset with at least 60 images per person
        # Convert images to list of lists - each row is an image, each column is a pixel
        images = lfw_people.data.tolist() # pyright: ignore[reportAttributeAccessIssue] - I use a type checker that complains here but this is correct
        labels = lfw_people.target.tolist() # pyright: ignore[reportAttributeAccessIssue] - I use a type checker for good input validation, and it complains here, but this is correct and doesn't cause runtime issues
        
        return images, labels, lfw_people # return the images, labels, and dataset object

    def __normalize_images(self, images):
        """
        Normalize pixel values to be between 0 and 1.
        You may note that in some sources, normalization is the very first step before anything else.
        However, we can only normalize after we actually have the data loaded.
        Therefore, normalization is done in the classifier class instead of the PCA class.
        This helps PCA work better since we want all features (pixel values) on the same scale.
        
        What this does:
        1. Finds the max pixel value in all images
        2. Divides every pixel by this max value
        3. Uses ONLY simple Python (no numpy!)
        
        Parameters:
        images (list of lists): Raw pixel values where each row is an image
        
        Returns:
        normalized_images (list of lists): Pixel values between 0 and 1 where each row is an image
        """

        try: 
            max_pixel = 0 # variable to hold the maximum pixel value

            if not images or not images[0]: # handle empty images case
                raise ValueError("Images list is empty or malformed")

            # Step 1: Find the max pixel value across all images
            for img_row in images:
                if not img_row:  # Check if image is empty
                    raise ValueError("Each image must be a non-empty list of pixel values")
                
                max_pixel = max(max_pixel, *img_row) # update max_pixel if a larger pixel value is found

            normalized_images = [] # list to hold normalized images

            # Step 2: Normalize each image by dividing by the max pixel value
            for img_row in images:
                # Ensure image is valid
                if not img_row:
                    raise ValueError("Each image must be a non-empty list of pixel values")

                # Normalize image by dividing each pixel by max_pixel
                normalized_image = [pixel / max_pixel for pixel in img_row]
                normalized_images.append(normalized_image) # add the normalized image to the list

        except (TypeError, ValueError): 
            print("Error normalizing images. Ensure images are lists of numerical pixel values.")
            return None
        
        return normalized_images # return the list of normalized images

    def __split_data_simple(self, images, labels, train_ratio=0.8):
        """
        Split data into training and testing sets.
        In machine learning, we often split our dataset into training and testing sets to evaluate how well our model generalizes to unseen data.
        The training set is used to train the model, while the testing set is used to evaluate its performance.

        What this does:
        1. Calculates how many samples go to training
        2. Splits using simple indexing
        3. Returns separate train and test lists
        
        Parameters:
        images (list): All images
        labels (list): All labels  
        train_ratio (float): How much data goes to training (0.8 = 80%)
        
        Returns:
        train_images, train_labels, test_images, test_labels
        """
        try:

            if not images or not labels or len(images) != len(labels): # Check if input is invalid
                raise ValueError("Invalid image or label data.")

            total_samples = len(images) # total number of samples/images

            # 1: Calculate number of training samples (80% by default)
            train_size = int(total_samples * train_ratio) 

            train_images = []
            train_labels = []
            test_images = []
            test_labels = []

            # 2: Split data using simple indexing
            for i in range(total_samples):
                if i < train_size:
                    train_images.append(images[i]) # add image to training set
                    train_labels.append(labels[i]) # add image label (who the image is of) to training labels
                else: 
                    test_images.append(images[i]) # add image to testing set
                    test_labels.append(labels[i]) # add image label (who the image is of) to testing labels

        except (TypeError, ValueError): # Handle errors
            print("Error splitting data. Ensure images and labels are lists of the same length and not empty.")
            return None, None, None, None

        return train_images, train_labels, test_images, test_labels # return the split data of the images and their corresponding labels

    
    def __train(self, images, labels):
        """
        Train the k-NN classifier on images and labels.
        
        What this does:
        1. Fits PCA algorithm on the images
        2. Transforms images to PCA space
        3. Stores the transformed images and labels for later comparison

        
        Parameters:
        images (list of lists): Training images where each row is an image
        labels (list): Training labels
        """
        try:

            if not images or not labels or len(images) != len(labels):
                raise ValueError("Invalid image or label data.")

            print("Training k-NN classifier...")

            # Step 1: Fit PCA on training images
            self.fit(images)

            # Step 2: Transform training images to PCA space
            images_pca = self.transform(images)

            if images_pca is None:
                raise ValueError("PCA transformation failed during training.")

            # Step 3: Store the training data in PCA space for k-NN using setters
            # This is the "training" for k-NN - we just memorize the training data!
            self.set_training_data_pca(images_pca)
            self.set_training_labels(labels)

            print(f"Stored {len(self.get_training_data_pca())} training samples in PCA space for k-NN (k={self.get_k()})")

        
        except (TypeError, ValueError):
            print("Error. Ensure images and labels are valid.")
            return
        
    
    def __predict(self, images):
        """
        Predict labels for new images using k-Nearest Neighbors.
        
        What this does:
        1. Transforms new images using the fitted PCA
        2. For each test image, finds the k closest training images
        3. Takes a majority vote among those k neighbors
        4. Returns the predicted class
        
        Parameters:
        images (list of lists): Images to classify where each row is an image
        
        Returns:
        predictions (list): Predicted class for each image
        """
        
        try:

            if not images:
                raise ValueError("Images list is empty.")

            if self.get_training_data_pca() is None or self.get_training_labels() is None:
                raise ValueError("Classifier must be trained before prediction.")

            # Step 1: Transform new images to PCA space
            images_pca = self.transform(images)

            if images_pca is None:
                raise ValueError("PCA transformation failed during prediction.")

            predictions = []
            training_data = self.get_training_data_pca()
            training_labels = self.get_training_labels()

            # Step 2: For each test image, find k nearest neighbors
            print(f"Predicting with k-NN (k={self.get_k()})...")
            for img_pca in tqdm(images_pca, desc="Classifying images", leave=False):
                
                # Convert row to column vector for vector operations if needed
                if not self.is_column_vector(img_pca):
                    img_pca_column = [[val] for val in img_pca]
                else:
                    img_pca_column = img_pca
                
                # Compute distances to all training samples
                distances = []
                
                for i in range(len(training_data)): # iterate over all training samples
                    training_sample_row = training_data[i] # get the i-th training sample (row format)
                    
                    # Convert training sample row to column vector for vector operations if needed
                    if not self.is_column_vector(training_sample_row):
                        training_sample_column = [[val] for val in training_sample_row]
                    else:
                        training_sample_column = training_sample_row

                    # Compute Euclidean distance using vector operations
                    distance_vector = self.vector_subtract(img_pca_column, training_sample_column)
                    distance = self.vector_magnitude(distance_vector)
                    
                    if distance is None:
                        raise ValueError("Error computing distance during prediction.")
                    
                    # Store (distance, label) pairs
                    distances.append((distance, training_labels[i]))
                
                # Step 3: Sort by distance and get k nearest neighbors
                distances.sort(key=lambda x: x[0])  # Sort by distance (first element of tuple)
                k_nearest = distances[:self.get_k()]  # Take first k elements
                
                # Step 4: Majority vote among k nearest neighbors
                label_counts = {}
                for distance, label in k_nearest:  # Each element is (distance, label) tuple
                    if label not in label_counts:  # Check if label is already in the dictionary
                        label_counts[label] = 0
                    label_counts[label] += 1  # Increment count for this label
                
                # Find the label with the most votes
                best_label = None
                max_votes = 0
                for label, count in label_counts.items():  # label is the person's ID, count is the number of votes
                    if count >= max_votes:
                        max_votes = count
                        best_label = label
                
                predictions.append(best_label)

        except (TypeError, ValueError) as e:
            print(f"Error during prediction: {e}")
            print("Ensure images are valid.")
            return None

        return predictions # return the list of predicted labels for the input images


    # Building Block 4: Simple Evaluation Functions
    def __compute_accuracy(self, true_labels, predicted_labels):
        """
        Compute how many predictions were correct.
        
        Parameters:
        true_labels (list): The actual correct labels
        predicted_labels (list): What our classifier predicted
        
        Returns:
        accuracy (float): Fraction correct (between 0 and 1)
        """
        
        try:

            if not true_labels or not predicted_labels or len(true_labels) != len(predicted_labels): # Check for valid input
                raise ValueError("Invalid true or predicted labels.")

            correct_count = 0
            total_count = len(true_labels) # total number of samples

            for i in range(total_count):
                if true_labels[i] == predicted_labels[i]: # if prediction is correct, increment count
                    correct_count += 1

            accuracy = correct_count / total_count if total_count > 0 else 0.0 # Compute accuracy as the correct predictions divided by total samples

        except (TypeError, ValueError): 
            print("Error computing accuracy. Ensure true and predicted labels are valid lists of the same length.")
            return 0.0

        return accuracy # return the accuracy value of correct predictions


    def __evaluate_classifier(self, test_images, test_labels):
        """
        Complete evaluation of the classifier.
        Simply uses our existing functions to predict, and compute accuracy
        
        Parameters:
        test_images (list): Test images
        test_labels (list): True labels for test images
        lfw_people: The LFW dataset object (for visualization)
        
        Returns:
        accuracy (float): Classification accuracy
        predictions (list): Predicted labels
        """
        
        try:

            if not test_images or not test_labels or len(test_images) != len(test_labels):
                raise ValueError("Invalid test images or labels.")

            # Step 1: Predict labels for test images
            predictions = self.__predict(test_images)

            if predictions is None:
                raise ValueError("Prediction failed during evaluation.")

            # Step 2: Compute accuracy
            accuracy = self.__compute_accuracy(test_labels, predictions)
            if accuracy is None:
                raise ValueError("Accuracy computation failed during evaluation.")
                

        except (TypeError, ValueError):
            print("Error evaluating classifier. Ensure test images and labels are valid.")
            return 0.0, None

        return accuracy, predictions # return accuracy and predictions
    

    def __visualize_predictions(self, test_images, test_labels, predictions, lfw_people, num_samples=30, save_plot=False):
        """
        Visualize test images with their predicted vs true labels.
        
        Parameters:
        test_images (list): Test images (row format - each row is an image)
        test_labels (list): True labels for test images
        predictions (list): Predicted labels
        lfw_people: The LFW dataset object (to get target names and image shape info)
        num_samples (int): Number of sample images to display (default=30)
        save_plot (bool): Whether to save the plot to a file

        Returns:
        None
        """
        
        try:
            if not test_images or not test_labels or not predictions:
                print("Error: Invalid data for visualization.")
                return
                
            if len(test_images) != len(test_labels) or len(test_labels) != len(predictions):
                print("Error: Mismatched lengths in visualization data.")
                return
                
            # Get the image dimensions from the dataset
            img_height, img_width = lfw_people.images.shape[1], lfw_people.images.shape[2]
            target_names = lfw_people.target_names
            
            print(f"Image dimensions: {img_height} x {img_width}")
            print(f"Number of unique people: {len(target_names)}")
            
            # Limit to available samples
            num_samples = min(num_samples, len(test_images))
            
            # Select random samples to display
            sample_indices = random.sample(range(len(test_images)), num_samples)
            
            # Create figure for plotting
            nrows = 5
            ncols = 6
            plt.figure(figsize=(15, 12))

            for i, sample in enumerate(sample_indices):
                # Create a subplot
                plt.subplot(nrows, ncols, i + 1)
                
                # Get the image (already in 1D row format)
                test_image = test_images[sample]
                
                # Reshape the flattened image back to 2D
                image_2d = np.array(test_image).reshape(img_height, img_width)
                
                # Plot the target image
                plt.imshow(image_2d, cmap="gray")
                
                # Get the names for true and predicted labels
                pred_label = target_names[predictions[sample]]
                truth_label = target_names[test_labels[sample]]

                # Create the title text of the plot
                title_text = f"Pred: {pred_label}\nTruth: {truth_label}"
                
                # Check for equality and change title colour accordingly
                if pred_label == truth_label:
                    plt.title(title_text, fontsize=8, c="g")  # green text if correct
                else:
                    plt.title(title_text, fontsize=8, c="r")  # red text if wrong
                
                plt.axis(False)
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_plot:
                filename = f"pca_classification_results_{num_samples}_samples.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Visualization saved as: {filename}")
            
            plt.show()
            
            # Print summary statistics
            correct_count = sum(1 for i in range(len(test_labels)) if test_labels[i] == predictions[i])
            accuracy = correct_count / len(test_labels) if test_labels else 0
            
            print(f"\nVisualization Summary:")
            print(f"Displayed {num_samples} sample predictions")
            print(f"Overall accuracy: {accuracy * 100:.2f}%")
            print(f"Correct predictions: {correct_count}/{len(test_labels)}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Continuing without visualization...")


    # Building Block 5: Main Classification Function
    def image_classification(self, n_components=None, k=None, variance_threshold=None):
        """
        Main function that puts all building blocks together.
        
        Parameters:
        n_components (int): Number of PCA components to use (uses current setting if None)
        k (int): Number of nearest neighbors for k-NN classification (uses current setting if None)
        variance_threshold (float): Variance threshold for automatic selection (uses current setting if None)

        Returns:
        None
        """
        # Use current values if not provided
        if n_components is None:
            n_components = self.get_n_components() # from PCA base class
        if k is None: # use current k value
            k = self.get_k() # from this class
        if variance_threshold is None:
            variance_threshold = self.get_variance_threshold()

        # Update settings if user provided new values
        self.set_n_components(n_components)
        self.set_k(k)
        self.set_variance_threshold(variance_threshold)
            
        print("=== Simple PCA Image Classification with k-NN ===")
        print(f"Using k={k} nearest neighbors")
        print()

        # Step 2: Load the dataset
        print("Step 1: Loading dataset...")
        images, labels, lfw_people = self.__load_faces_dataset()

        # Step 3: Normalize images
        print("Step 2: Normalizing images...")
        images = self.__normalize_images(images)
        # Step 4: Split into train and test
        print("Step 3: Splitting data...")
        train_images, train_labels, test_images, test_labels = self.__split_data_simple(images, labels)

        # Step 5: Train classifier
        print("Step 4: Creating classifier...")
        classifier = PCAImageClassifier(n_components=n_components, k=k)
        
        print("Step 5: Training classifier...")
        # Train the k-NN classifier
        classifier.__train(train_images, train_labels)

        # Step 6: Evaluate on test data
        print("Step 6: Evaluating...")

        accuracy, predictions = classifier.__evaluate_classifier(test_images, test_labels)
        if accuracy is not None:
            print(f"Accuracy: {accuracy * 100:.2f}%")

        # Step 7: Visualize results
        print("Step 7: Creating visualization...")

        if predictions is not None and test_images and test_labels: # Ensure we have data to visualize
            classifier.__visualize_predictions(test_images, test_labels, predictions, lfw_people, num_samples=30, save_plot=True)
        else:
            print("Skipping visualization due to missing data.")

        # Step 8: Summary
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        print(f"Dataset: LFW People")
        print(f"Total samples: {len(images) if images else 0}")
        print(f"Training samples: {len(train_images) if train_images else 0}")
        print(f"Test samples: {len(test_images) if test_images else 0}")
        print(f"Features per image: {len(images[0]) if images and images[0] else 0}")
        print(f"PCA components: {self.get_n_components()}")  # Use actual final number of components

        variance_ratios = self.explained_variance_ratio() # get variance ratios from PCA base class
        if variance_ratios:
            # variance_ratios already corresponds to the fitted/selected components
            # Extract the list of variance ratios from the tuple (first element)
            if isinstance(variance_ratios, tuple) and len(variance_ratios) > 0: # Check for tuple and non-empty
                variance_list = variance_ratios[0]

                if isinstance(variance_list, list):
                    cumulative_variance = sum(variance_list)
                    print(f"Variance retained: {cumulative_variance*100:.1f}%")

        print(f"k-NN parameter: k={k}")
        print(f"Final accuracy: {accuracy * 100:.2f}%" if accuracy is not None else "Final accuracy: N/A")
        
        # Show class distribution
        if train_labels:
            unique_classes = list(set(train_labels))
            print(f"Number of different people: {len(unique_classes)}")
        
        print("Classification completed!")


if __name__ == "__main__":
    # You can adjust these parameters:
    # - n_components: Number of PCA dimensions (higher = more detail, slower)
    # - k: Number of neighbors to consider (odd numbers work best to avoid ties)
    # - variance_threshold: Set to 0.0 to disable automatic selection
    classifier = PCAImageClassifier(n_components=10, k=9, variance_threshold=0.70)
    classifier.image_classification(n_components=10, k=9, variance_threshold=0.70)