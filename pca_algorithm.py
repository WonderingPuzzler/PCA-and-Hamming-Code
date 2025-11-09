import random
from tqdm import tqdm
from vector_matrix_operations import *

class PCA(VectorMatrixOperations):
    """
    A class to perform Principal Component Analysis (PCA) on a dataset. 
    Borrows vector and matrix operations from VectorMatrixOperations.
    
    Component Matrix :
    This PCA implementation stores components in standard (n_components, n_features) format:
    - ROWS: Each row represents one principal component (eigenvector)
    - COLUMNS: Each column position represents one feature's weight within each component
    - A feature is a specific measurable property or characteristic of the data being analyzed.
    - Component weights/features: The numerical coefficients in the component matrix that determine how much each original feature (pixel) contributes to each principal component. Each weight represents the importance of a specific pixel in defining a particular component.
    - A component is a new axis in the transformed feature space that captures maximum variance.
    - Maximizing variance means that the components are oriented in such a way that they capture the most significant patterns and variations in the data.
    
    Mathematical Interpretation:
    - components[i] = weights for component i across all features
    - components[i][j] = how much feature j contributes to component i
    - For transformation: X_new = X_centered matrix multiply components.T
    """

    def __init__(self, n_components, mean=None, components=None, variance_eigenvalues=None, fitted=False):
        """
        Initialize the PCA algorithm with the number of components to keep.
        

        Parameters:
        n_components (int): The number of principal components to retain. 
        components (list of lists): The principal components matrix in (n_components, n_features) format.
                                   Each row is a component direction, each column is a feature weight.
        variance_eigenvalues (list): The variance explained by each component. Variance is represented by the eigenvalues of the covariance matrix and measures how much information (variance) each principal component captures from the data.
        fitted (bool): Whether the model has been fitted to data. Fitted means that the principal components and variance have been computed from the data.
        mean (list): The mean of the data. 

        Returns:
        None
        """
        self.set_n_components(n_components) # number of principal components to retain
        self.set_components(components)  # principal components after fitting
        self.set_variance_eigenvalues(variance_eigenvalues) # variance explained by each component
        self.set_fitted(fitted) # whether the model has been fitted
        self.set_mean(mean) # mean of the data

    def get_n_components(self):
        """
        Get the number of principal components to retain.

        Parameters:
        None

        Returns:
        int: The number of principal components.
        """
        return self.__n_components
    
    def get_components(self):
        """
        Get the principal components matrix in (n_components, n_features) format.

        Parameters:
        None

        Returns:
        list of lists: The principal components matrix where:
                      - Each ROW is a principal component (eigenvector direction)
                      - Each COLUMN represents feature weights across components
                      - components[i][j] = contribution of feature j to component i
        """
        return self.__components
    
    def get_variance_eigenvalues(self):
        """
        Get the variance explained by each component.

        Parameters:
        None

        Returns:
        list: The variance explained by each component.
        """
        return self.__variance_eigenvalues
    
    def is_fitted(self):
        """
        Check if the model has been fitted to data.

        Parameters:
        None

        Returns:
        bool: True if the model is fitted, False otherwise.
        """
        return self.__fitted
    
    def get_mean(self):
        """
        Get the mean of the data.

        Parameters:
        None

        Returns:
        list: The mean of the data.
        """
        
        return self.__mean
    
    def set_n_components(self, n_components):
        """
        Set the number of principal components to retain.

        Parameters:
        n_components (int): The number of principal components.

        Returns:
        None
        """
        try:
            if not isinstance(n_components, int) or n_components <= 0: # Check if n_components is a positive integer and greater than zero
                raise ValueError()
            
        except (TypeError, ValueError):
            print("n_components must be a positive integer.")
            return
        
        self.__n_components = n_components

    def set_components(self, components):
        """
        Set the principal components matrix in standard (n_components, n_features) format.

        Parameters:
        components (list of lists): The principal components as (n_components, n_features) matrix.
                                  Each row is a component, each column is a feature.

        Returns:
        None
        """
        try:
            if components is not None:

                if not isinstance(components, list) or not components: # Check if it's a non-empty list
                    raise ValueError("Components must be a non-empty list.")
                
                # Validate that all components are proper rows (1D lists)
                for i, component in enumerate(components): # iterate over each component

                    if not isinstance(component, list) or not component: # Check if it's a non-empty list
                        raise ValueError(f"Component {i} must be a non-empty list.")
                    
                    # Validate all values are numerical
                    for j, value in enumerate(component):
                        if not isinstance(value, (int, float)): # Check if each value is numerical
                            raise ValueError(f"All component values must be numerical. Found {type(value)} at component {i}, position {j}.")
                        
        except (TypeError, ValueError) as e:
            print(f"Input validation error for components: {e}")
            return
        
        self.__components = components

    def set_variance_eigenvalues(self, variance_eigenvalues):
        """
        Set the variance explained by each component.

        Parameters:
        variance_eigenvalues (list): The variance explained by each component.

        Returns:
        None
        """
        try:
            if variance_eigenvalues is not None: # Check if variance eigenvalues is provided

                if not isinstance(variance_eigenvalues, list): # Check if it's a list
                    raise ValueError("Variance eigenvalues must be a list.")
                
                # Check if all values are numerical
                for i, value in enumerate(variance_eigenvalues): 

                    if not isinstance(value, (int, float)): # All values must be numerical (int (0) or float (0.0))
                        raise ValueError(f"All variance eigenvalues must be numerical. Found {type(value)} at position {i}.")
                    
        except (TypeError, ValueError) as e:
            print(f"Input validation error for variance_eigenvalues: {e}")
            return
        
        self.__variance_eigenvalues = variance_eigenvalues

    def set_fitted(self, fitted):
        """
        Set whether the model has been fitted to data.

        Parameters:
        fitted (bool): True if the model is fitted, False otherwise.

        Returns: 
        None
        """
        try:
            if not isinstance(fitted, bool): # Check if fitted is a boolean value (True or False)
                raise ValueError("Fitted flag must be a boolean value.")
            
        except (TypeError, ValueError):
            print("Fitted flag must be a boolean value.")
            return
        
        self.__fitted = fitted
    
    def set_mean(self, mean):
        """
        Set the mean of the data.

        Parameters:
        mean (list of lists): The mean of the data as 2D column vector only.
        """
        try:
            if mean is not None:

                # Only accept proper 2D column vectors
                if not self.is_column_vector(mean):
                    raise ValueError("Mean must be a proper 2D column vector format [[val1], [val2], ...]")
                
                # Check if all values are numerical
                for i, row in enumerate(mean):
                    
                    if not isinstance(row[0], (int, float)): # Check if each value is numerical (int (0) or float (0.0))
                        raise ValueError(f"All mean values must be numerical. Found {type(row[0])} at position {i}.")
                    
        except (TypeError, ValueError) as e:
            print(f"Input validation error for mean: {e}")
            return
        
        self.__mean = mean

    # PCA Specific Operations
    def __mean_calculation(self, X):
        """
        Compute the mean of each column/feature in the dataset.
        
        Parameters:
        X (list of lists): The input data.
        
        Returns:
        mean (list of lists): The mean of each column/feature as a 2D column vector.
        """

        try:

            if not X or not X[0]: # handle empty data case
                raise ValueError

            length_rows = len(X) # number of rows/samples
            width_columns = len(X[0]) # number of columns/features
            mean = [0.0 for _ in range(width_columns)] # create a list to hold the mean of each column/feature

            # Add progress bar for mean computation and get means
            for column in tqdm(range(width_columns), desc="Computing means", leave=False): # iterate over each column/feature and show progress with a progress bar (leave =False to avoid cluttering output)
                for row in range(length_rows):
                    mean[column] += X[row][column] # sum up the column/feature values
                mean[column] /= length_rows # compute the mean for the column/feature

            # Create 2D column vector directly without using conversion function
            mean_column = [[value] for value in mean]

        except (TypeError, IndexError, ValueError): 
            print("Input data X must be a 2D list of numerical values and non-empty.")
            return None

        return mean_column # return the mean as a 2D column vector

    def __center_data(self, X, mean):
        """
        Center the data by subtracting the mean of each feature (column) from each feature (column).
        
        Parameters:
        X (list of lists): The input data to be centered.
        mean (list of lists): The mean of each column/feature as a 2D column vector.
        
        Returns:
        X_centered (list of lists): The centered data.

        """

        try:

            if not X or not X[0]: # handle empty data case
                raise ValueError

            # Ensure mean is not None and is a valid 2D column vector
            if mean is None:
                raise ValueError("Mean cannot be None")

            # Only accept proper 2D column vectors 
            if not self.is_column_vector(mean):
                raise ValueError("Mean must be a proper 2D column vector format [[val1], [val2], ...].")
        
            length_rows = len(X) # number of rows/samples
            width_columns = len(X[0]) # number of columns/features

            self.set_mean(mean) # store the mean for later use (we pass __center_data the mean we computed in __mean_calculation)

            X_centered = [[0 for _ in range(width_columns)] for _ in range(length_rows)]  # create a new matrix to hold the centered data

            # Add progress bar for centering data
            for row in tqdm(range(length_rows), desc="Centering rows", leave=False): # iterate over each row/sample and show progress with a progress bar (leave =False to avoid cluttering output)
                for column in range(width_columns):
                    # Access mean directly from 2D column vector
                    X_centered[row][column] = X[row][column] - mean[column][0] # subtract the mean from each row/feature value
    

        except (TypeError, IndexError, ValueError):
            print("Input data X must be a 2D list of numerical values and non-empty.")
            return None

        return X_centered # return the centered data
    
    def __compute_covariance_matrix(self, X_centered):
        """
        Compute the covariance matrix of the centered data
        Handles arrays of any dimension using pure python structures and functions.
        Does so using Sample covariance formula.
        Uses equation Cov(X,X) = (1/n-1) * X_centered^T * X_centered
        n = number of samples, or in the language of matrices, the number of rows in X_centered
        Uses matrix_multiply function for computation.
        Note the exact equation for the covariance matrix is: Cov(X,X) = sum ((X - mean_X) * (X - mean_X))^T / (n - 1)
        But lets break down why we can use matrix multiplication instead: 

        For example, say basicaly that the centered matrix (with variables x and y) is: [[x1, y1], 
                                                                                         [x2, y2], 
                                                                                         [x3, y3]]
        X_centered^T would be: [[x1, x2, x3],
                                [y1, y2, y3]]
    
        Well, the covariance matrix asks us to compute: [[cov(x,x), cov(x,y)], 
                                                         [cov(x,y), cov(y,y)]]

        However, the rows of x_centered dot produceted with the columns of x_centered^T give us exactly that matrix, not the other way around!

        Therefore, the equation must be: Cov(X,X) = (1/n-1) * X_centered^T * X_centered!

        The covariance matrix is important because it shows us how the different parts of the matrix are related to each other.

        Parameters:
        X_centered (list of lists): The centered data.

        Returns:
        covariance_matrix (list of lists): The computed covariance matrix.
        """

        try:

            if not X_centered or not X_centered[0]: # handle empty data case
                raise ValueError

            length_rows = len(X_centered) # number of samples
            
            # Handle single sample case (avoid division by zero)
            # If we have only one sample, covariance matrix is undefined, so we return a zero matrix instead
            if length_rows <= 1:
                # For single sample, return zero covariance matrix
                n_features = len(X_centered[0])
                covariance_matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
                return covariance_matrix
            
            X_centered_T = self.transpose(X_centered) # transpose the centered data

            # Compute X_centered^T * X_centered
            covariance_matrix = self.matrix_multiply(X_centered_T, X_centered)
            
            # Multiply by 1/(n-1) (scalar multiplication)
            covariance_matrix = self.scalar_matrix_multiply(1.0 / (length_rows - 1), covariance_matrix)

        except (TypeError, IndexError, ValueError): 
            print("Input data X_centered must be a 2D list of numerical values and non-empty.")
            return None
        
        return covariance_matrix

    def __rayleigh_quotient(self, A, b):
        """
        Compute the Rayleigh quotient for matrix A and vector b.
        The Rayleigh quotient is defined as R(A, b) = (b^T * A * b) / (b^T * b).
        We use the Rayleigh quotient to estimate the dominant eigenvalue of a matrix given an approximate eigenvector.

        Parameters:
        A (list of lists): The input square matrix.
        b (list): The input vector.

        Returns:
        rayleigh_quotient (float): The computed Rayleigh quotient.
        """

        if not A or not A[0] or not b or len(b) == 0: # handle empty matrix or vector case
            raise ValueError
        
        try:
            Ab = self.matrix_vector_multiply(A, b) # compute A * b

            b_t_Ab = self.vector_dot_product(b, Ab) # compute b^T * A * b (using dot product equivalence)

            b_t_b = self.vector_dot_product(b, b) # compute b^T * b (using dot product equivalence)

            if b_t_b is None or b_t_Ab is None:
                raise ValueError("Error computing dot products.")
            elif b_t_b == 1e-15: # handle zero denominator case (float arithmetic (decimals) is rarely exact, so we use a small threshold instead of exact zero)
                raise ValueError("Denominator in Rayleigh quotient is zero.")
            
            rayleigh_quotient = b_t_Ab / b_t_b # compute the Rayleigh quotient

            return rayleigh_quotient # return the Rayleigh quotient

        except (TypeError, IndexError, ValueError):
            print("Input matrix A must be a 2D list of numerical values and vector b must be a list of numerical values.")
            return None

    # Eigen Decomposition Computations
    def __power_iteration(self, A, num_iterations= 50, tolerance=0.999999):
        """
        Perform power iteration to find the dominant eigenvalue and eigenvector of matrix A.
        We perform power iteration because it is an efficient algorithm for finding the largest eigenvalue and corresponding eigenvector of a matrix,
        at least ones as large as the ones we're dealing with.
        In image classification and PCA, the dominant eigenvector corresponds to the direction of maximum variance in the data, 
        which is crucial for dimensionality reduction and feature extraction.
        Dimensionality reduction helps in reducing computational costs and improving model performance 
        by eliminating redundant features while retaining the most informative aspects of the data.
        The power iteration equation used is: b_{k} = A * b_{k-1} / ||A * b_{k-1}||,
        
        Parameters:
        A (list of lists): The input square matrix.
        num_iterations (int): The maximum number of iterations to perform.
        tolerance (float): The convergence tolerance. Tolerance is used to determine when the algorithm has converged to a stable solution.
        
        Returns:
        eigenvalue (float): The dominant eigenvalue of the matrix.
        eigenvector (list): The dominant eigenvector of the matrix.
        """

        try: 

            if not A or not A[0]: # handle empty matrix case
                raise ValueError
            
            n = len(A) # number of rows
            
            # Check if matrix is all zeros (no variance case)
            # No variance means all eigenvalues are zero
            # This is important to handle to avoid division by zero errors
            all_zero = True
            for i in range(n):
                for j in range(n):
                    if abs(A[i][j]) > 1e-15: # handle non-zero case (float arithmetic (decimals) is rarely exact, so we use a small threshold instead of exact zero)
                        all_zero = False
                        break
                if not all_zero:
                    break
            
            if all_zero: # Because all eigenvalues are zero, we can directly return zero eigenvalue and a unit vector instead of running the full power iteration
                # Return zero eigenvalue and a unit column vector
                eigenvector_b = [[1.0] if i == 0 else [0.0] for i in range(n)]
                return 0.0, eigenvector_b
            
            # Initialize a random column vector which will converge to the dominant eigenvector
            eigenvector_b = [[random.random()] for _ in range(n)]
            eigenvector_b = self.normalize_vector(eigenvector_b)  # Normalize the initial vector - we normalize to avoid numerical instability

            # Progress bar for power iteration
            pbar = tqdm(range(num_iterations), desc="Power iteration", leave=False)
            for i in pbar:
                # calculate the matrix-by-vector product Ab
                Ab = self.matrix_vector_multiply(A, eigenvector_b)

                # Normalize the vector Ab so we can get the next approximation of the eigenvector
                eigenvector_b_next = self.normalize_vector(Ab)
                
                # Handle the case where normalization fails (zero vector)
                if eigenvector_b_next is None:
                    # If we get a zero vector, the eigenvalue is 0, return column vector
                    eigenvector_b = [[1.0] if i == 0 else [0.0] for i in range(n)] # this vector is arbitrary since eigenvalue is 0
                    return 0.0, eigenvector_b
                

                convergence_check = self.vector_dot_product(eigenvector_b_next, eigenvector_b) # This checks how much the eigenvector has changed from the last iteration
                if convergence_check is None:
                    raise ValueError("Error computing convergence check.")
                # Check for convergence and stop if converged, meaning the change is below the tolerance
                
                # Update progress bar description with convergence info
                pbar.set_postfix({'convergence': f'{abs(convergence_check):.6f}'})
                
                # Check if the change is below the tolerance 
                # (float arithmetic is rarely exact, so we use a small threshold instead of exact zero)
                if abs(convergence_check) >= tolerance: 
                    pbar.set_description("Power iteration (converged)")
                    pbar.close()
                    break

                eigenvector_b = eigenvector_b_next
            
            # Close progress bar if loop completed without early break
            if not pbar.disable:
                pbar.close()

            # Compute the Rayleigh quotient to estimate the eigenvalue, which gives us the dominant eigenvalue
            # the dominant eigenvalue corresponds to the direction of maximum variance in the data
            # and is crucial for dimensionality reduction and feature extraction in PCA
            eigenvalue = self.__rayleigh_quotient(A, eigenvector_b)

            return eigenvalue, eigenvector_b # return the dominant eigenvalue and eigenvector

        except (TypeError, IndexError, ValueError):
            print("Input matrix A must be a 2D list of numerical values and non-empty.")
            return None, None

    def __deflate_matrix(self, A, eigenvalue, b):
        """
        Deflate the matrix to find the next dominant eigenvalue and eigenvector.
        Deflation is used to remove the influence of the already found dominant eigenvalue and eigenvector from the matrix.
        This allows us to find the next dominant eigenvalue and eigenvector in subsequent iterations.
        Uses equation A_deflated = A - λ * (b * b^T)
        The equation is also known as Hotelling's deflation.
        where λ is the dominant eigenvalue, b is the dominant eigenvector, and A_deflated is the deflated matrix.

        Parameters:
        A (list of lists): The input square matrix.
        eigenvalue (float): The dominant eigenvalue.
        b (list): The dominant eigenvector.

        Returns:
        A_deflated (list of lists): The deflated matrix.
        """

        try:
            if not A or not A[0]: # handle empty matrix or vector case
                raise ValueError

            if eigenvalue is None or b is None or len(b) == 0: # handle invalid eigenvalue or eigenvector case
                raise ValueError("Eigenvalue and eigenvector must be valid.")

            # # compute the outer product b * b^T (note that by doing outer product,
            # we don't need to transpose b, as outer product is a calculation that inherently handles the transposition)
            outer_prod = self.outer_product(b, b)
            scaled_outer_prod = self.scalar_matrix_multiply(eigenvalue, outer_prod) # scale the outer product by the eigenvalue λ

            A_deflated = self.matrix_subtract(A, scaled_outer_prod) # compute the deflated matrix A_new = A - λ * (b * b^T)


        except (TypeError, IndexError, ValueError): 
            print("Input matrix A must be a 2D list of numerical values and non-empty. Eigenvalue must be a numerical value and eigenvector must be a list of numerical values.")
            return None
        
        return A_deflated # return the deflated matrix

    def __find_top_eigenpairs(self, covariance_matrix, num_components):
        """
        Find the top 'num_components' eigenvalue-eigenvector pairs using power iteration with deflation.
        
        This method uses an iterative approach:
        1. Use power iteration to find the dominant (largest) eigenvalue and eigenvector of the covariance matrix.
        2. Deflate the covariance matrix to remove the influence of the found eigenvalue and eigenvector.
        3. Repeat the process to find the next dominant eigenvalue and eigenvector until we have found 'num_components' of them.
        4. Return the list of top eigenvalues and eigenvectors.
        
        
        Parameters:
        covariance_matrix (list of lists): The covariance matrix of the data.
        num_components (int): The number of top eigenvalues and eigenvectors to compute.

        Returns:
        eigenvalues (list): The list of top eigenvalues.
        eigenvectors (list of lists): The list of top eigenvectors.
        """

        try:

            if not covariance_matrix or not covariance_matrix[0]: # handle empty matrix case
                raise ValueError

            eigenvalues = [] # list to hold the top eigenvalues
            eigenvectors = [] # list to hold the top eigenvectors
            A_current = covariance_matrix # start with the original covariance matrix

            # Progress bar for finding components
            component_pbar = tqdm(range(num_components), desc="Finding principal components", leave=False)

            for component in component_pbar: # iterate to find the top 'num_components' eigenvalue-eigenvector pairs

                component_pbar.set_description(f"Finding component {component + 1}/{num_components}") # update description with current component number
                eigenvalue, eigenvector = self.__power_iteration(A_current) # find the dominant eigenvalue and eigenvector using power iteration

                eigenvalues.append(eigenvalue) # append the found eigenvalue to the list
                eigenvectors.append(eigenvector)  # append the found eigenvector to the list
                
                # Update progress bar with eigenvalue info
                component_pbar.set_postfix({'eigenvalue': f'{eigenvalue:.4f}' if eigenvalue is not None else 'None'})

                A_current = self.__deflate_matrix(A_current, eigenvalue, eigenvector) # deflate the matrix to find the next dominant eigenvalue and eigenvector

        except (TypeError, IndexError, ValueError):
            print("Input covariance_matrix must be a 2D list of numerical values and non-empty.")
            return None, None

        return eigenvalues, eigenvectors # return the list of top eigenvalues and eigenvectors

    def explained_variance_ratio(self):
        """
        Calculate the ratio of variance explained by each component.
        Variance ratio shows how much information (variance) each principal component captures from the data.

        Parameters:
        None
        
        Returns:
        variance_ratios (list): List of variance ratios for each component
        total_variance (float): Total variance in the dataset
        """
        if not self.is_fitted() or self.get_variance_eigenvalues() is None: # Check if PCA is fitted
            print("PCA must be fitted before calculating explained variance ratios.")
            return None, None
            
        eigenvalues = self.get_variance_eigenvalues() # get the eigenvalues representing variance
        total_variance = sum(eigenvalues) # compute total variance
        
        if total_variance == 0: # handle zero variance case to avoid division by zero
            return [0.0] * len(eigenvalues), 0.0 # Because all variance is zero, all ratios are zero
            
        variance_ratios = [eigenval / total_variance for eigenval in eigenvalues] # compute variance ratio for each component

        return variance_ratios, total_variance # return the variance ratios and total variance

    def select_components_by_variance(self, variance_threshold=60.0):
        """
        Automatically select the number of components needed to retain a specified
        percentage of total variance.
        
        Parameters:
        variance_threshold (float): Desired percentage of variance to retain (0-100).
        
        Returns:
        optimal_components (int): Number of components needed
        cumulative_variance (float): Actual cumulative variance retained
        """

        if not self.is_fitted() or self.get_variance_eigenvalues() is None: # Check if PCA is fitted
            print("PCA must be fitted before selecting components by variance.")
            return None, None
            
        variance_ratios, _ = self.explained_variance_ratio() # get variance ratios (we ignore total variance here because we only need ratios, good practice is to use _ for unused variables)
        
        if variance_ratios is None: # handle error case
            return None, None
            
        cumulative_variance = 0.0
        optimal_components = 0
        
        for i, ratio in enumerate(variance_ratios): # iterate over variance ratios
            cumulative_variance += ratio # accumulate variance ratio
            optimal_components = i + 1 # update number of components needed
            
            if cumulative_variance >= variance_threshold: # check if we have reached the desired variance threshold (meaning if cumulative variance is greater than or equal to threshold)
                break
                
        return optimal_components, cumulative_variance # return the number of components needed and the actual cumulative variance retained

    # PCA Fitting and Transformation
    def fit(self, X):
        """
        Fit the PCA model to the data X.
        Fitting the PCA model involves computing the mean of the data, centering the data, computing the covariance matrix,
        and performing eigen decomposition to find the principal components and explained variance.
        By fitting the PCA model, we can later transform new data into the PCA space for dimensionality reduction and feature extraction.
        
        Parameters:
        X (list of lists): The input data to fit the PCA model.
        
        Returns:
        components (list of lists): The principal components of the fitted PCA model.
        """

        try:

            if not X or not X[0]: # handle empty data case
                raise ValueError

            print(f"Fitting PCA with {len(X)} samples and {len(X[0])} features...")
            
            # Progress through main PCA fitting steps
            with tqdm(total=4, desc="PCA Fitting", leave=False) as pbar:
                # Step 1: Compute the mean of each column/feature
                pbar.set_description("Computing mean")
                mean = self.__mean_calculation(X)
                pbar.update(1)

                # Step 2: Center the data by subtracting the mean
                pbar.set_description("Centering data")
                X_centered = self.__center_data(X, mean)
                pbar.update(1)

                # Step 3: Compute the covariance matrix of the centered data
                pbar.set_description("Computing covariance matrix")
                covariance_matrix = self.__compute_covariance_matrix(X_centered)
                pbar.update(1)

                # Step 4: Find the top 'n_components' eigenvalue-eigenvector pairs using power iteration with deflation
                pbar.set_description("Finding top eigenpairs")
                eigenvalues, eigenvectors = self.__find_top_eigenpairs(covariance_matrix, self.get_n_components())
                pbar.update(1)

            # Check if eigenvectors were computed successfully
            if eigenvectors is None:
                raise ValueError("Failed to compute principal components.")

            # Convert column vectors to row vectors for standard (n_components, n_features) storage
            components_matrix = []
            for eigenvector_col in eigenvectors:
                # Convert column vector [[a], [b], [c]] to row [a, b, c] 
                # We do this because:
                # - Power iteration returns eigenvectors as column vectors (mathematical convention)
                # - Standard PCA stores components as ROWS in (n_components, n_features) matrix
                # - Each ROW becomes a principal component direction in feature space
                # - Each COLUMN represents how a specific feature contributes across components
                component_row = [row[0] for row in eigenvector_col]
                components_matrix.append(component_row)

            self.set_components(components_matrix)  # set the principal components in standard row format
            self.set_variance_eigenvalues(eigenvalues)  # set the explained variance
            self.set_fitted(True)  # mark the model as fitted

            return self.get_components() # return the principal components (eigenvectors, or directions of maximum variance)

        except (TypeError, IndexError, ValueError):
            print("Input data X must be a 2D list of numerical values and non-empty.")
            return None
    
    def transform(self, X):
        """
        Transform the data X using the fitted PCA model.
        The transformation involves centering the data and projecting it onto the principal components.
        By transforming the data, we can reduce its dimensionality while retaining the most important features.
        These important features can then be used for further analysis or machine learning tasks.
        The equation used is: X_transformed = X_centered * components^T,
        
        Parameters:
        X (list of lists): The input data to be transformed.
        
        Returns:
        X_transformed (list of lists): The transformed data in the PCA space.
        """

        try:

            if not self.is_fitted(): # ensure the model is fitted before transforming
                print("PCA model is not fitted yet. Fitting now...")
                self.fit(X)

            if not X or not X[0]: # handle empty data case
                raise ValueError

            print(f"Transforming {len(X)} samples to {self.get_n_components()} dimensions...") # display transformation info 
            
            # Progress through transformation steps
            with tqdm(total=3, desc="PCA Transform", leave=False) as pbar:

                # Step 1: Center the data by subtracting the mean.
                # We center the data because PCA assumes that the data is centered around the origin.
                pbar.set_description("Centering data")
                X_centered = self.__center_data(X, self.get_mean())
                pbar.update(1) # update progress bar

                if X_centered is None:
                    raise ValueError("Failed to center data.")

                # Step 2: Project the centered data onto the principal components
                # Matrix Multiplication Explanation:
                # X_centered is (n_samples, n_features) - each row is a data sample
                # components is (n_components, n_features) - each row is a component direction
                # We need components.T which is (n_features, n_components) for multiplication

                # Result: X_transformed = X_centered @ components.T
                # - Each row of X_transformed is a transformed sample
                # - Each column of X_transformed is projection onto one component
                # - components.T[:, i] gives the i-th component direction as a column
                pbar.set_description("Preparing components transpose")
                
                # Components are already in (n_components, n_features) format
                # We need transpose to get (n_features, n_components) for multiplication
                components_T = self.transpose(self.get_components())
                pbar.update(1)
                
                if components_T is None: # handle error case
                    raise ValueError("Failed to transpose components.")

                pbar.set_description("Computing matrix multiplication") # update progress bar description

                X_transformed = self.matrix_multiply(X_centered, components_T) # project the centered data onto the principal components

                pbar.update(1)

            return X_transformed # return the transformed data in PCA space

        except (TypeError, IndexError, ValueError):
            print("Input data X must be a 2D list of numerical values and non-empty.")
            return None

    def step_by_step_visualize(self, X):
        """
        Show step by step what PCA is doing to the input data.
        This function demonstrates each step of the PCA algorithm with output.
        
        Parameters:
        X (list of lists): The input data matrix to analyze.
        
        Returns:
        X_transformed (list of lists): The final transformed data.
        """
        
        if not X or not X[0]: # handle empty data case
            print("Error: Input data must be a 2D list and non-empty.")
            return None
            
        print("PCA Step-by-Step Analysis")
        print("=" * 40)
        
        # Display original data
        print("Step 1: Original Data")
        print(f"Data shape: {len(X)} samples x {len(X[0])} features") # Display the shape (or dimensions) of the data
        print("Data matrix:")

        for i, row in enumerate(X): # This loop is basically just allowing us to print the data nicely
            formatted_row = [f"{val:.3f}" for val in row] # format each value to 3 decimal places
            print(f"  Sample {i + 1}: {formatted_row}")
        print()
        
        # Step 1: Compute mean
        print("Step 2: Calculate Mean")
        mean = self.__mean_calculation(X)

        if mean is None: # handle error case (basically for typing errors)
            print("Error computing mean.")
            return None
        
        # Convert 2D column vector to 1D for display formatting
        if self.is_column_vector(mean):
            formatted_mean = [f"{row[0]:.3f}" for row in mean] # format each mean value to 3 decimal places
        else:
            formatted_mean = [f"{m:.3f}" for m in mean] # format each mean value to 3 decimal places
        print(f"Mean of each feature: {formatted_mean}")
        print()
        
        # Step 2: Center the data
        print("Step 3: Center Data (subtract mean)")
        X_centered = self.__center_data(X, mean)

        if X_centered is None: # handle error case (basically for typing errors)
            print("Error centering data.")
            return None
        
        print("Centered data matrix:") 
        for i, row in enumerate(X_centered): 
            formatted_row = [f"{val:.3f}" for val in row] # format each value to 3 decimal places
            print(f"  Sample {i + 1}: {formatted_row}")
        print()
        
        # Step 3: Compute covariance matrix
        print("Step 4: Compute Covariance Matrix")
        covariance_matrix = self.__compute_covariance_matrix(X_centered)

        if covariance_matrix is None: # handle error case (basically for typing errors)
            print("Error computing covariance matrix.")
            return None
        
        print("Covariance matrix:")

        for i, row in enumerate(covariance_matrix): # For printing covariance matrix nicely
            formatted_row = [f"{val:.6f}" for val in row] # format each value to 6 decimal places
            print(f"  Row {i + 1}: {formatted_row}")
        print()
        
        # Step 4: Perform eigen decomposition
        print("Step 5: Find Principal Components")
        
        # Create a temporary PCA instance to fit on the data we've processed so far
        temp_pca = PCA(n_components=self.get_n_components())


        # Use the fit method to compute eigenvalues and eigenvectors
        components = temp_pca.fit(X)
        
        if components is None: # handle error case (basically for typing errors)
            print("Error in PCA fitting.")
            return None
            
        # Copy the results to this instance
        self.set_components(temp_pca.get_components())
        self.set_variance_eigenvalues(temp_pca.get_variance_eigenvalues())
        self.set_fitted(True)

        if self.get_variance_eigenvalues() is None or self.get_components() is None: # handle the case where fitting failed
            print("Error: Failed to compute PCA components.")
            return None
            
        print("Eigenvalues (variance explained):")
        for i, val in enumerate(self.get_variance_eigenvalues()): #
            print(f"  Component {i + 1}: {val:.6f}") # format each eigenvalue to 6 decimal places
        
        print("Eigenvectors (principal components):")
        print("Component Matrix Format: Each row = component, each column = feature weight")

        for i, vec in enumerate(self.get_components()):
            # Components are stored as row vectors [a, b, c] where:
            # - This entire row represents component i's direction in feature space  
            # - Each value shows how much the corresponding feature contributes to this component
            formatted_vec = [f"{v:.6f}" for v in vec] # format each component value to 6 decimal places
            print(f"  Component {i + 1} (row {i}): {formatted_vec}")
            print(f"    → Feature contributions: {[f'f{j}={v:.3f}' for j, v in enumerate(vec)]}")
        print()
        
        # Step 5: Transform the data
        print("Step 6: Transform Data to PCA Space")
        
        # Use the transform method to project the data
        X_transformed = self.transform(X)

        if X_transformed is None:
            print("Error transforming data.")
            return None
            
        print("Transformed data matrix:")

        # Display the transformed data
        for i, row in enumerate(X_transformed):
            formatted_row = [f"{val:.6f}" for val in row] # format each value to 6 decimal places
            print(f"  Sample {i + 1}: {formatted_row}")
        print()
        
        # Summary
        print("Summary:")
        print("- Each eigenvalue = variance captured by that principal component")
        print("- Total variance = sum of all eigenvalues = total data variance")
        print("- Variance ratio = (component eigenvalue) / (total variance)")
        print("- This shows what percentage of information each component preserves")
        print()
        
        if self.get_variance_eigenvalues() is not None: # handle case where eigenvalues are available

            # Use the new explained_variance_ratio method instead of manual calculation
            variance_ratios, total_variance = self.explained_variance_ratio()
            
            if variance_ratios is not None: # handle case where variance ratios are available
                print(f"Total variance in dataset: {total_variance:.6f}")
                
                # Display variance explained by each component using the new method
                cumulative_variance = 0.0
                print("Individual component analysis:")
                
                for i, (eigenvalue, ratio) in enumerate(zip(self.get_variance_eigenvalues(), variance_ratios)): # Go through each eigenvalue and its corresponding variance ratio
                    cumulative_variance += ratio # accumulate the variance ratio
                    print(f"  Component {i + 1}: eigenvalue={eigenvalue:.6f}, explains {ratio * 100:.1f}% of variance") # format each eigenvalue to 6 decimal places and ratio to 1 decimal place
                
                print(f"\nCumulative variance explained by all {len(self.get_variance_eigenvalues())} components: {cumulative_variance * 100:.1f}%") # format cumulative variance to 1 decimal place
                
            else:
                print("Error calculating variance ratios.")
        else:
            print("Error: No variance information available.")
        
        return X_transformed

def main():
    # Create sample 3D data for demonstration
    sample_data = [
        [2.5, 2.4, 1.2],
        [0.5, 0.7, 0.3],
        [2.2, 2.9, 1.1],
        [1.9, 2.2, 0.9],
        [3.1, 3.0, 1.5],
        [2.3, 2.7, 1.3],
        [2.0, 1.6, 0.8],
        [1.0, 1.1, 0.5],
        [1.5, 1.6, 0.7],
        [1.1, 0.9, 0.4]
    ]
    
    # Create PCA object and run step by step analysis
    pca = PCA(n_components=3)
    pca.step_by_step_visualize(sample_data)


if __name__ == "__main__":
    main()