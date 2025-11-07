import math
from tqdm import tqdm


class VectorMatrixOperations:
    """
    A class containing basic vector and matrix operations for linear algebra computations.
    This class provides fundamental operations that can be used by other classes for more complex algorithms.
    
    VECTOR REPRESENTATION:
    - Vectors are represented as 2D lists (column vectors): [[a], [b], [c]]
    
    """

    # Helper Functions for Vector Conversion
    def list_to_column_vector(self, vec_list):
        """
        Convert a 1D list to a 2D column vector.
        
        Parameters:
        vec_list (list): 1D list [a, b, c]
        
        Returns:
        list of lists: 2D column vector [[a], [b], [c]]
        """
        if not vec_list:
            return None
        return [[element] for element in vec_list] # wrap each element in its own list
    
    def is_column_vector(self, vec):
        """
        Check if input is a proper column vector (2D list with single column).
        
        Parameters:
        vec (list): Input to check
        
        Returns:
        bool: True if it's a column vector, False otherwise
        """
        if not vec or not isinstance(vec, list): # Check if input is a list
            return False
        if not all(isinstance(row, list) and len(row) == 1 for row in vec): # Check each row is a list of length 1
            return False
        return True

    def column_vector_to_list(self, vec_column):
        """
        Convert a 2D column vector to a 1D list.
        
        Parameters:
        vec_column (list of lists): 2D column vector [[a], [b], [c]]
        
        Returns:
        list: 1D list [a, b, c]
        """
        if not vec_column:
            return None
        if not self.is_column_vector(vec_column):
            return None
        return [row[0] for row in vec_column] # extract the single element from each row

    # Basic Linear Algebra Operations
    def vector_dot_product(self, vec1, vec2):
        """
        Compute the dot product of two column vectors.
        Dot product is defined as vec1^T * vec2.
        
        Parameters:
        vec1 (list of lists): The first input column vector [[a], [b], [c]].
        vec2 (list of lists): The second input column vector [[x], [y], [z]].
        
        Returns:
        dot_product (float): The resulting dot product.
        """

        try:
                
            if not vec1 or not vec2: # handle empty vector case
                raise ValueError

            if len(vec1) != len(vec2): # check if dimensions are compatible
                raise ValueError("Input vectors must be of the same length.")

            dot_product = 0.0 # initialize dot product

            vect1 = self.column_vector_to_list(vec1)

            if vect1 is None: # handle invalid input case
                raise ValueError("First input vector is not a valid column vector or 1D list.")

            for row in range(len(vect1)): # for each element in the vectors
                dot_product += vect1[row] * vec2[row][0] # accumulate the product of corresponding elements by using equation vec1^T * vec2

        except (TypeError, IndexError): # handle invalid input case
            print("Input vectors must be column vectors (2D lists) or 1D lists and non-empty.")
            return None
        
        return dot_product
    

    def scalar_vector_multiply(self, scalar, vector):
        """
        Multiply a column vector by a scalar.
        Multiplication is done by multiplying each element of the vector by the scalar.
        
        Parameters:
        scalar (float): The scalar value.
        vector (list of lists): The input column vector [[a], [b], [c]].
        
        Returns:
        result (list of lists): The resulting column vector after multiplication.
        """

        try:
                
            if not vector: # handle empty vector case
                raise ValueError

            result = [[0.0] for _ in range(len(vector))] 

            for row in range(len(vector)): 
                result[row][0] = scalar * vector[row][0] # perform multiplication for each element in the vector

        except (TypeError, ValueError, IndexError): # handle invalid input case
            print("Input vector must be a column vector (2D list) or 1D list and scalar must be numerical.")
            return None

        return result
    
    def vector_subtract(self, vec1, vec2):
        """
        Subtract one column vector from another.
        Subtraction is done simply by subtracting each element of the second vector 
        from the corresponding element of the first vector.
        
        Parameters:
        vec1 (list of lists): The first input column vector.
        vec2 (list of lists): The second input column vector.
        
        Returns:
        result (list of lists): The resulting column vector after subtraction.
        """

        try:
                
            if not vec1 or not vec2: # handle empty vector case
                raise ValueError

            if len(vec1) != len(vec2): # check if dimensions are compatible
                raise ValueError("Input vectors must be of the same length.")

            result = [[0.0] for _ in range(len(vec1))] # initialize result vector

            for row in range(len(vec1)):
                result[row][0] = vec1[row][0] - vec2[row][0] # compute the difference between corresponding elements

        except (TypeError, ValueError, IndexError): # handle invalid input case
            print("Input vectors must be column vectors (2D lists) or 1D lists and non-empty.")
            return None

        return result
    
    def vector_add(self, vec1, vec2):
        """
        Add two column vectors.
        Addition is done simply by adding each element of the two vectors.
        
        Parameters:
        vec1 (list of lists): The first input column vector.
        vec2 (list of lists): The second input column vector.
        
        Returns:
        result (list of lists): The resulting column vector after addition.
        """

        try:
                
            if not vec1 or not vec2: # handle empty vector case
                raise ValueError

            if len(vec1) != len(vec2): # check if dimensions are compatible
                raise ValueError("Input vectors must be of the same length.")

            result = [[0.0] for _ in range(len(vec1))] 

            for row in range(len(vec1)):
                result[row][0] = vec1[row][0] + vec2[row][0] # compute the sum of corresponding elements

        except (TypeError, ValueError, IndexError): # handle invalid input case
            print("Input vectors must be column vectors (2D lists) or 1D lists and non-empty.")
            return None

        return result
    
    

    # Vector Operations
    def vector_magnitude(self, vector):
        """
        Compute the magnitude/length of a column vector.
        The magnitude is defined as the square root of the sum of the squares of its elements.
        Uses vector_dot_product function for computation.
        
        Parameters:
        vector (list of lists): The input column vector.
        
        Returns:
        magnitude (float): The magnitude of the vector.
        """

        if not vector: # handle empty vector case
            raise ValueError 
        try:
                
            norm_squared = self.vector_dot_product(vector, vector) # compute the dot product of the vector with itself

            if norm_squared is None: # handle error case from dot product
                raise ValueError("Error computing dot product.")
            
            magnitude = math.sqrt(norm_squared) # compute the square root to get the magnitude

        except (TypeError, ValueError, IndexError): # handle invalid input case
            print("Input vector must be a column vector (2D list) or 1D list and non-empty.")
            return None
        
        return magnitude 


    def normalize_vector(self, vector):
        """
        Normalize a column vector to have unit magnitude.
        Uses vector_magnitude function for computation.
        Uses scalar_vector_multiply function for normalization.
        Normalization is done by dividing each element of the vector by its magnitude.
        We normalize a vector so that it is between 0 and 1 in length, which is useful when we want to avoid scaling issues in computations.

        
        Parameters:
        vector (list of lists): The input column vector.
        
        Returns:
        normalized_vector (list of lists): The normalized column vector.
        """

        try:
            if not vector or len(vector) == 0: # handle empty vector case
                raise ValueError

            magnitude = self.vector_magnitude(vector) # compute the magnitude of the vector

            if magnitude is None or magnitude <= 1e-15: # handle zero magnitude case or None return
                raise ValueError("Cannot normalize a zero vector or error computing magnitude.")
            
            normalized_vector = self.scalar_vector_multiply(1 / magnitude, vector) # normalize the vector using equation v_normalized = v / ||v||

        except (TypeError, ZeroDivisionError, ValueError, IndexError):
            print("Input vector must be a column vector (2D list) or 1D list.")
            return None

        return normalized_vector
    

    # Matrix Operations
    def transpose(self, matrix):
        """
        Transpose a given matrix.
        Transposition is done by swapping rows with columns.

        Parameters:
        matrix (list of lists): The input matrix to be transposed.
        
        Returns:
        transposed (list of lists): The transposed matrix.
        """

        try:
            if not matrix or not matrix[0]: # handle empty matrix case
                raise ValueError

            length_rows = len(matrix) 
            width_columns = len(matrix[0]) 
            transposed = [[0 for _ in range(length_rows)] for _ in range(width_columns)] 

            # Add progress bar for large matrix transpose
            row_iterator = tqdm(range(length_rows), desc="Transposing matrix", leave=False)

            for row in row_iterator: # for each row
                for column in range(width_columns): # for each column
                    transposed[column][row] = matrix[row][column] # transpose the element by swapping row and column indices

        except (TypeError, IndexError, ValueError):
            print("Input matrix must be a 2D list of numerical values and non-empty.")
            return None

        return transposed
    
    def matrix_vector_multiply(self, matrix, vector):
        """
        Multiply a matrix by a column vector.
        Uses vector_dot_product function for multiplication.
        Multiplication can be done by taking the dot product of each row of the matrix with the vector.
        
        Parameters:
        matrix (list of lists): The input matrix.
        vector (list of lists): The input column vector [[a], [b], [c]].
        
        Returns:
        result (list of lists): The resulting column vector after multiplication.
        """

        try:
                
            if not matrix or not vector: # handle empty matrix or vector case
                raise ValueError("Input matrix and vector must be non-empty.")
            
            result = [[0.0] for _ in range(len(matrix))]

            if len(matrix[0]) != len(vector): # check if dimensions are compatible
                raise ValueError("Number of columns in the matrix must be equal to the length of the vector.")
            
            for row in range(len(matrix)):
                # Convert matrix row to column vector for dot product
                matrix_row_as_col = self.list_to_column_vector(matrix[row])
                dot_result = self.vector_dot_product(matrix_row_as_col, vector) # compute the dot product of each matrix row with the vector

                if dot_result is not None: # handle error case from dot product
                    result[row][0] = dot_result

        except (TypeError, IndexError, ValueError):
            print("Input matrix must be a 2D list of numerical values and vector must be a column vector.")
            return None
        return result
    
    def matrix_subtract(self, matrix1, matrix2):
        """
        Subtract one matrix from another.
        Subtraction is done by subtracting corresponding elements.
        
        Parameters:
        matrix1 (list of lists): The first input matrix.
        matrix2 (list of lists): The second input matrix.
        
        Returns:
        result (list of lists): The resulting matrix after subtraction.
        """

        try:
            if not matrix1 or not matrix2 or not matrix1[0] or not matrix2[0]: # handle empty matrix case
                raise ValueError
            
            if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]): # check if dimensions are compatible
                raise ValueError("Input matrices must be of the same dimensions.")

            result = [[0.0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))] 

            for row in range(len(matrix1)): # for each row
                for col in range(len(matrix1[0])): # for each column
                    result[row][col] = matrix1[row][col] - matrix2[row][col] # compute the difference between corresponding elements

        except (TypeError, IndexError, ValueError):
            print("Input matrices must be 2D lists of numerical values and non-empty.")
            return None

        return result
    
    def matrix_add(self, matrix1, matrix2):
        """
        Add two matrices.
        Addition is done by adding corresponding elements.
        
        Parameters:
        matrix1 (list of lists): The first input matrix.
        matrix2 (list of lists): The second input matrix.
        
        Returns:
        result (list of lists): The resulting matrix after addition.
        """

        try:
            if not matrix1 or not matrix2 or not matrix1[0] or not matrix2[0]: # handle empty matrix case
                raise ValueError
            
            if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]): # check if dimensions are compatible
                raise ValueError("Input matrices must be of the same dimensions.")

            result = [[0.0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))] 

            for row in range(len(matrix1)):  # for each row
                for col in range(len(matrix1[0])):  # for each column
                    result[row][col] = matrix1[row][col] + matrix2[row][col] # compute the sum of corresponding elements

        except (TypeError, IndexError, ValueError):
            print("Input matrices must be 2D lists of numerical values and non-empty.")
            return None

        return result
    

    def scalar_matrix_multiply(self, scalar, matrix):
        """
        Multiply a matrix by a scalar.
        Uses scalar_vector_multiply function for computation.
        Multiplication is done by multiplying each row of the matrix by the scalar.
        
        Parameters:
        scalar (float): The scalar value.
        matrix (list of lists): The input matrix.
        
        Returns:
        result (list of lists): The resulting matrix after scalar multiplication.
        """

        try:
            if not matrix or not matrix[0]: # handle empty matrix case
                raise ValueError

            result = [[0.0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))] 

            for row in range(len(matrix)):  # for each row
                for col in range(len(matrix[0])):  # for each column
                    # Direct scalar multiplication: simply multiply each element by the scalar
                    result[row][col] = scalar * matrix[row][col]

        except (TypeError, IndexError, ValueError): 
            print("Input matrix must be a 2D list of numerical values and scalar must be a numerical value.")
            return None

        return result
    
    def matrix_multiply(self, A, B):
        """
        Multiply two matrices using the dot product definition of matrix multiplication.

        
        Matrix multiplication C = A × B can be computed as:
        C[i][j] = dot_product(row_i_of_A, column_j_of_B)
        
        This leverages the mathematical fact that each element of the result matrix
        is the dot product of a row from the first matrix with a column from the second matrix.
        
        Parameters:
        A (list of lists): The first input matrix (m × n).
        B (list of lists): The second input matrix (n × p).

        Returns:
        C (list of lists): The resulting matrix after multiplication (m × p).
        """

        try: 
            if not A or not B: # handle empty matrix case
                raise ValueError 
        
            if len(A[0]) != len(B): # check if dimensions are compatible
                raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")
            
            C = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))] 

            # Add progress bar for large matrix multiplications
            row_iterator = tqdm(range(len(A)), desc="Matrix multiply", leave=False) 
            
            for row in row_iterator: # for each row in A
                for col in range(len(B[0])): # for each column in B
                    # Convert matrix row to column vector format for dot product
                    row_vector = self.list_to_column_vector(A[row])
                    
                    # Extract column from B and convert to column vector format
                    column_data = [B[k][col] for k in range(len(B))]
                    column_vector = self.list_to_column_vector(column_data)

                    # Compute dot product using existing robust function
                    # This handles all edge cases and error checking automatically
                    dot_result = self.vector_dot_product(row_vector, column_vector)
                    
                    if dot_result is not None:
                        C[row][col] = dot_result
                    else:
                        raise ValueError("Error computing dot product during matrix multiplication.")

        except (TypeError, IndexError, ValueError):
            print("Input matrices must be 2D lists of numerical values and non-empty.")
            return None

        return C

    # Outer Product
    def outer_product(self, vec1, vec2):
        """
        Compute the outer product of two column vectors.
        Outer product is defined as a matrix where each element (i,j) is the product of vec1[i] and vec2[j].
        For column vectors: vec1 * vec2^T
        
        Parameters:
        vec1 (list of lists): The first input column vector.
        vec2 (list of lists): The second input column vector.
        
        Returns:
        outer_product (list of lists): The resulting matrix from the outer product.
        """

        try:
                
            if not vec1 or not vec2: # handle empty vector case
                raise ValueError

            outer_product = [[0.0 for _ in range(len(vec2))] for _ in range(len(vec1))] 

            for row in range(len(vec1)): # for each row in vec1
                for col in range(len(vec2)): # for each column in vec2
                    # Direct calculation: vec1[i] * vec2[j] for each position (i,j)
                    outer_product[row][col] = vec1[row][0] * vec2[col][0]

        except (TypeError, IndexError, ValueError):
            print("Input vectors must not be empty and must be column vectors or 1D lists.")
            return None

        return outer_product
    

    # Determinant Calculations

    def determinant(self, A):
        """
        Compute the determinant of a square matrix using recursion and Cofactor expansion.
        Cofactor expansion is defined as det(A) = sum over j of (-1)^(i+j) * A[i][j] * det(C(i,j))
        
        Parameters:
        A (list of lists): The input square matrix.
        
        Returns:
        det (float): The determinant of the matrix.
        """

        try:
            if not A or not A[0]: # handle empty matrix case
                raise ValueError

            if len(A) != len(A[0]): # check if the matrix is square
                raise ValueError("Input matrix must be square.")

            # Base case for 1x1 matrix
            if len(A) == 1:
                return A[0][0]  # Determinant is the single element

            # Base case for 2x2 matrix
            if len(A) == 2:
                det = A[0][0] * A[1][1] - A[0][1] * A[1][0] # compute determinant using ad-bc formula
                return det 

            # For larger matrices, use cofactor expansion along the first row
            det = 0.0
            
            # Go through each column in the first row
            for col in range(len(A)):
                # Get the element from the first row
                element = A[0][col]
                
                # Create the C matrix by removing the first row and current column
                C = []
                for row in range(1, len(A)):  # Skip the first row
                    new_row = []
                    for col_idx in range(len(A)): # col_idx stands for the column index, which must be different from col because we are removing that column
                        if col_idx != col:  # Skip the current column
                            new_row.append(A[row][col_idx]) # Add remaining elements to new row
                    C.append(new_row)

                # Calculate the sign: alternates +/- starting with +
                sign = (-1) ** col
                
                # Calculate the cofactor: sign * element * determinant of C
                # Recursive call (C is smaller than A, so we will eventually reach a case where len(C) == 2, 
                # then we can compute the determinant upwards)
                C_det = self.determinant(C)  
                cofactor = sign * element * C_det 

                # Add this cofactor to the total determinant
                det += cofactor

        except (TypeError, IndexError, ValueError):
            print("Input matrix must be a square 2D list of numerical values and non-empty.")
            return None

        return det
    
    def determinant_FF2(self, A):
        """
        Compute the determinant of a square matrix over Finite Field 2 using recursion and cofactor expansion.
        All arithmetic operations are performed modulo 2 (XOR for addition, AND for multiplication).
        
        Parameters:
        A (list of lists): The input square matrix with binary values.
        
        Returns:
        det (int): The determinant of the matrix in GF(2) (0 or 1).
        """
        try:
            if not A or not A[0]: # Check if we were passed an empty matrix
                raise ValueError("Matrix cannot be empty.")

            if len(A) != len(A[0]): # Check if the matrix is square
                raise ValueError("Input matrix must be square.")

            # Base case for 1x1 matrix
            if len(A) == 1:
                return A[0][0] % 2 # Determinant is the single element mod 2

            # Base case for 2x2 matrix in FF(2)
            if len(A) == 2:
                # det = (a*d - b*c) mod 2 in FF(2)
                det = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) % 2
                return det

            # For larger matrices, use cofactor expansion along the first row
            det = 0
            
            # Go through each column in the first row
            for col in range(len(A)):
                # Get the element from the first row
                element = A[0][col] % 2
                
                # Create the C matrix by removing the first row and current column
                C = []
                for row in range(1, len(A)):  # Skip the first row
                    new_row = []
                    for col_idx in range(len(A)): 
                        if col_idx != col:  # Skip the current column
                            new_row.append(A[row][col_idx] % 2) # Ensure elements are mod 2
                    C.append(new_row)
                
                # Calculate the sign: in FF(2), (-1)^col = 1 since -1 = 1 (mod 2)
                # So sign is always 1 in FF(2)
                sign = 1

                # Calculate the cofactor: sign * element * determinant of C (all mod 2)
                minor_det = self.determinant_FF2(C)  # Recursive call
                cofactor = (sign * element * minor_det) % 2

                # Add this cofactor to the total determinant (XOR in FF(2))
                det = (det + cofactor) % 2

        except (TypeError, IndexError, ValueError):
            raise ValueError("Input matrix must be a square 2D list of binary values.")

        return det
    
    def visualize_operations(self):
        """
        Demonstrate all vector and matrix operations with step-by-step visualization.
        This function shows examples of each operation with formatted input and output.

        Parameters:
        None

        Returns:
        None
        """
        print("=" * 60)
        print("Vector and Matrix Operations Visualization")
        print("=" * 60)
        print()
 
        # Part 1: Basic Vector Operations
        print("1: Basic Vector Operations")
        print("-" * 40)
        print()
        
        # Sample vectors for demonstration ( as proper column vectors)
        vec1 = [[3.0], [4.0], [2.0]]
        vec2 = [[1.0], [2.0], [5.0]]
        scalar = 2.5
        
        print("Sample vectors (column vectors):") # display input vectors
        print("  vec1 = ")
        for i, v in enumerate(vec1): # for each row in vec1, we print the value formatted to 3 decimal places, making it look like a column vector
            print(f"    [{v[0]:.3f}]") 
        print("  vec2 = ")
        for i, v in enumerate(vec2): # for each row in vec2, we print the value formatted to 3 decimal places, making it look like a column vector
            print(f"    [{v[0]:.3f}]")
        print(f"  scalar = {scalar}")
        print()
        
        # Vector dot product
        print("1. Vector Dot Product:")
        print("   vec1^T * vec2 (column vector dot product)")
        dot_result = self.vector_dot_product(vec1, vec2) 

        if dot_result is not None: # handle error case from dot product
            print(f"   Result: {dot_result:.6f}")
        print()
        

        # Scalar vector multiplication
        print("2. Scalar Vector Multiplication:")
        print(f"   {scalar} x vec1:") # multiply vec1 by scalar
        scalar_mult_result = self.scalar_vector_multiply(scalar, vec1)

        if scalar_mult_result is not None: # handle error case from scalar multiplication
            for i, v in enumerate(scalar_mult_result): # for each row in the result, we print the value formatted to 3 decimal places, making it look like a column vector
                print(f"     [{v[0]:.3f}]")
        print()
        
        # Vector addition
        print("3. Vector Addition:")
        print("   vec1 + vec2:")
        add_result = self.vector_add(vec1, vec2)
        if add_result is not None: 
            for i, v in enumerate(add_result):
                print(f"     [{v[0]:.3f}]")
        print()
        
        # Vector subtraction
        print("4. Vector Subtraction:")
        print("   vec1 - vec2:")
        sub_result = self.vector_subtract(vec1, vec2)
        if sub_result is not None:
            for i, v in enumerate(sub_result):
                print(f"     [{v[0]:.3f}]")
        print()
        
        # Vector magnitude
        print("5. Vector Magnitude:")
        print("   ||vec1||:")
        mag_result = self.vector_magnitude(vec1)
        if mag_result is not None:
            print(f"   Result: {mag_result:.6f}")
        print()
        
        # Vector normalization
        print("6. Vector Normalization:")
        print("   normalize(vec1) = vec1 / ||vec1||:")

        norm_result = self.normalize_vector(vec1)

        if norm_result is not None: # handle error case from normalization
            for i, v in enumerate(norm_result): # for each row in the result, we print the value formatted to 6 decimal places, making it look like a column vector
                print(f"     [{v[0]:.6f}]")

            # Verify it's unit length
            norm_mag = self.vector_magnitude(norm_result) 
            if norm_mag is not None: # handle error case from magnitude
                print(f"   Verification: ||normalized|| = {norm_mag:.6f}") # should be 1.0
        print()
        

        # Part 2: Matrix Operations
        print("Part 2: Matrix Operations")
        print("-" * 40)
        print()
        
        # Sample matrices for demonstration
        A = [[2.0, 1.0, 3.0],
             [4.0, 0.0, 1.0]] # 2x3 matrix
        
        B = [[1.0, 2.0],
             [3.0, 1.0],
             [0.0, 2.0]] # 3x2 matrix
        
        C = [[5.0, 2.0],
             [1.0, 3.0]] # 2x2 matrix
        
        D = [[2.0, 1.0],
             [4.0, 3.0]] # 2x2 matrix

        print("Sample matrices:")
        print("Matrix A (2x3):")

        for i, row in enumerate(A): # for each row in A, we print the values formatted to 3 decimal places, making it look like a matrix
            formatted_row = [f'{v:.3f}' for v in row]
            print(f"  Row {i+1}: {formatted_row}")
        print()
        
        print("Matrix B (3x2):")

        for i, row in enumerate(B): # for each row in B, we print the values formatted to 3 decimal places, making it look like a matrix
            formatted_row = [f'{v:.3f}' for v in row]
            print(f"  Row {i+1}: {formatted_row}")
        print()
        
        print("Matrix C (2x2):")

        for i, row in enumerate(C): # for each row in C, we print the values formatted to 3 decimal places, making it look like a matrix
            formatted_row = [f'{v:.3f}' for v in row]
            print(f"  Row {i+1}: {formatted_row}")
        print()
        
        print("Matrix D (2x2):")

        for i, row in enumerate(D): # for each row in D, we print the values formatted to 3 decimal places, making it look like a matrix
            formatted_row = [f'{v:.3f}' for v in row]
            print(f"  Row {i+1}: {formatted_row}")
        print()
        
        # Matrix transpose
        print("7. Matrix Transpose:")
        print("   A^T:")
        transpose_result = self.transpose(A) 

        if transpose_result is not None: # handle error case from transpose
            for i, row in enumerate(transpose_result):
                formatted_row = [f'{v:.3f}' for v in row] # format each value to 3 decimal places, making it look like a matrix
                print(f"     Row {i+1}: {formatted_row}")
        print()
        
        # Matrix-vector multiplication
        test_vec = [[1.0], [2.0], [1.0]]
        print("8. Matrix-Vector Multiplication:")
        print("   A x vec where vec =")

        for i, v in enumerate(test_vec): # for each row in test_vec, we print the value formatted to 3 decimal places, making it look like a column vector
            print(f"     [{v[0]:.3f}]")

        mv_result = self.matrix_vector_multiply(A, test_vec) # multiply A by test_vec

        if mv_result is not None: # handle error case from matrix-vector multiplication
            print("   Result:")
            for i, v in enumerate(mv_result): # for each row in the result, we print the value formatted to 3 decimal places, making it look like a column vector
                print(f"     [{v[0]:.3f}]")
        print()
        
        # Matrix multiplication
        print("9. Matrix Multiplication:")
        print("   A x B:")
        mult_result = self.matrix_multiply(A, B)
        if mult_result is not None: # handle error case from matrix multiplication
            print("   Result:")
            for i, row in enumerate(mult_result): # for each row in the result, we print the values formatted to 3 decimal places, making it look like a matrix
                formatted_row = [f'{v:.3f}' for v in row] 
                print(f"     Row {i+1}: {formatted_row}")
        print()
        
        # Matrix addition
        print("10. Matrix Addition:")
        print("    C + D:")
        add_result = self.matrix_add(C, D)
        if add_result is not None:
            for i, row in enumerate(add_result):

                formatted_row = [f'{v:.3f}' for v in row]
                print(f"      Row {i+1}: {formatted_row}")
        print()
        
        # Matrix subtraction
        print("11. Matrix Subtraction:")
        print("    C - D:")
        sub_result = self.matrix_subtract(C, D)

        if sub_result is not None:
            for i, row in enumerate(sub_result):
                formatted_row = [f'{v:.3f}' for v in row]
                print(f"      Row {i+1}: {formatted_row}")
        print()
        
        # Scalar matrix multiplication
        print("12. Scalar Matrix Multiplication:")
        print(f"    {scalar} x C:")
        scalar_matrix_result = self.scalar_matrix_multiply(scalar, C)
        if scalar_matrix_result is not None:

            for i, row in enumerate(scalar_matrix_result):

                formatted_row = [f'{v:.3f}' for v in row] # format each value to 3 decimal places, making it look like a matrix
                print(f"      Row {i+1}: {formatted_row}")
        print()
        
        # Outer product
        print("13. Outer Product:")
        outer_vec1 = [[2.0], [3.0]]
        outer_vec2 = [[1.0], [4.0], [2.0]]
        print("    vec1 Outer Product vec2^T where:")
        print("    vec1 =")

        for i, v in enumerate(outer_vec1): # for each row in outer_vec1, we print the value formatted to 3 decimal places, making it look like a column vector
            print(f"      [{v[0]:.3f}]")
        print("    vec2^T = [", end="")

        for i, v in enumerate(outer_vec2): # for each row in outer_vec2, we print the value formatted to 3 decimal places, making it look like a row vector
            if i > 0: print(", ", end="")
            print(f"{v[0]:.3f}", end="")
        print("]")

        outer_result = self.outer_product(outer_vec1, outer_vec2)

        if outer_result is not None:
            print("    Result:")
            for i, row in enumerate(outer_result): # for each row in the outer product result, we print the values formatted to 3 decimal places, making it look like a matrix
                formatted_row = [f'{v:.3f}' for v in row]
                print(f"      Row {i+1}: {formatted_row}")
        print()
        
        # Part 3: Determinant Calculations
        print("Part 3: Determinant Calculations")
        print("-" * 40)
        print()
        
        # 2x2 determinant
        matrix_2x2 = [[3.0, 2.0],
                      [1.0, 4.0]]
        
        print("14. Determinant (2x2 Matrix):")
        print("    Matrix:")
        for i, row in enumerate(matrix_2x2): # for each row in the matrix, we print the values formatted to 3 decimal places, making it look like a matrix
            formatted_row = [f'{v:.3f}' for v in row]
            print(f"      Row {i+1}: {formatted_row}")
        
        det_result = self.determinant(matrix_2x2) # compute determinant
        if det_result is not None:
            print(f"    Determinant: {det_result:.6f}") # print determinant formatted to 6 decimal places
        print()
        
        # 3x3 determinant
        matrix_3x3 = [[2.0, 1.0, 3.0],
                      [1.0, 4.0, 2.0],
                      [3.0, 2.0, 1.0]]
        
        print("15. Determinant (3x3 Matrix):")
        print("    Matrix:")
        for i, row in enumerate(matrix_3x3): # for each row in the matrix, we print the values formatted to 3 decimal places, making it look like a matrix
            formatted_row = [f'{v:.3f}' for v in row]
            print(f"      Row {i+1}: {formatted_row}")
        
        det_result_3x3 = self.determinant(matrix_3x3)
        if det_result_3x3 is not None:
            print(f"    Determinant: {det_result_3x3:.6f}") # print determinant formatted to 6 decimal places
        print()
        
        # FF2 determinant
        matrix_ff2 = [[1, 0, 1],
                      [1, 1, 0],
                      [0, 1, 1]]
        
        print("16. Determinant in Finite Field 2:")
        print("    Binary Matrix:")
        for i, row in enumerate(matrix_ff2): # for each row in the matrix, we print the values making it look like a matrix
            print(f"      Row {i+1}: {row}")

        det_ff2_result = self.determinant_FF2(matrix_ff2) # compute determinant in FF(2)
        if det_ff2_result is not None:
            print(f"    Determinant (mod 2): {det_ff2_result}")
        print()
        

def main():
    vmo = VectorMatrixOperations()
    vmo.visualize_operations()

if __name__ == "__main__":
    main()