from vector_matrix_operations import *
import math


class HammingCode(VectorMatrixOperations):
    """ 
    A general Hamming Code implementation using matrix operations.
    
    This implementation uses a generator matrix [G] for encoding and a parity check matrix [H] for decoding.
    It can work with any valid Hamming code by accepting custom [G] and [H] matrices.
    
    All matrix operations use modulo 2 arithmetic (addition is XOR, multiplication is AND).

    Remember, if we say a 1d row, we mean a single row matrix (e.g., [[0, 1, 1, 0]]).
    If we say a 2d column vector, we mean a single column matrix (e.g., [[0], [1], [1], [0]])
    """
    
    def __init__(self, generator_data=None, data=None):
        """
        Initialize the HammingCode class with generator and parity check matrices.
        
        Parameters:
        generator_data (list): A list of 4 integers (0-15) representing generator data.
        data (list): A 2D column vector [[bit], [bit], [bit], [bit]] representing the data to encode.
        """
        super().__init__()

        # Use setters to handle validation
        self.set_generator_data(generator_data)
        self.set_data(data)
        
        self.__data_matrix = self.__transform_data_to_four_bit_matrix(self.__generator_data)
        # Calculate parity bits from actual data bits, not generator data
        data_bits_1d = [row[0] for row in self.__data]  # Convert 2D data to 1D for parity calculation
        self.__parity_matrix = self.__calculate_parity_bits(data_bits_1d)  # Calculate parity bits based on data bits
        self.__G = self.__create_generator_matrix(self.__data_matrix, self.__parity_matrix) # Create G by combining data and parity matrices
        self.__H = self.__create_parity_check_matrix(self.__parity_matrix, self.__data_matrix) # Create H from parity and data matrices

    
    def get_G(self):
        """
        Getter for generator matrix [G].

        Returns:
        list of lists: The generator matrix [G].
        """

        if self.__G is None: # Sanity check for uninitialized G
            raise ValueError("Generator matrix [G] has not been initialized.")
        
        for i in range(len(self.__G)): # For each row in G
            for k in range(len(self.__G[i])): # For each column in G

                if self.__G[i][k] is None: # Sanity check for uninitialized elements
                    raise ValueError("Generator matrix [G] is not properly formed.")
        
        
        return self.__G
    

    def get_H(self):
        """
        Getter for parity check matrix [H].

        Returns:
        list of lists: The parity check matrix [H].
        """
        return self.__H
    

    def get_data(self):
        """
        Getter for data bits.

        Returns:
        list: The data bits.
        """
        return self.__data

    def get_generator_data(self):
        """
        Getter for generator data.

        Returns:
        list: The generator data.
        """
        return self.__generator_data

    def get_k(self):
        """
        Getter for number of data bits.

        Returns:
        int: The number of data bits.
        """
        return self.__k

    def get_data_matrix(self):
        """
        Getter for data matrix.

        Returns:
        list: The data matrix.
        """
        return self.__data_matrix

    def get_parity_matrix(self):
        """
        Getter for parity matrix.

        Returns:
        list: The parity matrix.
        """
        return self.__parity_matrix

    def set_generator_data(self, generator_data):
        """
        Setter for generator data with validation.

        Parameters:
        generator_data (list): A list of 4 integers (0-15) representing generator data.
        """
        if not isinstance(generator_data, list): # Check if it's a list
            raise TypeError("Generator data must be a list.")
        
        if len(generator_data) != 4: # Check for exactly 4 values
            raise ValueError("Generator data list must contain exactly 4 values.")
        
        for value in generator_data: 
            if not isinstance(value, int) or not (0 <= value <= 15): # Check if each value is an integer between 0 and 15 (since 4 bits can represent 0-15)
                raise ValueError("Generator data values must be integers between 0 and 15.")
        
        self.__generator_data = generator_data

    def set_data(self, data):
        """
        Setter for data bits with validation.

        Parameters:
        data (list): A 2D column vector [[bit], [bit], [bit], [bit]] representing the data to encode.
        """
        # Check if it's a 2D column vector
        if not isinstance(data, list):
            raise TypeError("Data must be a list (column vector).")
        
        if len(data) != 4: # Check for exactly 4 data bits
            raise ValueError("This implementation currently supports only (7,4) Hamming code with 4 data bits.")
        
        # Validate column vector format
        if not self._is_column_vector(data):
            raise ValueError("Data must be a 2D column vector format: [[bit], [bit], [bit], [bit]]")
        
        # Validate bit values
        for row in data:
            if row[0] not in [0, 1]: # Check if each bit is 0 or 1 (since we're in FF(2))
                raise ValueError("Data bits must be 0 or 1.")
        
        self.__data = data  # Store as 2D column vector
        self.__k = len(data)  # Number of data bits

    def __transform_data_to_four_bit_matrix(self, data, is_parity=False):
        """
        Transform input data into a binary matrix where each bit position is a column.
        Data values must be 0-15 in decimal to fit into 4 bits.
        Each column represents all values' bits at that position.
        If is_parity is True, expects parity bits (0-7) for 3 bits instead.
        
        
        Parameters:
        data (list): A list of integers (0-15) representing 4-bit values.
        
        Returns:
        list of lists: A matrix where each row represents a bit position,
                    and each column represents a data value.
        """
        if not data: # Check for empty data
            raise ValueError("Data cannot be empty.")
        
        if is_parity:
            if len(data) != 3: # We only care that there are 3 parity numbers
                raise ValueError("Parity bits list must contain exactly 3 values.")
        else:
            if len(data) != 4:
                raise ValueError("Data list must contain exactly 4 values.")
            
        bit_length = 4 # No matter what, we use 4 bits for data representation, even for parity)

        # Initialize rows (one for each bit position)
        binary_matrix = [[] for _ in range(bit_length)]

        for value in data:
            if not (0 <= value < 16): # Check if value fits in 4 bits 
                raise ValueError(f"Data value {value} must be in the range 0-15 to fit into 4 bits.")

            # Convert to binary string with appropriate bit length
            binary_str = format(value, f'0{bit_length}b')
            
            # Add each bit to its corresponding row (bit position)
            for bit_position, bit_char in enumerate(binary_str):
                binary_matrix[bit_position].append(int(bit_char)) # Convert char to int and append to the correct row
        
        return binary_matrix
    
    def __calculate_parity_bits(self, data_bits):
        """
        Calculate the parity bits as such:
        p1 = d2 + d3 + d4 
        p2 = d1 + d3 + d4 
        p3 = d1 + d2 + d4 
        Then return the parity bits as a matrix for further processing using transform_data_bits_to_four_bit_matrix.

        Parameters:
        data_bits (list): A list of 4 bits (0s and 1s) representing the data bits.

        Returns:
        list of lists: A matrix where each row represents a bit position,
                    and each column represents a parity value.
        """

        if len(data_bits) != 4:
            raise ValueError("Data bits list must contain exactly 4 bits.")

        d1, d2, d3, d4 = data_bits

        # Use XOR operations for FF(2) arithmetic (addition is XOR )
        # For example, 1 + 1 = 0 in FF(2), the same as if we just did XOR!
        p1 = (d2 + d3 + d4) % 2
        p2 = (d1 + d3 + d4) % 2
        p3 = (d1 + d2 + d4) % 2

        parity_bits = [p1, p2, p3] # List of parity bits

        parity_matrix = self.__transform_data_to_four_bit_matrix(parity_bits, is_parity=True) # Transform parity bits into matrix form (with each bit being a column vector)

        return parity_matrix
    

    def __is_matrix_linearly_independent(self, matrix):
        """
        Check if the data bits are linearly independent over FF(2).
        For 4 data bits, they are linearly independent if their determinant is non-zero in FF(2).

        Parameters:
        data_matrix (list of lists): The data matrix for data bits (4x4).

        Returns:
        bool: True if data bits are linearly independent, False otherwise.
        """

        # Calculate the determinant of the data matrix over FF(2)
        det = self.determinant_FF2(matrix)

        if det is None: # Sanity check for determinant calculation failure
            raise ValueError("Error calculating determinant.")

        return det != 0  # In FF(2), non-zero determinant means linear independence

       


    def __create_generator_matrix(self, data_matrix, parity_matrix):
        """
        Create the standard (7,4) Hamming code generator matrix [G].
        G is calculated as follows: [P | D]
        Where P is the parity matrix (4x3) and D is the data matrix (4x4).
        The generator matrix G will be 4x7.
        Note that the four data bits MUST be linearly independent for this to work.
        Error checking for this is implemented.

        Parameters:
        data_matrix (list of lists): The data matrix for data bits (4x4 identity).
        parity_matrix (list of lists): The parity matrix for parity bits (4x3).

        Returns:
        list of lists: The generator matrix [G] for (7,4) Hamming code.

        """

        G = []

        # Error checking: Ensure data bits are linearly independent

        for i in range(4):  # 4 data bits (rows of G)
            row = []
            # Parity bits (first 3 columns)
            for j in range(3):
                row.append(parity_matrix[i][j])  # Parity bits from parity matrix
            # Data bits (last 4 columns - identity matrix)
            for j in range(4):
                row.append(data_matrix[i][j])   # Data bits from data matrix

            G.append(row) # Append the constructed row to G

        if not self.__is_matrix_linearly_independent(data_matrix): # Check linear independence of data bits
            raise ValueError("Matrix is not linearly independent.")

        return G
    

    def __create_parity_check_matrix(self, parity_matrix, data_matrix):
        """
        Create the parity check matrix [H] for the Hamming code.
        For a standard (7,4) Hamming code, H is a 3x7 matrix such that
        row 1 contaains 1s in the positions corresponding to parity bit 1 and the data bits it checks,
        row 2 contains 1s in the positions corresponding to parity bit 2 and the data bits it checks,
        row 3 contains 1s in the positions corresponding to parity bit 3 and the data bits it checks,
        Where I is a 3x3 identity matrix and P^T is the transpose of the parity matrix.

        Parameters:
        parity_matrix (list of lists): The parity matrix for parity bits (4x3).
        data_matrix (list of lists): The data matrix for data bits (4x4).

        Returns:
        list of lists: The parity check matrix [H] for the Hamming code.
        """

        # First, get the transpose of the parity matrix using the inherited transpose function
        parity_matrix_transposed = self.transpose(parity_matrix)
        
        if parity_matrix_transposed is None:
            raise ValueError("Error during parity matrix transpose.")

        # Now, construct H by setting up row 1s, row 2s, and row 3s
        H = [[0 for _ in range(7)] for _ in range(3)]

        # Set up the parity check matrix H
        # Parity bit 1 is at position 1 (index 0) and checks d2, d3, d4
        H[0][0] = 1  # p1
        H[0][4] = 1  # d2
        H[0][5] = 1  # d3
        H[0][6] = 1  # d4

        # Parity bit 2 is at position 2 (index 1) and checks d1, d3, d4
        H[1][1] = 1  # p2
        H[1][3] = 1  # d1
        H[1][5] = 1  # d3
        H[1][6] = 1  # d4

        # Parity bit 3 is at position 3 (index 2) and checks d1, d2, d4
        H[2][2] = 1  # p3
        H[2][3] = 1  # d1
        H[2][4] = 1  # d2
        H[2][6] = 1  # d4

        if not self.__is_matrix_linearly_independent(data_matrix): # Check linear independence of data bits
            raise ValueError("Matrix is not linearly independent.")

        return H
    
    def __find_data_bit_positions(self):
        """
        Find the positions of data bits in the codeword by locating the identity matrix
        columns in the generator matrix. This makes the code robust to different
        generator_data orders.
        
        Returns:
        list: A list of column indices where the identity matrix is located in G.
        """
        data_positions = []
        
        # Look for identity matrix columns in G
        for row_idx in range(4):  # We have 4 data bits
            for col_idx in range(7):  # We have 7 total bits in codeword
                # Check if this column matches the expected identity column
                expected_identity_col = [0] * 4
                expected_identity_col[row_idx] = 1


                actual_col = [self.get_G()[i][col_idx] for i in range(4)] # Extract the actual column from G (4x7 matrix)

                if actual_col == expected_identity_col: # Check if this column matches the expected identity column
                    data_positions.append(col_idx) # Store the position
                    break
        
        if len(data_positions) != 4: # Ensure we found all 4 data bit positions
            raise ValueError("Could not find valid identity matrix in generator matrix.")
        
        return data_positions 

    def encode(self, data, generator_matrix):
        """
        Encode data bits into a Hamming codeword using the generator matrix [G].

        Standard encoding: codeword = data x G
        Where data is a 1x4 row matrix and G is a 4x7 matrix, producing a 1x7 codeword.
        
        Parameters:
        data (list of lists): A 2D column vector [[bit], [bit], [bit], [bit]] representing the data.
        generator_matrix (list of lists): The generator matrix [G] for encoding (4x7 format).

        Returns:
        list: A 1D row representing the encoded Hamming codeword.
        """
        if len(data) != 4:
            raise ValueError("Data list must contain exactly 4 bits for (7,4) Hamming code.")

        # Convert data from column vector format to row matrix format for multiplication
        data_row = [[row[0] for row in data]]  # Convert [[a], [b], [c], [d]] to [[a, b, c, d]]

        # Compute data x G (1x4 matrix x 4x7 matrix = 1x7 matrix)
        codeword_matrix = self.matrix_multiply(data_row, generator_matrix)

        if codeword_matrix is None: # Sanity check for multiplication failure
            raise ValueError("Error during matrix multiplication.")
        
        # Apply modulo 2 arithmetic and convert to 1D row vector
        codeword_1d = [int(bit) % 2 for bit in codeword_matrix[0]]

        return codeword_1d

    def __calculate_syndrome(self, received_bits, parity_check_matrix):
        """
        Calculate the syndrome for the received bits using the parity check matrix [H].
        
        Uses the formula: syndrome = H x received_bits

        If syndrome is all zeros, no error is detected.
        If syndrome is non-zero, it indicates the position of the error!
        For example, syndrome [1,0,1] indicates an error in position 5 (1-based index).
        Syndrome [0,1,1] indicates an error in position 6, etc.
        
        Parameters:
        received_bits (list): A 1D row matrix representing the received codeword.
        parity_check_matrix (list of lists): The parity check matrix [H] for decoding.

        Returns:
        list of lists: A 2D column vector representing the syndrome.
        """

        # Convert 1D received_bits to column vector format
        received_bits_column = [[bit] for bit in received_bits]  # Convert to 7x1 column vector

        syndrome = self.matrix_multiply(parity_check_matrix, received_bits_column) # H (3x7) x received_bits (7x1) = syndrome (3x1)

        if syndrome is None: # Sanity check for multiplication failure
            raise ValueError("Error during matrix-vector multiplication.")

        # Apply modulo 2 arithmetic to the column vector result
        syndrome_mod2 = [[int(row[0]) % 2] for row in syndrome]
     
        return syndrome_mod2
    
    
    def __correct_error(self, received_bits, syndrome):
        """
        Correct a single-bit error in the received bits based on the syndrome.
        Again, we can only correct single-bit errors with Hamming code.
        If we have more than one error, we cannot correct it with 100% accuracy!

        Parameters:
        received_bits (list): A 1D row matrix representing the received codeword.
        syndrome (list of lists): A 2D column vector representing the syndrome.

        Returns:
        list: A 1D row matrix representing the corrected codeword.
        """

        # Check if there's no error (syndrome is all zeros)
        if all(row[0] == 0 for row in syndrome):
            # No error detected
            return received_bits
        
        # Find which column in H matrix matches the syndrome
        error_position = None
        for pos in range(len(received_bits)):

            # Extract the actual column from H (3x7 matrix)
            # The code means to get the pos-th column of H (as a 3x1 column vector)
            column = [[self.get_H()[i][pos]] for i in range(len(self.get_H()))] 

            if column == syndrome: # If this column matches the syndrome, we found the error position! 
                error_position = pos
                break
        
        if error_position is None: # If no matching column found, raise an error
            raise ValueError("Syndrome does not match any column in the parity check matrix.")
        
        # Correct the error by flipping the bit at the error position (0-based index)
        corrected_bits = received_bits.copy()  # Make a copy of the 1D list
        corrected_bits[error_position] ^= 1  # Flip the bit using XOR (this is a basic property of XORing with 1!)
        
        return corrected_bits

    def decode(self, received_bits, parity_check_matrix, syndrome, correct_errors=True):   
        """
        Decode the received bits using the parity check matrix [H] and syndrome.
        
        Parameters:
        received_bits (list): A 1D row matrix representing the received codeword.
        parity_check_matrix (list of lists): The parity check matrix [H] for decoding.
        syndrome (list of lists): A 2D column vector representing the syndrome.
        correct_errors (bool): Whether to attempt error correction based on the syndrome.
        
        Returns:
        list of lists: A 2D column vector representing the decoded data bits.
        """
        # First, calculate the syndrome
        calculated_syndrome = self.__calculate_syndrome(received_bits, parity_check_matrix)

        if correct_errors: # If error correction is enabled
            corrected_bits = self.__correct_error(received_bits, calculated_syndrome)
        else:
            corrected_bits = received_bits # No correction, just use received bits

        # Extract the original data bits from the corrected codeword
        # Find where the identity matrix columns are located in the generator matrix
        data_positions = self.__find_data_bit_positions()
        
        # Extract data bits from their actual positions in the codeword
        decoded_data = []
        for pos in data_positions:
            decoded_data.append([corrected_bits[pos]])

        return decoded_data


    
    def print_matrices(self):
        """
        Print the generator matrix [G] and parity check matrix [H] in a readable format.
        Useful for debugging and understanding the code structure.
        """

        print(f"\nGenerator Matrix [G]:")
        print(self.get_G())

        print(f"\nParity Check Matrix [H]:")
        print(self.get_H())

        print (f"\nSyndrome for data {self.get_data()}:")
        syndrome = self.__calculate_syndrome(self.encode(self.get_data(), self.get_G()), self.get_H())
        print(syndrome)

        print(f"\nDecoding process for data {self.get_data()}:")
        decoded_data = self.decode(self.encode(self.get_data(), self.get_G()), self.get_H(), syndrome)
        print(decoded_data)


    def test_capabilities(self):
        """
        Demonstrate Hamming Code (7,4) functionality with step-by-step mathematical process visualization.
        Shows vector operations and matrix transformations throughout the encoding/decoding pipeline.
        """
        print("\n\n")
        print("=" * 60) 
        print("Hamming Code (7,4) Mathematical Demonstration")
        print("=" * 60)
        print()
        
        # Test 1: Step-by-Step Encoding Process with Vector Notation
        print("1. ENCODING PROCESS: Mathematical Transformation")
        print("-" * 50)
        print()
        
        # Initialize with clear data vector representation
        generator_data = [8, 4, 2, 1]  # Binary positions: [2^3, 2^2, 2^1, 2^0]
        data_vector = [[1], [0], [1], [0]]  # 4x1 column vector d = [d1, d2, d3, d4]^T
        
        print("Input Data Vector:")
        print("  d = data vector (4x1 column vector)")

        for i, row in enumerate(data_vector): # Print each data bit in a formatted way as column vector
            print(f"    [{row[0]}]  <- d{i+1}") # Data bits d1, d2, d3, d4
        print()

        hamming = HammingCode(generator_data, data_vector) # Initialize HammingCode instance

        # Show the generator matrix [G] construction
        print("Generator Matrix [G] (4x7):")

        G = hamming.get_G() # Get the generator matrix
        print("  G = [P | I4] where P = parity bits matrix, I4 = 4x4 identity")

        for i, row in enumerate(G): # Print each row of G in a formatted way
            parity_part = row[:3]  # First 3 columns (parity)

            identity_part = row[3:]  # Last 4 columns (identity)

            formatted_parity = [f'{v}' for v in parity_part]

            formatted_identity = [f'{v}' for v in identity_part]

            print(f"  Row {i+1}: {formatted_parity} | {formatted_identity}") # Print row with parity and identity parts
        print("        Parity bits ^     Data bits ^")
        print()
        
        # Demonstrate the encoding transformation: c = d x G
        print("ENCODING TRANSFORMATION:")
        print("  Mathematical equation: c = d x G")
        print("  Where:")
        print("    c = 1x7 codeword matrix")
        print("    d = 1x4 data vector")
        print("    G = 4x7 generator matrix")  
        print()
    
        encoded_vector = hamming.encode(data_vector, G) # Perform encoding

        print("Encoded Codeword Vector:")
        print("  c = codeword vector (1x7 row matrix)")

        codeword_labels = ["p1", "p2", "p3", "d1", "d2", "d3", "d4"]

        # Extract values from encoded vector (now 1D) and format as row matrix
        encoded_values = encoded_vector
        print(f"    {encoded_values}")
        # print each bit of the codeword with labels
        print(f"    {codeword_labels}")  # Codeword bits p1, p2, p3, d1, d2, d3, d4

        print()

        # Test 2: Vector Pattern Analysis
        print("2. Vector Pattern Analysis: Multiple Data Inputs")
        print("-" * 50)
        print()
        
        test_vectors = [
            [[0], [0], [0], [0]],  # Zero vector
            [[1], [1], [1], [1]],  # Ones vector  
            [[1], [0], [0], [1]],  # Alternating pattern
            [[0], [1], [1], [0]]   # Inverse alternating
        ]

        vector_names = ["0000", "1111", "1001", "0110"] # Names for each test vector
        print("Data Vectors and Corresponding Encoded Codewords:")

        for i, test_vector in enumerate(test_vectors): # For each test vector
            hamming = HammingCode(generator_data, test_vector) # Initialize HammingCode instance
            encoded = hamming.encode(test_vector, hamming.get_G()) # Encode the test vector
            
            # Format vectors for display with consistent formatting
            data_str = [f'{row[0]}' for row in test_vector]
            encoded_str = [f'{bit}' for bit in encoded]

            print(f"  {vector_names[i]}: d = {data_str} -> c = {encoded_str}") # Print data vector and corresponding codeword
        print()

        # Test 3: Syndrome Calculation with Matrix Operations
        print("3. Syndrome Calculation: Error Detection Mathematics")
        print("-" * 55)
        print()
        
        print("Parity Check Matrix [H] (3x7):")
        H = hamming.get_H() # Get the parity check matrix

        print("  H matrix detects errors using equation: s = H x r")

        for i, row in enumerate(H): # Print each row of H in a formatted way
            formatted_row = [f'{v}' for v in row] # Format each value for display
            print(f"  Row {i+1}: {formatted_row}") 
        print("  Each row checks specific bit combinations for parity violations")
        print()
        
        # For error-free codeword, syndrome is zero
        syndrome_vector = [[0], [0], [0]]

        print("Syndrome Computation:")
        print("  s = H x c (for error-free codeword)")
        print("  s = syndrome vector (3x1 column vector)")

        for i, row in enumerate(syndrome_vector): # Print each syndrome bit in a formatted way as column vector
            status = "no errors detected" if i == 1 else ""
            print(f"    [{row[0]}]  <- s{i+1} {status}")
        print()
        
        # Test 4: Error Correction with Vector Mathematics
        print("4. ERROR CORRECTION: Single-Bit Error Recovery")
        print("-" * 50)
        print()
        
        data_test = [[1], [0], [1], [1]]  # Test data vector
        hamming = HammingCode(generator_data, data_test) # Initialize HammingCode instance
        correct_codeword = hamming.encode(data_test, hamming.get_G())
        
        print("Original Process:") # Show original encoding based on data test
        print("  Encoding: c = d x G")
        print("  Data vector d =", [f'{row[0]}' for row in data_test]) 
        print("  Correct codeword c =", [f'{bit}' for bit in correct_codeword])
        print()
        
        # Introduce single-bit error at position 3 (0-based indexing)
        error_position = 3
        erroneous_vector = correct_codeword.copy()  # Copy the 1D list

        if erroneous_vector is None or error_position < 0 or error_position >= len(erroneous_vector): # Sanity check for valid error position
            raise ValueError("Error position is out of bounds for the codeword length.")
        
        erroneous_vector[error_position] ^= 1  # XOR flip bit at the third position

        print("Error Introduction:")
        print(f"  Error at position {error_position}: bit flip using XOR operation") # Explain the error introduction
        print("  Erroneous vector r =", [f'{bit}' for bit in erroneous_vector]) 
        print()
        
        # Calculate syndrome using the proper method
        error_syndrome = self.__calculate_syndrome(erroneous_vector, hamming.get_H())
        
        print("ERROR SYNDROME CALCULATION:")
        print("  s = H x r (for corrupted codeword)")
        print("  s = error syndrome vector (3x1 column vector)")

        for i, row in enumerate(error_syndrome): # Print each syndrome bit in a formatted way as column vector
            status = "non-zero syndrome indicates error!" if i == 1 else ""
            print(f"    [{row[0]}]  <- s{i+1} {status}")   # Print syndrome bits with indication of error
        print(f"  Syndrome pattern points to error location in position {error_position}")
        print()
        
        # Perform error correction using the proper method
        corrected_vector = self.__correct_error(erroneous_vector, error_syndrome)
        
        # Use the decode method to extract data bits
        final_decoded = self.decode(corrected_vector, hamming.get_H(), error_syndrome, correct_errors=False)
        
        print("ERROR CORRECTION PROCESS:")
        print("  Correction: flip bit at syndrome-indicated position using __correct_error method")
        print("  Corrected vector c' =", [f'{bit}' for bit in corrected_vector]) # Print corrected codeword
        print("  Final decoded data d' =", [f'{row[0]}' for row in final_decoded]) # Print final decoded data bits
        print("  Success: d' = d (original data restored)")
        print()

        # Test 5: Generator Matrix Variations
        print("5. GENERATOR MATRIX VARIATIONS: Different Constructions")
        print("-" * 55)
        print()
        
        # Different generator data configurations to show flexibility
        generator_lists = [
            ([8,4,2,1], "Standard binary"),
            ([1,2,4,8], "Reversed binary"), 
            ([4,1,8,2], "Random binary")
        ]
        test_data = [[1], [0], [1], [0]] # Same data vector for all tests
        
        print("Same data vector with different generator configurations:")
        print("  Test vector d =", [f'{row[0]}' for row in test_data]) # 4x1 column vector
        print()
        
        for generator_list, description in generator_lists: # For each generator configuration
            hamming = HammingCode(generator_list, test_data) # Initialize HammingCode instance
            encoded = hamming.encode(test_data, hamming.get_G()) # Encode the test data

            formatted_encoded = [f'{bit}' for bit in encoded] # Format encoded codeword for display
            print(f"  {description:15}: {generator_list} -> c = {formatted_encoded}")

        print()
        print("Mathematical Summary:")
        print("  * Encoding: c = d x G (1x4 vector x 4x7 matrix = 1x7 vector)")
        print("  * Syndrome: s = H x r (3x7 matrix x 7x1 vector = 3x1 vector)")
        print("  * Error correction: single-bit errors detected and corrected")
        print("  * All operations in FF(2): addition = XOR, multiplication = AND")


def main():
    """
    Run the demonstration of Hamming Code capabilities.
    """
    generator_data = [8, 4, 2, 1]
    data = [[1], [0], [1], [0]]
    
    hamming = HammingCode(generator_data, data)
    hamming.test_capabilities()

if __name__ == "__main__":
    main()  
