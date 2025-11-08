from vector_matrix_operations import *
import math


class HammingCode(VectorMatrixOperations):
    """ 
    A (7,4) Hamming Code implementation using matrix operations.
    (7,4) Hamming can correct and detect single-bit errors.
    
    This implementation follows the standard Hamming code approach where:
    Parity bits are defined as: p1 = i2 + i3 + i4, p2 = i1 + i3 + i4, p3 = i1 + i2 + i4
    Generator matrix [G] is constructed with parity and I-bit arrangements
    Parity check matrix [H] is the corresponding matrix for error detection/correction

    All matrix operations use modulo 2 arithmetic (addition is XOR, multiplication is AND).
    
    The bit_arrangement parameter determines how parity and I-bits are ordered in the codeword.
    Remember, 'I-bits' (i1,i2,i3,i4) refer to identity matrix positions in the bit arrangement,
    while 'input data' refers to the actual 4-bit data vector being encoded.

    Another important note: if we say i1, i2, i3, i4, we can also mean i[0], i[1], i[2], i[3] respectively in code.
    """
    
    def __init__(self, bit_arrangement=None, data=None):
        """
        Initialize the HammingCode class with a specific bit arrangement.
        
        Parameters:
        bit_arrangement (list): A list of 7 strings indicating bit positions, e.g.,
                               ['p1', 'p2', 'p3', 'i1', 'i2', 'i3', 'i4'] for standard arrangement,
                               ['i1', 'i2', 'i3', 'i4', 'p1', 'p2', 'p3'] for I-bits-first arrangement,
                               etc. (i1,i2,i3,i4 are identity matrix positions; p1,p2,p3 are parity)

        The bit arrangement is perhaps one of the most critical aspects of this implementation.
        As when the user specifies the arrangement, they are defining where the identity matrix and parity bits are located in the codeword.
        Which affects the construction of the generator [G] and parity check [H] matrices.
        data (list): A 2D column vector [[bit], [bit], [bit], [bit]] representing the input data to encode.

        Returns:
        None
        """
        super().__init__()

        # Set default standard arrangement if none provided
        if bit_arrangement is None:
            bit_arrangement = ['p1', 'p2', 'p3', 'i1', 'i2', 'i3', 'i4']  # Standard arrangement with parity bits first
        
        self.set_bit_arrangement(bit_arrangement) # Set and validate bit arrangement
        self.set_data(data) # Set and validate input data
        
        # Create generator and parity check matrices based on the bit arrangement
        self.__G = self.__create_generator_matrix()
        self.__H = self.__create_parity_check_matrix()

    
    def get_G(self):
        """
        Getter for generator matrix [G].

        Parameters:
        None

        Returns:
        G (list of lists): The generator matrix [G].
        """
        return self.__G
    

    def get_H(self):
        """
        Getter for parity check matrix [H].

        Parameters:
        None

        Returns:
        H (list of lists): The parity check matrix [H].
        """
        return self.__H
    

    def get_data(self):
        """
        Getter for data bits.

        Parameters:
        None

        Returns:
        Data (list): The data bits.
        """
        return self.__data

    def get_bit_arrangement(self):
        """
        Getter for bit arrangement.

        Parameters:
        None

        Returns:
        bit_arrangement (list): The bit arrangement.
        """
        return self.__bit_arrangement

    def get_k(self):
        """
        Getter for number of data bits.

        Parameters:
        None

        Returns:
        int: The number of data bits (always 4 for (7,4) Hamming code).
        """
        return 4

    def set_bit_arrangement(self, bit_arrangement):
        """
        Setter for bit arrangement with validation.

        Parameters:
        bit_arrangement (list): A list of 7 strings indicating bit positions.

        Returns:
        None
        """
        try:
            if not isinstance(bit_arrangement, list):
                raise TypeError("Bit arrangement must be a list.")
            
            if len(bit_arrangement) != 7:
                raise ValueError("Bit arrangement must contain exactly 7 positions.")
            
            # Check that we have exactly 4 I-bits and 3 parity bits
            ibits = [bit for bit in bit_arrangement if bit.startswith('i')]
            parity_bits = [bit for bit in bit_arrangement if bit.startswith('p')]
            
            # If the arrangement is not valid, raise an error
            if len(ibits) != 4 or len(parity_bits) != 3: 
                raise ValueError("Bit arrangement must contain exactly 4 I-bits (i1-i4) and 3 parity bits (p1-p3).")
            
            # Check that we have the right I-bit and parity bit names
            expected_ibits = {'i1', 'i2', 'i3', 'i4'}
            expected_parity = {'p1', 'p2', 'p3'}
            
            # If the expected bits are not present, raise an error
            if set(ibits) != expected_ibits or set(parity_bits) != expected_parity:
                raise ValueError("Bit arrangement must contain i1, i2, i3, i4, p1, p2, p3.")
            
            self.__bit_arrangement = bit_arrangement # Set the validated bit arrangement
            
        except (TypeError, ValueError) as e:
            raise e

    def set_data(self, data):
        """
        Setter for input data with validation.

        Parameters:
        data (list): A 2D column vector [[bit], [bit], [bit], [bit]] representing the input data to encode.

        Returns:
        None
        """
        try:
            if data is None:
                self.__data = None
                return
                
            # Check if it's a 2D column vector
            if not isinstance(data, list):
                raise TypeError("Input data must be a list (column vector).")
            
            # Check for correct number of input data bits
            if len(data) != 4:
                raise ValueError("This implementation supports only (7,4) Hamming code with 4 input data bits.")
            
            # Validate column vector format
            if not self.is_column_vector(data):
                raise ValueError("Input data must be a 2D column vector format: [[bit], [bit], [bit], [bit]]")
            
            # Validate bit values
            for row in data:
                if row[0] not in [0, 1]:
                    raise ValueError("Input data bits must be 0 or 1.")
             
            self.__data = data # Set the validated data
            
        except (TypeError, ValueError) as e:
            raise e

    def __create_generator_matrix(self):
        """
        Create the generator matrix [G] for (7,4) Hamming code based on bit arrangement.
        
        The generator matrix is constructed so that:
        - Each row corresponds to one identity data bit (identity[0], identity[1], identity[2], identity[3])
        - Each column corresponds to one codeword bit position
        - I-bit relationships: p1 = i2 + i3 + i4, p2 = i1 + i3 + i4, p3 = i1 + i2 + i4
        - i1,i2,i3,i4 represent the identity matrix positions in the bit arrangement

        For example, if the arrangement is ['p1', 'p2', 'p3', 'i1', 'i2', 'i3', 'i4'], then the matrix would be:
           [[0, 1, 1, 1, 0, 0, 0],  (identity[0] contributes to p2, p3, and is i1)
            [1, 0, 1, 0, 1, 0, 0],  (identity[1] contributes to p1, p3, and is i2)
            [1, 1, 0, 0, 0, 1, 0],  (identity[2] contributes to p1, p2, and is i3)
            [1, 1, 1, 0, 0, 0, 1]]  (identity[3] contributes to p1, p2, p3, and is i4)

        Parameters:
        None

        Returns:
        G (list of lists): The 4x7 generator matrix [G].
        """
        try:
            # Initialize 4x7 generator matrix (4 identity data bits x 7 codeword bits)
            G = [[0 for _ in range(7)] for _ in range(4)]
            
            # Find positions of each bit in the arrangement
            bit_arrangement = self.get_bit_arrangement()
            pos_p1 = bit_arrangement.index('p1')
            pos_p2 = bit_arrangement.index('p2')
            pos_p3 = bit_arrangement.index('p3')
            pos_i1 = bit_arrangement.index('i1')
            pos_i2 = bit_arrangement.index('i2')
            pos_i3 = bit_arrangement.index('i3')
            pos_i4 = bit_arrangement.index('i4')

            # Set up parity bit relationships
            # p1 = i2 + i3 + i4 (identity[1] + identity[2] + identity[3])
            G[1][pos_p1] = 1  # identity[1] contributes to p1
            G[2][pos_p1] = 1  # identity[2] contributes to p1
            G[3][pos_p1] = 1  # identity[3] contributes to p1

            # p2 = i1 + i3 + i4 (identity[0] + identity[2] + identity[3])
            G[0][pos_p2] = 1  # identity[0] contributes to p2
            G[2][pos_p2] = 1  # identity[2] contributes to p2
            G[3][pos_p2] = 1  # identity[3] contributes to p2

            # p3 = i1 + i2 + i4 (identity[0] + identity[1] + identity[3])
            G[0][pos_p3] = 1  # identity[0] contributes to p3
            G[1][pos_p3] = 1  # identity[1] contributes to p3
            G[3][pos_p3] = 1  # identity[3] contributes to p3

            # Set up identity matrix for I-bits
            # Row 0 corresponds to identity[0], Row 1 to identity[1], Row 2 to identity[2], Row 3 to identity[3]
            # These map to i1, i2, i3, i4 positions respectively
            G[0][pos_i1] = 1  # identity[0] contributes to i1 position
            G[1][pos_i2] = 1  # identity[1] contributes to i2 position
            G[2][pos_i3] = 1  # identity[2] contributes to i3 position
            G[3][pos_i4] = 1  # identity[3] contributes to i4 position

            return G
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error creating generator matrix: {e}")

    
    def __create_parity_check_matrix(self):
        """
        Create the parity check matrix [H] for the Hamming code based on bit arrangement.
        
        The parity check matrix is constructed so that:
        - Each row corresponds to one parity bit equation
        - H * G^T = 0 (in modulo 2 arithmetic)

        For example, if the arrangement is ['p1', 'p2', 'p3', 'i1', 'i2', 'i3', 'i4'], then the matrix would be:
            [[1, 0, 0, 0, 1, 1, 1],  (p1 has  i2, i3, i4 in its equation)
             [0, 1, 0, 1, 0, 1, 1],  (p2 has i1, i3, i4 in its equation)
             [0, 0, 1, 1, 1, 0, 1]]  (p3 has i1, i2, i4 in its equation)

        Parameters:
        None

        Returns:
        H (list of lists): The 3x7 parity check matrix [H].
        """
        try:
            # Initialize 3x7 parity check matrix (3 parity bits x 7 codeword bits)
            H = [[0 for _ in range(7)] for _ in range(3)]
            
            # Find positions of each bit in the arrangement
            bit_arrangement = self.get_bit_arrangement()
            pos_p1 = bit_arrangement.index('p1')
            pos_p2 = bit_arrangement.index('p2')
            pos_p3 = bit_arrangement.index('p3')
            pos_i1 = bit_arrangement.index('i1')
            pos_i2 = bit_arrangement.index('i2')
            pos_i3 = bit_arrangement.index('i3')
            pos_i4 = bit_arrangement.index('i4')
            
            # Set up parity check equations
            # Row 0: p1 + i2 + i3 + i4 = 0 (parity check for p1)
            H[0][pos_p1] = 1
            H[0][pos_i2] = 1
            H[0][pos_i3] = 1
            H[0][pos_i4] = 1
            
            # Row 1: p2 + i1 + i3 + i4 = 0 (parity check for p2)
            H[1][pos_p2] = 1
            H[1][pos_i1] = 1
            H[1][pos_i3] = 1
            H[1][pos_i4] = 1
            
            # Row 2: p3 + i1 + i2 + i4 = 0 (parity check for p3)
            H[2][pos_p3] = 1
            H[2][pos_i1] = 1
            H[2][pos_i2] = 1
            H[2][pos_i4] = 1
            
            return H
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error creating parity check matrix: {e}")
    
    def __find_ibit_positions(self):
        """
        Find the positions of I-bits in the codeword from the bit arrangement.

        Parameters:
        None
        
        Returns:
        ibit_positions (list): A list of column indices where the I-bits are located, in order i1,i2,i3,i4.
        """
        try:
            ibit_positions = []
            
            # Find positions of i1, i2, i3, i4 in the bit arrangement
            bit_arrangement = self.get_bit_arrangement()
            for i, bit_type in enumerate(bit_arrangement):
                if bit_type.startswith('i'): # If it's an I-bit
                    ibit_positions.append(i) # Store the position
            
            # Sort by I-bit number to ensure correct order (i1, i2, i3, i4)
            # For example, if arrangement is ['i3', 'p2', 'i1', 'p1', 'i4', 'p3', 'i2']
            # Positions would be [2, 6, 0, 4] corresponding to i1, i2, i3, i4
            
            # Simple bubble sort approach
            for i in range(len(ibit_positions)):
                for j in range(len(ibit_positions) - 1):
                    pos1 = ibit_positions[j] 
                    pos2 = ibit_positions[j + 1]
                    # Get the number from 'i1', 'i2', etc.
                    num1 = int(bit_arrangement[pos1][1])
                    num2 = int(bit_arrangement[pos2][1])

                    # Because the bit-arrangement should be the exact same between the generator and parity check matrix creation and this function,
                    # We know that the positions correspond to the same bits
                    # For example, if we look for ibit_positions[1] = 6,  that corresponds to 'i2' in the arrangement
                    # So we can use the same logic to compare and sort if i2 is in the wrong order compared to i1, i3, i4

                    # Swap if out of order
                    if num1 > num2:
                        ibit_positions[j], ibit_positions[j + 1] = ibit_positions[j + 1], ibit_positions[j]
            
            return ibit_positions
            
        except (IndexError, ValueError) as e:
            raise ValueError(f"Error finding I-bit positions: {e}")

    def __is_matrix_linearly_independent(self, matrix):
        """
        Check if the Identity bits are linearly independent over FF(2).
        For 4 Identity bits, they are linearly independent if their determinant is non-zero in FF(2).

        Parameters:
        matrix (list of lists): The identity matrix for identity bits (4x4).

        Returns:
        det (bool): True if data bits are linearly independent, False otherwise.
        """

        # Calculate the determinant of the identity matrix over FF(2)
        det = self.determinant_FF2(matrix)

        if det is None: # Sanity check for determinant calculation failure
            raise ValueError("Error calculating determinant.")

        det = (det != 0)

        return det # In FF(2), non-zero determinant means linear independence

    def __extract_identity_matrix(self):
        """
        Extract the 4x4 identity portion from the generator matrix [G].
        This corresponds to the I-bit columns in the generator matrix.

        Parameters:
        None

        Returns:
        identity_matrix (list of lists): The 4x4 identity matrix portion.
        """
        try:
            # Get I-bit positions in sorted order (i1, i2, i3, i4)
            ibit_positions = self.__find_ibit_positions()
            
            # Extract the identity matrix columns from G
            identity_matrix = []
            for row in self.get_G(): # For each row in G
                identity_row = [] # To store the identity bits for this row
                for pos in ibit_positions: # For each I-bit position
                    identity_row.append(row[pos]) # Extract the bit at that position
                identity_matrix.append(identity_row)  # Add the row to the identity matrix
            
            return identity_matrix
            
        except (IndexError, ValueError) as e:
            raise ValueError(f"Error extracting identity matrix: {e}")

    def encode(self, data, generator_matrix=None):
        """
        Encode data bits into a Hamming codeword using the generator matrix [G].

        Standard encoding: codeword = data x G
        Where data is a 1x4 row matrix and G is a 4x7 matrix, producing a 1x7 codeword.
        
        Parameters:
        data (list of lists): A 2D column vector [[bit], [bit], [bit], [bit]] representing the data.
        generator_matrix (list of lists): The generator matrix [G] for encoding (4x7 format).
                                        If None, uses the instance's generator matrix.

        Returns:
        codeword_1d (list): A 1D row representing the encoded Hamming codeword.
        """
        try:
            if generator_matrix is None: # Use instance's generator matrix if not provided
                generator_matrix = self.__G
                
            if len(data) != 4:
                raise ValueError("Data list must contain exactly 4 bits for (7,4) Hamming code.")

            # Convert data from column vector format to row matrix format for multiplication
            data_1d = self.column_vector_to_list(data)  # Convert [[a], [b], [c], [d]] to [a, b, c, d]
            data_row = [data_1d]  # Convert to 2D row matrix [[a, b, c, d]] for matrix multiplication

            # Compute data x G (1x4 matrix x 4x7 matrix = 1x7 matrix)
            codeword_matrix = self.matrix_multiply(data_row, generator_matrix)

            if codeword_matrix is None: # Error during multiplication
                raise ValueError("Error during matrix multiplication.")
            
            # Apply modulo 2 arithmetic and convert to 1D row vector
            codeword_1d = [int(bit) % 2 for bit in codeword_matrix[0]]

            return codeword_1d
            
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f"Error encoding data: {e}")

    def __calculate_syndrome(self, received_bits, parity_check_matrix=None):
        """
        Calculate the syndrome for the received bits using the parity check matrix [H].
        
        Uses the formula: syndrome = H x received_bits

        If syndrome is all zeros, no error is detected.
        If syndrome is non-zero, it indicates the position of the error.
        
        Parameters:
        received_bits (list): A 1D row matrix representing the received codeword.
        parity_check_matrix (list of lists): The parity check matrix [H] for decoding.
                                           If None, uses the instance's parity check matrix.

        Returns:
        syndrome_2d (list of lists): A 2D column vector representing the syndrome.
        """
        try:
            if parity_check_matrix is None:
                parity_check_matrix = self.__H

            # Convert 1D received_bits to column vector format using inherited method
            received_bits_column = self.list_to_column_vector(received_bits)

            syndrome = self.matrix_multiply(parity_check_matrix, received_bits_column) # H (3x7) x received_bits (7x1) = syndrome (3x1)

            if syndrome is None: # Error during multiplication
                raise ValueError("Error during matrix-vector multiplication.")

            syndrome_2d = syndrome  # syndrome is already in 2D column vector format

            # Apply modulo 2 arithmetic to the syndrome result using element-wise operations
            for i in range(len(syndrome)):
                syndrome_2d[i][0] = int(syndrome_2d[i][0]) % 2

            return syndrome_2d

        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f"Error calculating syndrome: {e}")
    
    def __correct_error(self, received_bits, syndrome):
        """
        Correct a single-bit error in the received bits based on the syndrome.
        Can only correct single-bit errors with Hamming code.

        Parameters:
        received_bits (list): A 1D row matrix representing the received codeword.
        syndrome (list of lists): A 2D column vector representing the syndrome.

        Returns:
        corrected_bits (list): A 1D row matrix representing the corrected codeword.
        """
        try:
            # Check if there's no error (syndrome is all zeros)
            if all(row[0] == 0 for row in syndrome):
                # No error detected
                return received_bits
            
            # Find which column in H matrix matches the syndrome
            error_position = None

            for pos in range(len(received_bits)): # Iterate through each column position in H (0 to 6)
                column = []

                # Extract the actual column from H (3x7 matrix)
                for i in range(len(self.get_H())): 
                    column.append(self.get_H()[i][pos]) # Get the bit at row i, column pos

                if column == syndrome: # If the column matches the syndrome, then we found the error position
                    error_position = pos
                    break
            
            if error_position is None:
                raise ValueError("Syndrome does not match any column in the parity check matrix.")
            
            # Correct the error by flipping the bit at the error position (0-based index)
            corrected_bits = received_bits.copy()  # Make a copy of the 1D list
            corrected_bits[error_position] ^= 1  # Flip the bit using XOR
            
            return corrected_bits
            
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f"Error correcting error: {e}")

    def decode(self, received_bits, parity_check_matrix=None, syndrome=None, correct_errors=True):   
        """
        Decode the received bits using the parity check matrix [H] and syndrome.
        
        Parameters:
        received_bits (list): A 1D row matrix representing the received codeword.
        parity_check_matrix (list of lists): The parity check matrix [H] for decoding.
                                           If None, uses the instance's parity check matrix.
        syndrome (list of lists): A 2D column vector representing the syndrome.
                                If None, calculates the syndrome.
        correct_errors (bool): Whether to attempt error correction based on the syndrome.
        
        Returns:
        decoded_data (list of lists): A 2D column vector representing the decoded data bits.
        """
        try:
            if parity_check_matrix is None:
                parity_check_matrix = self.__H
                
            # Calculate the syndrome if not provided
            if syndrome is None:
                calculated_syndrome = self.__calculate_syndrome(received_bits, parity_check_matrix)
            else:
                calculated_syndrome = syndrome

            if correct_errors:
                corrected_bits = self.__correct_error(received_bits, calculated_syndrome)
            else:
                corrected_bits = received_bits

            # Extract the original input data from the corrected codeword using I-bit positions
            ibit_positions = self.__find_ibit_positions()
            
            # Extract input data bits from their actual positions in the codeword
            decoded_data = []
            for pos in ibit_positions:
                decoded_data.append([corrected_bits[pos]])

            return decoded_data
            
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f"Error decoding: {e}")


    
    def print_matrices(self):
        """
        Print the generator matrix [G] and parity check matrix [H] in a readable format.
        Useful for debugging and understanding the code structure.

        Parameters:
        None

        Returns:
        None
        """

        print(f"\nGenerator Matrix [G]:")
        print(self.get_G())

        print(f"\nParity Check Matrix [H]:")
        print(self.get_H())

        if self.get_data() is not None: # If data is set, show encoding, syndrome calculation, and decoding
            print(f"\nEncoding process for data {self.get_data()}:")
            encoded = self.encode(self.get_data(), self.get_G())
            print(f"  Encoded data: {encoded}")

            print(f"\nSyndrome for data {self.get_data()}:")
            syndrome = self.__calculate_syndrome(encoded, self.get_H()) # Send data through encoder and calculate syndrome
            print(syndrome)

            print(f"\nDecoding process for data {self.get_data()}:")
            decoded_data = self.decode(self.encode(self.get_data(), self.get_G()), self.get_H(), syndrome) # Decode the encoded data by encoding it first
            print(decoded_data)



    def test_capabilities(self):
        """
        Demonstrate Hamming Code (7,4) functionality with step-by-step mathematical process visualization.
        Shows different bit arrangements and error correction capabilities.

        Parameters:
        None

        Returns:
        None
        """
        print("\n\n")
        print("=" * 60) 
        print("Hamming Code (7,4) Mathematical Demonstration")
        print("=" * 60)
        print()
        
        # Test 1: Different bit arrangements
        print("1. Different Bit Arrangements Testing")
        print("-" * 50)
        print()
        
        arrangements = [
            ['p1', 'p2', 'p3', 'i1', 'i2', 'i3', 'i4'],  # Standard arrangement - parity first
            ['i1', 'i2', 'i3', 'i4', 'p1', 'p2', 'p3'],  # Data first
            ['p1', 'i1', 'p2', 'i3', 'p3', 'i4', 'i2'],  # Random
        ]

        arrangement_names = ["Standard (P-first)", "Data-first", "Random"]
        test_data = [[1], [0], [1], [0]]  # Test data vector
        
        print("Testing same data with different bit arrangements:")
        test_data_values = self.column_vector_to_list(test_data) # Convert to 1D for display
        print(f"  Test data: {test_data_values}") # Display test data
        print()
        
        for i, (arrangement, name) in enumerate(zip(arrangements, arrangement_names)): # Iterate through each arrangement, and use its name for display
            try:
                print(f"  {name} arrangement: {arrangement}")
                hamming = HammingCode(arrangement, test_data) # Create HammingCode instance with specific arrangement
                encoded = hamming.encode(test_data) # Encode the test data
                print(f"    Encoded codeword: {encoded}")
                
                # Test error correction
                corrupted = encoded.copy() 
                corrupted[3] ^= 1  # Flip bit at position 3 to simulate error
                print(f"    Corrupted (pos 3): {corrupted}")
                
                decoded = hamming.decode(corrupted, correct_errors=True) # Decode and correct errors
                decoded_values = self.column_vector_to_list(decoded) # Convert to 1D for display
                print(f"    Decoded (corrected): {decoded_values}")
                print(f"    Correction successful: {decoded_values == test_data_values}")
                print()
                
            except Exception as e:
                print(f"    Error testing arrangement {name}: {e}")
                print()

        # Test 2: Matrix relationships verification
        print("2. Matrix Relationships Verification")
        print("-" * 50)
        print()
        
        # Use standard arrangement for detailed analysis
        hamming = HammingCode(['p1', 'p2', 'p3', 'i1', 'i2', 'i3', 'i4'], test_data)
        G = hamming.get_G() 
        H = hamming.get_H()
        
        print("Generator Matrix [G] (4x7):") 
        for i, row in enumerate(G): # Print each row of G with row number
            print(f"  Row {i+1}: {row}")
        print()
        
        print("Parity Check Matrix [H] (3x7):")
        for i, row in enumerate(H): # Print each row of H with row number
            print(f"  Row {i+1}: {row}")
        print()
        
        # Verify H * G^T = 0 (mod 2)
        print("Verification: H * G^T should equal zero matrix (mod 2)")
        try:
            G_transpose = hamming.transpose(G) # Transpose G to get G^T (7x4)
            print(f"  G^T (7x4):")

            if G_transpose is None:
                raise ValueError("Error transposing G matrix.")
            
            for i, row in enumerate(G_transpose):
                print(f"    Row {i+1}: {row}") 

            HG_product = hamming.matrix_multiply(H, G_transpose) # H (3x7) x G^T (7x4) = HG^T (3x4)
            
            if HG_product is not None: # Check multiplication success
                print(f"  H * G^T (3x4):")
                HG_mod2 = [[int(element) % 2 for element in row] for row in HG_product] # This loop applies mod 2 to each element in the product matrix
                print(f"  H * G^T = {HG_mod2}")
                is_zero = all(all(cell == 0 for cell in row) for row in HG_mod2) # Check if all elements are zero
                print(f"  Is zero matrix: {is_zero}")

        except (ValueError, TypeError) as e:
            print(f"  Error verifying matrix relationship: {e}")
        
        # Verify linear independence of identity bits
        print("\nVerification: Identity bits should be linearly independent over FF(2)")
        try:
            identity_matrix = hamming.__extract_identity_matrix() # Extract identity matrix from G
            print(f"  Identity matrix (4x4): {identity_matrix}")

            is_independent = hamming.__is_matrix_linearly_independent(identity_matrix) # Check linear independence
            det_value = hamming.determinant_FF2(identity_matrix) # Get determinant value for display
            print(f"  Determinant over FF(2): {det_value}")
            print(f"  Linearly independent: {is_independent}")
                
        except (ValueError, TypeError) as e:
            print(f"  Error verifying linear independence: {e}")
        
        # Demonstrate syndrome calculation for correct and corrupted codewords
        print("\nSyndrome Demonstration:")
        try:
            # Show syndrome for correct codeword (should be all zeros)
            correct_codeword = hamming.encode(test_data) # Encode to get correct codeword
            correct_syndrome = hamming.__calculate_syndrome(correct_codeword) # Calculate syndrome for correct codeword
            print(f"  Correct codeword: {correct_codeword}")
            print(f"  Syndrome for correct codeword: {[row[0] for row in correct_syndrome]} (should be [0,0,0])")
            
            # Show syndromes for single-bit errors at each position
            print(f"  Syndromes for single-bit errors:")
            for error_pos in range(7):
                corrupted = correct_codeword.copy()
                corrupted[error_pos] ^= 1  # Introduce single-bit error
                error_syndrome = hamming.__calculate_syndrome(corrupted)
                syndrome_values = [row[0] for row in error_syndrome]
                print(f"    Error at position {error_pos}: syndrome = {syndrome_values}")
                print(f"      Corrupted codeword: {corrupted}\n\n")
                
        except (ValueError, TypeError) as e:
            print(f"  Error demonstrating syndromes: {e}")
        print()

        # Test 3: Comprehensive error correction testing
        print("3. Comprehensive Error Correction Testing")
        print("-" * 50)
        print()
        
        test_vectors = [
            [[0], [0], [0], [0]],  # All zeros
            [[1], [1], [1], [1]],  # All ones
            [[1], [0], [1], [0]],  # Alternating
            [[0], [1], [1], [0]],  # Different pattern
        ]
        
        vector_names = ["0000", "1111", "1010", "0110"]
        
        print("Testing error correction for different data patterns:")
        
        for vec, name in zip(test_vectors, vector_names): # Iterate through each test vector with its name for display
            try:
                vec_values = self.column_vector_to_list(vec) # Convert to 1D for display
                print(f"\n  Data pattern {name}: {vec_values}") 
                hamming = HammingCode(['p1', 'p2', 'p3', 'i1', 'i2', 'i3', 'i4'], vec) # Create HammingCode instance
                
                correct_codeword = hamming.encode(vec) # Encode the data
                print(f"    Correct codeword: {correct_codeword}") # Display correct codeword
                
                # Test single-bit errors at each position
                error_corrections = []
                for error_pos in range(7): # Introduce single-bit error at each position
                    corrupted = correct_codeword.copy()
                    corrupted[error_pos] ^= 1  # Introduce error
                    
                    try:
                        decoded = hamming.decode(corrupted, correct_errors=True) # Decode and correct errors
                        # Check if decoded data matches original data
                        correction_success = [row[0] for row in decoded] == [row[0] for row in vec]
                        error_corrections.append(correction_success) # Store success status
                    except:
                        error_corrections.append(False) # If decoding fails, mark as unsuccessful
                
                successful_corrections = sum(error_corrections) # Count successful corrections
                print(f"    Single-bit error corrections: {successful_corrections}/7 successful")

            except Exception as e:
                print(f"    Error testing pattern {name}: {e}")


        print()
        print("Mathematical Summary:")
        print("  * Encoding: codeword = data x G (1x4 vector x 4x7 matrix = 1x7 vector)")
        print("  * Syndrome: syndrome = H x received_codeword (3x7 matrix x 7x1 vector = 3x1 vector)")
        print("  * Error correction: single-bit errors detected and corrected")
        print("  * All operations in FF(2): addition = XOR, multiplication = AND")


def main():
    """
    Run comprehensive demonstration of Hamming Code capabilities with different arrangements.
    """

    data = [[1], [0], [1], [0]]
    hamming = HammingCode(['p1', 'p2', 'p3', 'i1', 'i2', 'i3', 'i4'], data)
    hamming.test_capabilities()
    

if __name__ == "__main__":
    main()  
