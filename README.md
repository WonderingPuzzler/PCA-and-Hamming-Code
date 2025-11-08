# **PCA and Hamming Code Programs, with Custom-Made Vector and Matrix Operations!**

This is a project I created as part of a Honors Contract for Linear Algebra!

PCA is used on data matrices to extract the most important 'pieces' of that data. In the case of something like a digital image, these might be the distinguishing parts of the image, or pixels, since that's what the data matrix would be made out of, that could be used to classify what type of object or person the image is of. It's mainly used on huge data matrices to try and improve classification or data interpretation of Machine Learning Algorithms. Basically, you put the data through PCA, send it to the actual machine learning algorithm, and hopefully your output is better than if you hadn't used it!

The Linear Algebra aspect of PCA comes in not just because we're using matrices, and performing matrix operations on the data, to try and make it as easy as possible to distinguish the most important features of things like images, but because PCA is basically about finding eigenvectors and eigenvalues, data which correlates to the most important features of the image! However, again, PCA is about dealing with relatively large data matrices (in the hundreds and even thousands and above in their length and width). 

Of course, you can't reasonably find the eigenvectors and eigenvalues of a PCA-sized matrix with basic Linear Algebra techniques, so I had to use more complex ones. Specifically, I decided to use Power Iteration to find eigenvectors, Rayleigh Quotient to find eigenvalues, and then Hotelling's deflation to remove already found eigenvalues/eigenvectors.

Additionally, I created a whole bunch of matrix operations (matrix subtraction, matrix vector multiplication, matrix multiplication, length/magnitude of a column vector, and much more) because PCA uses lots of matrix and vector math along with what's needed in the three aforementioned algorithms. I created visualized output for that part too! To end off the PCA section, I made a very basic image classifier! I used a machining learning algorithm called K-Nearest Neighbors (with some help online) and fed it the PCA-transformed matrix! This repo has an example output as shown:

![alt text](https://github.com/WonderingPuzzler/PCA-and-Hamming-Code/blob/main/pca_classification_results_30_samples.png "PCA_Image_classification_Example")

After creating the PCA and Image Classifier, I then created Hamming Code program! This one uses principles like vector and matrix multiplication to find and correct errors within numerical input it receives. It also has features of linear independence and often involves the identity matrix, or a similar such matrix, in its error-finding and correcting algorithms. Example outputs for this one are provided as well!

## Usage

For details on usage, please look at the following section: [Usage.md ](https://github.com/WonderingPuzzler/PCA-and-Hamming-Code/blob/main/Usage.md)

## Credits

Big thanks to the following sources! They've been a huge help to me in the process of making these programs!

aishwarya.27. (2025, July 11). Principal component analysis(Pca). GeeksforGeeks. https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/

Bourke, D. (2024a, September 3). 03. Pytorch computer vision. Github. https://www.learnpytorch.io/03_pytorch_computer_vision/

Bourke, D. (2024b, September 11). 01. Pytorch workflow fundamentals. Learnpytorch.Io; GitHub. https://www.learnpytorch.io/01_pytorch_workflow/

Bourke, D. (2025, January 6). 02. Pytorch neural network classification. GitHub. https://www.learnpytorch.io/02_pytorch_classification/

Dipperstein, M. (2018, December 27). Hamming (7,4) code discussion and implementation. GitHub. https://michaeldipperstein.github.io/hamming.html

Gupta, V. (2025, May 8). Outer Product on Vector. GeeksforGeeks. https://www.geeksforgeeks.org/python/outer-product-on-vector/

Jaadi, Z. (2025, June 23). Principal component analysis (Pca): Explained step-by-step. Built In. https://builtin.com/data-science/step-step-explanation-principal-component-analysis

kartik. (2025, August 23). K-nearest neighbor(Knn) algorithm. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/

Kulhandjian , M., & Rudra, A. (2007, September 7). Lecture 5: Linear Codes. Buffalo.Edu; Department of Computer Science and Engineering - University at Buffalo. https://cse.buffalo.edu/faculty/atri/courses/coding-theory/lectures/lect5.pdf

Lanhenke, M. (2023, November 16). Eigendecomposition explained. Built In. https://builtin.com/data-science/eigendecomposition

Lay, D. C., Lay, S. R., & McDonald, J. (2021). Linear algebra and its applications (Sixth edition. Rental edition.). Pearson.

Liu, H. (2012, September 29). ERROR-CORRECTING CODES AND FINITE FIELDS. Math.UChicago; University of Chicago. https://math.uchicago.edu/~may/REU2012/REUPapers/LiuH.pdf

Love, B. (2016a, January 9). Linear Algebra—Determinants (1 of 2). YouTube. https://www.youtube.com/watch?v=WAg-ozv8Trg

Love, B. (2016b, January 9). Linear Algebra—Determinants (2 of 2). YouTube. https://www.youtube.com/watch?v=-SfhIxoiD7I

Love, B. (2016c, January 9). Linear Algebra—Eigenvalues and Eigenvectors. YouTube. https://www.youtube.com/watch?v=5_CLdaQSE6U

Love, B. (2016d, January 9). Linear Algebra—Inner Product, Vector Length, Orthogonality. YouTube. https://www.youtube.com/watch?v=-DDsguw-M2w

Love, B. (2016e, January 9). Linear Algebra—Linear Independence. YouTube. https://www.youtube.com/watch?v=FM6DbT6J1XQ

Love, B. (2016f, January 9). Linear Algebra—Matrix Diagonalization. YouTube. https://www.youtube.com/watch?v=jkG8kF8BM20

Love, B. (2016g, January 9). Linear Algebra—Matrix Operations. YouTube. https://www.youtube.com/watch?v=rUrFNrmp3s4

Love, B. (2016h, January 9). Linear Algebra—The Matrix Equation Ax = b (1 of 2). YouTube. https://www.youtube.com/watch?v=Yz2qH7c5yQM&t=1s

Love, B. (2016i, January 9). Linear Algebra—The Matrix Equation Ax = b (2 of 2). YouTube. https://www.youtube.com/watch?v=-pvnRTwXxqY

Love, B. (2016j, January 9). Linear Algebra—Vector Equations (1 of 2). YouTube. https://www.youtube.com/watch?v=ghfTtjLz7bI

Love, B. (2016k, January 9). Linear Algebra—Vector Equations (2 of 2). YouTube. https://www.youtube.com/watch?v=VjkZful09sA

Margalit, D., Rabinoff, J., & Rolen, L. (2019). Interactive linear algebra.  Georgia Institute of Technology. https://textbooks.math.gatech.edu/ila/ila.pdf (Original work published 2017)

prabhjotkushparmar. (2025, July 23). Covariance matrix. GeeksforGeeks. https://www.geeksforgeeks.org/maths/covariance-matrix/

Raschka, S., & Mirjalili, V. (04). Python machine learning: Machine learning and deep learning with Python, scikit-learn, and TensorFlow (Second edition, fourth release,[fully revised and updated]). Packt Publishing.

Roberts , S., & Term , M. (n.d.). Computation of matrix eigenvalues and eigenvectors . University of Oxford.

Savasta, M. (2019, April 7). Pca: A linear transformation. Analytics Vidhya. https://medium.com/analytics-vidhya/pca-a-linear-transformation-f8aacd4eb007

vinayedula. (2025, April 28). Power method—Determine largest eigenvalue and eigenvector in python. GeeksforGeeks. https://www.geeksforgeeks.org/python/power-method-determine-largest-eigenvalue-and-eigenvector-in-python/

Weisstein, Eric W. "Finite Field." From MathWorld--A Wolfram Resource. https://mathworld.wolfram.com/FiniteField.html

