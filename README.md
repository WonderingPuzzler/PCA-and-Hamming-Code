# **PCA and Hamming Code Programs, with Custom-Made Vector and Matrix Operations!**

This is a project I created as part of a Honors Contract for Linear Algebra!

PCA is used on data matrices to extract the most important 'pieces' of that data. In the case of something like a digital image, these might be the distinguishing parts of the image, or pixels, since that's what the data matrix would be made out of, that could be used to classify what type of object or person the image is of. It's mainly used on huge data matrices to try and improve classification or data interpretation of Machine Learning Algorithms. Basically, you put the data through PCA, send it to the actual machine learning algorithm, and hopefully your output is better than if you hadn't used it!

The Linear Algebra aspect of PCA comes in not just because we're using matrices, and performing matrix operations on the data, to try and make it as easy as possible to distinguish the most important features of things like images, but because PCA is basically about finding eigenvectors and eigenvalues, data which correlates to the most important features of the image! However, again, PCA is about dealing with relatively large data matrices (in the hundreds and even thousands and above in their length and width). 

Of course, you can't reasonably find the eigenvectors and eigenvalues of a PCA-sized matrix with basic Linear Algebra techniques, so I had to use more complex ones. Specifically, I decided to use Power Iteration to find eigenvectors, Rayleigh Quotient to find eigenvalues, and then Hotelling's deflation to remove already found eigenvalues/eigenvectors.

Additionally, I created a whole bunch of matrix operations (matrix subtraction, matrix vector multiplication, matrix multiplication, length/magnitude of a column vector, and much more) because PCA uses lots of matrix and vector math along with what's needed in the three aforementioned algorithms. I created visualized output for that part too! To end off the PCA section, I made a very basic image classifier! I used a machining learning algorithm called K-Nearest Neighbors (with some help online) and fed it the PCA-transformed matrix! This repo has an example output as shown:

![alt text](https://github.com/WonderingPuzzler/PCA-and-Hamming-Code/blob/main/pca_classification_results_30_samples.png "PCA_Image_classification_Example")

After creating the PCA and Image Classifier, I then created Hamming Code program! This one uses principles like vector and matrix multiplication to find and correct errors within numerical input it receives. It also has features of linear independence and often involves the identity matrix, or a similar such matrix, in its error-finding and correcting algorithms. Example outputs for this one are provided as well!

## Credits

Big thanks to the following sources! They've been huge helps to me in the process of creating these programs!

Michael Dipperstein - https://michaeldipperstein.github.io/hamming.html 
Vandita Gupta - https://www.geeksforgeeks.org/python/outer-product-on-vector/
Zakaria Jaadi https://builtin.com/data-science/step-step-explanation-principal-component-analysis
aishwarya - https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/
Michel Kulhandjian & Atri Rudra - https://cse.buffalo.edu/faculty/atri/courses/coding-theory/lectures/lect5.pdf
Haoru Liu - https://math.uchicago.edu/~may/REU2012/REUPapers/LiuH.pdf
Marvin Lanhenke - https://builtin.com/data-science/eigendecomposition
Mirko Savasta - https://medium.com/analytics-vidhya/pca-a-linear-transformation-f8aacd4eb007
Stephen Roberts & Michaelmas Term - https://www.robots.ox.ac.uk/~sjrob/Teaching/EngComp/ecl4.pdf
Weisstein, Eric W. "Finite Field." From MathWorld--A Wolfram Resource. https://mathworld.wolfram.com/FiniteField.html
Dan Margalit, Joseph Rabinoff, & Larry Rolen - https://textbooks.math.gatech.edu/ila/determinants-cofactors.html
vinayedula - https://www.geeksforgeeks.org/python/power-method-determine-largest-eigenvalue-and-eigenvector-in-python/
kartik - https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/

