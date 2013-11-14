import Jama.*;
import java.util.*;

/**
 * Class for the Steepest Descent Algorithm, first it computes a basic matrix 
 * solution. Then it runs and outputs the results of 5 10x10 random matrix 
 * equations(the A values are positive definite), and their solutions. 
 * 
 * @author Jonathan Foster 11/13/2013
 */
public class SteepestDescent {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		// This keeps track of how many steps it takes to find the solution
		int count = 0;
		
		// First we set up our array of matrices
		ArrayList<Matrix> Xmatrices = new ArrayList();
		ArrayList<Matrix> Dmatrices = new ArrayList();
		
		// Prescribe our accuracy
		final double ACCURACY = 0.00001;
		
		// Create our A and B matrices. 
		double[][] arrayA = {{5, 2}, {2, 1}};
		Matrix A = new Matrix(arrayA);
		double[][] arrayB = {{1,}, {1}};
		Matrix b = new Matrix(arrayB);
		System.out.print("Matrix A: ");
		A.print(1, 0);
		System.out.print("Matrix b: ");
		b.print(1, 0);
		
		// create any vector initialX, add this to the matrices array
		// we'll start will all 1's, but see if other things work better. 
		double[][] arrayX = {{1,}, {1}};
		Matrix x = new Matrix(arrayX);
		Xmatrices.add(x);
		
		// compute D, this will be d_k-1 because each iteration begins after we have our initial x. 
		Matrix d = computeD(A, b, x);
		Dmatrices.add(d);
		
		// Here is our first simple test, outputs the answer of the above matrices
		System.out.println("Solution vector x to matrices A and b: ");
		
		// call our algorithm
		Matrix answer = steepestDescentDPAlgorithm(Xmatrices, Dmatrices, A, b, ACCURACY, count);
		
		/********************************************************************************************
		 * Now that we know our algorithm works, we'll increase the value of x_0 to see how it 
		 * affects the number of iterations it takes to get to a solution. 
		 ********************************************************************************************/
		
		for (int i=0; i<5; i++) {
			System.out.println("*****************************************************************************");
			System.out.println("Increasing x, Iteration #: " +(i+1));
			Xmatrices.clear();
			Dmatrices.clear();
			count = 0;
		
			// Print our values
			System.out.print("Matrix A: ");
			A.print(1, 0);
			System.out.print("Matrix b: ");
			b.print(1, 0);
			
			// Computes any x as a starting point
			double[][] newarrayx = {{i+1}, {0}};
			Matrix newx = new Matrix(newarrayx);
			Xmatrices.add(newx);
			
			// compute D, this will be d_k-1 because each iteration begins after we have our initial x. 
			Matrix newd = computeD(A, b, newx);
			Dmatrices.add(newd);
			
			System.out.println("Solution vector x to matrices A and b: ");
			Matrix newAnswer = steepestDescentDPAlgorithm(Xmatrices, Dmatrices, A, b, ACCURACY, count);		
		}
		
		System.out.println("Now we have seen that as you increase X_0, the number of iterations it takes" +
				" to find solutions decreases!!!!!!!!!!!!!!!\n");
		
		/********************************************************************************************
		 * Now that we have seen that our algorithm correctly finds solutions, we are going to create
		 * 5 random 10x10 positive definite matrices A and vectors b and solve Ax = b
		 ********************************************************************************************/
		
		for (int i=0; i<5; i++) {
			System.out.println("*****************************************************************************");
			System.out.println("Random Iteration #: " +(i+1));
			Xmatrices.clear();
			Dmatrices.clear();
			count = 0;
			Matrix randb = Matrix.random(10, 1);
			Matrix B = generateRandomMatrix();
			Matrix randA = computeRandomA(B);
			
			// Print our values
			System.out.print("Matrix A: ");
			randA.print(1, 0);
			System.out.print("Matrix b: ");
			randb.print(1, 0);
			
			// Computes any x as a starting point
			Matrix newx = Matrix.random(10, 1);
			Xmatrices.add(newx);
			
			// compute D, this will be d_k-1 because each iteration begins after we have our initial x. 
			Matrix newd = computeD(randA, randb, newx);
			Dmatrices.add(newd);
			
			System.out.println("Solution vector x to matrices A and b: ");
			Matrix newAnswer = steepestDescentDPAlgorithm(Xmatrices, Dmatrices, randA, randb, ACCURACY, count);		
		}
		
	}
	
	/**
	 * Here is our main algorithm, it follows the formula of computing vector x, while
	 * the norm of d is greater than our prescribed accuracy.  
	 * 
	 * @param Xmatrices array holding our x matrices
	 * @param Dmatrices array for holding our D matrices
	 * @param A the matrix A
	 * @param b the matrix b
	 * @param accuracy the accuracy to which our solution is calculated
	 * @param count the number of iterations our algorithm does
	 * @return Matrix answer, our solution matrix
	 */
	public static Matrix steepestDescentDPAlgorithm(ArrayList<Matrix> Xmatrices, 
			ArrayList<Matrix> Dmatrices, Matrix A, Matrix b, double accuracy, int count) {
		int i = 1;
		double dNorm = Dmatrices.get(0).normF();
		while (dNorm >= accuracy) {
			Matrix tempX = computeX(Xmatrices.get(i-1), Dmatrices.get(i-1), A);
			Xmatrices.add(tempX);
			Matrix nextD = computeD(A, b, Xmatrices.get(i));
			Dmatrices.add(nextD);
			dNorm = Dmatrices.get(i-1).normF();
			i++;
			count++;
		}
		Matrix answer = Xmatrices.get(Xmatrices.size()-1);
		System.out.print("Number of iterations: " +count);
		answer.print(1, 0);
		return answer;
	}
	
	/**
	 * Here we use the formula -(A*x_k - b) to calculate our value for d_k
	 * 
	 * @param A - the Matrix A
	 * @param b - the matrix b
	 * @param x - the prev x value
	 * @return - our new d value
	 */
	public static Matrix computeD(Matrix A, Matrix b, Matrix x) {
		Matrix temp = A.times(x);
		temp = temp.minus(b);
		temp = temp.times(-1);
		return temp;
	}
	
	/**
	 * Check to see that the top part is supposed to be the norm or absolute value.
	 * 
	 * @param prevX the previous X matrix
	 * @param prevD the previous D matrix
	 * @param A the matrix A
	 * @return our new x value
	 */
	public static Matrix computeX(Matrix prevX, Matrix prevD, Matrix A) {
		Matrix x;
		// calculate the top part of our equation. 
		double prevDSquared = prevD.normF();
		prevDSquared *= prevDSquared;
		Math.abs(prevDSquared);
		
		// here we do the denominator
		Matrix denom = A.times(prevD);
		double dot = dotproduct(prevD, denom);	
		
		// this gets the whole phrase in the middle of the equation.
		double coefficient = prevDSquared/dot;
		
		denom = prevD.times(coefficient);
		x = prevX.plus(denom);
				
		return x;
	}
	
	/** Determines if a given matrix is a row vector, that is, it has only one row.
	 * 
	 * @param m the matrix.
	 * @return whether the given matrix is a row vector (whether it has only one row).
	 */
	public static boolean isRowVector(Matrix m) {
		return m.getRowDimension()==1;
	}
	
	
	/** Determines if a given matrix is a column vector
	 * 
	 * @param m the matrix.
	 * @return whether the given matrix is a column vector. 
	 */
	public static boolean isColumnVector(Matrix m) {
		return m.getColumnDimension()==1;
	}
	
	/** Transforms the given matrix into a column vector, that is, a matrix with one column.
	 * The matrix must be a vector (row or column) to begin with.
	 * 
	 * @param m
	 * @return <code>m.transpose()</code> if m is a row vector,
	 *         <code>m</code> if m is a column vector.
	 * @throws IllegalArgumentException if m is not a row vector or a column vector.
	 */
	public static Matrix makeColumnVector(Matrix m) {	
		if (isColumnVector(m))
			return m;
		else if (isRowVector(m))
			return m.transpose();
		else
			throw new IllegalArgumentException("m is not a vector.");
	}
	
	/** Computes the dot product of two vectors.
	 * 
	 * @param m1
	 * @param m2
	 * @return the dot product of the two vectors.
	 */        
	public static double dotproduct(Matrix m1, Matrix m2) {
		
		Matrix m1colVector = makeColumnVector(m1);
		Matrix m2colVector = makeColumnVector(m2);
		
		int n = m1colVector.getRowDimension();
		if (n != m2colVector.getRowDimension()) {
			throw new IllegalArgumentException("m1 and m2 must have the same number of elements.");
		}
		
		double scalarProduct = 0;
		for (int row=0; row<n; row++) {
			scalarProduct += m1colVector.get(row,0) * m2colVector.get(row,0);
		}
		
		return scalarProduct;
		
	}
	
	/**
	 * Generates a random matrix
	 * @return random matrix
	 */
	public static Matrix generateRandomMatrix() {
		Matrix B = Matrix.random(10, 10);
		return B;
	}
	
	/**
	 * Computes B_transposed times B to get A
	 * @param B the random matrix
	 * @return the answer 
	 */
	public static Matrix computeRandomA(Matrix B) {
		Matrix BTrans = B.transpose();
		Matrix A = BTrans.times(B);
		return A;
	}

}
