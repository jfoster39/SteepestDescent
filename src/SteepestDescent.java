import Jama.*;
import java.util.*;

public class SteepestDescent {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] array = {{5, 2},{2, 1}};
		Matrix P = new Matrix(array);
		double[][] array1 = {{1},{1}};
		Matrix q = new Matrix(array1);
		Matrix z = P.solve(q);
		
		P.print(1, 0);
		q.print(1, 0);
		z.print(1, 0);
	
		
		// First we set up our array of matrices
		ArrayList<Matrix> Xmatrices = new ArrayList();
		ArrayList<Matrix> Dmatrices = new ArrayList();
		
		// Prescribe our accuracy
		double accuracy = 0.00001;
		
		// Create our A and B matrices. 
		double[][] arrayA = {{5, 2}, {2, 1}};
		Matrix A = new Matrix(arrayA);
		double[][] arrayB = {{1,}, {1}};
		Matrix b = new Matrix(arrayB);
		A.print(1, 0);
		b.print(1, 0);
		
		// create any vector initialX, add this to the matrices array
		// we'll start will all 1's, but see if other things work better. 
		double[][] arrayX = {{1,}, {1}};
		Matrix x = new Matrix(arrayX);
		Xmatrices.add(x);
		// x.print(1, 0);
		
		// compute D, this will be d_k-1 because each iteration begins after we have our inital x. 
		Matrix d = computeD(A, b, x);
		Dmatrices.add(d);
		//d.print(1, 0);
		
		Matrix test = computeX(x, d, A);
		//test.print(1, 0);
		
		// call our algorithm
		Matrix answer = steepestDescentDPAlgorithm(Xmatrices, Dmatrices, A, b, accuracy);
		
		answer.print(1, 0);
	}
	
	public static Matrix steepestDescentDPAlgorithm(ArrayList<Matrix> Xmatrices, 
			ArrayList<Matrix> Dmatrices, Matrix A, Matrix b, double accuracy) {
		int i = 1;
		double dNorm = Dmatrices.get(0).normF();
		while (dNorm >= accuracy) {
			Matrix tempX = computeX(Xmatrices.get(i-1), Dmatrices.get(i-1), A);
			Xmatrices.add(tempX);
			Matrix nextD = computeD(A, b, Xmatrices.get(i));
			Dmatrices.add(nextD);
			dNorm = Dmatrices.get(i-1).normF();
			i++;
		}
		Matrix answer = Xmatrices.get(Xmatrices.size()-1);
		return answer;
	}
	
	// Here we use the formula -(A*x_k - b) to calculate our value for d_k
	public static Matrix computeD(Matrix A, Matrix b, Matrix x) {
		Matrix temp = A.times(x);
		temp = temp.minus(b);
		temp = temp.times(-1);
		return temp;
	}
	
	// check to see that the top part is supposed to be the norm or absolute value
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
	 * @param m the matrix.
	 * @return whether the given matrix is a row vector (whether it has only one row).
	 */
	public static boolean isRowVector(Matrix m) {
		return m.getRowDimension()==1;
	}
	
	
	/** Determines if a given matrix is a column vector, that is, it has only one column.
	 * @param m the matrix.
	 * @return whether the given matrix is a column vector (whether it has only one column).
	 */
	public static boolean isColumnVector(Matrix m) {
		return m.getColumnDimension()==1;
	}
	
	/** Transforms the given matrix into a column vector, that is, a matrix with one column.
	 * The matrix must be a vector (row or column) to begin with.
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
	
	/** Computes the dot product of two vectors.  Both must be either row or column vectors.
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

}
