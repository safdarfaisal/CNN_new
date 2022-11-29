package in.faisal.safdar;

import org.ejml.simple.SimpleMatrix;

import java.security.KeyPair;

public class SimpleMatrixEx {
    //v0.41 does not have elementMax, only elementMaxAbs
    private SimpleMatrix matrix;

    SimpleMatrixEx(SimpleMatrix m) {
        matrix = m;
    }

    public double elementMax() {
        double max = matrix.get(0, 0);
        for (int i = 0; i < matrix.numRows(); i++) {
            for (int j = 0; j < matrix.numCols(); j++) {
                double d = matrix.get(i, j);
                if (max < d) {
                    max = d;
                }
            }
        }
        return max;
    }

    //return e^(a[i,j])
    public SimpleMatrix exp() {
        SimpleMatrix m = new SimpleMatrix(matrix);
        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numCols(); j++) {
                double d = m.get(i, j);
                m.set(i, j, Math.exp(d));
            }
        }
        return m;
    }

    //return index of max element
    public IndexPair indexOfMax() {
        double max = matrix.get(0, 0);
        IndexPair p = new IndexPair(0, 0);
        for (int i = 0; i < matrix.numRows(); i++) {
            for (int j = 0; j < matrix.numCols(); j++) {
                double d = matrix.get(i, j);
                if (max < d) {
                    max = d;
                    p.row = i;
                    p.column = j;
                }
            }
        }
        return p;
    }
}
