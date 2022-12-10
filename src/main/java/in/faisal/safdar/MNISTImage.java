package in.faisal.safdar;

import org.ejml.simple.SimpleMatrix;

import java.awt.image.BufferedImage;

public interface MNISTImage {
    public SimpleMatrix simpleMatrixFloat0To1();
    public int getDigit();
    public BufferedImage bufferedImage();
    public SampleId id();
}
