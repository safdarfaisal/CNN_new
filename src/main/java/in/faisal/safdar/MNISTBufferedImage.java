package in.faisal.safdar;

import org.ejml.simple.SimpleMatrix;

import java.awt.image.BufferedImage;

/*
represents a single image in MNIST database.
 */
public class MNISTBufferedImage implements MNISTImage {
    private final BufferedImage image;
    private final int digit;
    private final SampleId id;

    MNISTBufferedImage(BufferedImage img, int d, SampleId i) {
        image = img;
        digit = d;
        id = i;
    }

    public SimpleMatrix simpleMatrixFloat0To1() {
        int width = image.getWidth();
        int height = image.getHeight();
        int[] pxi = image.getRGB(0, 0, width, height, null, 0, width);
        float[] pxf = new float[pxi.length];
        for(int j = 0; j < pxi.length; j++) {
            pxf[j] = (pxi[j]>>16&0xff)/255.0f;
        }
        return new SimpleMatrix(width, height, true, pxf);
    }

    public int getDigit() {
        return digit;
    }

    public BufferedImage bufferedImage() {
        return image;
    }

    @Override
    public SampleId id() {
        return id;
    }
}
