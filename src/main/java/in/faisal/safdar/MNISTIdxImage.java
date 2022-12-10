package in.faisal.safdar;

import org.ejml.simple.SimpleMatrix;

import java.awt.image.BufferedImage;

//TODO: Inheritance vs delegate handling for all cases

public class MNISTIdxImage implements MNISTImage {
    private byte[] imgBytes;
    private int digit;
    private int width;
    private int height;
    private SampleId id;

    MNISTIdxImage(byte[] bytes, int d, int w, int h, SampleId i) {
        imgBytes = bytes;
        digit = d;
        width = w;
        height = h;
        id = i;
    }

    public SimpleMatrix simpleMatrixFloat0To1() {
        /* TODO: The commented code below should work, but it does not.
        We are creating a BufferedImage and converting back to byte values to keep the code path
        the same as reading from PNG, and things work beautifully. Need to check how we can get
        identical results by reading directly from IDX file.
         */
        /*
        float[] pxf = new float[imgBytes.length];
        for(int j = 0; j < imgBytes.length; j++) {
            pxf[j] = (imgBytes[j])/255.0f;
        }
        return new SimpleMatrix(width, height, true, pxf);
        */
        BufferedImage image = bufferedImage();
        int width = image.getWidth();
        int height = image.getHeight();
        int[] pxi = image.getRGB(0, 0, width, height, null, 0, width);
        float[] pxf = new float[pxi.length];
        for (int j = 0; j < pxi.length; j++) {
            pxf[j] = (pxi[j] >> 16 & 0xff) / 255.0f;
        }
        return new SimpleMatrix(width, height, true, pxf);
    }

    public int getDigit() {
        return digit;
    }

    public BufferedImage bufferedImage() {
        BufferedImage newBi = null;
        newBi = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        int numberOfPixels = width * height;
        int[] imgPixels = new int[numberOfPixels];
        for (int p = 0; p < numberOfPixels; p++) {
            int gray = imgBytes[p];
            imgPixels[p] = 0xFF000000 | (gray << 16) | (gray << 8) | gray;
        }
        newBi.setRGB(0, 0, width, height, imgPixels, 0, width);
        return newBi;
    }

    public SampleId id() {
        return id;
    }
}
