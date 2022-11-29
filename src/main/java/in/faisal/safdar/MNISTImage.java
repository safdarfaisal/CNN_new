package in.faisal.safdar;

import java.awt.image.BufferedImage;

/*
represents a single image in MNIST database.
 */
public class MNISTImage {
    public BufferedImage image;
    public int digit;

    MNISTImage(BufferedImage img, int d) {
        image = img;
        digit = d;
    }
}
