package in.faisal.safdar;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;
import javax.imageio.ImageIO;

//represents the dataset with MNIST digit images
//Runs on MNIST digit data represented as PNG images, downloaded from
//https://www.kaggle.com/datasets/jidhumohan/mnist-png. Not using the original
//MNIST database from http://yann.lecun.com/exdb/mnist/ to avoid implementing
//idx file reading in Java.
//TODO: Implement IDX support and use original MNIST data.
public class MNISTDataset {
    public static final String MNIST_TRAINING_DATA_FOLDER = "data/mnist_png/training";
    public static final String MNIST_TEST_DATA_FOLDER = "data/mnist_png/testing";

    private static int randomDigit() {
        //random number from 0 to 9.
        return ThreadLocalRandom.current().nextInt(0, 10);
    }
    private static String randomImageFilePath(String dataFolderName, int rand) {
        String imageFolderPath = dataFolderName + "/" + rand;
        //System.out.println(imageFolderPath);
        String[] imageFilePathList = (new File(imageFolderPath)).list();
        int r = ThreadLocalRandom.current().nextInt(0, imageFilePathList.length);
        return imageFolderPath + "/" + imageFilePathList[r];
    }

    public static String test() {
        return randomImageFilePath(MNIST_TRAINING_DATA_FOLDER, randomDigit());
    }

    public static MNISTImage testSample() throws IOException {
        int digit = randomDigit();
        return new MNISTImage(ImageIO.read(new File(randomImageFilePath(MNIST_TEST_DATA_FOLDER, digit))), digit);
    }

    public static MNISTImage trainingSample() throws IOException {
        int digit = randomDigit();
        return new MNISTImage(ImageIO.read(new File(randomImageFilePath(MNIST_TRAINING_DATA_FOLDER, digit))), digit);
    }
}
