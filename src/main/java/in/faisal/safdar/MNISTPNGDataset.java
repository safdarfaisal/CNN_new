package in.faisal.safdar;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import javax.imageio.ImageIO;

//represents the dataset with MNIST digit images
//Runs on MNIST digit data represented as PNG images, downloaded from
//https://www.kaggle.com/datasets/jidhumohan/mnist-png. Not using the original
//MNIST database from http://yann.lecun.com/exdb/mnist/ to avoid implementing
//idx file reading in Java.
//TODO: Implement IDX support and use original MNIST data.
public class MNISTPNGDataset implements MNISTDataset {
    public static final String MNIST_TRAINING_DATA_FOLDER = "data/mnist_png/training";
    public static final String MNIST_TEST_DATA_FOLDER = "data/mnist_png/testing";

    private String randomImageFilePath(String dataFolderName, int rand) {
        String imageFolderPath = dataFolderName + "/" + rand;
        //System.out.println(imageFolderPath);
        String[] imageFilePathList = (new File(imageFolderPath)).list();
        assert imageFilePathList != null;
        int r = ThreadLocalRandom.current().nextInt(0, imageFilePathList.length);
        return imageFolderPath + "/" + imageFilePathList[r];
    }

    public String test() {
        return randomImageFilePath(MNIST_TRAINING_DATA_FOLDER, Utils.randomDigit());
    }

    public MNISTBufferedImage testSample() throws IOException {
        int digit = Utils.randomDigit();
        String path = randomImageFilePath(MNIST_TEST_DATA_FOLDER, digit);
        return new MNISTBufferedImage(ImageIO.read(new File(path)), digit, new SampleId(path));
    }

    public MNISTBufferedImage trainingSample() throws IOException {
        int digit = Utils.randomDigit();
        String path = randomImageFilePath(MNIST_TRAINING_DATA_FOLDER, digit);
        return new MNISTBufferedImage(ImageIO.read(new File(path)), digit, new SampleId(path));
    }

    @Override
    public MNISTImage trainingSample(SampleId index) throws IOException {
        return null;
    }

    @Override
    public int trainingSampleCount() {
        return 0;
    }

    @Override
    public int testSampleCount() {
        return 0;
    }

    @Override
    public List<SampleId> trainingSubset(int count) {
        return null;
    }

    @Override
    public List<SampleId> testSubset(int count) {
        return null;
    }

    public MNISTBufferedImage validationSample() throws IOException {
        return null;
    }

    @Override
    public MNISTBufferedImage validationSample(SampleId index) throws IOException {
        return null;
    }
}
