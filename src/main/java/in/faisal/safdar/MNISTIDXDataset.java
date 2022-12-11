package in.faisal.safdar;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class MNISTIDXDataset implements MNISTDataset {
    /*
    Load IDX training dataset from file into memory and divide into training and validation sets.
     */
    public static final String MNIST_TRAINING_DATA_FILE = "data/mnist_idx/train-images-idx3-ubyte";
    public static final String MNIST_TEST_DATA_FILE = "data/mnist_idx/t10k-images-idx3-ubyte";
    public static final String MNIST_TRAINING_LABELS_FILE = "data/mnist_idx/train-labels-idx1-ubyte";
    public static final String MNIST_TEST_LABELS_FILE = "data/mnist_idx/t10k-labels-idx1-ubyte";
    public static final int MNIST_IMAGE_WIDTH = 28;
    public static final int MNIST_IMAGE_HEIGHT = 28;

    MappedByteBuffer trainingImages;
    MappedByteBuffer testImages;
    MappedByteBuffer trainingLabels;
    MappedByteBuffer testLabels;
    int trainingImageCount;
    int testImageCount;

    MNISTIDXDataset() {
        trainingImages = mmapIDXFile(MNISTIDXDataset.MNIST_TRAINING_DATA_FILE);
        testImages = mmapIDXFile(MNISTIDXDataset.MNIST_TEST_DATA_FILE);
        trainingLabels = mmapIDXFile(MNISTIDXDataset.MNIST_TRAINING_LABELS_FILE);
        testLabels = mmapIDXFile(MNISTIDXDataset.MNIST_TEST_LABELS_FILE);
        assert(trainingImages.getInt(0) == 2051);
        assert(trainingLabels.getInt(0) == 2049);
        assert(trainingImages.getInt(4) == trainingLabels.getInt(4));
        assert(testImages.getInt(0) == 2051);
        assert(testLabels.getInt(0) == 2049);
        assert(testImages.getInt(4) == testLabels.getInt(4));
        assert(trainingImages.getInt(8) == MNIST_IMAGE_HEIGHT);
        assert(trainingImages.getInt(12) == MNIST_IMAGE_WIDTH);
        assert(testImages.getInt(8) == MNIST_IMAGE_HEIGHT);
        assert(testImages.getInt(12) == MNIST_IMAGE_WIDTH);
        trainingImageCount = trainingImages.getInt(4);
        testImageCount = testImages.getInt(4);
    }

    private MappedByteBuffer mmapIDXFile(String filePath) {
        Path pathToRead = Paths.get(filePath);
        MappedByteBuffer mappedByteBuffer = null;
        try (FileChannel fileChannel =
                     (FileChannel) Files.newByteChannel(pathToRead, EnumSet.of(StandardOpenOption.READ))) {
            mappedByteBuffer = fileChannel
                    .map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());

        } catch (IOException e) {
            e.printStackTrace();
        }
        return mappedByteBuffer;
    }

    private MNISTIdxImage sampleRandom(int imageCount, MappedByteBuffer images, MappedByteBuffer labels) {
        int r = ThreadLocalRandom.current().nextInt(0, imageCount);
        return sampleNth(r, imageCount, images, labels);
    }
    private MNISTIdxImage sampleNth(int n, int imageCount, MappedByteBuffer images, MappedByteBuffer labels) {
        int s = MNIST_IMAGE_HEIGHT*MNIST_IMAGE_WIDTH;
        byte[] b = new byte[s];
        images.get(n*s+16, b, 0, s);
        int l = (int)(labels.get(n+8));
        return new MNISTIdxImage(b, l, MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, new SampleId(n));
    }
    public MNISTImage testSample() throws IOException {
        return sampleRandom(testImageCount, testImages, testLabels);
    }

    public MNISTImage trainingSample() throws IOException {
        return sampleRandom(trainingImageCount, trainingImages, trainingLabels);
    }

    @Override
    public MNISTImage trainingSample(SampleId index) throws IOException {
        return sampleNth(Integer.parseInt(index.value()), trainingImageCount, trainingImages, trainingLabels);
    }

    @Override
    public int trainingSampleCount() {
        return trainingImageCount;
    }

    @Override
    public int testSampleCount() {
        return testImageCount;
    }

    private List<SampleId> sampleSubset(int count, int max) {
        List<SampleId> l = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            l.add(new SampleId(ThreadLocalRandom.current().nextInt(0, max)));
        }
        return l;
    }
    @Override
    public List<SampleId> trainingSubset(int count) {
        return sampleSubset(count, trainingImageCount);
    }

    @Override
    public List<SampleId> testSubset(int count) {
        return sampleSubset(count, testImageCount);
    }

    public MNISTBufferedImage validationSample() throws IOException {
        return null;
    }

    @Override
    public MNISTBufferedImage validationSample(SampleId index) throws IOException {
        return null;
    }
}
