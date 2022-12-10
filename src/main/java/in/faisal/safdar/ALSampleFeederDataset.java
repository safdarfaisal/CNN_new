package in.faisal.safdar;

import java.io.IOException;
import java.util.List;
import java.util.ListIterator;

public class ALSampleFeederDataset implements MNISTDataset {
    ListIterator<SampleId> trainingIter;
    ListIterator<SampleId> testIter;
    List<SampleId> trainingSS;
    List<SampleId> testSS;
    MNISTDataset ds;
    int trainingSampleCount;
    int testSampleCount;

    private ALSampleFeederDataset() {

    }

    private ALSampleFeederDataset(List<SampleId> trainingSet, List<SampleId> testSet,
                                  MNISTDataset dataSet, int trainSampleCount, int testSampleCount) {
        trainingSS = trainingSet;
        testSS = testSet;
        trainingIter = trainingSet.listIterator();
        testIter = testSet.listIterator();
        ds = dataSet;
        this.trainingSampleCount = trainSampleCount;
        this.testSampleCount = testSampleCount;
    }

    public static ALSampleFeederDataset create(List<SampleId> trainingSet,
                                               List<SampleId> testSet, MNISTDataset dataSet,
                                               int trainSampleCount, int testSampleCount) {
        return new ALSampleFeederDataset(trainingSet, testSet, dataSet, trainSampleCount, testSampleCount);
    }

    public ALSampleFeederDataset refurbishedClone() {
        return new ALSampleFeederDataset(trainingSS, testSS, ds, trainingSampleCount, testSampleCount);
    }

    @Override
    public MNISTImage testSample() throws IOException {
        if (testIter.hasNext()) {
            //we are drawing test samples from the original MNIST training pool
            return ds.trainingSample(testIter.next());
        }
        return null;
    }

    @Override
    public MNISTBufferedImage validationSample() throws IOException {
        return null;
    }

    @Override
    public MNISTBufferedImage validationSample(SampleId index) throws IOException {
        return null;
    }

    @Override
    public MNISTImage trainingSample() throws IOException {
        if (trainingIter.hasNext()) {
            return ds.trainingSample(trainingIter.next());
        }
        return null;
    }

    @Override
    public MNISTImage trainingSample(SampleId index) throws IOException {
        return ds.trainingSample(index);
    }

    @Override
    public int trainingSampleCount() {
        return trainingSampleCount;
    }

    @Override
    public int testSampleCount() {
        return testSampleCount;
    }

    @Override
    public List<SampleId> trainingSubset(int count) {
        return null;
    }

    @Override
    public List<SampleId> testSubset(int count) {
        return null;
    }
}
