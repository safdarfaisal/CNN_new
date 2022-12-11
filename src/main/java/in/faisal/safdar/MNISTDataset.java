package in.faisal.safdar;

import java.io.IOException;
import java.util.List;

public interface MNISTDataset {
    MNISTImage testSample() throws IOException;
    MNISTBufferedImage validationSample() throws IOException;
    MNISTBufferedImage validationSample(SampleId index) throws IOException;
    MNISTImage trainingSample() throws IOException;
    MNISTImage trainingSample(SampleId index) throws IOException;
    int trainingSampleCount();
    int testSampleCount();
    List<SampleId> trainingSubset(int count);
    List<SampleId> testSubset(int count);
}
