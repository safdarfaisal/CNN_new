package in.faisal.safdar;

import java.io.IOException;
import java.util.List;

public interface MNISTDataset {
    public MNISTImage testSample() throws IOException;
    public MNISTBufferedImage validationSample() throws IOException;
    public MNISTBufferedImage validationSample(SampleId index) throws IOException;
    public MNISTImage trainingSample() throws IOException;
    public MNISTImage trainingSample(SampleId index) throws IOException;
    public int trainingSampleCount();
    public int testSampleCount();
    public List<SampleId> trainingSubset(int count);
    public List<SampleId> testSubset(int count);
}
