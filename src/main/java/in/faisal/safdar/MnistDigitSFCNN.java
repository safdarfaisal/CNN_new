package in.faisal.safdar;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;

import org.apache.commons.lang3.tuple.Pair;
import org.ejml.simple.SimpleMatrix;

/*
CNN to classify MNIST DB. Written from scratch in Java.

28x28 input image
3x3 Conv Layer with 8 filters
2x2 Maxpool Layer
Softmax activation
 */
public class MnistDigitSFCNN implements MNISTModel {
    public static final int MNIST_CNN_TRAIN_STEPS_MAX = 10000;
    public static final int CONV_FILTER_COUNT = 12;
    public static final int CONV_FILTER_ROWS = 3;
    public static final int CONV_FILTER_COLUMNS = 3;
    public static final int INPUT_IMG_WIDTH = 28;
    public static final int INPUT_IMG_HEIGHT = 28;
    public static int fanIn = CONV_FILTER_ROWS*CONV_FILTER_COLUMNS;
    public static int fanOut = (INPUT_IMG_WIDTH-CONV_FILTER_COLUMNS+1)*(INPUT_IMG_HEIGHT-CONV_FILTER_ROWS+1);
    public static final int MAXPOOL_FILTER_ROWS = 2;
    public static final int MAXPOOL_FILTER_COLUMNS = 2;
    public static final float LEARNING_RATE = 0.003f;
    private static final int outputLayerInputNodeCount =
            fanOut*CONV_FILTER_COUNT/(MAXPOOL_FILTER_ROWS*MAXPOOL_FILTER_COLUMNS);

    private MNISTDataset dataSet;
    //Convolution filters, 8 3x3 matrices
    private final SimpleMatrix[] convFilters = new SimpleMatrix[CONV_FILTER_COUNT];
    //Output layer parameters
    private SimpleMatrix outputLayerFlatInput;
    private SimpleMatrix outputLayerWeights;
    private SimpleMatrix outputSoftMaxExp;
    private SimpleMatrix outputBias;
    private SimpleMatrix outputResults;

    MnistDigitSFCNN() {
        //dataSet = new MNISTPNGDataset();
        //dataSet = new MNISTIDXDataset();
    }

    public EvalResultMap eval(int count, EvalResultMap result, boolean print) {
        int accuracy = 0;
        for (int i=0; i < count; i++) {
            SamplePredictor p = test();
            if (result != null) {
                result.resultMap.put(p.index.value(), p);
            }
            if (print) {
                System.out.println("Sample label: " + p.correctLabel + ", Predicted label: " + p.predictedLabel +
                        ":" + p.index.value() + ":" + p.probForCorrectLabel);
            }
            accuracy += (p.correctLabel == p.predictedLabel) ? 1 : 0;
        }
        Float percentAccuracy = accuracy * 100.0f/count;
        if (result != null) {
            result.metricsMap.put("PercentAccuracy", percentAccuracy );
        }
        if (print) {
            System.out.println("Accuracy: " + percentAccuracy + "%");
        }
        return result;
    }

    public static void main(String[] args) {
        //TODO: Write trained model to file and read from there for eval.
        MnistDigitSFCNN cnn = new MnistDigitSFCNN();
        cnn.init(new MNISTIDXDataset());
        cnn.train(MNIST_CNN_TRAIN_STEPS_MAX);
        cnn.eval(100, null, true);
    }

    public MNISTDataset getDataSet() {
        return dataSet;
    }

    private SimpleMatrix[] convolve(SimpleMatrix input, SimpleMatrix[] filters) {
        SimpleMatrix[] outputs = new SimpleMatrix[filters.length];
        for(int fn=0; fn < filters.length; fn++) {
            SimpleMatrix filter = filters[fn];
            int filterH = filter.numRows();
            int filterW = filter.numCols();
            int inputH = input.numRows();
            int inputW = input.numCols();
            int outCols = inputW - filterW + 1;
            int outRows = inputH - filterH + 1;
            SimpleMatrix output = new SimpleMatrix(outRows, outCols);
            for (int i = 0; i < outRows; i++) {
                for (int j = 0; j < outCols; j++) {
                    output.set(
                            i, j,
                            filter.elementMult(input.extractMatrix(i, i + filterH, j, j + filterW))
                                    .elementSum()
                    );
                }
            }
            outputs[fn] = output;
        }
        return outputs;
    }

    private void deconvolve(SimpleMatrix input, SimpleMatrix[] outputGradients) {
        int filterH = convFilters[0].numRows();
        int filterW = convFilters[0].numCols();
        int fLen = convFilters.length;
        SimpleMatrix[] fts = new SimpleMatrix[fLen];
        int inputH = input.numRows();
        int inputW = input.numCols();
        int outCols = inputW - filterW + 1;
        int outRows = inputH - filterH + 1;
        for(int fn = 0; fn < fLen; fn++) {
            SimpleMatrix f = new SimpleMatrix(filterH, filterW);
            f.zero();
            fts[fn] = f;
        }
        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                for(int fn = 0; fn < fLen; fn++) {
                    SimpleMatrix x = input.extractMatrix(i, i + filterH, j, j + filterW);
                    SimpleMatrix outputGradient = outputGradients[fn];
                    fts[fn] = fts[fn].plus(x.scale(outputGradient.get(i,j)));
                }
            }
        }
        for(int fn = 0; fn < fLen; fn++) {
            convFilters[fn] = convFilters[fn].plus(fts[fn].scale(-LEARNING_RATE));
        }
    }

    private SimpleMatrix[] maxPool(SimpleMatrix[] inputs) {
        SimpleMatrix[] outputs = new SimpleMatrix[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            SimpleMatrix input = inputs[i];
            int inputH = input.numRows();
            int inputW = input.numCols();
            int outputH = inputH/MAXPOOL_FILTER_ROWS;
            int outputW = inputW/MAXPOOL_FILTER_COLUMNS;
            outputs[i] = new SimpleMatrix(outputH, outputW);
            for (int j = 0; j < outputH; j++) {
                for (int k = 0; k < outputW; k++) {
                    int p = j*MAXPOOL_FILTER_ROWS;
                    int q = k*MAXPOOL_FILTER_COLUMNS;
                    SimpleMatrix x = input.extractMatrix(p, p+MAXPOOL_FILTER_ROWS,
                            q, q+MAXPOOL_FILTER_COLUMNS);
                    outputs[i].set(j, k, new SimpleMatrixEx(x).elementMax());
                }
            }
        }
        return outputs;
    }

    private SimpleMatrix[] maxPoolBackward(SimpleMatrix[] inputs,
                                                  SimpleMatrix[] outputs, SimpleMatrix[] outputGradients) {
        SimpleMatrix[] inputGradients = new SimpleMatrix[CONV_FILTER_COUNT];
        for (int i=0; i<outputs.length; i++) {
            SimpleMatrix output = outputs[i];
            SimpleMatrix input = inputs[i];
            SimpleMatrix inputGradient =
                    new SimpleMatrix(INPUT_IMG_HEIGHT-CONV_FILTER_ROWS+1,
                            INPUT_IMG_WIDTH-CONV_FILTER_COLUMNS+1);
            inputGradients[i] = inputGradient;
            SimpleMatrix outputGradient = outputGradients[i];
            inputGradient.zero();
            int outputH = output.numRows();
            int outputW = output.numCols();
            for (int j = 0; j < outputH; j++) {
                for (int k = 0; k < outputW; k++) {
                    int p = j*MAXPOOL_FILTER_ROWS;
                    int q = k*MAXPOOL_FILTER_COLUMNS;
                    SimpleMatrix x = input.extractMatrix(p, p+MAXPOOL_FILTER_ROWS,
                            q, q+MAXPOOL_FILTER_COLUMNS);
                    Pair<Integer, Integer> indices = new SimpleMatrixEx(x).indexOfMax();
                    inputGradient.set(p+indices.getLeft(), q+indices.getRight(), outputGradient.get(j, k));
                }
            }
        }
        return inputGradients;
    }

    private SimpleMatrix outputForward(SimpleMatrix[] inputs) {
        outputLayerFlatInput = new SimpleMatrix(inputs.length, inputs[0].getNumElements());
        for (int i = 0; i < inputs.length; i++) {
            SimpleMatrix x = new SimpleMatrix(inputs[i]);
            x.reshape(1, x.getNumElements());
            outputLayerFlatInput.insertIntoThis(i, 0, x);
        }
        outputLayerFlatInput.reshape(1, outputLayerFlatInput.getNumElements());
        SimpleMatrix ol = outputLayerFlatInput.mult(outputLayerWeights).plus(outputBias);
        outputSoftMaxExp = new SimpleMatrixEx(ol).exp();
        float outputSoftMaxInvSum = (float)(1/outputSoftMaxExp.elementSum());
        return outputSoftMaxExp.scale(outputSoftMaxInvSum);
    }

    private SimpleMatrix[] outputBackward(SimpleMatrix outGradient) {
        assert(outGradient.numRows()==1);

        SimpleMatrix inputLosses = new SimpleMatrix(outputLayerInputNodeCount,1);
        inputLosses.zero();
        float outputSoftMaxSum = (float)(outputSoftMaxExp.elementSum());
        for(int i=0; i < outGradient.numCols(); i++) {
            float gradient = (float)(outGradient.get(0, i));
            if (gradient != 0) {
                float exp = (float)(outputSoftMaxExp.get(0,i));
                float smSumSquared = outputSoftMaxSum*outputSoftMaxSum;
                SimpleMatrix tm =
                        outputSoftMaxExp.scale(
                                -exp/smSumSquared);
                tm.set(0, i, exp*(outputSoftMaxSum-exp)/smSumSquared);
                SimpleMatrix ts = tm.scale(gradient);
                inputLosses = outputLayerWeights.mult(ts.transpose());
                //update weights and biases
                outputLayerWeights =
                        outputLayerFlatInput.transpose().mult(ts).scale(-LEARNING_RATE).plus(outputLayerWeights);
                outputBias = ts.scale(-LEARNING_RATE).plus(outputBias);
            }
        }
        inputLosses = inputLosses.transpose();
        inputLosses.reshape(CONV_FILTER_COUNT, fanOut/(MAXPOOL_FILTER_ROWS*MAXPOOL_FILTER_COLUMNS));
        SimpleMatrix[] inputLossesArray = new SimpleMatrix[CONV_FILTER_COUNT];
        for (int i=0; i < inputLosses.numRows(); i++) {
            SimpleMatrix m = inputLosses.rows(i, i);
            m.reshape((INPUT_IMG_HEIGHT-CONV_FILTER_ROWS+1)/MAXPOOL_FILTER_ROWS,
                    (INPUT_IMG_WIDTH-CONV_FILTER_COLUMNS+1)/MAXPOOL_FILTER_COLUMNS);
            inputLossesArray[i] = m;
        }
        return inputLossesArray;
    }

    public void init(MNISTDataset ds) {
        dataSet = (ds != null) ? ds : new MNISTIDXDataset();
        for (int i = 0; i < CONV_FILTER_COUNT; i++) {
            /*
            Using Xavier Uniform initialization.
            For each filter, there are 9 inputs per output, so fan_in = 9. Since each filter
            produces 26x26 values over the input layer with valid padding, fan_out = 676.
            https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks
            https://github.com/deeplearning4j/deeplearning4j/blob/e2b92619b299e13a181791bf5ecd53304cb393e0/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitXavierUniform.java#L33
             */
            float s = (float)(Math.sqrt(6.0f) / Math.sqrt(fanIn + fanOut));
            convFilters[i] = SimpleMatrix.random_FDRM(
                    CONV_FILTER_ROWS, CONV_FILTER_COLUMNS, -s, s, new Random(System.currentTimeMillis()+i));
        }
        //Output layer, fully connected.
        outputLayerWeights = SimpleMatrix
                .random_FDRM(outputLayerInputNodeCount, 10, 0, 1,
                        new Random(System.currentTimeMillis()))
                .divide((float)outputLayerInputNodeCount);
        outputBias = new SimpleMatrix(1, 10);
        outputBias.zero();
    }

    @Override
    public void setDataset(MNISTDataset ds) {
        dataSet = ds;
    }

    /*
    We are just going to feed random sample images N times. Not implementing
    batches with multiple epochs over the same data set, at this point.
    TODO: Support mini-batches and epochs.
     */
    public void train(int stepCount)
    {
        //Loss function
        float loss = 0;
        int accuracy = 0;
        int accTotal = 0;

        for (int i = 0; i < stepCount; i++) {
            try {
                //load a random image from MNIST DB along with its label
                MNISTImage sample = dataSet.trainingSample();
                int sampleLabel = sample.getDigit();
                SimpleMatrix pixelMatrix = sample.simpleMatrixFloat0To1();
                //forward propagation
                SimpleMatrix[] convOutput = convolve(pixelMatrix, convFilters);
                SimpleMatrix[] maxPoolOutput = maxPool(convOutput);
                outputResults = outputForward(maxPoolOutput);
                //loss computation
                loss += -Math.log(outputResults.get(0, sampleLabel));
                int result = new SimpleMatrixEx(outputResults).indexOfMax().getRight();
                accuracy += ((sampleLabel == result) ? 1 : 0);
                //backward propagation
                SimpleMatrix lossGradient = new SimpleMatrix(1, 10);
                lossGradient.zero();
                lossGradient.set(0, sampleLabel, -1/outputResults.get(0, sampleLabel));
                deconvolve(
                        pixelMatrix,
                        maxPoolBackward(
                                convOutput, maxPoolOutput,
                                outputBackward(lossGradient)
                        )
                );
                if(i%100 == 0) {
                    System.out.println(" Step: "+ i+ " loss: "+ loss/100.0+" accuracy: "+ accuracy);
                    loss = 0;
                    accTotal += accuracy;
                    accuracy = 0;
                }

            } catch (IOException e) {
                e.printStackTrace();
                System.exit(-1);
            }
        }
        //System.out.println("Average accuracy: " + (accTotal*100.0f)/stepCount + "%");
    }

    //returns digit values from 0 to 9, or negative numbers for error
    public SamplePredictor test()
    {
        SamplePredictor pr = new SamplePredictor();
        try {
            //load an image from MNIST DB along with its label. Selection
            //strategy depends on logic in the data set class.
            MNISTImage sample = dataSet.testSample();
            SimpleMatrix pixelMatrix = sample.simpleMatrixFloat0To1();
            pr.index = sample.id();
            pr.correctLabel = sample.getDigit();
            //forward propagation
            SimpleMatrix[] convOutput = convolve(pixelMatrix, convFilters);
            SimpleMatrix[] maxPoolOutput = maxPool(convOutput);
            outputResults = outputForward(maxPoolOutput);
            SimpleMatrixEx outputEx = new SimpleMatrixEx(outputResults);
            pr.predictedLabel = outputEx.indexOfMax().getRight();
            pr.probForCorrectLabel = (float)(outputResults.get(0, pr.correctLabel));
            pr.probForPrediction = (float)(outputEx.elementMax());
            List <Double> l = outputEx.elementList();
            l.sort(Collections.reverseOrder());
            pr.topTwoProbDiff = (float)(l.get(0) - l.get(1));
            pr.largestProbDiff = (float)(l.get(0) - l.get(l.size()-1));
            ListIterator<Double> iter = l.listIterator();
            double entropy = 0;
            while (iter.hasNext()) {
                double p = iter.next();
                entropy = entropy - p*Math.log10(p);
            }
            pr.entropy = (float)entropy;
            pr.classProbs = outputEx.elementList();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return pr;
    }

    @Override
    public MNISTModel deepCopy() {
        MnistDigitSFCNN newModel = new MnistDigitSFCNN();
        //dataset is not deep copied. If the specific dataset maintains state,
        //it is upto the caller to set a new one with the desired state.
        newModel.dataSet = dataSet;
        newModel.outputBias = outputBias.copy();
        for (int i=0; i < CONV_FILTER_COUNT; i++) {
            newModel.convFilters[i] = convFilters[i].copy();
        }
        newModel.outputLayerFlatInput = outputLayerFlatInput.copy();
        newModel.outputLayerWeights = outputLayerWeights.copy();
        newModel.outputSoftMaxExp = outputSoftMaxExp.copy();
        newModel.outputResults = outputResults.copy();
        return newModel;
    }
}