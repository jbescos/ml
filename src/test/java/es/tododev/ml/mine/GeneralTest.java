package es.tododev.ml.mine;

import static org.junit.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.ZipInputStream;

import org.junit.Ignore;
import org.junit.Test;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;

import es.tododev.ml.mine.Trainer.TestData;

public class GeneralTest {

    private static final int LIMIT_PLAYERS = 100;

    @Test
    @Ignore
    public void predictNumbersMine() throws IOException, ClassNotFoundException {
        List<TestData> train = fromZip("/mnist_train.zip");
        int inputs = train.get(0).getIn().length;
        int outputs = train.get(0).getOut().length;
        List<Net> players = IntStream.rangeClosed(0, LIMIT_PLAYERS).mapToObj(i -> {
            Net net = new Net();
            net.addLayer(inputs);
            net.addLayer(4);
            net.addLayer(4);
            net.addLayer(outputs);
            return net;
        }).collect(Collectors.toList());
        Trainer trainer = new Trainer();
        Net best = trainer.train(players, train);
        System.out.println("Best score is: " + ( 1 - best.getCost()));
        float test = 1 - best.cost(train);
        System.out.println("Precission with trained data: " + test);
        train = fromZip("/mnist_test.zip");
        test = 1 - best.cost(train);
        System.out.println("Precission with test data: " + test);

        // Test file load/save
        Path file = Files.createTempFile("Net", ".ml");
        best.save(file.toFile());
        Net loaded = Net.load(file.toFile());

        // Should be the same result
        assertEquals(Float.toString(best.cost(train)), Float.toString(loaded.cost(train)));
    }

    @Test
    @Ignore
    public void predictNumbersNeuroph() throws IOException, ClassNotFoundException {
        List<TestData> train = fromZip("/mnist_train.zip");
        int inputs = train.get(0).getIn().length;
        int outputs = train.get(0).getOut().length;

        NeuralNetwork ann = new NeuralNetwork();
        addLayer(ann, inputs);
        addLayer(ann, 10);
        addLayer(ann, 10);
        addLayer(ann, outputs);
        ann.setInputNeurons(ann.getLayers()[0].getNeurons());
        ann.setOutputNeurons(ann.getLayers()[ann.getLayersCount() - 1].getNeurons());
        train(ann, train);

        int successN = 0;
        int total = 0;

        train = fromZip("/mnist_test.zip");
        for (TestData data : train) {
            float[] in = data.getIn();
            float[] out = data.getOut();
            double[] inD = IntStream.range(0, in.length).mapToDouble(i -> in[i]).toArray();
            double[] outD = IntStream.range(0, out.length).mapToDouble(i -> out[i]).toArray();
            ann.setInput(inD);
            ann.calculate();
            double[] result = ann.getOutput();
            boolean success = Arrays.equals(outD, result);
            total++;
            if (success) {
                successN++;
            } else {
                System.out.println("Expected: " + getFromArray(outD) + " but was: " + getFromArray(result));
            }
        }
        System.out.println("Total: " + total + ", Success: " + successN + ", performance: " + (double) ( (double) successN /  (double) total));
    }

    @Test
    public void predictNumbersDeep4j() {
        
    }

    @Test
    public void irisNeuroph() throws IOException {
        List<TestData> train = irisData();
        int inputs = train.get(0).getIn().length;
        int outputs = train.get(0).getOut().length;
        NeuralNetwork ann = new NeuralNetwork();
        addLayer(ann, inputs);
        addLayer(ann, 4);
        addLayer(ann, 4);
        addLayer(ann, outputs);
        
        ann.setInputNeurons(ann.getLayers()[0].getNeurons());
        ann.setOutputNeurons(ann.getLayers()[ann.getLayersCount() - 1].getNeurons());
        train(ann, train);

        int successN = 0;
        int total = 0;

        // Test data set
//        train = TODO
        for (TestData data : train) {
            float[] in = data.getIn();
            float[] out = data.getOut();
            double[] inD = IntStream.range(0, in.length).mapToDouble(i -> in[i]).toArray();
            double[] outD = IntStream.range(0, out.length).mapToDouble(i -> out[i]).toArray();
            ann.setInput(inD);
            ann.calculate();
            double[] result = ann.getOutput();
            boolean success = Arrays.equals(outD, result);
            total++;
            if (success) {
                successN++;
            } else {
                System.out.println("Expected: " + Arrays.toString(outD) + " but was: " + Arrays.toString(result));
            }
        }
        System.out.println("Total: " + total + ", Success: " + successN + ", performance: " +  (double) ( (double) successN /  (double) total));
    }

    private List<TestData> irisData() throws IOException {
        List<TestData> test = new ArrayList<>();
        try (InputStream in = TrainerTest.class.getResourceAsStream("/iris.csv");
                BufferedReader reader = new BufferedReader(new InputStreamReader(in));) {
            while (reader.ready()) {
                String line = reader.readLine();
                String[] cells = line.split(",");
                float[] input = new float[cells.length - 1];
                float[] output = new float[3];
                for (int i = 0; i < cells.length; i++) {
                    if (i == cells.length - 1) {
                        int result = Integer.parseInt(cells[i]);
                        output[result] = 1;
                    } else {
                        input[i] = Float.parseFloat(cells[i]);
                    }
                }
                TestData testData = new TestData(input, output);
                System.out.println("Adding test entry. IN" + Arrays.toString(input) + " OUT" + Arrays.toString(output));
                test.add(testData);
            }
        }
        Trainer.normalize(test);
        return test;
    }

    private int getFromArray(double[] array) {
        for (int i = 0; i < array.length; i++) {
            if (array[i] == 1) {
                return i;
            }
        }
        return -1;
    }
    
    private void train(NeuralNetwork ann, List<TestData> train) {
        int inputs = train.get(0).getIn().length;
        int outputs = train.get(0).getOut().length;
        DataSet ds = new DataSet(inputs, outputs);
        for (TestData data : train) {
            float[] in = data.getIn();
            float[] out = data.getOut();
            double[] inD = IntStream.range(0, in.length).mapToDouble(i -> in[i]).toArray();
            double[] outD = IntStream.range(0, out.length).mapToDouble(i -> out[i]).toArray();
            assertEquals(inputs, inD.length);
            assertEquals(outputs, outD.length);
            DataSetRow row = new DataSetRow(inD, outD);
            ds.addRow(row);
        }
        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(1000);
        System.out.println("Learning ...");
        ann.learn(ds, backPropagation);
    }
    
    private void addLayer(NeuralNetwork ann, int size) {
        Layer layer = new Layer();
        for (int i = 0; i < size; i++) {
            layer.addNeuron(new Neuron());
        }
        ann.addLayer(layer);
        assertEquals(size, ann.getLayerAt(ann.getLayersCount() - 1).getNeuronsCount());
        int layerCount = ann.getLayersCount();
        if (layerCount > 1) {
            ConnectionFactory.fullConnect(ann.getLayerAt(layerCount - 2), ann.getLayerAt(layerCount - 1));
        }
    }
    
    private List<TestData> fromZip(String resource) throws IOException {
        List<TestData> test = new ArrayList<>();
        try (InputStream in = GeneralTest.class.getResourceAsStream(resource);
                ZipInputStream zipFile = new ZipInputStream(in)) {
            zipFile.getNextEntry();
            try (Scanner sc = new Scanner(zipFile)) {
                while (sc.hasNextLine()) {
                    String line = sc.nextLine();
                    test.add(convert(line));
                }
            }
        }
        System.out.println(test.size() + " elements loaded from " + resource);
        return test;
    }

    private TestData convert(String line) {
        String[] values = line.split(",");
        float[] out = new float[10];
        int idx = Integer.parseInt(values[0]);
        out[idx] = 1;
        float[] in = new float[values.length - 1];
        for (int i = 1; i < values.length; i++) {
            in[i - 1] = Float.parseFloat(values[i]);
        }
        return new TestData(in, out);
    }
}
