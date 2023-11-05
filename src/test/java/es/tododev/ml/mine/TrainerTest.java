package es.tododev.ml.mine;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.Test;

import es.tododev.ml.mine.Trainer.TestData;

public class TrainerTest {

    private static final int LIMIT_PLAYERS = 10000;
    private static final int TEST_ITEMS = 1000;

    @Test
    public void pairEven() {
        List<Net> players = IntStream.rangeClosed(0, LIMIT_PLAYERS).mapToObj(i -> {
            Net net = new Net();
            net.addLayer(1);
            net.addLayer(16);
            net.addLayer(2);
            return net;
        }).collect(Collectors.toList());
        Trainer trainer = new Trainer();
        Net best = trainer.train(players, generatePairEvenTestData(TEST_ITEMS));
        System.out.println("Best score is: " + (1 - best.getCost()));
        float test = 1 - best.cost(generatePairEvenTestData(100));
        System.out.println("Precission: " + test);
    }

    @Test
    public void headTail() {
        List<Net> players = IntStream.rangeClosed(0, LIMIT_PLAYERS).mapToObj(i -> {
            Net net = new Net();
            net.addLayer(2);
            net.addLayer(16);
            net.addLayer(2);
            return net;
        }).collect(Collectors.toList());
        Trainer trainer = new Trainer();
        Net best = trainer.train(players, generateHeadTailTestData(TEST_ITEMS));
        System.out.println("Best score is: " + (1 - best.getCost()));
        float test = 1 - best.cost(generateHeadTailTestData(100));
        System.out.println("Precission: " + test);
    }

    @Test
    public void iris() throws IOException {
        List<Net> players = IntStream.rangeClosed(0, LIMIT_PLAYERS).mapToObj(i -> {
            Net net = new Net();
            net.addLayer(4);
            net.addLayer(4);
            net.addLayer(1);
            return net;
        }).collect(Collectors.toList());
        Trainer trainer = new Trainer();
        Net best = trainer.train(players, irisData());
        System.out.println("Best score is: " + (1 - best.getCost()));
        float test = 1 - best.cost(irisData());
        System.out.println("Precission: " + test);
    }

    private List<TestData> irisData() throws IOException {
        List<TestData> test = new ArrayList<>();
        try (InputStream in = TrainerTest.class.getResourceAsStream("/iris.csv");
                BufferedReader reader = new BufferedReader(new InputStreamReader(in));) {
            while (reader.ready()) {
                String line = reader.readLine();
                String[] cells = line.split(",");
                float[] input = new float[cells.length - 1];
                float[] output = new float[1];
                for (int i = 0; i < cells.length; i++) {
                    if (i == cells.length - 1) {
                        output[0] = Float.parseFloat(cells[i]);
                    } else {
                        input[i] = Float.parseFloat(cells[i]);
                    }
                }
                TestData testData = new TestData(input, output);
                test.add(testData);
            }
        }
        Trainer.normalize(test);
        return test;
    }

    private List<TestData> generatePairEvenTestData(int items) {
        List<TestData> test = new ArrayList<>();
        float[] pair = new float[] { 0, 1 };
        float[] even = new float[] { 1, 0 };
        for (int i = 0; i < items; i++) {
            int value = new Random().nextInt(1000);
            float[] in = new float[1];
            in[0] = (float) value;
            TestData data;
            if (value % 2 == 0) {
                data = new TestData(in, pair);
            } else {
                data = new TestData(in, even);
            }
            test.add(data);
        }
        return test;
    }

    private List<TestData> generateHeadTailTestData(int items) {
        List<TestData> test = new ArrayList<>();
        float[] head = new float[] { 0, 1 };
        float[] tail = new float[] { 1, 0 };
        Random random = new Random();
        for (int i = 0; i < items; i++) {
            TestData data;
            boolean headB = random.nextBoolean();
            if (headB) {
                data = new TestData(head, head);
            } else {
                data = new TestData(tail, tail);
            }
            test.add(data);
        }
        return test;
    }

}
