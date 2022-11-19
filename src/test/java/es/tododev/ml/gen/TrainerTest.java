package es.tododev.ml.gen;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.Test;

import es.tododev.ml.gen.Trainer.TestData;

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
        System.out.println("Best score is: " + best.getScore());
        verify(trainer, best, generatePairEvenTestData(10));
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
        System.out.println("Best score is: " + best.getScore());
        verify(trainer, best, generateHeadTailTestData(100));
    }
    
    private void verify(Trainer trainer, Net best, List<TestData> unkown) {
        int accurate = 0;
        for (TestData data : unkown) {
            best.calculate(data.getIn());
            int idxWinner = best.result();
            boolean correct = data.getOut()[idxWinner] == 1;
            if (correct) {
                accurate++;
            }
        }
        System.out.println("Precisison: " + (accurate / unkown.size()));
    }
    
    private List<TestData> generatePairEvenTestData(int items) {
        List<TestData> test = new ArrayList<>();
        float[] pair = new float[]{0, 1};
        float[] even = new float[]{1, 0};
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
        float[] head = new float[]{0, 1};
        float[] tail = new float[]{1, 0};
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
