package es.tododev.ml.gen;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Trainer {
    
    private int generations = 50;
    private int choosen = 5;

    public Net train(List<Net> players, List<TestData> data) {
        List<Net> results = null;
        for (int i = 0; i < generations; i++) {
            results = IntStream.rangeClosed(0, players.size() - 1).parallel()
                    .mapToObj(j -> player(players.get(j), new ArrayList<>(data)))
                    .sorted()
                    .collect(Collectors.toList())
                    .subList(0, choosen);
            createNewGen(players, results);
            System.out.println("Gen: " + i + " score " + results.get(0).getScore());
        }
        return results.get(0);
    }
    
    private void createNewGen(List<Net> players, List<Net> results) {
        int limit = players.size();
        players.clear();
        players.addAll(results);
        while (players.size() < limit) {
            int selected = new Random().nextInt(results.size());
            Net best = results.get(selected);
            Net net = best.copy();
            net.mutate();
            players.add(net);
        }
    }
    
    private Net player(Net copy, List<TestData> data) {
        Collections.shuffle(data);
        float score = 0;
        for (TestData test : data) {
            Neuron[] out = copy.calculate(test.in);
            score = score + score(out, test.out);
        }
        copy.setScore(score / data.size());
        return copy;
    }
    
    private float score(Neuron[] out, float[] expected) {
        float error = 0;
        for (int i = 0; i < expected.length; i++) {
            error = error + Math.abs(out[i].getValue() - expected[i]);
        }
        return 1 - (error / expected.length);
    }
    
    public static class TestData {

        private final float[] in;
        private final float[] out;

        public TestData(float[] in, float[] out) {
            this.in = in;
            this.out = out;
        }

        public float[] getIn() {
            return in;
        }

        public float[] getOut() {
            return out;
        }
        
    }
    
}
