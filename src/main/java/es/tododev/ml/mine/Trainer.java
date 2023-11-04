package es.tododev.ml.mine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Trainer {
    
    private int generations = 10;
    private int choosen = 5;

    public Net train(List<Net> players, List<TestData> data) {
        List<Net> results = null;
        for (int i = 0; i < generations; i++) {
            long init = System.currentTimeMillis();
            results = IntStream.rangeClosed(0, players.size() - 1).parallel()
                    .mapToObj(j -> calculate(players.get(j), new ArrayList<>(data)))
                    .sorted()
                    .collect(Collectors.toList())
                    .subList(0, choosen);
            createNewGen(players, results);
            System.out.println("Generation " + i + ", Score " + (1 - results.get(0).getCost()) + ". Time " + (System.currentTimeMillis() - init) / 1000 + "s.");
        }
        return results.get(0);
    }
    
    private void createNewGen(List<Net> players, List<Net> results) {
        int limit = players.size();
        players.clear();
        players.addAll(results);
        Random rand = new Random();
        while (players.size() < limit) {
            int selected = rand.nextInt(results.size());
            Net best = results.get(selected);
            Net net = best.copy();
            net.mutate();
            players.add(net);
        }
    }
    
    private Net calculate(Net copy, List<TestData> data) {
        Collections.shuffle(data);
        copy.cost(data);
        return copy;
    }
    
    public static class TestData {

        private final float[] in;
        private final float[] out;
        private int winner = -1;

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
        
        public int winnerIdx() {
            if (winner == -1) {
                winner = 0;
                for (int i = 1; i < out.length; i++) {
                    if (out[i] > out[i - 1]) {
                        winner = i;
                    }
                }
            }
            return winner;
        }
        
    }
    
}
