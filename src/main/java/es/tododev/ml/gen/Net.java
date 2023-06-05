package es.tododev.ml.gen;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import es.tododev.ml.gen.Trainer.TestData;

public class Net implements Comparable<Net>, Serializable {

    private static final long serialVersionUID = 1L;
    private final List<Neuron[]> layers = new ArrayList<>();
    private Float cost = 0f;
    private int[] mistakes;

    public void addLayer(int size) {
        Neuron[] layer = new Neuron[size];
        Neuron[] previous = layers.isEmpty() ? Neuron.EMPTY : layers.get(layers.size() - 1);
        for (int i = 0; i < size; i++) {
            layer[i] = new Neuron(previous);
        }
        layers.add(layer);
    }

    public Neuron[] calculate(float... input) {
        Neuron[] inputLayer = layers.get(0);
        if (input.length != inputLayer.length) {
            throw new IllegalArgumentException("Input length must be " + inputLayer.length);
        }
        for (int i = 0; i < input.length; i++) {
            inputLayer[i].setValue(input[i]);
        }
        for (int i = 1; i < layers.size(); i++) {
            Neuron[] layer = layers.get(i);
            for (Neuron neuron : layer) {
                neuron.calculate();
            }
        }
        return layers.get(layers.size() - 1);
    }

    public void mutate() {
        // Back propagating
        for (int i = 0; i < mistakes.length; i++) {
            if (mistakes[i] > 0) {
                int mostWrongIdx = i;
                int currentLayer = layers.size() - 1;
                while (currentLayer > 0 && mostWrongIdx > -1) {
                    Neuron[] layer = layers.get(currentLayer);
                    Neuron mostWrongNeuron = layer[mostWrongIdx];
                    mostWrongIdx = mostWrongNeuron.mutate();
                    currentLayer--;
                }
            }
        }
    }

    public Net copy() {
        Net copy = new Net();
        if (mistakes != null) {
            copy.mistakes = new int[mistakes.length];
            System.arraycopy(mistakes, 0, copy.mistakes, 0, mistakes.length);
        }
        for (Neuron[] layer : layers) {
            Neuron[] previous = copy.layers.isEmpty() ? Neuron.EMPTY : copy.layers.get(copy.layers.size() - 1);
            copy.addLayer(layer.length);
            Neuron[] copied = copy.layers.get(copy.layers.size() - 1);
            for (int i = 0; i < layer.length; i++) {
                layer[i].copy(copied[i], previous);
            }
        }
        return copy;
    }

    public float cost(List<TestData> test) {
        mistakes = new int[test.get(0).getOut().length];
        float cost = 0;
        for (TestData data : test) {
            calculate(data.getIn());
            int idxWinner = result();
            boolean correct = data.getOut()[idxWinner] == 1;
            if (!correct) {
                mistakes[idxWinner] = mistakes[idxWinner] + 1;
                cost++;
//                System.out.println("Prediction correct, expected idx is " + idxWinner);
            } else {
//                System.out.println("Prediction failed, expected output is " + from(data.getOut()) + " and the prediction was " + from(layers.get(layers.size() - 1)));
            }
        }
        this.cost = cost / test.size();
        return this.cost;
    }

    public Float getCost() {
        return cost;
    }

    public void setCost(Float cost) {
        this.cost = cost;
    }

    public int result() {
        Neuron[] outLayer = layers.get(layers.size() - 1);
        int winnerIdx = 0;
        for (int i = 1; i < outLayer.length; i++) {
            if (outLayer[i].getValue() > outLayer[i - 1].getValue()) {
                winnerIdx = i;
            }
        }
        return winnerIdx;
    }

    public float resultValue(int idx) {
        return layers.get(layers.size() - 1)[idx].getValue();
    }

    @Override
    public int hashCode() {
        return Objects.hash(layers);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Net other = (Net) obj;
        for (int i = 0; i < layers.size(); i++) {
            boolean equals = Objects.equals(Arrays.asList(layers.get(i)), Arrays.asList(other.layers.get(i)));
            if (!equals) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < layers.size(); i++) {
            builder.append("Layer " + i);
            builder.append(Arrays.asList(layers.get(i)));
        }
        return builder.toString();
    }

    @Override
    public int compareTo(Net o) {
        return cost.compareTo(o.cost);
    }

    public void save(File file) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file))) {
            out.writeObject(this);
        }
        System.out.println(file.getAbsolutePath() + " saved");
    }
    
    public static Net load(File file) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(file))) {
            System.out.println(file.getAbsolutePath() + " loaded");
            return (Net) in.readObject();
        }
    }
}
