package es.tododev.ml.gen;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class Net implements Comparable<Net>, Serializable {

    private static final long serialVersionUID = 1L;
    private final List<Neuron[]> layers = new ArrayList<>();
    private Float score = 0f;
    
    public void addLayer(int size) {
        Neuron[] layer = new Neuron[size];
        Neuron[] previous = layers.isEmpty() ? Neuron.EMPTY : layers.get(layers.size() - 1);
        for (int i = 0; i < size; i++) {
            layer[i] = new Neuron(previous);
        }
        layers.add(layer);
    }
    
    public Neuron[] calculate(float ... input) {
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
        for (int i = 1; i < layers.size(); i++) {
            for (Neuron neuron : layers.get(i)) {
                neuron.mutate();
            }
        }
    }
    
    public Net copy() {
        Net copy = new Net();
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

    public Float getScore() {
        return score;
    }

    public void setScore(Float score) {
        this.score = score;
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
        return o.score.compareTo(score);
    }
    
}
