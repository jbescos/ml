package es.tododev.ml.gen;

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
    
    public void mutate(int mutations) {
        for (int i = 1; i < layers.size(); i++) {
            for (Neuron neuron : layers.get(i)) {
                neuron.mutate(mutations);
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

    public float test(List<TestData> test) {
        float accurate = 0;
        for (TestData data : test) {
            calculate(data.getIn());
            int idxWinner = result();
            boolean correct = data.getOut()[idxWinner] == 1;
            if (correct) {
                accurate++;
//                System.out.println("Prediction correct, expected idx is " + idxWinner);
            } else {
//                System.out.println("Prediction failed, expected output is " + from(data.getOut()) + " and the prediction was " + from(layers.get(layers.size() - 1)));
            }
        }
        return accurate / test.size();
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
    
    private List<Float> from(float[] floats) {
        List<Float> list = new ArrayList<>();
        for (float f : floats) {
            list.add(f);
        }
        return list;
    }
    
    private List<Float> from(Neuron[] neurons) {
        List<Float> list = new ArrayList<>();
        for (Neuron n : neurons) {
            list.add(n.getValue());
        }
        return list;
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
    
}
