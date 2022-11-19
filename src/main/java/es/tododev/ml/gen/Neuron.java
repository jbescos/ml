package es.tododev.ml.gen;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

public class Neuron implements Serializable {

    private static final long serialVersionUID = 1L;
    public static final Neuron[] EMPTY = new Neuron[0];
    private float[] inputWeights;
    private Neuron[] inputNeurons;
    private float bias;
    private float value;
    
    public Neuron(Neuron ... inputNeurons) {
        this.inputNeurons = inputNeurons;
        this.inputWeights = new float[inputNeurons.length];
        for (int i = 0; i < inputWeights.length; i++) {
            inputWeights[i] = getRandom();
        }
        bias = getRandom(-1, 1);
    }
    
    public Neuron() {
        this(EMPTY);
    }

    private float getRandom(double rangeMin, double rangeMax) {
        Random r = new Random();
        return (float) (rangeMin + (rangeMax - rangeMin) * r.nextDouble());
    }
    
    private float getRandom() {
        return getRandom(-1, 1);
    }

    public void calculate() {
        float calculated = 0;
        for (int i = 0; i < inputNeurons.length; i++) {
            calculated = calculated + (inputNeurons[i].value * inputWeights[i]);
        }
        calculated = calculated + bias;
        value = sigmoid(calculated);
    }

    static float sigmoid(float x) {
        return (float) (1 / (1 + Math.pow(Math.E, (-1 * x))));
    }
    
    static float maxX(float x) {
        return x < 0 ? 0 : 1;
    }
    
    public void mutate() {
        int rand = new Random().nextInt(inputWeights.length + 1);
        if (rand == inputWeights.length) {
            bias = getRandom();
        } else {
            inputWeights[rand] = getRandom();
        }
    }

    public void setValue(float value) {
        this.value = value;
    }

    public float getValue() {
        return value;
    }

    void copy(Neuron copy, Neuron[] previousLayer) {
        copy.bias = bias;
        copy.value = value;
        copy.inputNeurons = previousLayer;
        copy.inputWeights = new float[inputWeights.length];
        for (int i = 0; i < inputWeights.length; i++) {
            copy.inputWeights[i] = inputWeights[i];
        }
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + Arrays.hashCode(inputWeights);
        result = prime * result + Objects.hash(bias, value);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Neuron other = (Neuron) obj;
        return Float.floatToIntBits(bias) == Float.floatToIntBits(other.bias)
                && Arrays.equals(inputWeights, other.inputWeights)
                && Float.floatToIntBits(value) == Float.floatToIntBits(other.value);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("{");
        for (int i = 0; i < inputNeurons.length; i++) {
            builder.append("v").append(i).append("=").append(inputNeurons[i].value).append(",").append("w").append(i).append("=").append(inputWeights[i]);
        }
        builder.append(",bias=").append(bias);
        builder.append(",value=").append(value).append("}");
        return builder.toString();
    }
    
}