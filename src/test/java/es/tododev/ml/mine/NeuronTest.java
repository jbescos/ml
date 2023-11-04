package es.tododev.ml.mine;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import es.tododev.ml.mine.Net;
import es.tododev.ml.mine.Neuron;

public class NeuronTest {

    @Test
    public void string() {
        Neuron neuron = new Neuron(new Neuron(), new Neuron());
        System.out.println(neuron.toString());
        neuron.mutate();
        System.out.println(neuron.toString());
    }
    
    @Test
    public void sigmoid() {
        assertEquals("0.11920292", Float.toString(Neuron.sigmoid(-2)));
        assertEquals("0.99752736", Float.toString(Neuron.sigmoid(6)));
    }

    @Test
    public void copyNeuron() {
        Neuron n1 = new Neuron();
        Neuron n2 = new Neuron();
        n1.copy(n2, Neuron.EMPTY);
        assertEquals(n1, n2);
    }

    @Test
    public void copy() {
        Net net = new Net();
        net.addLayer(2);
        net.addLayer(1);
        Net copy = net.copy();
        assertEquals(net, copy);
    }
}
