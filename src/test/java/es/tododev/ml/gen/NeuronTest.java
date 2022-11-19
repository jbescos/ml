package es.tododev.ml.gen;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import org.junit.Test;

public class NeuronTest {

    @Test
    public void string() {
        Neuron neuron = new Neuron(new Neuron(), new Neuron());
        System.out.println(neuron.toString());
        neuron.mutate(1);
        System.out.println(neuron.toString());
    }
    
    @Test
    public void sigmoid() {
        assertEquals("0.11920292", Float.toString(Neuron.sigmoid(-2)));
        assertEquals("0.99752736", Float.toString(Neuron.sigmoid(6)));
    }
    
    @Test
    public void net() {
        Net net = new Net();
        net.addLayer(2);
        net.addLayer(4);
        net.addLayer(2);
        Neuron[] out = net.calculate(1, 2);
        System.out.println(out[0].toString());
        System.out.println(out[1].toString());
        out = net.calculate(1, 2);
        System.out.println(out[0].toString());
        System.out.println(out[1].toString());
        net.mutate(2);
        out = net.calculate(1, 2);
        System.out.println(out[0].toString());
        System.out.println(out[1].toString());
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
        copy.mutate(2);
        assertNotEquals(net, copy);
    }
}
