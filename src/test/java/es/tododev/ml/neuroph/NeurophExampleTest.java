package es.tododev.ml.neuroph;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;

public class NeurophExampleTest {

    @Test
    public void example() {
        NeuralNetwork neuralNetwork = inst();
        training(neuralNetwork);
        
        verify(neuralNetwork, 0, 1, 1);
        verify(neuralNetwork, 1, 1, 0);
        verify(neuralNetwork, 0, 0, 0);
        verify(neuralNetwork, 1, 0, 1);
    }

    private void verify(NeuralNetwork neuralNetwork, int in0, int in1, int expectedOutput) {
        neuralNetwork.setInput(in0, in1);
        neuralNetwork.calculate();
        double[] networkOutputOne = neuralNetwork.getOutput();
        assertEquals(expectedOutput, (int) networkOutputOne[0]);
    }
    
    private void training(NeuralNetwork network) {
        int inputSize = 2;
        int outputSize = 1;
        DataSet ds = new DataSet(inputSize, outputSize);
        DataSetRow rOne = new DataSetRow(new double[] { 0, 1 }, new double[] { 1 });
        ds.addRow(rOne);
        DataSetRow rTwo = new DataSetRow(new double[] { 1, 1 }, new double[] { 0 });
        ds.addRow(rTwo);
        DataSetRow rThree = new DataSetRow(new double[] { 0, 0 }, new double[] { 0 });
        ds.addRow(rThree);
        DataSetRow rFour = new DataSetRow(new double[] { 1, 0 }, new double[] { 1 });
        ds.addRow(rFour);
        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(1000);
        network.learn(ds, backPropagation);
    }

    private NeuralNetwork inst() {
        Layer inputLayer = new Layer();
        inputLayer.addNeuron(new Neuron());
        inputLayer.addNeuron(new Neuron());

        Layer hiddenLayerOne = new Layer();
        hiddenLayerOne.addNeuron(new Neuron());
        hiddenLayerOne.addNeuron(new Neuron());
        hiddenLayerOne.addNeuron(new Neuron());
        hiddenLayerOne.addNeuron(new Neuron());

        Layer hiddenLayerTwo = new Layer();
        hiddenLayerTwo.addNeuron(new Neuron());
        hiddenLayerTwo.addNeuron(new Neuron());
        hiddenLayerTwo.addNeuron(new Neuron());
        hiddenLayerTwo.addNeuron(new Neuron());

        Layer outputLayer = new Layer();
        outputLayer.addNeuron(new Neuron());

        NeuralNetwork<?> ann = new NeuralNetwork<>();
        ann.addLayer(0, inputLayer);
        ann.addLayer(1, hiddenLayerOne);
        ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(1));
        ann.addLayer(2, hiddenLayerTwo);
        ConnectionFactory.fullConnect(ann.getLayerAt(1), ann.getLayerAt(2));
        ann.addLayer(3, outputLayer);
        ConnectionFactory.fullConnect(ann.getLayerAt(2), ann.getLayerAt(3));
        ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(ann.getLayersCount() - 1), false);
        ann.setInputNeurons(inputLayer.getNeurons());
        ann.setOutputNeurons(outputLayer.getNeurons());
        return ann;
    }

}
