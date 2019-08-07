package es.tododev.ml;

import static org.junit.Assert.assertEquals;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

@RunWith(SpringRunner.class)
@SpringBootTest
public class MlApplicationTests {
	
	@Test
	public void calculateCost() {
		Model model = new Model();
		INeuron neuron1 = new Neuron();
		INeuron neuron2 = new Neuron("label");
		INeuron[] output = new INeuron[] {neuron1, neuron2};
		
		neuron1.setValue(0);
		neuron2.setValue(0);
		assertEquals(0.5, model.calculateCost(neuron2.getLabel(), output), 0.01);
		neuron1.setValue(0);
		neuron2.setValue(1);
		assertEquals(0, model.calculateCost(neuron2.getLabel(), output), 0.01);
		neuron1.setValue(1);
		neuron2.setValue(0);
		assertEquals(1, model.calculateCost(neuron2.getLabel(), output), 0.01);
		neuron1.setValue(0.2);
		neuron2.setValue(0.97);
		assertEquals(0.03, model.calculateCost(neuron2.getLabel(), output), 0.01);
	}
	
	@Test
	public void numbers() throws FileNotFoundException, IOException {
		List<Data> training = Utils.getFromCSV(this.getClass().getResourceAsStream("/models/numbers/mnist_train.zip"));
		int inputLayer = training.get(0).getInputValues().length;
		IModel model = new Model();
		model.addLayer(inputLayer);
		model.addLayer(10);
		model.addOutput("0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
		model.train(10, training);
		List<Data> tests = Utils.getFromCSV(this.getClass().getResourceAsStream("/models/numbers/mnist_test.zip"));
		int success = 0;
		for(Data test : tests) {
			String result = model.getResultLabel(test.getInputValues());
			if(test.getExpectedLabel().equals(result)) {
				success++;
			}
		}
		System.out.println(model.toString());
		System.out.println("Success "+success+"/"+tests.size());
	}
	

}
