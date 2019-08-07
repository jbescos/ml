package es.tododev.ml;

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
	public void sigmoid() {
		System.out.println(Utils.sigmoid(9999));
		System.out.println(Utils.sigmoid(1));
		System.out.println(Utils.sigmoid(0));
		System.out.println(Utils.sigmoid(-0.4));
		System.out.println(Utils.sigmoid(-9999));
	}
	
	@Test
	public void numbers() throws FileNotFoundException, IOException {
		List<Data> training = Utils.getFromCSV(this.getClass().getResourceAsStream("/models/numbers/mnist_train.zip"));
		int inputLayer = training.get(0).getInputValues().length;
		IModel model = new Model();
		model.addLayer(inputLayer);
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
		System.out.println("Success "+success+"/"+tests.size());
	}
	

}
