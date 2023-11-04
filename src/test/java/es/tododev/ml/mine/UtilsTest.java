package es.tododev.ml.mine;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import es.tododev.ml.mine.Utils;

public class UtilsTest {

    @Test
    public void maxIndex() {
        int[] values = new int[] {2, 5, 6, 1};
        assertEquals(2, Utils.maxIndex(values));
        values = new int[] {1, 1, 1, 1};
        assertEquals(-1, Utils.maxIndex(values));
    }
}
