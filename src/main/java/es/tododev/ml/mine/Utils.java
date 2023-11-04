package es.tododev.ml.mine;

import java.util.Arrays;
import java.util.function.BiFunction;

public class Utils {

    public static <T> int indexOfPredicate(BiFunction<T, T, Boolean> operator, T[] array) {
        int idx = -1;
        for (int i = 1; i < array.length; i++) {
            if (operator.apply(array[i], array[i - 1])) {
                idx = i;
            }
        }
        return idx;
    }
    
    public static int maxIndex(int[] array) {
        BiFunction<Integer, Integer, Boolean> operator = (a, b) -> a > b;
        return indexOfPredicate(operator, Arrays.stream(array).boxed().toArray(Integer[]::new));
    }
}
