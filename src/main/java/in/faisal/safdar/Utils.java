package in.faisal.safdar;

import java.util.concurrent.ThreadLocalRandom;

public class Utils {
    public static int randomDigit() {
        //random number from 0 to 9.
        return ThreadLocalRandom.current().nextInt(0, 10);
    }
}
