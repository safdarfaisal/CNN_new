package in.faisal.safdar;

public class SampleId {
    String id;

    SampleId(String identifier) {
        id = identifier;
    }

    SampleId(Integer identifier) {
        id = String.valueOf(identifier);
    }

    String value() {
        return id;
    }
}
