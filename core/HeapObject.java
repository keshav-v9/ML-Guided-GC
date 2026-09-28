package core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class HeapObject {
    private final int id;
    private final String name;
    private boolean marked;
    private final List<Integer> references = new ArrayList<>();

    public HeapObject(int id, String name) {
        this.id = id;
        this.name = name;
    }

    public int getId() { return id; }
    public String getName() { return name; }

    public void addReference(int address) {
        if (!references.contains(address)) references.add(address);
    }

    public void removeReference(int address) { references.remove(Integer.valueOf(address)); }

    /** Kept for compatibility with the original misspelled API. */
    @Deprecated
    public void removeRefrence(int address) { removeReference(address); }

    public List<Integer> getReferences() { return Collections.unmodifiableList(references); }

    public void replaceReferences(List<Integer> newReferences) {
        references.clear();
        for (int address : newReferences) addReference(address);
    }

    public boolean isMarked() { return marked; }
    public void setMarked(boolean marked) { this.marked = marked; }

    @Override
    public String toString() {
        return name + " { id=" + id + ", marked=" + marked + ", refs=" + references + " }";
    }
}
