package core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class HeapObject {
    private final int id;
    private final String name;
    private final int sizeBytes;
    private final long allocationTick;
    private final double heapUtilizationAtAllocation;
    private final double allocationRate;
    private final String allocationSite;
    private boolean marked;
    private Long deathTick;
    private int age;
    private Generation generation = Generation.YOUNG;
    private int incomingReferenceCount;
    private int gcCyclesSurvived;
    private final List<Integer> references = new ArrayList<>();

    public HeapObject(int id, String name) {
        this(id, name, 1, 0L, 0.0, 0.0, "default");
    }

    public HeapObject(int id, String name, int sizeBytes, long allocationTick,
                      double heapUtilizationAtAllocation, double allocationRate,
                      String allocationSite) {
        if (sizeBytes <= 0) throw new IllegalArgumentException("sizeBytes must be positive");
        this.id = id;
        this.name = name;
        this.sizeBytes = sizeBytes;
        this.allocationTick = allocationTick;
        this.heapUtilizationAtAllocation = heapUtilizationAtAllocation;
        this.allocationRate = allocationRate;
        this.allocationSite = allocationSite == null ? "default" : allocationSite;
    }

    public int getId() { return id; }
    public String getName() { return name; }
    public int getSizeBytes() { return sizeBytes; }
    public long getAllocationTick() { return allocationTick; }
    public Long getDeathTick() { return deathTick; }
    public int getAge() { return age; }
    public Generation getGeneration() { return generation; }
    public int getIncomingReferenceCount() { return incomingReferenceCount; }
    public int getOutgoingReferenceCount() { return references.size(); }
    public int getGcCyclesSurvived() { return gcCyclesSurvived; }
    public double getHeapUtilizationAtAllocation() { return heapUtilizationAtAllocation; }
    public double getAllocationRate() { return allocationRate; }
    public String getAllocationSite() { return allocationSite; }

    public void setGeneration(Generation generation) {
        if (generation == null) throw new IllegalArgumentException("generation must not be null");
        this.generation = generation;
    }

    void setDeathTick(long deathTick) {
        if (deathTick < allocationTick)
            throw new IllegalArgumentException("deathTick must not precede allocationTick");
        if (this.deathTick == null) this.deathTick = deathTick;
    }

    void setIncomingReferenceCount(int count) { incomingReferenceCount = count; }

    void survivedCollection() {
        age++;
        gcCyclesSurvived++;
    }

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
        return name + " { id=" + id + ", generation=" + generation + ", age=" + age
                + ", marked=" + marked + ", refs=" + references + " }";
    }
}
