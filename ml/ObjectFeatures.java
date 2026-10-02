package ml;

/** Leakage-safe features available before an object's lifetime is known. */
public final class ObjectFeatures {
    private final int sizeBytes;
    private final int incomingReferenceCount;
    private final int outgoingReferenceCount;
    private final double heapUtilizationAtAllocation;
    private final double allocationRate;
    private final int gcCyclesSurvived;
    private final int ageAtPrediction;
    private final String allocationSite;

    public ObjectFeatures(int sizeBytes, int incomingReferenceCount, int outgoingReferenceCount,
                          double heapUtilizationAtAllocation, double allocationRate,
                          int gcCyclesSurvived, int ageAtPrediction, String allocationSite) {
        this.sizeBytes = sizeBytes;
        this.incomingReferenceCount = incomingReferenceCount;
        this.outgoingReferenceCount = outgoingReferenceCount;
        this.heapUtilizationAtAllocation = heapUtilizationAtAllocation;
        this.allocationRate = allocationRate;
        this.gcCyclesSurvived = gcCyclesSurvived;
        this.ageAtPrediction = ageAtPrediction;
        this.allocationSite = allocationSite == null ? "__MISSING__" : allocationSite;
    }

    public int getSizeBytes() { return sizeBytes; }
    public int getIncomingReferenceCount() { return incomingReferenceCount; }
    public int getOutgoingReferenceCount() { return outgoingReferenceCount; }
    public double getHeapUtilizationAtAllocation() { return heapUtilizationAtAllocation; }
    public double getAllocationRate() { return allocationRate; }
    public int getGcCyclesSurvived() { return gcCyclesSurvived; }
    public int getAgeAtPrediction() { return ageAtPrediction; }
    public String getAllocationSite() { return allocationSite; }
}
