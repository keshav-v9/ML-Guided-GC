package telemetry;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** One labeled object row. Post-death fields are never included in model features. */
public final class ObjectTelemetry {
    public static final String CSV_HEADER = "object_id,workload_type,seed,size_bytes,"
            + "incoming_reference_count,outgoing_reference_count,"
            + "heap_utilization_at_allocation,allocation_rate,gc_cycles_survived,"
            + "age_at_prediction,allocation_site,lifetime_ticks,label,censored,labeling_rule";

    public static final List<String> MODEL_FEATURE_COLUMNS = Collections.unmodifiableList(
            Arrays.asList("size_bytes", "incoming_reference_count",
                    "outgoing_reference_count", "heap_utilization_at_allocation",
                    "allocation_rate", "gc_cycles_survived", "age_at_prediction",
                    "allocation_site"));

    private final int objectId;
    private final String workloadType;
    private final long seed;
    private final int sizeBytes;
    private final int incomingReferenceCount;
    private final int outgoingReferenceCount;
    private final double heapUtilizationAtAllocation;
    private final double allocationRate;
    private final int gcCyclesSurvived;
    private final int ageAtPrediction;
    private final String allocationSite;
    private final long lifetimeTicks;
    private final LifetimeLabel label;
    private final boolean censored;
    private final CensoringPolicy labelingRule;

    public ObjectTelemetry(int objectId, String workloadType, long seed, int sizeBytes,
                           int incomingReferenceCount, int outgoingReferenceCount,
                           double heapUtilizationAtAllocation, double allocationRate,
                           int gcCyclesSurvived, int ageAtPrediction, String allocationSite,
                           long lifetimeTicks, LifetimeLabel label, boolean censored,
                           CensoringPolicy labelingRule) {
        this.objectId = objectId;
        this.workloadType = workloadType;
        this.seed = seed;
        this.sizeBytes = sizeBytes;
        this.incomingReferenceCount = incomingReferenceCount;
        this.outgoingReferenceCount = outgoingReferenceCount;
        this.heapUtilizationAtAllocation = heapUtilizationAtAllocation;
        this.allocationRate = allocationRate;
        this.gcCyclesSurvived = gcCyclesSurvived;
        this.ageAtPrediction = ageAtPrediction;
        this.allocationSite = allocationSite;
        this.lifetimeTicks = lifetimeTicks;
        this.label = label;
        this.censored = censored;
        this.labelingRule = labelingRule;
    }

    public int getObjectId() { return objectId; }
    public String getWorkloadType() { return workloadType; }
    public long getSeed() { return seed; }
    public int getSizeBytes() { return sizeBytes; }
    public int getIncomingReferenceCount() { return incomingReferenceCount; }
    public int getOutgoingReferenceCount() { return outgoingReferenceCount; }
    public double getHeapUtilizationAtAllocation() { return heapUtilizationAtAllocation; }
    public double getAllocationRate() { return allocationRate; }
    public int getGcCyclesSurvived() { return gcCyclesSurvived; }
    public int getAgeAtPrediction() { return ageAtPrediction; }
    public String getAllocationSite() { return allocationSite; }
    public long getLifetimeTicks() { return lifetimeTicks; }
    public LifetimeLabel getLabel() { return label; }
    public boolean isCensored() { return censored; }
    public CensoringPolicy getLabelingRule() { return labelingRule; }

    String toCsvRow() {
        return objectId + "," + csv(workloadType) + "," + seed + "," + sizeBytes + ","
                + incomingReferenceCount + "," + outgoingReferenceCount + ","
                + heapUtilizationAtAllocation + "," + allocationRate + ","
                + gcCyclesSurvived + "," + ageAtPrediction + "," + csv(allocationSite)
                + "," + lifetimeTicks + "," + label + "," + censored + "," + labelingRule;
    }

    private static String csv(String value) {
        if (value == null) return "";
        if (value.indexOf(',') < 0 && value.indexOf('"') < 0
                && value.indexOf('\n') < 0 && value.indexOf('\r') < 0) return value;
        return '"' + value.replace("\"", "\"\"") + '"';
    }
}
