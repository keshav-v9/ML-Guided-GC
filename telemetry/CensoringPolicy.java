package telemetry;

/** Policy for objects that are still reachable when a workload ends. */
public enum CensoringPolicy {
    EXCLUDE,
    LABEL_LONG_IF_THRESHOLD_EXCEEDED
}
