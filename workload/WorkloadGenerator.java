package workload;

import core.Heap;
import core.HeapObject;
import gc.GarbageCollector;
import gc.MarkAndSweepCollector;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/** Creates deterministic object graphs for demos and benchmarks. */
public final class WorkloadGenerator {
    private static final class RootLease {
        private final int objectId;
        private final long expiresAt;

        private RootLease(int objectId, long expiresAt) {
            this.objectId = objectId;
            this.expiresAt = expiresAt;
        }
    }

    private WorkloadGenerator() {}

    public static Heap generate(int capacity, int objectCount, long seed) {
        if (objectCount < 0 || objectCount > capacity)
            throw new IllegalArgumentException("objectCount must be between 0 and capacity");
        Heap heap = new Heap(capacity);
        Random random = new Random(seed);
        for (int i = 0; i < objectCount; i++) heap.allocate("object-" + (i + 1));
        if (objectCount == 0) return heap;

        int rootCount = Math.max(1, objectCount / 10);
        for (int i = 0; i < rootCount; i++) heap.addRoot(random.nextInt(objectCount));
        for (int from = 0; from < objectCount; from++) {
            int referenceCount = random.nextInt(4);
            for (int j = 0; j < referenceCount; j++)
                heap.addReference(from, random.nextInt(objectCount));
        }
        return heap;
    }

    /** Runs a complete deterministic workload with mark-and-sweep collection. */
    public static WorkloadResult run(WorkloadConfig config) {
        return run(config, new MarkAndSweepCollector());
    }

    /** Runs a workload using the supplied collector without relying on stable addresses. */
    public static WorkloadResult run(WorkloadConfig config, GarbageCollector collector) {
        if (config == null || collector == null)
            throw new IllegalArgumentException("config and collector must not be null");
        Heap heap = new Heap(config.getHeapCapacity());
        Random random = new Random(config.getSeed());
        List<RootLease> roots = new ArrayList<>();
        List<Integer> allocatedIds = new ArrayList<>();
        int collections = 0;

        for (int index = 0; index < config.getObjectCount(); index++) {
            expireRoots(heap, roots);
            if (index > 0 && index % config.getCollectionInterval() == 0) {
                collector.collect(heap);
                collections++;
            }

            if (heap.freeSlots() == 0) {
                collector.collect(heap);
                collections++;
            }
            if (heap.freeSlots() == 0) {
                throw new IllegalStateException("heap capacity and lifetime settings cannot sustain "
                        + config.getObjectCount() + " allocations");
            }

            boolean longLived = isLongLived(config, index, random);
            int lifetime = longLived
                    ? config.getLongLifetimeTicks() + random.nextInt(config.getLongLifetimeTicks() + 1)
                    : 1 + random.nextInt(config.getShortLifetimeTicks());
            int sizeBytes = 16 * (1 + random.nextInt(32));
            String site = allocationSite(config, index);
            int address = heap.allocate(site + "-object-" + (index + 1), sizeBytes, site);
            HeapObject object = heap.get(address);
            heap.addRoot(address);
            roots.add(new RootLease(object.getId(), heap.getCurrentTick() + lifetime));
            addGraphEdges(config.getType(), heap, allocatedIds, object.getId(), index);
            allocatedIds.add(object.getId());
            heap.advanceTick(1);
        }

        // Let short-lived objects expire, while intentionally leaving some long-lived
        // objects right-censored for the telemetry policy to handle explicitly.
        for (int i = 0; i < config.getShortLifetimeTicks(); i++) {
            expireRoots(heap, roots);
            if ((i + 1) % config.getCollectionInterval() == 0) {
                collector.collect(heap);
                collections++;
            }
            heap.advanceTick(1);
        }
        expireRoots(heap, roots);
        collector.collect(heap);
        collections++;
        return new WorkloadResult(config, heap, collections);
    }

    private static boolean isLongLived(WorkloadConfig config, int index, Random random) {
        switch (config.getType()) {
            case SHORT_LIVED:
                return false;
            case LONG_LIVED:
                return true;
            case MIXED:
                return random.nextDouble() < config.getLongLivedFraction();
            case PHASE_CHANGING:
                boolean secondPhase = index >= config.getObjectCount() / 2;
                double probability = secondPhase ? 0.80 : 0.10;
                return random.nextDouble() < probability;
            case GRAPH_STRESS:
                return index % 5 == 0;
            default:
                throw new IllegalStateException("unsupported workload type " + config.getType());
        }
    }

    private static String allocationSite(WorkloadConfig config, int index) {
        if (config.getType() == WorkloadType.PHASE_CHANGING)
            return index < config.getObjectCount() / 2 ? "phase-short" : "phase-long";
        if (config.getType() == WorkloadType.GRAPH_STRESS)
            return "graph-pattern-" + (index % 4);
        return config.getType().name().toLowerCase().replace('_', '-');
    }

    private static void expireRoots(Heap heap, List<RootLease> leases) {
        boolean changed = false;
        boolean needsTrace = false;
        Iterator<RootLease> iterator = leases.iterator();
        while (iterator.hasNext()) {
            RootLease lease = iterator.next();
            if (lease.expiresAt > heap.getCurrentTick()) continue;
            int address = heap.addressOfObject(lease.objectId);
            if (address >= 0) {
                heap.removeRoot(address);
                needsTrace |= !heap.noteUnreachableIfUnreferenced(address);
            }
            iterator.remove();
            changed = true;
        }
        if (changed && needsTrace) heap.noteUnreachableObjects();
    }

    private static void addGraphEdges(WorkloadType type, Heap heap, List<Integer> ids,
                                      int newId, int index) {
        if (ids.isEmpty()) return;
        int newAddress = heap.addressOfObject(newId);
        int previousAddress = heap.addressOfObject(ids.get(ids.size() - 1));
        if (type != WorkloadType.GRAPH_STRESS) return;

        int pattern = index % 8;
        if ((pattern == 1 || pattern == 2) && previousAddress >= 0
                && heap.isRoot(previousAddress)) {
            heap.addReference(previousAddress, newAddress);
            if (pattern == 2) heap.addReference(newAddress, previousAddress);
        } else if (pattern >= 3 && pattern <= 5) {
            int hubIndex = Math.max(0, index - pattern);
            int hubAddress = heap.addressOfObject(ids.get(hubIndex));
            if (hubAddress >= 0 && heap.isRoot(hubAddress))
                heap.addReference(hubAddress, newAddress);
        }
        // Patterns 0, 6, and 7 intentionally create isolated components.
    }
}
