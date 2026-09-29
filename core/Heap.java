package core;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.LinkedHashSet;
import java.util.Set;

/** A small, address-based heap used by the collector simulations. */
public class Heap {
    private static final int DEFAULT_ALLOCATION_RATE_WINDOW = 10;

    private final HeapObject[] slots;
    private final Set<Integer> roots = new LinkedHashSet<>();
    private final List<HeapObject> reclaimedObjects = new ArrayList<>();
    private final Deque<Long> recentAllocationTicks = new ArrayDeque<>();
    private final int allocationRateWindow;
    private int nextId = 1;
    private long currentTick;

    public Heap(int capacity) {
        this(capacity, DEFAULT_ALLOCATION_RATE_WINDOW);
    }

    public Heap(int capacity, int allocationRateWindow) {
        if (capacity < 0) throw new IllegalArgumentException("capacity must not be negative");
        if (allocationRateWindow <= 0)
            throw new IllegalArgumentException("allocationRateWindow must be positive");
        slots = new HeapObject[capacity];
        this.allocationRateWindow = allocationRateWindow;
    }

    public int capacity() { return slots.length; }

    public HeapObject get(int address) {
        checkAddress(address);
        return slots[address];
    }

    public void set(int address, HeapObject object) {
        checkAddress(address);
        slots[address] = object;
        if (object == null) roots.remove(address);
    }

    public boolean isFree(int address) { return get(address) == null; }

    /** Allocates in the first free slot and returns its address, or -1 when full. */
    public int allocate(String name) {
        return allocate(name, 1, "default");
    }

    /** Allocates an instrumented object while preserving the original allocation API. */
    public int allocate(String name, int sizeBytes, String allocationSite) {
        if (sizeBytes <= 0) throw new IllegalArgumentException("sizeBytes must be positive");
        for (int address = 0; address < slots.length; address++) {
            if (slots[address] == null) {
                discardExpiredAllocationTicks();
                recentAllocationTicks.addLast(currentTick);
                double utilization = (usedSlots() + 1.0) / capacity();
                double rate = recentAllocationTicks.size() / (double) allocationRateWindow;
                slots[address] = new HeapObject(nextId++, name, sizeBytes, currentTick,
                        utilization, rate, allocationSite);
                return address;
            }
        }
        return -1;
    }

    public long getCurrentTick() { return currentTick; }

    public void advanceTick(long ticks) {
        if (ticks < 0) throw new IllegalArgumentException("ticks must not be negative");
        currentTick += ticks;
        discardExpiredAllocationTicks();
    }

    public int getAllocationRateWindow() { return allocationRateWindow; }

    public void addReference(int from, int to) {
        if (!hasObject(from) || !hasObject(to)) return;
        if (!slots[from].getReferences().contains(to)) {
            slots[from].addReference(to);
            slots[to].setIncomingReferenceCount(slots[to].getIncomingReferenceCount() + 1);
        }
    }

    public void removeReference(int from, int to) {
        if (!hasObject(from) || !isValidAddress(to)) return;
        if (slots[from].getReferences().contains(to)) {
            slots[from].removeReference(to);
            if (hasObject(to))
                slots[to].setIncomingReferenceCount(
                        Math.max(0, slots[to].getIncomingReferenceCount() - 1));
        }
    }

    /** Kept for compatibility with the original misspelled API. */
    @Deprecated
    public void removeRefrence(int from, int to) { removeReference(from, to); }

    public void addRoot(int address) {
        if (!hasObject(address)) return;
        roots.add(address);
    }

    public void removeRoot(int address) {
        if (!isValidAddress(address)) return;
        roots.remove(address);
    }

    public boolean isRoot(int address) {
        return isValidAddress(address) && roots.contains(address);
    }

    public Set<Integer> getRoots() {
        return Collections.unmodifiableSet(new LinkedHashSet<>(roots));
    }

    /** Replaces roots after a moving collector has calculated new addresses. */
    public void replaceRoots(Set<Integer> newRoots) {
        for (int address : newRoots) requireObject(address);
        roots.clear();
        roots.addAll(newRoots);
    }

    public int usedSlots() {
        int used = 0;
        for (HeapObject object : slots) if (object != null) used++;
        return used;
    }

    public int freeSlots() { return capacity() - usedSlots(); }

    /** Records an object's ground-truth death and removes it from the heap. */
    public void reclaim(int address) {
        requireObject(address);
        HeapObject object = slots[address];
        recordDeath(address);
        slots[address] = null;
        roots.remove(address);
        object.setMarked(false);
    }

    /** Records death without removing a slot, for collectors that move live objects. */
    public void recordDeath(int address) {
        requireObject(address);
        HeapObject object = slots[address];
        if (object.getDeathTick() == null) {
            object.setDeathTick(currentTick);
            reclaimedObjects.add(object);
        }
    }

    public List<HeapObject> getReclaimedObjects() {
        return Collections.unmodifiableList(new ArrayList<>(reclaimedObjects));
    }

    /** Updates per-cycle metadata after a collector has established the final live heap. */
    public void finishCollection() {
        for (HeapObject object : slots) {
            if (object != null) object.survivedCollection();
        }
        recomputeIncomingReferenceCounts();
    }

    public void recomputeIncomingReferenceCounts() {
        for (HeapObject object : slots) {
            if (object != null) object.setIncomingReferenceCount(0);
        }
        for (HeapObject object : slots) {
            if (object == null) continue;
            for (int target : object.getReferences()) {
                if (hasObject(target))
                    slots[target].setIncomingReferenceCount(
                            slots[target].getIncomingReferenceCount() + 1);
            }
        }
    }

    private void discardExpiredAllocationTicks() {
        long firstIncludedTick = currentTick - allocationRateWindow + 1L;
        while (!recentAllocationTicks.isEmpty()
                && recentAllocationTicks.peekFirst() < firstIncludedTick)
            recentAllocationTicks.removeFirst();
    }

    private void requireObject(int address) {
        checkAddress(address);
        if (slots[address] == null) throw new IllegalArgumentException("no object at address " + address);
    }

    private boolean hasObject(int address) {
        return isValidAddress(address) && slots[address] != null;
    }

    private boolean isValidAddress(int address) {
        return address >= 0 && address < slots.length;
    }

    private void checkAddress(int address) {
        if (address < 0 || address >= slots.length)
            throw new IndexOutOfBoundsException("invalid heap address: " + address);
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        for (int address = 0; address < slots.length; address++) {
            result.append(String.format("%2d%s: %s%n", address,
                    roots.contains(address) ? "*" : " ", slots[address]));
        }
        return result.toString();
    }
}
