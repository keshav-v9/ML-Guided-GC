package core;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.LinkedHashSet;
import java.util.Set;

/** A small, address-based heap used by the collector simulations. */
public class Heap {
    private static final int DEFAULT_ALLOCATION_RATE_WINDOW = 10;

    private final HeapObject[] slots;
    private final Set<Integer> roots = new LinkedHashSet<>();
    private final List<HeapObject> allocatedObjects = new ArrayList<>();
    private final List<HeapObject> reclaimedObjects = new ArrayList<>();
    private final Set<Integer> reclaimedObjectIds = new HashSet<>();
    private final Deque<Long> recentAllocationTicks = new ArrayDeque<>();
    private final int allocationRateWindow;
    private int nextId = 1;
    private int occupiedSlots;
    private int firstFreeAddress;
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
        HeapObject previous = slots[address];
        slots[address] = object;
        if (previous == null && object != null) occupiedSlots++;
        if (previous != null && object == null) occupiedSlots--;
        if (object == null) {
            roots.remove(address);
            firstFreeAddress = Math.min(firstFreeAddress, address);
        } else if (address == firstFreeAddress) {
            advanceFirstFreeAddress();
        }
    }

    public boolean isFree(int address) { return get(address) == null; }

    /** Allocates in the first free slot and returns its address, or -1 when full. */
    public int allocate(String name) {
        return allocate(name, 1, "default");
    }

    /** Allocates an instrumented object while preserving the original allocation API. */
    public int allocate(String name, int sizeBytes, String allocationSite) {
        if (sizeBytes <= 0) throw new IllegalArgumentException("sizeBytes must be positive");
        advanceFirstFreeAddress();
        if (firstFreeAddress >= slots.length) return -1;
        int address = firstFreeAddress;
        discardExpiredAllocationTicks();
        recentAllocationTicks.addLast(currentTick);
        double utilization = (occupiedSlots + 1.0) / capacity();
        double rate = recentAllocationTicks.size() / (double) allocationRateWindow;
        HeapObject object = new HeapObject(nextId++, name, sizeBytes, currentTick,
                utilization, rate, allocationSite);
        slots[address] = object;
        occupiedSlots++;
        allocatedObjects.add(object);
        firstFreeAddress++;
        advanceFirstFreeAddress();
        return address;
    }

    public long getCurrentTick() { return currentTick; }

    public void advanceTick(long ticks) {
        if (ticks < 0) throw new IllegalArgumentException("ticks must not be negative");
        currentTick += ticks;
        discardExpiredAllocationTicks();
    }

    public int getAllocationRateWindow() { return allocationRateWindow; }

    /** Returns the current address for a stable object id, or -1 if it is not live. */
    public int addressOfObject(int objectId) {
        for (int address = 0; address < slots.length; address++) {
            HeapObject object = slots[address];
            if (object != null && object.getId() == objectId) return address;
        }
        return -1;
    }

    public boolean isReachable(int address) {
        return isValidAddress(address) && reachableAddresses().contains(address);
    }

    /** Fast exact check for an unrooted object with no incoming references. */
    public boolean noteUnreachableIfUnreferenced(int address) {
        if (!isValidAddress(address) || slots[address] == null) return true;
        HeapObject object = slots[address];
        if (object.getDeathTick() != null) return true;
        if (roots.contains(address) || object.getIncomingReferenceCount() > 0) return false;
        object.setDeathTick(currentTick);
        return true;
    }

    /** Records the semantic death tick before a later collection reclaims storage. */
    public void noteUnreachableObjects() {
        Set<Integer> reachable = reachableAddresses();
        for (int address = 0; address < slots.length; address++) {
            HeapObject object = slots[address];
            if (object != null && !reachable.contains(address) && object.getDeathTick() == null)
                object.setDeathTick(currentTick);
        }
    }

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
        return occupiedSlots;
    }

    public int freeSlots() { return capacity() - usedSlots(); }

    /** Records an object's ground-truth death and removes it from the heap. */
    public void reclaim(int address) {
        requireObject(address);
        HeapObject object = slots[address];
        recordDeath(address);
        slots[address] = null;
        occupiedSlots--;
        firstFreeAddress = Math.min(firstFreeAddress, address);
        roots.remove(address);
        object.setMarked(false);
    }

    /** Records death without removing a slot, for collectors that move live objects. */
    public void recordDeath(int address) {
        requireObject(address);
        HeapObject object = slots[address];
        if (object.getDeathTick() == null) object.setDeathTick(currentTick);
        if (reclaimedObjectIds.add(object.getId()))
            reclaimedObjects.add(object);
    }

    public List<HeapObject> getReclaimedObjects() {
        return Collections.unmodifiableList(new ArrayList<>(reclaimedObjects));
    }

    /** Includes both live and reclaimed objects in stable allocation order. */
    public List<HeapObject> getAllocatedObjects() {
        return Collections.unmodifiableList(new ArrayList<>(allocatedObjects));
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

    private void advanceFirstFreeAddress() {
        while (firstFreeAddress < slots.length && slots[firstFreeAddress] != null)
            firstFreeAddress++;
    }

    private Set<Integer> reachableAddresses() {
        Set<Integer> reachable = new HashSet<>();
        Deque<Integer> work = new ArrayDeque<>(roots);
        while (!work.isEmpty()) {
            int address = work.pop();
            if (!isValidAddress(address) || slots[address] == null || !reachable.add(address))
                continue;
            for (int reference : slots[address].getReferences()) work.push(reference);
        }
        return reachable;
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
