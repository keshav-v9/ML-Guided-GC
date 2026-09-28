package core;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/** A small, address-based heap used by the collector simulations. */
public class Heap {
    private final HeapObject[] slots;
    private final Set<Integer> roots = new LinkedHashSet<>();
    private int nextId = 1;

    public Heap(int capacity) {
        if (capacity < 0) throw new IllegalArgumentException("capacity must not be negative");
        slots = new HeapObject[capacity];
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
        for (int address = 0; address < slots.length; address++) {
            if (slots[address] == null) {
                slots[address] = new HeapObject(nextId++, name);
                return address;
            }
        }
        return -1;
    }

    public void addReference(int from, int to) {
        if (!hasObject(from) || !hasObject(to)) return;
        slots[from].addReference(to);
    }

    public void removeReference(int from, int to) {
        if (!hasObject(from) || !isValidAddress(to)) return;
        slots[from].removeReference(to);
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
