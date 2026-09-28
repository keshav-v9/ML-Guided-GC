package gc;

import core.Heap;
import core.HeapObject;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

/** Copies reachable objects in breadth-first order, updating every address. */
public class CopyingCollector implements GarbageCollector {
    @Override
    public void collect(Heap heap) {
        if (heap == null) throw new IllegalArgumentException("heap must not be null");
        Set<Integer> oldRoots = heap.getRoots();
        Map<Integer, Integer> forwarding = new LinkedHashMap<>();
        Queue<Integer> work = new ArrayDeque<>();
        for (int root : oldRoots) forward(heap, root, forwarding, work);
        while (!work.isEmpty()) {
            HeapObject object = heap.get(work.remove());
            for (int reference : object.getReferences())
                forward(heap, reference, forwarding, work);
        }

        HeapObject[] copied = new HeapObject[forwarding.size()];
        for (Map.Entry<Integer, Integer> entry : forwarding.entrySet()) {
            HeapObject object = heap.get(entry.getKey());
            List<Integer> references = new ArrayList<>();
            for (int reference : object.getReferences()) {
                Integer moved = forwarding.get(reference);
                if (moved != null) references.add(moved);
            }
            object.replaceReferences(references);
            object.setMarked(false);
            copied[entry.getValue()] = object;
        }
        for (int address = 0; address < heap.capacity(); address++)
            heap.set(address, address < copied.length ? copied[address] : null);

        Set<Integer> newRoots = new LinkedHashSet<>();
        for (int root : oldRoots) newRoots.add(forwarding.get(root));
        heap.replaceRoots(newRoots);
    }

    private void forward(Heap heap, int address, Map<Integer, Integer> forwarding,
                         Queue<Integer> work) {
        if (address < 0 || address >= heap.capacity() || heap.get(address) == null
                || forwarding.containsKey(address)) return;
        forwarding.put(address, forwarding.size());
        work.add(address);
    }
}
