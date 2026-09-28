package gc;

import core.Heap;
import core.HeapObject;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Marks live objects and slides them toward the beginning of the heap. */
public class MarkAndCompactCollector implements GarbageCollector {
    @Override
    public void collect(Heap heap) {
        if (heap == null) throw new IllegalArgumentException("heap must not be null");
        MarkAndSweepCollector marker = new MarkAndSweepCollector();
        for (int root : heap.getRoots()) marker.mark(heap, root);

        Map<Integer, Integer> forwarding = new LinkedHashMap<>();
        int destination = 0;
        for (int source = 0; source < heap.capacity(); source++) {
            HeapObject object = heap.get(source);
            if (object != null && object.isMarked()) forwarding.put(source, destination++);
        }

        Set<Integer> oldRoots = heap.getRoots();
        HeapObject[] live = new HeapObject[forwarding.size()];
        for (Map.Entry<Integer, Integer> move : forwarding.entrySet()) {
            HeapObject object = heap.get(move.getKey());
            List<Integer> references = new ArrayList<>();
            for (int oldReference : object.getReferences()) {
                Integer newReference = forwarding.get(oldReference);
                if (newReference != null) references.add(newReference);
            }
            object.replaceReferences(references);
            object.setMarked(false);
            live[move.getValue()] = object;
        }

        for (int address = 0; address < heap.capacity(); address++)
            heap.set(address, address < live.length ? live[address] : null);

        Set<Integer> newRoots = new LinkedHashSet<>();
        for (int oldRoot : oldRoots) {
            Integer newRoot = forwarding.get(oldRoot);
            if (newRoot != null) newRoots.add(newRoot);
        }
        heap.replaceRoots(newRoots);
    }
}
