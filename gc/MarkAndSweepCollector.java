package gc;

import core.Heap;
import core.HeapObject;
import java.util.ArrayDeque;
import java.util.Deque;

/** Non-moving mark-and-sweep collection. */
public class MarkAndSweepCollector implements GarbageCollector {
    public void mark(Heap heap, int address) {
        if (heap == null || address < 0 || address >= heap.capacity()) return;
        Deque<Integer> work = new ArrayDeque<>();
        work.push(address);
        while (!work.isEmpty()) {
            int current = work.pop();
            if (current < 0 || current >= heap.capacity()) continue;
            HeapObject object = heap.get(current);
            if (object == null || object.isMarked()) continue;
            object.setMarked(true);
            for (int reference : object.getReferences()) work.push(reference);
        }
    }

    @Override
    public void collect(Heap heap) {
        if (heap == null) throw new IllegalArgumentException("heap must not be null");
        for (int root : heap.getRoots()) mark(heap, root);
        for (int address = 0; address < heap.capacity(); address++) {
            HeapObject object = heap.get(address);
            if (object == null) continue;
            if (!object.isMarked()) heap.set(address, null);
            else object.setMarked(false);
        }
    }
}
