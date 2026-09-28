package gc;

import core.Heap;

public interface GarbageCollector {
    void collect(Heap heap);
}
