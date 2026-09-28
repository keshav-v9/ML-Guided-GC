import core.Allocator;
import core.Heap;
import gc.MarkAndCompactCollector;

public class Main {
    public static void main(String[] args) {
        Heap heap = new Heap(8);
        int application = heap.allocate("application");
        int cache = heap.allocate("cache");
        int session = heap.allocate("session");
        heap.allocate("unreachable-1");
        heap.allocate("unreachable-2");
        heap.addRoot(application);
        heap.addReference(application, cache);
        heap.addReference(cache, session);

        System.out.println("Before collection:");
        System.out.print(heap);

        Allocator allocator = new Allocator(heap, new MarkAndCompactCollector());
        // Fill the heap, then one more allocation automatically triggers GC.
        allocator.allocate("temporary-1");
        allocator.allocate("temporary-2");
        allocator.allocate("temporary-3");
        int newAddress = allocator.allocate("allocated-after-GC");

        System.out.println("\nAfter collection and allocation at address " + newAddress + ":");
        System.out.print(heap);
    }
}
