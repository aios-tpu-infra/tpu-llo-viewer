# LLO Dump Timeline Prototype

Open `index.html` in a Chromium-based browser, then choose the LLO dump folder with `Import Folder`.

The page looks for these files per kernel group:

- `*-final_bundles.txt`: cycle timeline and instruction blocks
- `*-final_hlo-static-per-bundle-utilization.txt`: hardware utilization rows when present
- `*-post-llo-dependency-graph-optimizations.txt`: discovered for context, not required by the first parser

Dependency arrows are inferred in layers:

- SSA data dependencies from `%id` use-def chains
- exact memory dependencies for static `space:[#allocation + offset]` references
- conservative memory dependencies for dynamic addresses
- DMA/semaphore dependencies from `dma.general`, `dma.done`, and semaphore-like ops when resolvable
- control/order dependencies where the text makes a conservative relationship visible

Use the zoom slider or Ctrl+wheel over the canvas to zoom the cycle axis. Blocks show only color when compact, hardware labels when wider, and short instruction labels when there is enough room.
