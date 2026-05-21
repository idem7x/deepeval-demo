// SSE parser for fetch() responses.
//
// Why not the browser's EventSource? Because EventSource only supports GET,
// and our /chat endpoint is a POST. We use fetch() with ReadableStream and
// parse the text/event-stream format manually.
//
// Format:
//     event: <name>\n
//     data: <payload>\n
//     \n
//
// Payloads in this project are always JSON.

import type { StreamEvent } from "./types";

/**
 * Async generator that yields parsed StreamEvents from a fetch Response.
 *
 * Caller is responsible for closing the iteration (early return / break)
 * if it wants to abort midway; doing so cancels the underlying stream.
 */
export async function* parseSSE(response: Response): AsyncGenerator<StreamEvent> {
  if (!response.body) {
    throw new Error("Response has no body");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE messages are separated by a blank line. Drain complete ones
      // out of the buffer; leave any trailing partial for the next read.
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const evt = parseEvent(raw);
        if (evt) yield evt;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseEvent(raw: string): StreamEvent | null {
  let eventName: string | null = null;
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
    // ignore comments, id:, retry:, etc — backend doesn't use them.
  }
  if (!eventName || dataLines.length === 0) return null;
  const payload = dataLines.join("\n");
  try {
    return { type: eventName, data: JSON.parse(payload) } as StreamEvent;
  } catch {
    return null;
  }
}
