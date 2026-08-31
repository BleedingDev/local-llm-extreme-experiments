#!/usr/bin/env node

let buffer = Buffer.alloc(0);
let note = "controlled fixture note";

const send = (message) => {
  const body = Buffer.from(JSON.stringify(message), "utf8");
  process.stdout.write(`Content-Length: ${body.length}\r\n\r\n`);
  process.stdout.write(body);
};

const parseFrames = () => {
  while (buffer.length > 0) {
    const headerEnd = buffer.indexOf("\r\n\r\n");
    if (headerEnd < 0) {
      return;
    }
    const header = buffer.subarray(0, headerEnd).toString("utf8");
    const lengthLine = header.split("\r\n").find((line) => line.toLowerCase().startsWith("content-length:"));
    const contentLength = Number(lengthLine?.slice("content-length:".length).trim());
    if (!Number.isInteger(contentLength) || contentLength < 0) {
      throw new Error("bad content-length");
    }
    const bodyStart = headerEnd + 4;
    const bodyEnd = bodyStart + contentLength;
    if (buffer.length < bodyEnd) {
      return;
    }
    const raw = buffer.subarray(bodyStart, bodyEnd).toString("utf8");
    buffer = buffer.subarray(bodyEnd);
    void handle(JSON.parse(raw));
  }
};

const tools = [
  {
    name: "read_note",
    title: "Read Note",
    description: "Read the controlled fixture note.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string" },
      },
      required: ["id"],
      additionalProperties: false,
    },
    outputSchema: {
      type: "object",
      properties: {
        id: { type: "string" },
        note: { type: "string" },
      },
    },
    annotations: { readOnlyHint: true },
  },
  {
    name: "write_note",
    title: "Write Note",
    description: "Write the controlled fixture note.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string" },
        note: { type: "string" },
      },
      required: ["id", "note"],
      additionalProperties: false,
    },
    annotations: { destructiveHint: true },
  },
  {
    name: "slow_read",
    title: "Slow Read",
    description: "Read the controlled fixture note after a delay.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string" },
        delayMs: { type: "number" },
      },
      required: ["id", "delayMs"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true },
  },
  {
    name: "fail_read",
    title: "Fail Read",
    description: "Read the controlled fixture note through a failure path.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string" },
      },
      required: ["id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true },
  },
];

const toolResult = (structuredContent) => ({
  content: [{ type: "text", text: JSON.stringify(structuredContent) }],
  structuredContent,
});

const handle = async (message) => {
  if (message.id === undefined || message.id === null) {
    return;
  }

  try {
    if (message.method === "initialize") {
      send({
        jsonrpc: "2.0",
        id: message.id,
        result: {
          protocolVersion: "2024-11-05",
          capabilities: { tools: {} },
          serverInfo: { name: "controlled-mcp-fixture", version: "1.0.0" },
        },
      });
      return;
    }

    if (message.method === "tools/list") {
      send({ jsonrpc: "2.0", id: message.id, result: { tools } });
      return;
    }

    if (message.method === "tools/call") {
      const name = message.params?.name;
      const args = message.params?.arguments ?? {};
      if (name === "read_note") {
        send({ jsonrpc: "2.0", id: message.id, result: toolResult({ id: args.id, note }) });
        return;
      }
      if (name === "write_note") {
        note = args.note;
        send({ jsonrpc: "2.0", id: message.id, result: toolResult({ id: args.id, written: true }) });
        return;
      }
      if (name === "slow_read") {
        await new Promise((resolve) => setTimeout(resolve, args.delayMs));
        send({ jsonrpc: "2.0", id: message.id, result: toolResult({ id: args.id, note }) });
        return;
      }
      if (name === "fail_read") {
        send({
          jsonrpc: "2.0",
          id: message.id,
          error: { code: -32001, message: "controlled fixture failure", data: { id: args.id } },
        });
        return;
      }
    }

    send({
      jsonrpc: "2.0",
      id: message.id,
      error: { code: -32601, message: `Unknown method: ${message.method}` },
    });
  } catch (error) {
    send({
      jsonrpc: "2.0",
      id: message.id,
      error: { code: -32000, message: error instanceof Error ? error.message : "fixture error" },
    });
  }
};

process.stdin.on("data", (chunk) => {
  buffer = Buffer.concat([buffer, chunk]);
  parseFrames();
});

process.stdin.resume();
