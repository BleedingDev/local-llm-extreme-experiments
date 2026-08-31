import process from "node:process";
import { loadConfig } from "../src/config";
import { createLlmRouter } from "../src/llm";

const main = async () => {
  const config = loadConfig(process.cwd());
  const router = createLlmRouter(config);

  console.log(JSON.stringify({
    masterAvailable: router.masterAvailable,
    masterModel: config.master.model,
    masterBaseUrl: config.master.baseUrl,
    localModel: config.local.model,
    localBaseUrl: config.local.baseUrl,
  }, null, 2));

  const masterStart = performance.now();
  const masterReply = await router.chatText({
    role: "master",
    maxTokens: 64,
    messages: [
      { role: "system", content: "You are a curt assistant. Reply in <= 8 words." },
      { role: "user", content: "Confirm BAG master wiring is working." },
    ],
  });
  const masterMs = Math.round(performance.now() - masterStart);

  const localStart = performance.now();
  const localReply = await router.chatText({
    role: "local",
    maxTokens: 64,
    messages: [
      { role: "system", content: "You are a curt assistant. Reply in <= 8 words." },
      { role: "user", content: "Confirm BAG local wiring is working." },
    ],
  });
  const localMs = Math.round(performance.now() - localStart);

  console.log(JSON.stringify({
    master: { reply: masterReply, ms: masterMs },
    local: { reply: localReply, ms: localMs },
  }, null, 2));
};

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
