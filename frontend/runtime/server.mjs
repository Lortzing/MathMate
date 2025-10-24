import { createServer } from "node:http";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  LangGraphHttpAgent,
  copilotRuntimeNodeHttpEndpoint,
} from "@copilotkit/runtime";

const agentName = process.env.COPILOTKIT_AGENT_NAME ?? "mathmate";
const agentUrl =
  process.env.COPILOTKIT_AGENT_URL ?? "http://localhost:8000";

const serviceAdapter = new ExperimentalEmptyAdapter();

const runtime = new CopilotRuntime({
  agents: {
    [agentName]: new LangGraphHttpAgent({ url: agentUrl }),
  },
});

const handler = copilotRuntimeNodeHttpEndpoint({
  endpoint: "/mathmate",
  runtime,
  serviceAdapter,
});

const port = Number.parseInt(process.env.PORT ?? "4000", 10);

const server = createServer((req, res) => handler(req, res));

server.listen(port, () => {
  console.log(
    `Copilot runtime listening at http://localhost:${port}/mathmate for agent '${agentName}' -> ${agentUrl}`,
  );
});
